from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import DiscordConfig
from artifex.discord.models import CommandName
from artifex.discord.preflight import effective_channel_permissions, probe_discord


def _config() -> DiscordConfig:
    return DiscordConfig(
        enabled=True,
        guild_id=77,
        channel_id=88,
        allowed_user_ids=(100,),
        token_env="ARTIFEX_DISCORD_TEST_TOKEN",
    )


def _endpoints(
    *,
    history: bool = True,
    send: bool = True,
    commands: bool = True,
    guild: str = "77",
):
    permissions = (1 << 10) | ((1 << 11) if send else 0)
    permissions |= (1 << 16) if history else 0
    return {
        "/users/@me": {"id": "300", "bot": True},
        "/applications/@me": {"id": "301"},
        "/guilds/77/members/300": {"roles": ["79"]},
        "/guilds/77/roles": [
            {"id": "77", "permissions": "0"},
            {"id": "79", "permissions": str(permissions)},
        ],
        "/channels/88": {
            "guild_id": guild,
            "type": 0,
            "permission_overwrites": [],
        },
        "/applications/301/guilds/77/commands": [
            {
                "name": "artifex",
                "type": 1,
                "options": [
                    {"name": item.value, "type": 1} for item in CommandName
                    if commands or item is not CommandName.NOTICE_DISMISS
                ],
            }
        ],
        "/channels/88/messages": [],
    }


def _client(
    endpoints: dict[str, object],
    calls: list[str],
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers["Authorization"] == "Bot secret-test-token"
        calls.append(str(request.url))
        endpoint = request.url.path.removeprefix("/api/v10")
        if endpoint not in endpoints:
            return httpx.Response(404, json={"message": "missing"})
        return httpx.Response(200, json=endpoints[endpoint])

    return httpx.AsyncClient(
        base_url="https://discord.com/api/v10",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_local_probe_is_network_free_and_does_not_claim_live_delivery() -> None:
    result = await probe_discord(_config(), token="secret-test-token")
    assert result["ready"] is True
    assert result["live_verified"] is False
    assert result["requests_sent"] == 0
    assert result["writes_performed"] == 0
    assert "secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_live_probe_uses_only_reads_and_verifies_commands_and_permissions() -> None:
    calls: list[str] = []
    async with _client(_endpoints(), calls) as client:
        result = await probe_discord(
            _config(), live=True, client=client, token="secret-test-token",
        )
    assert result["ready"] is True, result
    assert result["live_verified"] is True
    assert result["writes_performed"] == 0
    assert len(calls) == 7
    assert all("/api/v10/" in call for call in calls)
    assert "secret-test-token" not in json.dumps(result)
    assert result["checks"]["guild_slash_commands_synced"] is True
    assert result["checks"]["read_message_history"] is True
    assert result["checks"]["history_api_accessible"] is True


@pytest.mark.asyncio
async def test_live_probe_fails_closed_without_commands_or_permissions() -> None:
    for attrs in (
        {"commands": False},
        {"send": False},
        {"history": False},
        {"guild": "999"},
    ):
        calls: list[str] = []
        async with _client(_endpoints(**attrs), calls) as client:
            result = await probe_discord(
                _config(), live=True, client=client, token="secret-test-token",
            )
        assert result["ready"] is False, attrs
        assert result["live_verified"] is False
        assert result["writes_performed"] == 0
        assert "secret-test-token" not in json.dumps(result)


@pytest.mark.asyncio
async def test_live_probe_reports_403_without_leaking_error_body_or_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            403,
            json={"message": "Bot secret-test-token is invalid"},
        )

    async with httpx.AsyncClient(
        base_url="https://discord.com/api/v10",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await probe_discord(
            _config(), live=True, client=client, token="secret-test-token",
        )
    assert result["ready"] is False
    assert result["requests_sent"] == 1
    assert "403" in result["problems"][0]
    assert "secret-test-token" not in json.dumps(result)


@pytest.mark.asyncio
async def test_disabled_or_unconfigured_bot_makes_no_network_calls() -> None:
    calls: list[str] = []
    async with _client(_endpoints(), calls) as client:
        result = await probe_discord(
            DiscordConfig(),
            live=True,
            client=client,
            token="secret-test-token",
        )
    assert result["ready"] is False
    assert result["requests_sent"] == 0
    assert calls == []


def test_overwrite_precedence_and_administrator_override() -> None:
    channel = {
        "permission_overwrites": [
            {"id": "77", "type": 0, "deny": str(1 << 10), "allow": "0"},
            {"id": "79", "type": 0, "deny": "0", "allow": str(1 << 10)},
            {"id": "300", "type": 1, "deny": str(1 << 11), "allow": "0"},
        ]
    }
    permissions = effective_channel_permissions(
        guild_id="77", user_id="300", role_ids=["79"],
        roles=[
            {"id": "77", "permissions": "0"},
            {"id": "79", "permissions": str(1 << 11)},
        ],
        channel=channel,
    )
    assert permissions & (1 << 10)
    assert not permissions & (1 << 11)

    admin = effective_channel_permissions(
        guild_id="77", user_id="300", role_ids=["79"],
        roles=[
            {"id": "77", "permissions": "0"},
            {"id": "79", "permissions": str(1 << 3)},
        ],
        channel=channel,
    )
    assert admin & (1 << 11)
    assert admin & (1 << 16)


def test_discord_probe_cli_is_registered_without_attempting_live_access() -> None:
    result = CliRunner().invoke(app, ["discord", "probe", "--help"])
    assert result.exit_code == 0, result.output
    assert "--live" in result.output
