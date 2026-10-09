from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.onboarding_pair import (
    PairConfigurationPreview,
    apply_controller_pair,
    discover_controller_pair,
)

NOW = datetime(2026, 10, 9, 11, 0, tzinfo=UTC)
TOKEN = "test-private-token-which-is-long-enough"
ATTEST = "http://192.168.50.8:8190"


def _attestation(*, age_seconds: int = 0, name: str = "illustration.safetensors") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "node_id": "gpu-b",
        "created_at": (NOW - timedelta(seconds=age_seconds)).isoformat(),
        "hostname": "render-box",
        "os": {"system": "Windows"},
        "nvidia_gpus": ({"name": "RTX 3060"},),
        # Do not trust this loopback upstream as PC-A's actual gateway!
        "comfyui_base_url": "http://127.0.0.1:8188",
        "assets": ({
            "label": "production_checkpoint",
            "path": "D:\\AI\\ComfyUI\\models\\checkpoints\\" + name,
            "sha256": "a" * 64,
            "bytes": 123456,
        },),
        "loras": (),
        "inventory_errors": (),
    }


def _client(
    payload: dict[str, Any] | None = None,
    *, status: int = 200,
    gateway_status: int = 200,
) -> tuple[httpx.Client, list[str]]:
    requested: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer " + TOKEN
        if request.url.path == "/v1/attestation":
            assert request.url.params["fresh"] == "1"
            return httpx.Response(status, json=payload or _attestation())
        if request.url.path == "/system_stats":
            assert str(request.url) == "http://192.168.50.8:8191/system_stats"
            return httpx.Response(
                gateway_status,
                json={"system": {"comfyui_version": "0.3.0"}, "devices": []},
            )
        pytest.fail("unexpected URL or GPU-submitting API request")

    return httpx.Client(transport=httpx.MockTransport(handle)), requested


def _discover(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> PairConfigurationPreview:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", TOKEN)
    http, _ = _client(kwargs.pop("payload", None), status=kwargs.pop("status", 200),
                      gateway_status=kwargs.pop("gateway_status", 200))
    with http:
        return discover_controller_pair(ATTEST, now=NOW, client=http, **kwargs)


def test_derives_only_authenticated_pc_b_gateway_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", TOKEN)
    http, requests = _client()
    with http:
        preview = discover_controller_pair(ATTEST, now=NOW, client=http)
    assert preview.node_id == "gpu-b"
    assert preview.renderer_gateway_url == "http://192.168.50.8:8191"
    assert preview.attestation_url == ATTEST
    assert preview.checkpoint_filename == "illustration.safetensors"
    assert preview.checkpoint_sha256 == "a" * 64
    assert preview.gpu_count == 1
    assert preview.authenticated and preview.gateway_reachable
    assert not preview.configuration_written
    assert not preview.services_mutated and not preview.gpu_jobs_submitted
    assert not preview.production_qualified
    assert len(requests) == 2
    assert "127.0.0.1" not in " ".join(requests)
    assert TOKEN not in preview.model_dump_json()


@pytest.mark.parametrize("bad_url", [
    "http://localhost:8190",
    "http://127.0.0.1:8190",
    "http://0.0.0.0:8190",
    "http://user:pw@192.168.50.8:8190",
    "http://192.168.50.8:8190/private",
    "http://192.168.50.8:8190?token=secret",
    "file:///tmp/anything",
    "http://192.168.50.8",
])
def test_lan_url_rejects_insecure_destinations(
    monkeypatch: pytest.MonkeyPatch, bad_url: str,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", TOKEN)
    with pytest.raises(ValueError):
        discover_controller_pair(bad_url)


@pytest.mark.parametrize("scenario", ["missing_token", "short_token", "auth",
                                     "gateway_down", "stale", "future", "unsafe_asset"])
def test_fails_closed_before_writing_controller_config(
    monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    monkeypatch.setenv(
        "ARTIFEX_RENDER_NODE_TOKEN",
        "bad" if scenario == "short_token" else TOKEN,
    )
    if scenario == "missing_token":
        monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    att = _attestation(age_seconds=360 if scenario == "stale" else
                      -120 if scenario == "future" else 0,
                      name="../bad.safetensors" if scenario == "unsafe_asset" else
                      "illustration.safetensors")
    # Unsafe asset is invalid because the resulting basename is not trusted
    if scenario == "unsafe_asset":
        att["assets"][0]["path"] = r"D:\AI\ComfyUI\models\checkpoints\bad|name.safetensors"
    http, _ = _client(att, status=401 if scenario == "auth" else 200,
                      gateway_status=503 if scenario == "gateway_down" else 200)
    with http, pytest.raises((ValueError, httpx.HTTPStatusError)):
        discover_controller_pair(ATTEST, now=NOW, client=http)


def test_does_not_redirect_on_attestation_or_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", TOKEN)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://another-host.example"})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http,
        pytest.raises(httpx.HTTPStatusError),
    ):
        discover_controller_pair(ATTEST, now=NOW, client=http)


def test_explicit_apply_merges_without_overwriting_or_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    preview = _discover(monkeypatch)
    dest = tmp_path / "config" / "local.yaml"
    settings = ArtifexSettings()
    applied = apply_controller_pair(preview, settings, dest)
    assert applied.configuration_written
    assert applied.output_path == str(dest)
    assert not applied.services_mutated and not applied.gpu_jobs_submitted
    assert dest.is_file()
    import yaml

    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert data["render_nodes"]["primary"] == "gpu-b"
    node = data["render_nodes"]["nodes"]["gpu-b"]
    assert node["base_url"] == "http://192.168.50.8:8191"
    assert node["attestation_url"] == ATTEST
    assert data["production"]["checkpoint"] == "illustration.safetensors"
    assert TOKEN not in dest.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="--update"):
        apply_controller_pair(preview, settings, dest)
    data["storage"] = {"database_url": "sqlite:///operator-owned.db"}  # unrelated
    dest.write_text(yaml.safe_dump(data), encoding="utf-8")
    second = apply_controller_pair(preview, settings, dest, update=True)
    assert second.configuration_written
    updated = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert updated["storage"]["database_url"] == "sqlite:///operator-owned.db"


def test_refuses_symlinked_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preview = _discover(monkeypatch)
    outside = tmp_path / "outside.yaml"
    outside.write_text("{}")
    linked = tmp_path / "linked.yaml"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    with pytest.raises(ValueError, match="Symlinked"):
        apply_controller_pair(preview, ArtifexSettings(), linked, update=True)
    assert outside.read_text() == "{}"


def test_cli_preview_does_not_modify_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.onboarding_pair as module

    config = tmp_path / "pc-a.yaml"
    seen: list[str] = []
    preview = PairConfigurationPreview(
        node_id="gpu-b", attestation_url=ATTEST,
        renderer_gateway_url="http://192.168.50.8:8191",
        hostname="render-box", attestation_age_seconds=0.0,
        checkpoint_filename="illustration.safetensors",
        checkpoint_sha256="a" * 64, gpu_count=1,
        gateway_reachable=True, authenticated=True,
    )

    def discover(*args: Any, **kwargs: Any) -> PairConfigurationPreview:
        seen.append("discover")
        return preview

    def apply(*args: Any, **kwargs: Any) -> PairConfigurationPreview:
        pytest.fail("Default preview must not write config")

    monkeypatch.setattr(module, "discover_controller_pair", discover)
    monkeypatch.setattr(module, "apply_controller_pair", apply)
    result = CliRunner().invoke(app, [
        "onboard", "pair-sync", "--attestation-url", ATTEST,
        "--output", str(config), "--json",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["configuration_written"] is False
    assert seen == ["discover"]
    assert not config.exists()


def test_cli_update_requires_explicit_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.onboarding_pair as module

    monkeypatch.setattr(
        module, "discover_controller_pair",
        lambda *args, **kwargs: pytest.fail("No probe without --apply for --update"),
    )
    result = CliRunner().invoke(app, [
        "onboard", "pair-sync", "--attestation-url", ATTEST, "--update",
    ])
    assert result.exit_code == 1
    assert "--update requires --apply" in result.output
