from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifex.discord.bot import find_completion_in_history
from artifex.discord.notifications import completion_marker


class FakeChannel:
    def __init__(self, messages: list[object]) -> None:
        self.messages = messages
        self.limits: list[int] = []

    def history(self, *, limit: int):
        self.limits.append(limit)

        async def records():
            for message in self.messages[:limit]:
                yield message

        return records()


def _msg(
    *,
    mid: int,
    user: int,
    pack_id: str,
    heading: str = "**Artifex Pack completed**",
    marker: str | None = None,
):
    suffix = completion_marker(pack_id) if marker is None else marker
    return SimpleNamespace(
        id=mid,
        author=SimpleNamespace(id=user),
        content=f"{heading}\nPack: {pack_id}\n{suffix}",
    )


@pytest.mark.asyncio
async def test_reconcile_only_verifies_same_bots_exact_completion_marker() -> None:
    channel = FakeChannel(
        [
            _msg(mid=1, user=7, pack_id="pack-1"),  # other bot/user
            _msg(mid=2, user=9, pack_id="pack-10"),  # different Pack
            _msg(
                mid=3, user=9, pack_id="pack-1",
                heading="**Some other report**",
            ),
            _msg(
                mid=4, user=9, pack_id="pack-1",
                marker="Artifex-Completion-ID: pack-1-extra",
            ),
            _msg(mid=5, user=9, pack_id="pack-1"),
        ]
    )
    result = await find_completion_in_history(
        channel, pack_id="pack-1", bot_user_id=9,
    )
    assert result == "5"
    assert channel.limits == [200]


@pytest.mark.asyncio
async def test_reconcile_history_absence_is_not_proof_message_never_arrived() -> None:
    channel = FakeChannel(
        [_msg(mid=1, user=7, pack_id="pack-1")]
        + [_msg(mid=i, user=9, pack_id="other") for i in range(2, 6)]
    )
    assert await find_completion_in_history(
        channel, pack_id="pack-1", bot_user_id=9, limit=3,
    ) is None
    assert channel.limits == [3]


@pytest.mark.asyncio
async def test_reconcile_propagates_history_permission_errors() -> None:
    class DeniedChannel:
        def history(self, *, limit: int):
            async def records():
                raise PermissionError("missing read_message_history")
                yield None

            return records()

    with pytest.raises(PermissionError, match="read_message_history"):
        await find_completion_in_history(
            DeniedChannel(), pack_id="pack-1", bot_user_id=9,
        )
