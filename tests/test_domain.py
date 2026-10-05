from __future__ import annotations

import pytest
from pydantic import ValidationError

from artifex.domain import CharacterProfile, LoRAPolicy


def test_character_profile_is_strict_and_bounded() -> None:
    profile = CharacterProfile(
        id="example",
        display_name="Example",
        namespace="test",
        lora_policy=LoRAPolicy.REQUIRED,
        readiness=0.75,
    )
    assert profile.lora_policy is LoRAPolicy.REQUIRED

    with pytest.raises(ValidationError):
        CharacterProfile(
            id="bad",
            display_name="Bad",
            namespace="test",
            readiness=1.5,
        )
