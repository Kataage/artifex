from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from artifex.domain import CharacterProfile
from artifex.loras import LoRAPlan
from artifex.packs import ScenePlan
from artifex.prompts.models import CompiledPrompt


class PromptAdapter(Protocol):
    model_families: frozenset[str]

    def compile(
        self,
        scene: ScenePlan,
        characters: Sequence[CharacterProfile],
        lora_plan: LoRAPlan,
    ) -> CompiledPrompt: ...
