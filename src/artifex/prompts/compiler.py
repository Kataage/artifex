from __future__ import annotations

from collections.abc import Iterable

from artifex.characters import CharacterRegistry
from artifex.loras import LoRAPlan
from artifex.packs import ScenePlan
from artifex.prompts.base import PromptAdapter
from artifex.prompts.models import CompiledPrompt


class PromptCompilerError(RuntimeError):
    pass


class PromptCompiler:
    def __init__(
        self,
        characters: CharacterRegistry,
        adapters: Iterable[PromptAdapter],
    ) -> None:
        self._characters = characters
        self._adapters: dict[str, PromptAdapter] = {}
        for adapter in adapters:
            for family in adapter.model_families:
                if family in self._adapters:
                    raise ValueError(f"duplicate prompt adapter for model family: {family}")
                self._adapters[family] = adapter

    def compile(
        self,
        scene: ScenePlan,
        lora_plan: LoRAPlan,
    ) -> CompiledPrompt:
        if tuple(scene.character_ids) != tuple(lora_plan.character_ids):
            raise PromptCompilerError(
                "scene character_ids must exactly match the resolved LoRA plan"
            )

        adapter = self._adapters.get(lora_plan.model_family)
        if adapter is None:
            raise PromptCompilerError(
                f"no prompt adapter for model family: {lora_plan.model_family}"
            )

        characters = tuple(
            self._characters.require(character_id)
            for character_id in scene.character_ids
        )
        for character in characters:
            if character.model_families and lora_plan.model_family not in character.model_families:
                raise PromptCompilerError(
                    f"character {character.id} is incompatible with "
                    f"{lora_plan.model_family}"
                )

        return adapter.compile(scene, characters, lora_plan)
