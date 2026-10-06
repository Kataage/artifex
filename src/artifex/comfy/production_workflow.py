from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from artifex.comfy.models import (
    WorkflowAssetRequirement,
    WorkflowPatchRequest,
    WorkflowRequirements,
)
from artifex.comfy.source_workflow import (
    SourceWorkflowInfo,
    UiWorkflowCompiler,
    load_illust_main_source,
)


def _replace_reference(
    graph: dict[str, dict[str, Any]],
    source_node: str,
    source_output: int,
    replacement: Any,
) -> None:
    target = [source_node, source_output]
    for node in graph.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for name, value in tuple(inputs.items()):
            if value == target:
                inputs[name] = replacement


def _normalize_asset_name(value: str) -> str:
    return Path(value.replace("\\", "/")).stem.casefold()


class IllustMainWorkflowTemplate:
    """Artifex-safe projection of the operator's real illust Main workflow.

    Runtime changes are addressed by stable Artifex role names stored in the
    packaged UI source. The original graph topology is never LLM-generated and
    is not mutated outside this explicit contract.
    """

    version = 1
    model_family = "ilxl"
    _ASPECT_NODE_ID = "81"
    _EMPTY_LATENT_NODE_ID = "900001"
    _LORA_NODE_START = 900010

    _SEED_ROLES = (
        "base_seed",
        "refiner_seed",
        "upscale_pass_1",
        "upscale_pass_2",
        "detailer_hair",
        "detailer_hand",
        "detailer_face",
        "detailer_foot",
        "detailer_eyes",
        "detailer_nipple",
        "detailer_body",
        "detailer_clothes",
    )

    def __init__(self, *, repair: bool) -> None:
        self.repair = repair
        self.template_id = (
            "illust_main_repair_v1" if repair else "illust_main_v1"
        )
        pristine = UiWorkflowCompiler(load_illust_main_source())
        self.source_sha256 = pristine.source_sha256()

    def patch(self, request: WorkflowPatchRequest) -> dict[str, dict[str, Any]]:
        compiler = UiWorkflowCompiler(load_illust_main_source())

        compiler.set_mode_for_role("context_primary", 0)
        compiler.set_mode_for_role("context_wai", UiWorkflowCompiler.MODE_NEVER)
        compiler.set_mode_for_role("context_holo", UiWorkflowCompiler.MODE_NEVER)
        compiler.set_mode_for_role("context_merge", UiWorkflowCompiler.MODE_NEVER)

        compiler.set_mode_for_role(
            "interactive_image_filter",
            UiWorkflowCompiler.MODE_BYPASS,
        )

        detail_mode = 0 if self.repair else UiWorkflowCompiler.MODE_BYPASS
        compiler.set_mode_for_role("detailer_body", detail_mode)
        compiler.set_mode_for_role("detailer_clothes", detail_mode)
        compiler.set_mode_for_role("repair_body_paste", detail_mode)
        compiler.set_mode_for_role("repair_clothes_paste", detail_mode)

        compiler.set_widget_for_role(
            "base_checkpoint",
            "ckpt_name",
            request.checkpoint,
        )
        compiler.set_widget_for_role(
            "refiner_checkpoint",
            "ckpt_name",
            request.refiner_checkpoint,
        )
        compiler.set_widget_for_role("vae_loader", "vae_name", request.vae)
        compiler.set_widget_for_role(
            "upscale_model",
            "model_name",
            request.upscale_model,
        )
        compiler.set_widget_for_role(
            "positive_prompt",
            "prompt",
            request.positive_prompt,
        )
        compiler.set_widget_for_role(
            "negative_prompt",
            "prompt",
            request.negative_prompt,
        )

        for part in ("part1", "part2", "part3", "part4"):
            compiler.set_widget_for_role("legacy_positive_parts", part, "")
        compiler.set_input_link_for_role(
            "positive_prompt_parts_mux",
            "part3",
            None,
        )
        compiler.set_input_link_for_role(
            "positive_prompt_parts_mux",
            "part4",
            None,
        )

        compiler.set_widget_for_role("runtime_lora_stack", "text", "")
        compiler.set_widget_for_role("runtime_lora_stack", "loras", [])
        compiler.set_widget_for_role("runtime_lora_lbw", "rules_text", "")
        compiler.set_widget_for_role(
            "runtime_lora_lbw",
            "fill_missing_with_off",
            False,
        )
        compiler.set_widget_for_role(
            "runtime_lora_lbw",
            "remove_missing_from_result",
            False,
        )

        for role in self._SEED_ROLES:
            input_name = (
                "seed"
                if role.startswith(("upscale", "detailer"))
                else "noise_seed"
            )
            compiler.set_widget_for_role(role, input_name, request.seed)

        compiler.set_widget_for_role("base_cfg", "cfg", request.base_cfg)
        compiler.set_widget_for_role(
            "base_sampler",
            "sampler_name",
            request.base_sampler,
        )
        compiler.set_widget_for_role(
            "base_scheduler",
            "scheduler",
            request.base_scheduler,
        )
        compiler.set_widget_for_role(
            "base_scheduler",
            "steps",
            request.base_steps,
        )
        compiler.set_widget_for_role(
            "base_scheduler",
            "denoise",
            request.base_denoise,
        )

        compiler.set_widget_for_role("refiner_cfg", "cfg", request.refiner_cfg)
        compiler.set_widget_for_role(
            "refiner_sampler",
            "sampler_name",
            request.refiner_sampler,
        )
        compiler.set_widget_for_role(
            "refiner_scheduler",
            "scheduler",
            request.refiner_scheduler,
        )
        compiler.set_widget_for_role(
            "refiner_scheduler",
            "steps",
            request.refiner_steps,
        )
        compiler.set_widget_for_role(
            "refiner_scheduler",
            "denoise",
            request.refiner_denoise,
        )

        for role in ("upscale_pass_1", "upscale_pass_2"):
            compiler.set_widget_for_role(role, "steps", request.upscale_steps)
            compiler.set_widget_for_role(role, "cfg", request.upscale_cfg)
            compiler.set_widget_for_role(
                role,
                "sampler_name",
                request.upscale_sampler,
            )
            compiler.set_widget_for_role(
                role,
                "scheduler",
                request.upscale_scheduler,
            )
            compiler.set_widget_for_role(
                role,
                "denoise",
                request.upscale_denoise,
            )

        compiler.set_widget_for_role(
            "save_image",
            "filename_prefix",
            request.output_prefix,
        )

        graph, _ = compiler.compile_role("save_image")
        self._replace_aspect_ratio(graph, request)
        self._inject_runtime_loras(
            graph,
            request,
            static_apply_node=str(
                compiler.node_id_for_role("static_lora_apply")
            ),
        )
        return graph

    def source_info(self) -> SourceWorkflowInfo:
        compiler = UiWorkflowCompiler(load_illust_main_source())
        _, info = compiler.compile_role("save_image")
        return info

    def requirements(self, request: WorkflowPatchRequest) -> WorkflowRequirements:
        graph = self.patch(request)
        node_types = tuple(
            sorted(
                {
                    str(node.get("class_type", ""))
                    for node in graph.values()
                    if node.get("class_type")
                }
            )
        )
        return WorkflowRequirements(
            template_id=self.template_id,
            source_sha256=self.source_sha256,
            node_types=node_types,
            assets=tuple(self._asset_requirements(graph)),
        )

    @classmethod
    def _replace_aspect_ratio(
        cls,
        graph: dict[str, dict[str, Any]],
        request: WorkflowPatchRequest,
    ) -> None:
        _replace_reference(graph, cls._ASPECT_NODE_ID, 0, request.width)
        _replace_reference(graph, cls._ASPECT_NODE_ID, 1, request.height)
        _replace_reference(graph, cls._ASPECT_NODE_ID, 2, 1.0)
        _replace_reference(graph, cls._ASPECT_NODE_ID, 3, request.batch_size)
        _replace_reference(
            graph,
            cls._ASPECT_NODE_ID,
            4,
            [cls._EMPTY_LATENT_NODE_ID, 0],
        )
        graph.pop(cls._ASPECT_NODE_ID, None)
        graph[cls._EMPTY_LATENT_NODE_ID] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": request.width,
                "height": request.height,
                "batch_size": request.batch_size,
            },
            "_meta": {"title": "Artifex runtime latent"},
        }

    @classmethod
    def _inject_runtime_loras(
        cls,
        graph: dict[str, dict[str, Any]],
        request: WorkflowPatchRequest,
        *,
        static_apply_node: str,
    ) -> None:
        if not request.loras:
            return

        previous_model: list[Any] = [static_apply_node, 0]
        previous_clip: list[Any] = [static_apply_node, 1]
        inserted: list[tuple[str, dict[str, Any]]] = []
        for offset, lora in enumerate(request.loras):
            node_id = str(cls._LORA_NODE_START + offset)
            node = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": previous_model,
                    "clip": previous_clip,
                    "lora_name": lora.name,
                    "strength_model": lora.weight_model,
                    "strength_clip": lora.weight_clip,
                },
                "_meta": {"title": f"Artifex LoRA: {lora.name}"},
            }
            inserted.append((node_id, node))
            previous_model = [node_id, 0]
            previous_clip = [node_id, 1]

        _replace_reference(graph, static_apply_node, 0, previous_model)
        _replace_reference(graph, static_apply_node, 1, previous_clip)
        for node_id, node in inserted:
            graph[node_id] = node

    @staticmethod
    def _asset_requirements(
        graph: dict[str, dict[str, Any]],
    ) -> Iterable[WorkflowAssetRequirement]:
        seen: set[tuple[str, str, str]] = set()

        def add(
            *,
            label: str,
            node_class: str,
            input_name: str,
            value: object,
        ) -> tuple[WorkflowAssetRequirement, ...]:
            if not isinstance(value, str) or not value.strip():
                return ()
            key = (node_class, input_name, value)
            if key in seen:
                return ()
            seen.add(key)
            return (
                WorkflowAssetRequirement(
                    label=label,
                    node_class=node_class,
                    input_name=input_name,
                    value=value,
                ),
            )

        for node in graph.values():
            class_type = str(node.get("class_type", ""))
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            if class_type == "CheckpointLoaderSimple":
                yield from add(
                    label="checkpoint",
                    node_class=class_type,
                    input_name="ckpt_name",
                    value=inputs.get("ckpt_name"),
                )
            elif class_type == "VAELoader":
                yield from add(
                    label="vae",
                    node_class=class_type,
                    input_name="vae_name",
                    value=inputs.get("vae_name"),
                )
            elif class_type == "UpscaleModelLoader":
                yield from add(
                    label="upscale_model",
                    node_class=class_type,
                    input_name="model_name",
                    value=inputs.get("model_name"),
                )
            elif class_type == "UltralyticsDetectorProvider":
                yield from add(
                    label="detector_model",
                    node_class=class_type,
                    input_name="model_name",
                    value=inputs.get("model_name"),
                )
            elif class_type == "LoraLoader":
                yield from add(
                    label="lora",
                    node_class=class_type,
                    input_name="lora_name",
                    value=inputs.get("lora_name"),
                )
            elif class_type == "Lora Stacker (LoraManager)":
                raw_loras = inputs.get("loras")
                if isinstance(raw_loras, list):
                    for raw in raw_loras:
                        if not isinstance(raw, dict) or raw.get("active") is not True:
                            continue
                        try:
                            strength = float(raw.get("strength", 0))
                        except (TypeError, ValueError):
                            strength = 0.0
                        if strength == 0:
                            continue
                        yield from add(
                            label="static_lora",
                            node_class="LoraLoader",
                            input_name="lora_name",
                            value=raw.get("name"),
                        )

    @staticmethod
    def asset_name_matches(configured: str, available: str) -> bool:
        left = _normalize_asset_name(configured)
        right = _normalize_asset_name(available)
        return left == right or configured.casefold() == available.casefold()
