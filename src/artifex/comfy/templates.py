from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.comfy.models import WorkflowPatchRequest, WorkflowRequirements


class PatchPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    input: str


class ConsumerPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    input: str


class LoRAChainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_node: str
    model_output_index: int = 0
    clip_output_index: int = 1
    model_consumers: tuple[ConsumerPatch, ...]
    clip_consumers: tuple[ConsumerPatch, ...]
    start_node_id: int = Field(default=1000, ge=1)


class WorkflowTemplateLike(Protocol):
    template_id: str
    version: int
    model_family: str

    def patch(self, request: WorkflowPatchRequest) -> dict[str, dict[str, Any]]: ...

    def requirements(self, request: WorkflowPatchRequest) -> WorkflowRequirements: ...


class WorkflowTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str
    version: int = Field(ge=1)
    model_family: str
    prompt: dict[str, dict[str, Any]]
    patches: dict[str, PatchPoint]
    lora_chain: LoRAChainSpec | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> WorkflowTemplate:
        required = {
            "positive_prompt",
            "negative_prompt",
            "checkpoint",
            "seed",
            "width",
            "height",
            "batch_size",
            "output_prefix",
        }
        missing = required - set(self.patches)
        if missing:
            raise ValueError(
                "workflow template missing required patches: "
                + ", ".join(sorted(missing))
            )
        for name, point in self.patches.items():
            node = self.prompt.get(point.node)
            if node is None:
                raise ValueError(f"patch {name} references missing node {point.node}")
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or point.input not in inputs:
                raise ValueError(
                    f"patch {name} references missing input "
                    f"{point.node}.{point.input}"
                )
        if self.lora_chain is not None:
            if self.lora_chain.checkpoint_node not in self.prompt:
                raise ValueError("LoRA chain checkpoint node is missing")
            for consumer in (
                *self.lora_chain.model_consumers,
                *self.lora_chain.clip_consumers,
            ):
                node = self.prompt.get(consumer.node)
                if node is None:
                    raise ValueError(
                        f"LoRA chain consumer node is missing: {consumer.node}"
                    )
                inputs = node.get("inputs")
                if not isinstance(inputs, dict) or consumer.input not in inputs:
                    raise ValueError(
                        f"LoRA chain consumer input is missing: "
                        f"{consumer.node}.{consumer.input}"
                    )
        return self

    def patch(self, request: WorkflowPatchRequest) -> dict[str, dict[str, Any]]:
        graph = copy.deepcopy(self.prompt)
        values: dict[str, Any] = {
            "positive_prompt": request.positive_prompt,
            "negative_prompt": request.negative_prompt,
            "checkpoint": request.checkpoint,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
            "batch_size": request.batch_size,
            "output_prefix": request.output_prefix,
        }
        for name, value in values.items():
            point = self.patches[name]
            graph[point.node]["inputs"][point.input] = value

        if request.loras:
            self._patch_lora_chain(graph, request)
        return graph

    def requirements(self, request: WorkflowPatchRequest) -> WorkflowRequirements:
        graph = self.patch(request)
        raw = json.dumps(
            self.prompt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return WorkflowRequirements(
            template_id=self.template_id,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            node_types=tuple(
                sorted(
                    {
                        str(node.get("class_type", ""))
                        for node in graph.values()
                        if node.get("class_type")
                    }
                )
            ),
        )

    def _patch_lora_chain(
        self,
        graph: dict[str, dict[str, Any]],
        request: WorkflowPatchRequest,
    ) -> None:
        spec = self.lora_chain
        if spec is None:
            raise ValueError("workflow template does not support LoRAs")

        previous_model: list[Any] = [spec.checkpoint_node, spec.model_output_index]
        previous_clip: list[Any] = [spec.checkpoint_node, spec.clip_output_index]
        used_node_ids = {int(node_id) for node_id in graph if node_id.isdigit()}
        node_id = spec.start_node_id

        for lora in request.loras:
            while node_id in used_node_ids:
                node_id += 1
            current = str(node_id)
            graph[current] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": previous_model,
                    "clip": previous_clip,
                    "lora_name": lora.name,
                    "strength_model": lora.weight_model,
                    "strength_clip": lora.weight_clip,
                },
            }
            previous_model = [current, 0]
            previous_clip = [current, 1]
            used_node_ids.add(node_id)
            node_id += 1

        for consumer in spec.model_consumers:
            graph[consumer.node]["inputs"][consumer.input] = previous_model
        for consumer in spec.clip_consumers:
            graph[consumer.node]["inputs"][consumer.input] = previous_clip


class WorkflowTemplateRegistry:
    def __init__(self) -> None:
        self._templates: dict[str, WorkflowTemplateLike] = {}

    def register(self, template: WorkflowTemplateLike) -> None:
        if template.template_id in self._templates:
            raise ValueError(
                f"duplicate workflow template id: {template.template_id}"
            )
        self._templates[template.template_id] = template

    def require(self, template_id: str) -> WorkflowTemplateLike:
        try:
            return self._templates[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow template: {template_id}") from exc

    @classmethod
    def with_packaged_templates(cls) -> WorkflowTemplateRegistry:
        registry = cls()
        root = files("artifex.comfy").joinpath("workflow_templates")
        for name in ("ilxl_base_v1.json", "ilxl_repair_v1.json"):
            raw = json.loads(root.joinpath(name).read_text(encoding="utf-8"))
            registry.register(WorkflowTemplate.model_validate(raw))

        from artifex.comfy.production_workflow import IllustMainWorkflowTemplate

        registry.register(IllustMainWorkflowTemplate(repair=False))
        registry.register(IllustMainWorkflowTemplate(repair=True))
        return registry
