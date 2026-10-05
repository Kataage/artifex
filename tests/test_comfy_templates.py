from __future__ import annotations

from artifex.comfy import WorkflowLoRA, WorkflowPatchRequest, WorkflowTemplateRegistry


def _request(*, with_loras: bool = True) -> WorkflowPatchRequest:
    loras = (
        (
            WorkflowLoRA(name="kanata.safetensors", weight_model=0.8, weight_clip=0.7),
            WorkflowLoRA(name="style.safetensors", weight_model=0.5, weight_clip=0.5),
        )
        if with_loras
        else ()
    )
    return WorkflowPatchRequest(
        positive_prompt="amane_kanata, smile",
        negative_prompt="bad_hands",
        checkpoint="ilxl.safetensors",
        seed=12345,
        width=832,
        height=1216,
        batch_size=2,
        output_prefix="ARTIFEX/pack-1/scene-1",
        loras=loras,
    )


def test_packaged_template_has_stable_identity() -> None:
    template = WorkflowTemplateRegistry.with_packaged_templates().require("ilxl_base_v1")

    assert template.template_id == "ilxl_base_v1"
    assert template.version == 1
    assert template.model_family == "ilxl"


def test_template_patch_updates_only_contract_and_builds_lora_chain() -> None:
    template = WorkflowTemplateRegistry.with_packaged_templates().require("ilxl_base_v1")
    graph = template.patch(_request())

    assert graph["1"]["inputs"]["ckpt_name"] == "ilxl.safetensors"
    assert graph["2"]["inputs"]["text"] == "amane_kanata, smile"
    assert graph["3"]["inputs"]["text"] == "bad_hands"
    assert graph["4"]["inputs"] == {
        "width": 832,
        "height": 1216,
        "batch_size": 2,
    }
    assert graph["5"]["inputs"]["seed"] == 12345
    assert graph["7"]["inputs"]["filename_prefix"] == "ARTIFEX/pack-1/scene-1"

    assert graph["1000"]["class_type"] == "LoraLoader"
    assert graph["1000"]["inputs"]["model"] == ["1", 0]
    assert graph["1000"]["inputs"]["clip"] == ["1", 1]
    assert graph["1001"]["inputs"]["model"] == ["1000", 0]
    assert graph["1001"]["inputs"]["clip"] == ["1000", 1]
    assert graph["5"]["inputs"]["model"] == ["1001", 0]
    assert graph["2"]["inputs"]["clip"] == ["1001", 1]
    assert graph["3"]["inputs"]["clip"] == ["1001", 1]


def test_template_without_loras_keeps_checkpoint_connections() -> None:
    template = WorkflowTemplateRegistry.with_packaged_templates().require("ilxl_base_v1")
    graph = template.patch(_request(with_loras=False))

    assert "1000" not in graph
    assert graph["5"]["inputs"]["model"] == ["1", 0]
    assert graph["2"]["inputs"]["clip"] == ["1", 1]
