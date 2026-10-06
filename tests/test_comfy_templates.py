from __future__ import annotations

from artifex.comfy import (
    IllustMainWorkflowTemplate,
    WorkflowLoRA,
    WorkflowPatchRequest,
    WorkflowTemplateRegistry,
)


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



def test_real_illust_main_source_has_stable_production_identity() -> None:
    template = IllustMainWorkflowTemplate(repair=False)
    info = template.source_info()

    assert template.template_id == "illust_main_v1"
    assert template.version == 1
    assert template.model_family == "ilxl"
    assert info.workflow_id == "5535dc36-85cd-486e-9f9a-a9c1d403c415"
    assert info.node_count == 418
    assert len(template.source_sha256) == 64


def test_real_production_patch_disables_manual_and_unused_model_branches() -> None:
    template = IllustMainWorkflowTemplate(repair=False)
    graph = template.patch(_request())

    assert graph["15"]["inputs"]["ckpt_name"] == "ilxl.safetensors"
    assert "16" not in graph
    assert "17" not in graph
    assert graph["31"]["inputs"]["ckpt_name"] == "anime-refiner-beta1.1.safetensors"
    assert "556" not in graph
    assert "76" not in graph
    assert "77" not in graph

    assert graph["70"]["inputs"]["prompt"] == "amane_kanata, smile"
    assert graph["72"]["inputs"]["prompt"] == "bad_hands"
    assert graph["73"]["inputs"]["part1"] == ""
    assert graph["73"]["inputs"]["part2"] == ""
    assert graph["73"]["inputs"]["part4"] == ""

    assert "81" not in graph
    assert graph["900001"]["class_type"] == "EmptyLatentImage"
    assert graph["900001"]["inputs"] == {
        "width": 832,
        "height": 1216,
        "batch_size": 2,
    }

    assert graph["152"]["inputs"]["noise_seed"] == 12345
    assert graph["175"]["inputs"]["noise_seed"] == 12345
    assert graph["153"]["inputs"]["cfg"] == 5.5
    assert graph["154"]["inputs"]["sampler_name"] == "dpmpp_3m_sde_gpu"
    assert graph["156"]["inputs"]["scheduler"] == "karras"
    assert graph["156"]["inputs"]["steps"] == 48
    assert graph["63"]["inputs"]["filename_prefix"] == "ARTIFEX/pack-1/scene-1"


def test_validated_character_loras_are_injected_after_operator_static_stack() -> None:
    graph = IllustMainWorkflowTemplate(repair=False).patch(_request())

    assert graph["900010"]["class_type"] == "LoraLoader"
    assert graph["900010"]["inputs"]["model"] == ["53", 0]
    assert graph["900010"]["inputs"]["clip"] == ["53", 1]
    assert graph["900010"]["inputs"]["lora_name"] == "kanata.safetensors"
    assert graph["900011"]["inputs"]["model"] == ["900010", 0]
    assert graph["900011"]["inputs"]["clip"] == ["900010", 1]
    assert graph["900011"]["inputs"]["lora_name"] == "style.safetensors"

    assert graph["74"]["inputs"]["text"] == ""
    assert graph["74"]["inputs"]["loras"] == []
    assert graph["75"]["inputs"]["rules_text"] == ""


def test_repair_template_enables_real_body_and_clothes_detail_stages() -> None:
    base = IllustMainWorkflowTemplate(repair=False).patch(
        _request(with_loras=False)
    )
    repair = IllustMainWorkflowTemplate(repair=True).patch(
        _request(with_loras=False)
    )

    assert "441" not in base
    assert "444" not in base
    assert repair["441"]["class_type"] == "SEGSDetailer"
    assert repair["444"]["class_type"] == "SEGSDetailer"
    assert len(repair) > len(base)


def test_production_requirements_exclude_unused_checkpoints_and_cover_real_assets() -> None:
    requirements = IllustMainWorkflowTemplate(repair=False).requirements(
        _request()
    )
    values = {item.value for item in requirements.assets}

    assert "ilxl.safetensors" in values
    assert "anime-refiner-beta1.1.safetensors" in values
    assert "pppanimixVAE_ilxl.safetensors" in values
    assert "4xRealisticrescaler_100000G.pt" in values
    assert "kanata.safetensors" in values
    assert "waiIllustriousSDXL_v170.safetensors" not in values
    assert "kirazuriLazuliNoobaiV_v21.safetensors" not in values
    assert "UltralyticsDetectorProvider" in requirements.node_types
    assert "SEGSDetailer" in requirements.node_types
    assert "LoraLoader" in requirements.node_types


def test_real_workflow_patch_is_deterministic() -> None:
    template = IllustMainWorkflowTemplate(repair=False)
    first = template.patch(_request())
    second = template.patch(_request())

    assert first == second
    assert {
        node_id: node["class_type"]
        for node_id, node in first.items()
    } == {
        node_id: node["class_type"]
        for node_id, node in second.items()
    }
