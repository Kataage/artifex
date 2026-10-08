from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from artifex.config.models import ArtifexSettings
from artifex.setup import configure_two_pc


def test_configure_two_pc_probes_comfy_and_writes_minimal_overrides(
    tmp_path: Path,
) -> None:
    async_payload = {
        "system": {"comfyui_version": "0.9.0"},
        "devices": [{"name": "RTX 3060"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://192.168.1.50:8188/system_stats")
        return httpx.Response(200, json=async_payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = ArtifexSettings()
    output = tmp_path / "config" / "local.yaml"

    result = configure_two_pc(
        settings,
        comfyui_base_url="http://192.168.1.50:8188/",
        output_path=output,
        render_node_id="rtx3060",
        client=client,
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert result.comfyui_version == "0.9.0"
    assert result.devices == ("RTX 3060",)
    assert result.attestation_url == "http://192.168.1.50:8190"
    assert payload["render_nodes"]["primary"] == "rtx3060"
    assert (
        payload["render_nodes"]["nodes"]["rtx3060"]["base_url"]
        == "http://192.168.1.50:8188"
    )
    assert payload["render_nodes"]["nodes"]["rtx3060"]["output_mode"] == "api"
    assert payload["qualification"]["asset_paths"]["llm_model"].endswith(
        "Spark-X2.5-4B-Heretic-jp-Q8_0.gguf"
    )
    client.close()


def test_configure_two_pc_retry_reuses_identical_config(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"system": {}, "devices": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    output = tmp_path / "local.yaml"
    settings = ArtifexSettings()
    first = configure_two_pc(
        settings,
        comfyui_base_url="http://render.test:8188",
        output_path=output,
        client=client,
    )
    initial_bytes = output.read_bytes()
    second = configure_two_pc(
        settings,
        comfyui_base_url="http://render.test:8188",
        output_path=output,
        client=client,
    )
    assert first == second
    assert output.read_bytes() == initial_bytes
    client.close()


def test_configure_two_pc_refuses_to_replace_config_without_force(
    tmp_path: Path,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"system": {}, "devices": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    output = tmp_path / "local.yaml"
    output.write_text("existing: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        configure_two_pc(
            ArtifexSettings(),
            comfyui_base_url="http://render.test:8188",
            output_path=output,
            client=client,
        )
    client.close()


@pytest.mark.parametrize(
    "url",
    (
        "render.test:8188",
        "ftp://render.test",
        "http://render.test:8188/path",
        "http://render.test:8188?x=1",
    ),
)
def test_configure_two_pc_rejects_ambiguous_comfy_urls(
    tmp_path: Path,
    url: str,
) -> None:
    with pytest.raises(ValueError, match="ComfyUI URL"):
        configure_two_pc(
            ArtifexSettings(),
            comfyui_base_url=url,
            output_path=tmp_path / "local.yaml",
        )
