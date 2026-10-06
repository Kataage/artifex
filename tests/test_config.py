from __future__ import annotations

from pathlib import Path

import pytest

from artifex.config import load_settings


def test_config_precedence(tmp_path: Path) -> None:
    user = tmp_path / "config.yaml"
    user.write_text(
        """
production:
  retry_limit: 8
planner:
  candidate_count: 4
""",
        encoding="utf-8",
    )

    settings = load_settings(
        user_config=user,
        env={
            "ARTIFEX_PRODUCTION_RETRY_LIMIT": "5",
            "ARTIFEX_LLM_BASE_URL": "http://llm.test:9000",
        },
        overrides={"production": {"retry_limit": 2}},
    )

    assert settings.planner.candidate_count == 4
    assert settings.production.retry_limit == 2
    assert settings.llm.base_url == "http://llm.test:9000"


def test_nested_environment_override() -> None:
    settings = load_settings(
        env={
            "ARTIFEX_PLANNER__CANDIDATE_COUNT": "12",
            "ARTIFEX_DISCORD__ENABLED": "true",
            "ARTIFEX_DISCORD__ALLOWED_USER_IDS": "[123]",
        }
    )
    assert settings.planner.candidate_count == 12
    assert settings.discord.enabled is True
    assert settings.discord.allowed_user_ids == (123,)


def test_planner_mix_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        load_settings(
            env={},
            overrides={
                "planner": {
                    "mix": {
                        "evergreen": 1.0,
                        "trend": 1.0,
                        "seasonal": 0.0,
                        "exploration": 0.0,
                    }
                }
            },
        )


def test_primary_render_node_overlays_legacy_comfy_settings() -> None:
    settings = load_settings(
        env={},
        overrides={
            "render_nodes": {
                "primary": "gpu-box",
                "nodes": {
                    "gpu-box": {
                        "base_url": "http://192.168.1.50:8188",
                        "output_mode": "api",
                        "download_dir": "data/remote-cache",
                        "attestation_url": "http://192.168.1.50:8190",
                    }
                },
            }
        },
    )

    assert settings.comfyui.base_url == "http://192.168.1.50:8188"
    assert settings.comfyui.output_mode == "api"
    assert settings.comfyui.render_node_id == "gpu-box"
    assert settings.comfyui.download_dir == Path("data/remote-cache")


def test_default_llm_bootstrap_profile_is_replaceable() -> None:
    settings = load_settings(env={})

    selected = settings.llm.bootstrap.selected()
    assert settings.llm.bootstrap.profile == "spark-x2.5-4b-heretic-jp-q8_0"
    assert selected.filename == "Spark-X2.5-4B-Heretic-jp-Q8_0.gguf"

    custom = load_settings(
        env={},
        overrides={
            "llm": {
                "bootstrap": {
                    "profile": "custom",
                    "profiles": {
                        "custom": {
                            "source": "local",
                            "path": "D:/Models/custom.gguf",
                        }
                    },
                }
            }
        },
    )
    assert custom.llm.bootstrap.selected().source == "local"
    assert custom.llm.bootstrap.model_path() == Path("D:/Models/custom.gguf")
