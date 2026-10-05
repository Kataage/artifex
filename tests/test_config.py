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
        }
    )
    assert settings.planner.candidate_count == 12
    assert settings.discord.enabled is True


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
