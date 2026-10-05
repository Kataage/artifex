from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from artifex.config.models import ArtifexSettings

_ENV_PREFIX = "ARTIFEX_"
_KNOWN_SECTIONS = {
    "AGENT",
    "PLANNER",
    "PRODUCTION",
    "LLM",
    "COMFYUI",
    "DISCORD",
    "RIGHTS",
    "STORAGE",
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _load_yaml_text(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError("configuration root must be a mapping")
    return dict(loaded)


def _load_yaml_path(path: Path) -> dict[str, Any]:
    return _load_yaml_text(path.read_text(encoding="utf-8"))


def _parse_env_value(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, dict):
        return value
    if isinstance(parsed, list):
        if any(isinstance(item, (dict, list)) for item in parsed):
            return value
        return parsed
    return parsed


def _env_path(name: str) -> list[str] | None:
    if not name.startswith(_ENV_PREFIX):
        return None
    suffix = name[len(_ENV_PREFIX) :]
    if suffix == "DISCORD_TOKEN":
        return None
    if "__" in suffix:
        return [part.lower() for part in suffix.split("__") if part]

    for section in _KNOWN_SECTIONS:
        prefix = f"{section}_"
        if suffix.startswith(prefix):
            return [section.lower(), suffix[len(prefix) :].lower()]
    return [suffix.lower()]


def _environment_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for name, raw_value in env.items():
        path = _env_path(name)
        if not path:
            continue
        cursor = root
        for part in path[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError(f"environment configuration collision at {name}")
            cursor = child
        cursor[path[-1]] = _parse_env_value(raw_value)
    return root


def load_settings(
    *,
    user_config: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ArtifexSettings:
    """Load settings using default -> user YAML -> environment -> explicit overrides."""

    default_text = files("artifex.config").joinpath("default.yaml").read_text(encoding="utf-8")
    merged = _load_yaml_text(default_text)

    if user_config is not None:
        merged = _deep_merge(merged, _load_yaml_path(user_config))

    merged = _deep_merge(merged, _environment_overrides(env if env is not None else os.environ))

    if overrides is not None:
        merged = _deep_merge(merged, overrides)

    return ArtifexSettings.model_validate(merged)
