from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from artifex.config import load_settings


def _merge(base: dict[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in changes.items():
        old = result.get(key)
        if isinstance(old, dict) and isinstance(value, Mapping):
            result[key] = _merge(old, value)
        else:
            result[key] = value
    return result


def read_override(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    return loaded


def write_override(
    path: Path,
    changes: Mapping[str, Any],
    *,
    update: bool = False,
    force: bool = False,
) -> Path:
    """Validate and atomically write config; never erase unrelated settings.

    Default: refuse overwriting nonidentical configuration.
    --update: recursively merge only supplied keys into the existing config.
    --force: explicitly replace the entire override file.
    """
    if update and force:
        raise ValueError("--update and --force are mutually exclusive")
    target = path.expanduser().resolve(strict=False)
    payload = dict(changes)
    if target.exists():
        existing = read_override(target)
        if update:
            payload = _merge(existing, payload)
        elif existing != payload and not force:
            raise FileExistsError(
                f"configuration already exists with different settings: {target}; "
                "use --update to preserve other values or --force to replace it"
            )
        elif existing == payload:
            return target

    # Validate the full effective model, not only the edited section.
    load_settings(env={}, overrides=payload)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(
                payload, allow_unicode=True, sort_keys=False, default_flow_style=False
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
