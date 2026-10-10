"""Fail-closed resolution of ComfyUI output files on a shared filesystem.

ComfyUI history is a remote response, not authority to read arbitrary PC-A
paths. The user-configured filesystem output directory is the only trusted
root. A successful prompt is not a completed local image delivery unless the
real file is accessible, regular and nonempty on the controller.
"""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Iterable

from artifex.comfy.errors import ComfyUIExecutionError, ComfyUIProtocolError
from artifex.comfy.models import ComfyOutput


def _invalid_path() -> ComfyUIProtocolError:
    return ComfyUIProtocolError(
        "ComfyUI history returned an unsafe or out-of-root output path"
    )


def _safe_relative_output(output: ComfyOutput) -> Path:
    filename = output.filename
    subfolder = output.subfolder.replace("\\", "/")
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or "\x00" in subfolder
        or PureWindowsPath(filename).drive
        or PureWindowsPath(subfolder).drive
        or subfolder.startswith("/")
    ):
        raise _invalid_path()
    parts = subfolder.split("/") if subfolder else []
    if any(part in {".", ".."} for part in parts):
        raise _invalid_path()
    return Path(*parts, filename)


def resolve_existing_comfy_outputs(
    output_dir: Path, outputs: Iterable[ComfyOutput],
) -> tuple[Path, ...]:
    """Return trusted existing files or a non-retryable image delivery failure.

    Never submit another GPU prompt just to compensate for a missing shared
    folder: that case needs operator diagnosis, not duplicate generation.
    """
    try:
        root = output_dir.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise OSError("configured filesystem output root is not a directory")
    except (OSError, ValueError) as exc:
        raise ComfyUIExecutionError(
            "ComfyUI image completed, but the configured filesystem output_dir "
            "is inaccessible on PC-A; check the shared folder mapping",
            retryable=False,
        ) from exc

    paths: list[Path] = []
    for output in outputs:
        relative = _safe_relative_output(output)
        candidate = root / relative
        # Forbid symlinks even if they resolve back into root, including
        # subdirectories. A symlink may change between checks and evaluation.
        current = candidate
        while current != root:
            if current.is_symlink():
                raise _invalid_path()
            parent = current.parent
            if parent == current:
                raise _invalid_path()
            current = parent
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise _invalid_path()
            if not resolved.is_file() or resolved.stat().st_size <= 0:
                raise OSError("shared output is not a nonempty regular file")
        except (OSError, ValueError) as exc:
            raise ComfyUIExecutionError(
                "ComfyUI reported completed, but an output image is missing, "
                "empty or unreadable on PC-A; check the shared folder before "
                "resubmitting to avoid duplicate GPU generation",
                retryable=False,
            ) from exc
        paths.append(resolved)
    return tuple(paths)
