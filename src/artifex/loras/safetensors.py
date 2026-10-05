from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class SafeTensorMetadataError(ValueError):
    pass


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_metadata(
    path: Path,
    *,
    max_header_bytes: int,
) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafeTensorMetadataError(f"invalid safetensors header: {path}")
        header_length = int.from_bytes(prefix, byteorder="little", signed=False)
        if header_length <= 0 or header_length > max_header_bytes:
            raise SafeTensorMetadataError(
                f"safetensors header length out of range: {header_length}"
            )
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise SafeTensorMetadataError(f"truncated safetensors header: {path}")

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeTensorMetadataError(f"invalid safetensors JSON header: {path}") from exc

    if not isinstance(header, dict):
        raise SafeTensorMetadataError(f"safetensors header must be an object: {path}")
    metadata = header.get("__metadata__", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise SafeTensorMetadataError(f"invalid __metadata__ object: {path}")

    cleaned: dict[str, Any] = {}
    for index, (key, value) in enumerate(metadata.items()):
        if index >= 256:
            break
        name = str(key)[:256]
        if isinstance(value, str):
            cleaned[name] = value[:16384]
        elif isinstance(value, (int, float, bool)) or value is None:
            cleaned[name] = value
        else:
            cleaned[name] = str(value)[:16384]
    return cleaned
