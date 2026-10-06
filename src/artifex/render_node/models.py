from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RenderNodeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RenderAssetDigest(RenderNodeModel):
    label: str = Field(min_length=1)
    path: str
    sha256: str = Field(min_length=64, max_length=64)
    bytes: int = Field(ge=0)
    file_count: int = Field(default=1, ge=1)


class RenderLoRAInventoryItem(RenderNodeModel):
    name: str = Field(min_length=1)
    path: str
    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderNodeAttestation(RenderNodeModel):
    schema_version: int = 1
    node_id: str = Field(min_length=1)
    created_at: datetime
    hostname: str
    os: dict[str, str]
    nvidia_gpus: tuple[dict[str, object], ...] = ()
    comfyui_base_url: str
    assets: tuple[RenderAssetDigest, ...] = ()
    loras: tuple[RenderLoRAInventoryItem, ...] = ()
    inventory_errors: tuple[dict[str, str], ...] = ()
