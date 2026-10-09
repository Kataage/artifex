from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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

class RendererOwnerAuditCheck(RenderNodeModel):
    status: Literal["pass", "fail", "unknown"]
    reason: str


class RendererOwnerAudit(RenderNodeModel):
    """Point-in-time PC-B observation, not an entitlement to restart/render."""

    schema_version: Literal[1]
    captured_utc: datetime
    status: Literal["observed_stable", "blocked", "inconclusive", "unsupported"]
    checks: dict[str, RendererOwnerAuditCheck]
    actual_listener_pid: int | None = Field(default=None, ge=1)
    actual_process_started_utc: str | None = None
    launcher_pid: int | None = Field(default=None, ge=1)
    receipt_schema: int | None = None
    scheduler_state: str | None = None
    process_observation_verified: bool
    restart_authorized: Literal[False]
    child_survival_qualified: Literal[False]
    production_qualified: Literal[False]
    mutated_services: Literal[False]


class RemoteRendererOwnerAudit(RenderNodeModel):
    node_id: str = Field(min_length=1)
    audit: RendererOwnerAudit


class RemoteOwnerReadinessEvidence(RenderNodeModel):
    """Bearer-gated PC-B memory snapshot; never authorization to change services."""

    schema_version: Literal[1] = 1
    node_id: str = Field(min_length=1)
    report: dict[str, Any]
