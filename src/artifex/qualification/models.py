from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class QualificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QualificationStage(StrEnum):
    DOCTOR = "doctor"
    SINGLE_CHARACTER = "single_character"
    LORA_REQUIRED = "lora_required"
    DUO = "duo"
    GROUP = "group"
    PUBLIC_MEMBER = "public_member"
    SERIES_CONTINUATION = "series_continuation"
    FORCED_RETRY = "forced_retry"
    RESTART_GENERATION = "restart_generation"
    BACKEND_RECOVERY = "backend_recovery"
    UNATTENDED_MULTI_PACK = "unattended_multi_pack"
    OVERNIGHT_SOAK = "overnight_soak"
    DISCORD_CONTROLS = "discord_controls"
    ARCHIVE_REPRODUCTION = "archive_reproduction"


class QualificationStatus(StrEnum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


REQUIRED_STAGES: tuple[QualificationStage, ...] = (
    QualificationStage.DOCTOR,
    QualificationStage.SINGLE_CHARACTER,
    QualificationStage.LORA_REQUIRED,
    QualificationStage.DUO,
    QualificationStage.GROUP,
    QualificationStage.PUBLIC_MEMBER,
    QualificationStage.SERIES_CONTINUATION,
    QualificationStage.FORCED_RETRY,
    QualificationStage.RESTART_GENERATION,
    QualificationStage.BACKEND_RECOVERY,
    QualificationStage.UNATTENDED_MULTI_PACK,
    QualificationStage.OVERNIGHT_SOAK,
    QualificationStage.DISCORD_CONTROLS,
    QualificationStage.ARCHIVE_REPRODUCTION,
)


class QualificationStageEvidence(QualificationModel):
    stage: QualificationStage
    status: QualificationStatus = QualificationStatus.PENDING
    recorded_at: datetime | None = None
    pack_ids: tuple[str, ...] = ()
    note: str | None = None
    details: dict[str, object] = Field(default_factory=dict)


class AssetDigest(QualificationModel):
    label: str
    path: str
    sha256: str
    bytes: int = Field(ge=0)
    file_count: int = Field(default=1, ge=1)
    source: str = "local"
    node_id: str | None = None
    attested_at: datetime | None = None


class QualificationSession(QualificationModel):
    schema_version: int = 2
    session_id: str
    created_at: datetime
    updated_at: datetime
    hostname: str
    environment: dict[str, object]
    configuration: dict[str, object]
    workflow: dict[str, object]
    assets: tuple[AssetDigest, ...]
    loras: tuple[dict[str, object], ...]
    doctor_ready: bool
    doctor: dict[str, object]
    stages: dict[str, QualificationStageEvidence]
    notes: tuple[str, ...] = ()
    # Independently recorded read-only PC-B owner snapshots. These are
    # *not* any of the 14 stage PASS records and never permit GPU restart.
    renderer_owner_observations: tuple[dict[str, object], ...] = ()

    def stage(self, stage: QualificationStage) -> QualificationStageEvidence:
        try:
            return self.stages[stage.value]
        except KeyError as exc:
            raise KeyError(f"qualification stage is missing: {stage.value}") from exc
