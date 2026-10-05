from __future__ import annotations

from enum import StrEnum


class AgentState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    STOPPING = "stopping"
    STOPPED = "stopped"


class CharacterStatus(StrEnum):
    ACTIVE = "active"
    AFFILIATE = "affiliate"
    GRADUATED = "graduated"
    RETIRED = "retired"
    TERMINATED = "terminated"
    HISTORICAL = "historical"


class PackState(StrEnum):
    IDEA = "idea"
    PLANNED = "planned"
    POLICY_CHECK = "policy_check"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    REVIEW = "review"
    FINALIZED = "finalized"
    FAILED = "failed"
    BLOCKED = "blocked"


class SceneState(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    RETRY = "retry"
    REVIEW = "review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    BLOCKED = "blocked"


class LoRAPolicy(StrEnum):
    NONE = "none"
    OPTIONAL = "optional"
    PREFERRED = "preferred"
    REQUIRED = "required"


class LoRAState(StrEnum):
    DISCOVERED = "discovered"
    PENDING = "pending"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DISABLED = "disabled"
    FAILED = "failed"


class ResultState(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class PublicationTier(StrEnum):
    PUBLIC = "public"
    MEMBER = "member"
    PRIVATE_REVIEW = "private_review"
    BLOCKED = "blocked"
