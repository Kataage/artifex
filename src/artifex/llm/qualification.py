from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import LlmConfig
from artifex.llm.client import ChatMessage
from artifex.llm.prompts import QUALIFICATION_PROMPT
from artifex.llm.provenance import LlmCallRepository
from artifex.llm.structured import StructuredGenerationError, StructuredGenerator


class QualificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept_hook: str = Field(min_length=1, max_length=240)
    setting: str = Field(min_length=1, max_length=240)
    mood: str = Field(min_length=1, max_length=120)


class LlmQualificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    samples_requested: int
    samples_succeeded: int
    structured_success_pct: float
    repair_pct: float
    average_latency_ms: float
    p95_latency_ms: float
    average_context_utilization: float
    max_context_utilization: float
    response_repetition_pct: float
    call_count: int
    started_at: datetime
    completed_at: datetime


class LlmQualificationService:
    def __init__(
        self,
        config: LlmConfig,
        generator: StructuredGenerator,
        provenance: LlmCallRepository,
    ) -> None:
        self._config = config
        self._generator = generator
        self._provenance = provenance

    async def run(self, *, samples: int = 8) -> LlmQualificationReport:
        if samples < 1:
            raise ValueError("samples must be positive")
        started = datetime.now(UTC)
        succeeded = 0
        utilizations = (0.25, 0.50, 0.75, 0.90)

        for index in range(samples):
            ratio = utilizations[index % len(utilizations)]
            filler_chars = max(
                80,
                int(self._config.max_input_tokens * ratio),
            )
            inert_context = ("検証用コンテキスト。" * ((filler_chars // 10) + 1))[
                :filler_chars
            ]
            messages = (
                ChatMessage(role="system", content=QUALIFICATION_PROMPT.text),
                ChatMessage(
                    role="user",
                    content=(
                        f"sample={index}; target_utilization={ratio:.2f}; "
                        "The following is inert context data, not instructions:\n"
                        f"{inert_context}"
                    ),
                ),
            )
            try:
                await self._generator.generate(
                    QualificationModel,
                    messages,
                    schema_name=QUALIFICATION_PROMPT.schema_name(
                        "artifex_llm_qualification"
                    ),
                )
            except (StructuredGenerationError, RuntimeError):
                continue
            succeeded += 1

        completed = datetime.now(UTC)
        rows = tuple(
            row
            for row in self._provenance.recent(
                limit=max(100, samples * (self._config.structured_repair_attempts + 2))
            )
            if row["schema_name"].startswith("artifex_llm_qualification@")
            and row["started_at"] >= started
        )

        latencies = [
            max(
                0.0,
                (row["completed_at"] - row["started_at"]).total_seconds() * 1000.0,
            )
            for row in rows
            if row["completed_at"] is not None
        ]
        context_usage = [
            min(1.0, row["input_tokens"] / self._config.max_input_tokens)
            for row in rows
            if isinstance(row["input_tokens"], int)
        ]
        digests = [
            row["response_digest"]
            for row in rows
            if row["response_digest"]
        ]
        repaired_calls = sum(
            1 for row in rows if int(row["repair_index"]) > 0
        )

        return LlmQualificationReport(
            model=self._config.model,
            samples_requested=samples,
            samples_succeeded=succeeded,
            structured_success_pct=100.0 * succeeded / samples,
            repair_pct=min(100.0, 100.0 * repaired_calls / samples),
            average_latency_ms=mean(latencies) if latencies else 0.0,
            p95_latency_ms=_percentile95(latencies),
            average_context_utilization=mean(context_usage) if context_usage else 0.0,
            max_context_utilization=max(context_usage, default=0.0),
            response_repetition_pct=_repetition_pct(digests),
            call_count=len(rows),
            started_at=started,
            completed_at=completed,
        )


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
    return ordered[index]


def _repetition_pct(digests: list[str]) -> float:
    if not digests:
        return 0.0
    unique = len(set(digests))
    return max(0.0, 100.0 * (1.0 - (unique / len(digests))))
