# Initial Complete Acceptance Evidence

This document maps the normative acceptance criteria in
`docs/specification.md` to implementation and automated evidence.

The automated qualification tier is GPU-independent by design. Real local
LLM/ComfyUI/Discord hardware qualification is a separate deployment test tier,
as specified in the testing strategy.

| # | Acceptance criterion | Evidence | Status |
|---|---|---|---|
| 1 | Clean Windows installation with `uv` | GitHub Actions `windows-install` job runs `uv sync --extra dev`, imports Artifex, and runs `artifex --help` on `windows-latest`. | Automated |
| 2 | `artifex doctor` reports LLM, ComfyUI, DB, storage, Discord readiness | `DoctorService`, `HealthChecker`, CLI `doctor`; `tests/test_doctor.py`, `tests/test_operations_health.py`. | Automated |
| 3 | Daemon start/stop/pause/resume preserves state | Persisted `RuntimeStore` state machine and `RuntimeDaemon`; `tests/test_daemon.py`, `tests/test_runtime_store.py`, `tests/test_runtime_transitions.py`. | Automated |
| 4 | Planner replenishes idea inventory automatically | Scheduler inventory decisions and `ProductionCoordinator.replenish_ideas`; daemon scheduler tests and sustained-run wiring. | Automated |
| 5 | Character selection respects readiness and recent use | `PlanningContextBuilder` filters readiness/required assets and computes recent-use penalty; planner scoring includes readiness and recent-character penalty. | Automated/structural |
| 6 | Required LoRA policy is enforced | `LoRAResolver`; `tests/test_lora_resolver.py` covers missing required LoRA and multi-character prerequisites. | Automated |
| 7 | LoRA discovery detects additions and validation state | `LoRADiscovery`, checksum/change reset, validation lifecycle; `tests/test_lora_discovery.py`, `tests/test_lora_validation.py`. | Automated |
| 8 | Single-character Pack runs end-to-end | Full `ProductionCoordinator` fake-backend qualification in `tests/test_production_qualification.py`. | Automated |
| 9 | Duo Pack resolves prerequisites and generates | Same qualification suite plus `LoRAResolver` duo conflict tests. | Automated |
| 10 | Group Pack resolves prerequisites and generates | Same qualification suite plus maximum LoRA-stack/group prerequisite tests. | Automated |
| 11 | Multi-scene Pack continuity is maintained | Structured full-Pack planning, continuity bible and Scene persistence; `tests/test_pack_planner.py`, `tests/test_pack_repository.py`. | Automated |
| 12 | Series continuation references prior Packs | `SeriesRepository` and Pack planner history validation; `tests/test_series.py`, `tests/test_pack_planner.py`. | Automated |
| 13 | Runtime prompt compilation produces target-model prompts | Deterministic model-family compiler; exact regression fixture in `tests/test_prompt_compiler.py`. | Automated |
| 14 | ComfyUI uses stable versioned templates | Packaged `ilxl_base_v1` template and bounded patch contract; `tests/test_comfy_templates.py`, `tests/test_comfy_client.py`. | Automated |
| 15 | Generation failures are classified and retried by reason | `RetryPolicy`, infrastructure/creative retry separation; `tests/test_retry_policy.py`. | Automated |
| 16 | Evaluation accepts, reviews and rejects outputs | Multi-signal `EvaluationEngine`; `tests/test_evaluation.py`, repository/selection tests. | Automated |
| 17 | Concept and image similarity affect planning/selection | Similarity service/provider and planner penalty integration; `tests/test_similarity.py`, planner scoring tests. | Automated |
| 18 | Pack progress survives process restart | Persisted checkpoints plus `RecoveryManager`; completed-history, queued, and lost-prompt cases in `tests/test_recovery_manager.py`. | Automated |
| 19 | Temporary backend outage recovers automatically | Bounded reconnect/backoff in Comfy client and infrastructure retry policy; mocked transient connection recovery in `tests/test_comfy_client.py`. | Automated |
| 20 | Persistent backend failure surfaces actionable Discord alert | `HealthSupervisor` escalation and notification sink; `test_persistent_backend_failure_blocks_and_surfaces_backend_alert`. | Automated |
| 21 | Discord can approve/reject/retry/pause/resume/status | Authorized router, durable review queue, slash commands/buttons; `tests/test_discord_operations.py`, `tests/test_discord_router.py`. | Automated |
| 22 | Policy gates independently block publication/use | Separate versioned Policy Engine; `tests/test_policy.py` proves planner/Scene intent cannot bypass policy. | Automated |
| 23 | Finalized Pack has complete provenance/archive metadata | `PackArchive` manifest, selected/rejected outputs and SHA; `tests/test_archive.py` and production qualification. | Automated |
| 24 | Daemon proceeds to next Pack without operator instruction | Sustained `RuntimeDaemon.run_once()` qualification processes five Packs consecutively. | Automated |
| 25 | Sustained run completes multiple Packs without unbounded state/resource growth | Five-Pack qualification asserts bounded attempts/reviews/nonterminal state; telemetry retention is explicitly bounded by `tests/test_telemetry.py`. | Automated |
| 26 | Required CI passes | Linux Ruff + mypy + pytest and Windows clean-install jobs are required on the final acceptance PR. | CI gate |

## Deployment qualification

After the automated Initial Complete gate, a real-machine qualification should
be run on the target Windows host with the intended local services and assets:

- configured llama.cpp/Spark endpoint,
- the user's ComfyUI installation and production ILXL checkpoint,
- production Character/LoRA registry,
- configured vision evaluator,
- optional Discord bot credentials/channel.

Run `uv run artifex doctor --config <config>` first. A green doctor report is
the deployment prerequisite before starting `uv run artifex daemon`.

This real-backend qualification validates the environment; it does not replace
or weaken the reproducible automated acceptance gate above.
