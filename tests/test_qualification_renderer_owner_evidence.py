from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.models import QualificationSession
from artifex.qualification.renderer_owner_evidence import (
    persist_owner_observation,
    verify_owner_observation,
)
from artifex.qualification.service import QualificationService
from artifex.render_node.attestation import serve_attestation
from artifex.render_node.client import fetch_renderer_owner_audit
from artifex.render_node.models import RemoteRendererOwnerAudit

CHECKS = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _observation(
    *, node_id: str = "gpu-b", status: str = "observed_stable",
    captured: datetime | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "audit": {
            "schema_version": 1,
            "captured_utc": (captured or datetime.now(UTC)).isoformat(),
            "status": status,
            "checks": {
                key: {"status": "pass", "reason": "verified mock"}
                for key in CHECKS
            },
            "actual_listener_pid": 404,
            "actual_process_started_utc": "2026-10-09T00:00:00Z",
            "launcher_pid": 402,
            "receipt_schema": 2,
            "scheduler_state": "Running",
            "process_observation_verified": status == "observed_stable",
            "restart_authorized": False,
            "child_survival_qualified": False,
            "production_qualified": False,
            "mutated_services": False,
        },
    }


def _record(
    root: Path, snapshot: dict[str, Any],
    *, session_id: str = "test-session",
) -> tuple[dict[str, object], ...]:
    (root / session_id).mkdir(parents=True, exist_ok=True)
    observed = RemoteRendererOwnerAudit.model_validate(snapshot)
    return (persist_owner_observation(root, session_id, observed),)


def test_owner_snapshot_session_binding_and_hash(tmp_path: Path) -> None:
    current = datetime.now(UTC)
    records = _record(tmp_path, _observation(captured=current))
    assert verify_owner_observation(
        tmp_path, "test-session", "gpu-b", records,
        created_at=current - timedelta(minutes=1), now=current,
    ) is None
    name = records[0]["file"]
    assert isinstance(name, str)
    target = tmp_path / "test-session" / name
    data = json.loads(target.read_text())
    assert data["session_id"] == "test-session"
    assert data["node_id"] == "gpu-b"
    assert data["observed"]["audit"]["restart_authorized"] is False
    assert verify_owner_observation(
        tmp_path, "other-session", "gpu-b", records,
        created_at=current - timedelta(minutes=1), now=current,
    ) is not None
    assert verify_owner_observation(
        tmp_path, "test-session", "other-node", records,
        created_at=current - timedelta(minutes=1), now=current,
    ) is not None
    target.write_text(target.read_text().replace('"gpu-b"', '"forged"'))
    assert "hash" in str(verify_owner_observation(
        tmp_path, "test-session", "gpu-b", records,
        created_at=current - timedelta(minutes=1), now=current,
    ))


@pytest.mark.parametrize("kind", [
    "stale", "future", "before_session", "blocked", "failed_check",
    "no_pid", "missing", "invalid_path", "missing_check",
])
def test_qualification_never_accepts_unproven_owner(
    tmp_path: Path, kind: str,
) -> None:
    now = datetime.now(UTC)
    captured = now
    if kind == "stale":
        captured = now - timedelta(hours=1)
    if kind == "future":
        captured = now + timedelta(hours=1)
    if kind == "before_session":
        captured = now - timedelta(minutes=3)
    payload = _observation(
        captured=captured,
        status="blocked" if kind == "blocked" else "observed_stable",
    )
    if kind == "failed_check":
        payload["audit"]["checks"]["process_identity"]["status"] = "unknown"
    if kind == "no_pid":
        payload["audit"]["actual_listener_pid"] = None
    if kind == "missing_check":
        del payload["audit"]["checks"]["scheduler_policy"]
    records = _record(tmp_path, payload)
    if kind == "missing":
        name = records[0]["file"]
        assert isinstance(name, str)
        (tmp_path / "test-session" / name).unlink()
    if kind == "invalid_path":
        records[0]["file"] = "../escape.json"
    result = verify_owner_observation(
        tmp_path, "test-session", "gpu-b", records,
        created_at=now - timedelta(minutes=1), now=now,
    )
    assert result is not None


def test_most_recent_observation_cannot_fall_back_to_old_success(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    previous = _record(tmp_path, _observation(captured=now))
    latest = _record(tmp_path, _observation(status="blocked", captured=now))
    result = verify_owner_observation(
        tmp_path, "test-session", "gpu-b", (*previous, *latest),
        created_at=now - timedelta(minutes=1), now=now,
    )
    assert result is not None and "blocked" in result


def test_remote_reader_rejects_wrong_node_and_unsafe_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "integration-secret")
    config = RenderNodeConfig(base_url="http://example.invalid:8188", attestation_url="http://example.invalid:8190")
    routes: list[str] = []

    def respond(req: httpx.Request) -> httpx.Response:
        routes.append(str(req.url))
        assert req.headers["authorization"] == "Bearer integration-secret"
        return httpx.Response(200, json=_observation())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        out = fetch_renderer_owner_audit("gpu-b", config, client=client)
    assert out.node_id == "gpu-b"
    assert routes == ["http://example.invalid:8190/v1/owner-audit"]

    def mismatched(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_observation(node_id="other"))

    with (
        httpx.Client(transport=httpx.MockTransport(mismatched)) as client,
        pytest.raises(ValueError, match="node ID"),
    ):
        fetch_renderer_owner_audit("gpu-b", config, client=client)

    unsafe = _observation()
    unsafe["audit"]["restart_authorized"] = True

    def unsafe_reply(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=unsafe)

    with (
        httpx.Client(transport=httpx.MockTransport(unsafe_reply)) as client,
        pytest.raises(ValidationError),
    ):
        fetch_renderer_owner_audit("gpu-b", config, client=client)


def test_remote_reader_refuses_unauthenticated_or_missing_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    cfg = RenderNodeConfig(base_url="http://localhost:8188", attestation_url="http://localhost:8190")
    with pytest.raises(ValueError, match="token is missing"):
        fetch_renderer_owner_audit("gpu-b", cfg)
    with pytest.raises(ValueError, match="no attestation_url"):
        fetch_renderer_owner_audit(
            "gpu-b", RenderNodeConfig(base_url="http://localhost:8188"),
        )


def test_attestation_endpoint_authenticates_and_refuses_missing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.render_node.owner_audit as audit_module

    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.bind_host = "127.0.0.1"
    settings.render_agent.require_token = True
    settings.render_agent.token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "integration-secret")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.render_agent.port = sock.getsockname()[1]
    config = tmp_path / "render-node.yaml"
    config.write_text("render_agent: {}\n", encoding="utf-8")
    counts: list[int] = []

    def observe(settings: ArtifexSettings, *, config: Path) -> dict[str, Any]:
        counts.append(1)
        return _observation()["audit"]

    monkeypatch.setattr(audit_module, "observe_renderer_owner", observe)
    stop = threading.Event()
    thread = threading.Thread(
        target=serve_attestation,
        args=(settings,),
        kwargs={"stop_event": stop, "owner_config": config},
        daemon=True,
    )
    thread.start()
    base = f"http://127.0.0.1:{settings.render_agent.port}"
    try:
        with httpx.Client(trust_env=False, timeout=3) as http:
            for _ in range(30):
                try:
                    if http.get(base + "/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    time.sleep(0.1)
            else:
                pytest.fail("Disposable attestation server did not bind")

            unauthorized = http.get(base + "/v1/owner-audit")
            assert unauthorized.status_code == 401
            assert not counts
            authorized = http.get(
                base + "/v1/owner-audit",
                headers={"Authorization": "Bearer integration-secret"},
            )
            assert authorized.status_code == 200, authorized.text
            assert RemoteRendererOwnerAudit.model_validate(authorized.json()).node_id == "gpu-b"
            assert counts == [1]

            config.unlink()
            missing = http.get(
                base + "/v1/owner-audit",
                headers={"Authorization": "Bearer integration-secret"},
            )
            assert missing.status_code == 503
            assert counts == [1]
    finally:
        stop.set()
        thread.join(timeout=4)



def test_qualification_service_persists_owner_snapshot_without_stage_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.service as module

    now = datetime.now(UTC)
    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8191",
        attestation_url="http://127.0.0.1:8190",
    )
    service = object.__new__(QualificationService)
    service._root = tmp_path
    service._settings = settings
    session = QualificationSession(
        session_id="trial-01", created_at=now - timedelta(seconds=20),
        updated_at=now, hostname=socket.gethostname(),
        environment={}, configuration={}, workflow={},
        assets=(), loras=(), doctor_ready=False,
        doctor={}, stages={},
    )
    service._write(session)
    calls: list[str] = []

    def fetch(node_id: str, config: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        calls.append(node_id)
        return RemoteRendererOwnerAudit.model_validate(
            _observation(captured=now)
        )

    monkeypatch.setattr(module, "fetch_renderer_owner_audit", fetch)
    outcome = service.collect_renderer_owner_observation("trial-01")
    assert outcome["observation_status"] == "observed_stable"
    assert outcome["production_qualified"] is False
    assert outcome["stages_changed"] is False
    assert calls == ["gpu-b"]
    stored = service.load("trial-01")
    assert not stored.stages
    assert len(stored.renderer_owner_observations) == 1
    assert verify_owner_observation(
        tmp_path, "trial-01", "gpu-b",
        stored.renderer_owner_observations,
        created_at=stored.created_at,
        now=now,
    ) is None
    service.collect_renderer_owner_observation("trial-01")
    assert len(service.load("trial-01").renderer_owner_observations) == 2



def test_owner_endpoint_requires_token_even_when_public_attestation_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.bind_host = "127.0.0.1"
    settings.render_agent.require_token = False
    settings.render_agent.token_env = None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.render_agent.port = sock.getsockname()[1]
    stop = threading.Event()
    thread = threading.Thread(
        target=serve_attestation,
        args=(settings,),
        kwargs={"stop_event": stop},
        daemon=True,
    )
    thread.start()
    base = f"http://127.0.0.1:{settings.render_agent.port}"
    try:
        with httpx.Client(trust_env=False, timeout=3) as http:
            for _ in range(30):
                try:
                    if http.get(base + "/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    time.sleep(0.1)
            else:
                pytest.fail("Test attestation did not start")
            public = http.get(base + "/v1/attestation")
            assert public.status_code == 200
            protected = http.get(base + "/v1/owner-audit")
            assert protected.status_code == 401
    finally:
        stop.set()
        thread.join(timeout=4)



def test_foreign_controller_owner_evidence_refuses_network_and_file_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.service as module

    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8191",
        attestation_url="http://127.0.0.1:8190",
    )
    service = object.__new__(QualificationService)
    service._root = tmp_path
    service._settings = settings
    now = datetime.now(UTC)
    imported = QualificationSession(
        session_id="foreign-owner", created_at=now - timedelta(seconds=10),
        updated_at=now, hostname="foreign-pc-a",
        environment={}, configuration={}, workflow={}, assets=(), loras=(),
        doctor_ready=False, doctor={}, stages={},
    )
    service._write(imported)
    before = (tmp_path / "foreign-owner" / "qualification.json").read_bytes()
    calls: list[str] = []

    def disallowed_fetch(node_id: str, config: RenderNodeConfig) -> Any:
        calls.append(node_id)
        raise AssertionError("No PC-B read on foreign session")

    monkeypatch.setattr(module, "fetch_renderer_owner_audit", disallowed_fetch)
    with pytest.raises(ValueError, match="another controller"):
        service.collect_renderer_owner_observation("foreign-owner")
    assert not calls
    assert (tmp_path / "foreign-owner" / "qualification.json").read_bytes() == before
    assert not list((tmp_path / "foreign-owner").glob("owner-*"))


def test_parallel_remote_owner_snapshots_are_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.service as module

    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8191",
        attestation_url="http://127.0.0.1:8190",
    )
    service = object.__new__(QualificationService)
    service._root = tmp_path
    service._settings = settings
    now = datetime.now(UTC)
    service._write(QualificationSession(
        session_id="parallel-owner", created_at=now - timedelta(seconds=10),
        updated_at=now, hostname=socket.gethostname(), environment={},
        configuration={}, workflow={}, assets=(), loras=(),
        doctor_ready=False, doctor={}, stages={},
    ))
    monkeypatch.setattr(
        module, "fetch_renderer_owner_audit",
        lambda node_id, cfg: RemoteRendererOwnerAudit.model_validate(
            _observation(captured=now),
        ),
    )
    gate = threading.Barrier(3)

    def collect() -> dict[str, object]:
        gate.wait(timeout=5)
        return service.collect_renderer_owner_observation("parallel-owner")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(collect)
        second = executor.submit(collect)
        gate.wait(timeout=5)
        outcomes = [first.result(timeout=15), second.result(timeout=15)]
    assert all(x["stages_changed"] is False for x in outcomes)
    current = service.load("parallel-owner")
    assert current.stages == {}
    assert len(current.renderer_owner_observations) == 2
    assert len({
        str(observation["file"])
        for observation in current.renderer_owner_observations
    }) == 2
    for observation in current.renderer_owner_observations:
        assert (tmp_path / "parallel-owner" / str(observation["file"])).exists()
