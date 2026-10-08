from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_two_pc_readiness import _native, _snapshot
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig, RenderNodesConfig
from artifex.deployment import DeploymentCheck, DeploymentReport, RenderSmokeEvidence
from artifex.pair_render_proof import (
    PairRenderProof,
    run_pair_render_proof,
)
from artifex.render_node.models import RenderAssetDigest, RenderNodeAttestation
from artifex.two_pc_readiness import PairReadiness, RendererEvidence


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes = RenderNodesConfig(
        primary="gpu-b",
        nodes={
            "gpu-b": RenderNodeConfig(
                base_url="http://192.0.2.10:8188",
                output_mode="api",
                attestation_url="http://192.0.2.10:8190",
            )
        },
    )
    return settings


def _pair(*, ready: bool = True) -> PairReadiness:
    now = datetime.now(UTC)
    from artifex.two_pc_readiness import PairCheck

    return PairReadiness(
        preflight_ready=ready,
        checks=(
            PairCheck(
                name="live_authenticated_renderer", ready=ready,
                detail="authenticated" if ready else "wrong token",
            ),
        ),
        snapshot_at=now,
        checked_at=now,
        controller_native=_native("controller"),
        controller_deployment=DeploymentReport(
            role="controller", ready=ready,
            checks=(
                DeploymentCheck(
                    name="controller:comfyui_lan",
                    ready=ready, blocking=True,
                    detail="reachable" if ready else "connection refused",
                ),
            ),
        ),
    )


def _smoke(tmp_path: Path, *, healthy: bool = True) -> DeploymentReport:
    path = tmp_path / "image.png"
    return DeploymentReport(
        role="controller", ready=healthy,
        checks=(
            DeploymentCheck(
                name="render_smoke", ready=healthy, blocking=True,
                detail="downloaded and verified" if healthy else "invalid image",
            ),
        ),
        smoke=(
            RenderSmokeEvidence(
                template_id="illust_main_v1", prompt_id="real-id-123",
                image_path=path, image_sha256="a" * 64,
                width=512, height=512, bytes=1024,
            ) if healthy else None
        ),
    )


def _remote(
    *,
    changed: str | None = None,
    old: bool = False,
    hostname: str = "render-pc",
    gpu: bool = True,
    missing: str | None = None,
) -> RenderNodeAttestation:
    original = _snapshot()
    return RenderNodeAttestation(
        hostname=hostname,
        node_id="gpu-b",
        created_at=datetime.now(UTC) - timedelta(hours=2)
        if old else datetime.now(UTC),
        os={"system": "Windows"},
        comfyui_base_url="http://comfy:8188",
        nvidia_gpus=({"name": "RTX 3060"},) if gpu else (),
        assets=tuple(
            RenderAssetDigest(
                label=label,
                path=f"D:/models/{label}.safetensors",
                sha256="f" * 64 if label == changed
                else original.asset_sha256[label],
                bytes=original.asset_bytes[label],
            )
            for label in original.asset_sha256 if label != missing
        ),
        loras=(),
    )


def _snapshot_no_lora() -> RendererEvidence:
    return _snapshot().model_copy(update={"lora_count": 0})


@pytest.mark.asyncio
async def test_pair_smoke_gates_all_gpu_work_when_preflight_fails() -> None:
    calls: list[str] = []

    async def preflight(*args: Any, **kwargs: Any) -> PairReadiness:
        calls.append("preflight")
        return _pair(ready=False)

    async def no_render(*args: Any, **kwargs: Any) -> DeploymentReport:
        raise AssertionError("real GPU render must not be attempted")

    def no_attestation(*args: Any, **kwargs: Any) -> RenderNodeAttestation:
        raise AssertionError("post-render attestation must not run without render")

    report = await run_pair_render_proof(
        _settings(), _snapshot_no_lora(), controller_host="pc-a",
        preflight_check=preflight,
        deployment_check=no_render,
        attestation_fetch=no_attestation,
    )
    assert calls == ["preflight"]
    assert not report.render_attempted
    assert not report.actual_render_verified
    assert not report.ready_for_qualification
    assert not report.production_qualified
    assert "not started" in report.failures[0]


@pytest.mark.asyncio
async def test_pair_smoke_passes_only_one_render_and_post_model_check(tmp_path: Path) -> None:
    calls: list[str] = []

    async def preflight(*args: Any, **kwargs: Any) -> PairReadiness:
        calls.append("preflight")
        return _pair()

    async def smoke(*args: Any, **kwargs: Any) -> DeploymentReport:
        calls.append("smoke")
        assert kwargs["render_smoke"] is True
        assert kwargs["role"] == "controller"
        assert kwargs["width"] == 512 and kwargs["height"] == 640
        return _smoke(tmp_path)

    def attest(node: str, cfg: RenderNodeConfig) -> RenderNodeAttestation:
        calls.append("after")
        assert node == "gpu-b"
        return _remote()

    report = await run_pair_render_proof(
        _settings(), _snapshot_no_lora(), controller_host="pc-a",
        width=512, height=640,
        preflight_check=preflight, deployment_check=smoke,
        attestation_fetch=attest,
    )
    assert calls == ["preflight", "smoke", "after"]
    assert report.preflight_ready and report.actual_render_verified
    assert report.asset_stability_verified and report.ready_for_qualification
    assert report.output is not None and report.output.prompt_id == "real-id-123"
    assert report.production_qualified is False
    assert not report.failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("vae", "vae file digest/size changed"),
        ("missing_checkpoint", "production_checkpoint file digest/size changed"),
        ("gpu", "GPU inventory changed"),
        ("host", "host or node changed"),
        ("stale", "stale or clock-skewed"),
        ("lora", "LoRA inventory count changed"),
    ],
)
async def test_post_render_integrity_failure_blocks_qualification(
    tmp_path: Path, drift: str, message: str,
) -> None:
    async def preflight(*args: Any, **kwargs: Any) -> PairReadiness:
        return _pair()

    async def smoke(*args: Any, **kwargs: Any) -> DeploymentReport:
        return _smoke(tmp_path)

    def attest(*args: Any, **kwargs: Any) -> RenderNodeAttestation:
        remote = _remote(
            changed="vae" if drift == "vae" else None,
            missing="production_checkpoint" if drift == "missing_checkpoint" else None,
            gpu=drift != "gpu",
            hostname="different-pc" if drift == "host" else "render-pc",
            old=drift == "stale",
        )
        if drift == "lora":
            from artifex.render_node.models import RenderLoRAInventoryItem

            remote = remote.model_copy(update={
                "loras": (
                    RenderLoRAInventoryItem(
                        name="lora.safetensors", path="D:/models/lora.safetensors",
                        relative_path="lora.safetensors", sha256="e" * 64, bytes=100,
                    ),
                )
            })
        return remote

    result = await run_pair_render_proof(
        _settings(), _snapshot_no_lora(), controller_host="pc-a",
        preflight_check=preflight, deployment_check=smoke,
        attestation_fetch=attest,
    )
    assert result.render_attempted and result.actual_render_verified
    assert not result.asset_stability_verified
    assert not result.ready_for_qualification
    assert any(message in failure for failure in result.failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["broken", "exception", "post_failure", "missing_check"])
async def test_render_failure_never_promotes_readiness(
    tmp_path: Path, mode: str,
) -> None:
    async def preflight(*args: Any, **kwargs: Any) -> PairReadiness:
        return _pair()

    async def smoke(*args: Any, **kwargs: Any) -> DeploymentReport:
        if mode == "exception":
            raise RuntimeError("queue timed out")
        if mode == "broken":
            return _smoke(tmp_path, healthy=False)
        if mode == "missing_check":
            return DeploymentReport(role="controller", ready=True, checks=(), smoke=_smoke(tmp_path).smoke)
        return _smoke(tmp_path)

    def attest(*args: Any, **kwargs: Any) -> RenderNodeAttestation:
        if mode == "post_failure":
            raise ConnectionError("post-render attestation offline")
        return _remote()

    result = await run_pair_render_proof(
        _settings(), _snapshot_no_lora(), controller_host="pc-a",
        preflight_check=preflight, deployment_check=smoke,
        attestation_fetch=attest,
    )
    assert result.render_attempted
    assert not result.ready_for_qualification
    assert result.failures
    if mode == "post_failure":
        assert result.actual_render_verified
        assert not result.asset_stability_verified
    else:
        assert not result.actual_render_verified


@pytest.mark.asyncio
async def test_identity_mismatch_after_preflight_never_starts_render() -> None:
    settings = _settings()
    settings.render_nodes.primary = "other"

    async def preflight(*args: Any, **kwargs: Any) -> PairReadiness:
        return _pair()

    async def no_render(*args: Any, **kwargs: Any) -> DeploymentReport:
        raise AssertionError("no render after mismatched primary")

    report = await run_pair_render_proof(
        settings, _snapshot_no_lora(), controller_host="pc-a",
        preflight_check=preflight, deployment_check=no_render,
    )
    assert not report.render_attempted
    assert not report.ready_for_qualification


def test_cli_requires_explicit_confirmation_and_does_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    renderer_report = tmp_path / "pc-b.json"
    renderer_report.write_text(_snapshot_no_lora().model_dump_json(), encoding="utf-8")
    output = tmp_path / "pair-smoke.json"
    args = [
        "deployment", "pair-smoke", "--config", str(config),
        "--renderer-report", str(renderer_report), "--output", str(output),
    ]
    runner = CliRunner()
    no = runner.invoke(app, args)
    assert no.exit_code == 1 and not output.exists()
    assert "--confirm-render" in no.output

    monkeypatch.setattr(
        "artifex.cli.run_pair_render_proof", _fake_proof(tmp_path)
    )
    yes = runner.invoke(app, [*args, "--confirm-render"])
    assert yes.exit_code == 0, yes.output
    assert output.exists()
    assert '"actual_render_verified": true' in yes.output
    assert '"production_qualified": false' in yes.output
    preserved = output.read_bytes()
    again = runner.invoke(app, [*args, "--confirm-render"])
    assert again.exit_code == 1
    assert output.read_bytes() == preserved


def _fake_proof(tmp_path: Path) -> Any:
    async def handler(*args: Any, **kwargs: Any) -> PairRenderProof:
        now = datetime.now(UTC)
        return PairRenderProof(
            started_at=now, finished_at=now,
            controller_host="pc-a", renderer_host="pc-b",
            renderer_id="gpu-b",
            preflight_ready=True, render_attempted=True,
            actual_render_verified=True, asset_stability_verified=True,
            ready_for_qualification=True, production_qualified=False,
            failures=(), preflight=_pair(),
            deployment=_smoke(tmp_path), output=_smoke(tmp_path).smoke,
        )
    return handler


def test_cli_retains_failed_proof_and_fails_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    renderer_report = tmp_path / "pc-b.json"
    renderer_report.write_text(_snapshot_no_lora().model_dump_json(), encoding="utf-8")
    output = tmp_path / "failed.json"

    async def failure(*args: Any, **kwargs: Any) -> PairRenderProof:
        full = await _fake_proof(tmp_path)(*args, **kwargs)
        return full.model_copy(update={
            "actual_render_verified": False,
            "ready_for_qualification": False,
            "failures": ("Model mismatch",),
            "output": None,
        })

    monkeypatch.setattr("artifex.cli.run_pair_render_proof", failure)
    result = CliRunner().invoke(
        app, [
            "deployment", "pair-smoke", "--config", str(config),
            "--renderer-report", str(renderer_report), "--output", str(output),
            "--confirm-render",
        ],
    )
    assert result.exit_code == 1
    assert output.is_file()
    assert '"ready_for_qualification": false' in output.read_text(encoding="utf-8")


def test_render_size_is_validated_before_attempt() -> None:
    import asyncio

    with pytest.raises(ValueError, match="between 64 and 8192"):
        asyncio.run(
            run_pair_render_proof(
                _settings(), _snapshot_no_lora(),
                controller_host="pc-a", width=16,
            )
        )
