from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig, RenderNodesConfig
from artifex.deployment import DeploymentCheck, DeploymentReport
from artifex.native_dependencies import DependencyCheck, NativeDependencyReport
from artifex.render_node.models import (
    RenderAssetDigest,
    RenderLoRAInventoryItem,
    RenderNodeAttestation,
)
from artifex.render_node.preflight import RenderNodePreflight
from artifex.two_pc_readiness import (
    RendererEvidence,
    collect_renderer_evidence,
    read_renderer_evidence,
    verify_pair_readiness,
)

_ASSETS = (
    "production_checkpoint", "refiner_checkpoint", "vae", "upscale_model",
)
_SHA = {
    name: (f"{i}" * 64)
    for i, name in enumerate(_ASSETS, start=1)
}


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes = RenderNodesConfig(
        primary="gpu-b",
        nodes={
            "gpu-b": RenderNodeConfig(
                base_url="http://192.0.2.10:8188",
                output_mode="api",
                attestation_url="http://192.0.2.10:8190",
                attestation_token_env="ARTIFEX_RENDER_NODE_TOKEN",
            )
        },
    )
    return settings


def _native(role: str, *, ready: bool = True, torch: bool = True) -> NativeDependencyReport:
    checks = (
        DependencyCheck(name="windows", ready=ready, blocking=True, detail="windows"),
        DependencyCheck(name="uv", ready=True, blocking=True, detail="uv present"),
    )
    if role == "renderer" and torch:
        checks += (
            DependencyCheck(
                name="torch_cuda", ready=ready, blocking=True, detail="CUDA present",
            ),
        )
    return NativeDependencyReport(
        role=role, ready=ready, platform="Windows", checks=checks
    )


def _preflight(*, ready: bool = True) -> RenderNodePreflight:
    return RenderNodePreflight(
        ready=ready, node_id="gpu-b",
        comfyui_base_url="http://localhost:8188",
        comfyui_version="0.9", gpu_count=1,
        asset_labels=_ASSETS, lora_count=1,
        issues=() if ready else ("model directory unavailable",),
    )


def _remote(
    *,
    at: datetime | None = None,
    hostname: str = "render-pc",
    node_id: str = "gpu-b",
    missing: str | None = None,
    override_hash: str | None = None,
    gpu: bool = True,
) -> RenderNodeAttestation:
    assets = tuple(
        RenderAssetDigest(
            label=name, path=f"D:/models/{name}.safetensors",
            sha256=override_hash if override_hash is not None and name == "vae"
            else _SHA[name], bytes=100,
        )
        for name in _ASSETS if name != missing
    )
    return RenderNodeAttestation(
        node_id=node_id, hostname=hostname,
        created_at=at or datetime.now(UTC),
        os={"system": "Windows"},
        comfyui_base_url="http://localhost:8188",
        nvidia_gpus=({"name": "RTX 3060"},) if gpu else (),
        assets=assets,
        loras=(
            RenderLoRAInventoryItem(
                name="lora.safetensors", path="D:/lora.safetensors",
                relative_path="lora.safetensors", sha256="f" * 64, bytes=100,
            ),
        ),
    )


def _snapshot(
    *,
    time: datetime | None = None,
    host: str = "render-pc",
    native_ready: bool = True,
    renderer_ready: bool = True,
    torch: bool = True,
) -> RendererEvidence:
    return RendererEvidence(
        created_at=time or datetime.now(UTC), hostname=host, node_id="gpu-b",
        gpu_count=1, asset_sha256=_SHA, asset_bytes=dict.fromkeys(_ASSETS, 100),
        lora_count=1, torch_probed=torch,
        native=_native("renderer", ready=native_ready, torch=torch),
        preflight=_preflight(ready=renderer_ready),
    )


def _deployment(*, ready: bool = True) -> DeploymentReport:
    return DeploymentReport(
        role="controller", ready=ready,
        checks=(
            DeploymentCheck(
                name="controller:comfyui_lan", ready=ready,
                blocking=True, detail="reachable" if ready else "no route",
            ),
            DeploymentCheck(
                name="windows_autostart", ready=False,
                blocking=False, detail="not enabled",
            ),
            DeploymentCheck(
                name="workflow:illust_main_v1", ready=True,
                blocking=True, detail="node choices present",
            ),
        ),
    )


def test_renderer_snapshot_uses_one_attestation_and_readonly_checks(tmp_path: Path) -> None:
    counts = {"attest": 0, "native": 0, "preflight": 0}

    def native(
        settings: ArtifexSettings, *, role: str,
        comfy_root: Path | None = None, probe_torch: bool = False,
    ) -> NativeDependencyReport:
        counts["native"] += 1
        assert role == "renderer" and comfy_root == tmp_path
        assert probe_torch
        return _native(role)

    def attest(settings: ArtifexSettings) -> RenderNodeAttestation:
        counts["attest"] += 1
        return _remote()

    def preflight(
        settings: ArtifexSettings, *, attestation: RenderNodeAttestation,
    ) -> RenderNodePreflight:
        counts["preflight"] += 1
        assert attestation.hostname == "render-pc"
        return _preflight()

    result = collect_renderer_evidence(
        _settings(), comfy_root=tmp_path, native_check=native,
        attestation_builder=attest, renderer_check=preflight,
    )
    assert counts == {"attest": 1, "native": 1, "preflight": 1}
    assert result.gpu_count == 1
    assert result.asset_sha256 == _SHA
    assert result.torch_probed is True
    assert "ARTIFEX_RENDER_NODE_TOKEN" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_real_pair_status_checks_local_and_live_remote_without_render() -> None:
    called: list[str] = []

    def native(settings: ArtifexSettings, *, role: str) -> NativeDependencyReport:
        called.append("native:" + role)
        return _native(role)

    async def deployment(settings: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        called.append("deployment")
        assert kwargs == {"role": "controller", "render_smoke": False}
        return _deployment()

    def remote(node_id: str, node: RenderNodeConfig) -> RenderNodeAttestation:
        called.append("remote")
        assert node_id == "gpu-b" and node.output_mode == "api"
        return _remote()

    report = await verify_pair_readiness(
        _settings(), _snapshot(), local_hostname="controller-pc",
        native_check=native, deployment_check=deployment,
        attestation_fetch=remote,
    )
    assert called == ["native:controller", "deployment", "remote"]
    assert report.preflight_ready
    assert report.actual_render_verified is False
    assert all(item.ready for item in report.checks)
    assert len([item for item in report.checks if item.name.startswith("asset_agreement")]) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("problem", "failed_check"),
    [
        ("stale", "renderer_evidence_freshness"),
        ("same_host", "separate_hosts"),
        ("torch", "renderer_torch_probe"),
        ("native", "renderer_local_native"),
        ("renderer", "renderer_preflight"),
        ("hash", "asset_agreement:vae"),
        ("missing", "asset_agreement:vae"),
        ("gpu", "gpu_count_agreement"),
        ("hostname", "live_authenticated_renderer"),
        ("remote_stale", "live_authenticated_renderer"),
        ("node", "live_authenticated_renderer"),
    ],
)
async def test_pair_fail_closed(
    problem: str, failed_check: str,
) -> None:
    snapshot = _snapshot(
        time=datetime.now(UTC) - timedelta(minutes=90) if problem == "stale" else None,
        native_ready=problem != "native", renderer_ready=problem != "renderer",
        torch=problem != "torch",
    )
    live = _remote(
        missing="vae" if problem == "missing" else None,
        override_hash="0" * 64 if problem == "hash" else None,
        gpu=problem != "gpu",
        hostname="another-pc" if problem == "hostname" else "render-pc",
        at=datetime.now(UTC) - timedelta(hours=2) if problem == "remote_stale" else None,
        node_id="wrong-node" if problem == "node" else "gpu-b",
    )

    async def deployment(settings: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        return _deployment()

    report = await verify_pair_readiness(
        _settings(), snapshot, local_hostname="render-pc"
        if problem == "same_host" else "controller-pc",
        native_check=lambda settings, **kwargs: _native("controller"),
        deployment_check=deployment,
        attestation_fetch=lambda node, config: live,
    )
    assert not report.preflight_ready
    assert any(
        c.name == failed_check and not c.ready and c.next_action
        for c in report.checks
    )
    assert report.actual_render_verified is False


@pytest.mark.asyncio
async def test_auth_lan_failure_reports_actionable_check_not_success() -> None:
    async def deployment(settings: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        return _deployment(ready=False)

    def unavailable(node_id: str, node: RenderNodeConfig) -> RenderNodeAttestation:
        raise ConnectionError("401 unauthorized: invalid token")

    result = await verify_pair_readiness(
        _settings(), _snapshot(), local_hostname="controller-pc",
        native_check=lambda settings, **kwargs: _native("controller"),
        deployment_check=deployment, attestation_fetch=unavailable,
    )
    assert not result.preflight_ready
    assert any(
        c.name == "live_authenticated_renderer" and "bearer token" in c.next_action
        for c in result.checks if c.next_action
    )
    assert any(
        c.name == "controller_deployment:controller:comfyui_lan" and not c.ready
        for c in result.checks
    )


@pytest.mark.asyncio
async def test_missing_primary_config_fails_without_remote_fetch() -> None:
    settings = ArtifexSettings()

    async def deployment(settings: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        return _deployment(ready=False)

    def unexpectedly_called(node: str, cfg: RenderNodeConfig) -> RenderNodeAttestation:
        raise AssertionError("Must not fetch unconfigured attestation")

    result = await verify_pair_readiness(
        settings, _snapshot(), local_hostname="controller-pc",
        native_check=lambda settings, **kwargs: _native("controller"),
        deployment_check=deployment, attestation_fetch=unexpectedly_called,
    )
    assert not result.preflight_ready
    assert any(
        c.name == "primary_renderer_configured" and not c.ready
        for c in result.checks
    )


def test_renderer_evidence_file_rejects_invalid_symlink_and_oversize(tmp_path: Path) -> None:
    stored = tmp_path / "pc-b.json"
    stored.write_text(_snapshot().model_dump_json(), encoding="utf-8")
    assert read_renderer_evidence(stored).node_id == "gpu-b"
    stored.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError):
        read_renderer_evidence(stored)
    stored.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        read_renderer_evidence(stored)
    other = tmp_path / "link.json"
    try:
        other.symlink_to(stored)
    except (OSError, NotImplementedError):
        pytest.skip("No symlink permission on this host")
    with pytest.raises(ValueError, match="non-symlinked"):
        read_renderer_evidence(other)


def test_pair_export_cli_is_exclusive_and_does_not_send_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.write_text("{}\n", encoding="utf-8")
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    output = tmp_path / "renderer-evidence.json"
    observed: list[tuple[Path, bool]] = []

    def collect(
        settings: ArtifexSettings, *, comfy_root: Path, probe_torch: bool,
    ) -> RendererEvidence:
        observed.append((comfy_root, probe_torch))
        return _snapshot()

    monkeypatch.setattr("artifex.cli.collect_renderer_evidence", collect)
    cli = CliRunner()
    args = [
        "deployment", "pair-export", "--config", str(config),
        "--comfy-root", str(comfy), "--output", str(output),
    ]
    first = cli.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert observed == [(comfy, True)]
    assert read_renderer_evidence(output).hostname == "render-pc"
    existing = output.read_bytes()
    second = cli.invoke(app, args)
    assert second.exit_code == 1
    assert output.read_bytes() == existing


def test_pair_check_cli_emits_structured_failure_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    evidence = tmp_path / "pc-b.json"
    evidence.write_text(_snapshot().model_dump_json(), encoding="utf-8")

    async def verify(
        settings: ArtifexSettings, renderer: RendererEvidence, **kwargs: Any,
    ) -> Any:
        assert kwargs["max_age_minutes"] == 30
        return await verify_pair_readiness(
            _settings(), renderer, local_hostname="controller-pc",
            native_check=lambda settings, **opts: _native("controller", ready=False),
            deployment_check=lambda settings, **opts: _failed_deployment(),
            attestation_fetch=lambda node, cfg: _remote(),
        )

    async def _failed_deployment() -> DeploymentReport:
        return _deployment(ready=False)

    monkeypatch.setattr("artifex.cli.verify_pair_readiness", verify)
    result = CliRunner().invoke(
        app, [
            "deployment", "pair-check", "--config", str(config),
            "--renderer-report", str(evidence), "--max-age-minutes", "30",
        ],
    )
    assert result.exit_code == 1
    assert '"preflight_ready": false' in result.output
    assert '"actual_render_verified": false' in result.output
