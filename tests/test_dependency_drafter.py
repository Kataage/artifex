from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy import dependency_drafter
from artifex.comfy.dependency_installer import DependencyManifest
from artifex.comfy.dependency_resolver import (
    DependencyResolution,
    NodeCandidate,
)

SHA = "a" * 40
LICENSE_BYTES = b"MIT License\nCopyright test only\n"
REPO = "example-owner/example-nodes"
REPO_URL = "https://github.com/" + REPO
API = "https://api.github.com/repos/" + REPO
ARCHIVE_URL = "https://codeload.github.com/" + REPO + "/zip/" + SHA


def _zip(
    *,
    license_bytes: bytes = LICENSE_BYTES,
    init_name: str = "__init__.py",
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("example-nodes-aaa/" + init_name, "NODE_CLASS_MAPPINGS = {}")
        archive.writestr("example-nodes-aaa/LICENSE", license_bytes)
    return output.getvalue()


def _resolution(
    *,
    candidates: tuple[NodeCandidate, ...] | None = None,
) -> DependencyResolution:
    return DependencyResolution(
        manager_commit="b" * 40,
        source="https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/" + "b" * 40,
        missing_node_types=("FooNode",),
        candidates=candidates if candidates is not None else (
            NodeCandidate(
                repository_url=REPO_URL,
                matched_node_types=("FooNode",),
                description="Example node",
            ),
        ),
        unresolved_node_types=(),
        missing_model_choices=("checkpoint:missing.safetensors",),
        unverifiable_model_choices=("vae:unknown.safetensors",),
    )


def _http(
    *,
    license_id: str = "MIT",
    license_bytes: bytes = LICENSE_BYTES,
    zip_bytes: bytes | None = None,
    no_license: bool = False,
    repo_identity: str = REPO,
    recorded: list[str] | None = None,
) -> httpx.Client:
    archive = zip_bytes if zip_bytes is not None else _zip()

    def handler(request: httpx.Request) -> httpx.Response:
        uri = str(request.url)
        if recorded is not None:
            recorded.append(uri)
        if uri == API:
            return httpx.Response(
                200,
                json={"full_name": repo_identity, "default_branch": "main"},
            )
        if uri == API + "/commits/main":
            return httpx.Response(200, json={"sha": SHA})
        if uri == API + "/license?ref=" + SHA:
            if no_license:
                return httpx.Response(404, json={"message": "Not found"})
            return httpx.Response(
                200,
                json={
                    "license": {"spdx_id": license_id},
                    "path": "LICENSE",
                    "encoding": "base64",
                    "content": base64.b64encode(license_bytes).decode("ascii"),
                },
            )
        if uri == ARCHIVE_URL:
            return httpx.Response(200, content=archive)
        raise AssertionError(f"unexpected HTTP call: {uri}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_verified_candidate_yields_installer_compatible_manifest() -> None:
    calls: list[str] = []
    with _http(recorded=calls) as client:
        result = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client,
        )
    assert len(result.manifest.artifacts) == 1
    artifact = result.manifest.artifacts[0]
    assert artifact.repository == REPO
    assert artifact.commit == SHA
    assert artifact.kind == "custom_node"
    assert artifact.url == ARCHIVE_URL
    assert artifact.name == "example-nodes"
    assert artifact.sha256 == hashlib.sha256(_zip()).hexdigest()
    assert artifact.size_bytes == len(_zip())
    assert artifact.license_id == "MIT"
    assert artifact.license_url == (
        REPO_URL + "/blob/" + SHA + "/LICENSE"
    )
    assert artifact.id == "node-example-nodes"
    assert result.evidence[0].license_file_sha256 == hashlib.sha256(
        LICENSE_BYTES
    ).hexdigest()
    assert result.missing_model_choices == ("checkpoint:missing.safetensors",)
    assert result.unverifiable_model_choices == ("vae:unknown.safetensors",)
    assert result.unresolved_node_types == ()
    assert calls == [
        API, API + "/commits/main", API + "/license?ref=" + SHA, ARCHIVE_URL
    ]


@pytest.mark.parametrize("license_id", ["NOASSERTION", "OTHER", "NONE", "???"])
def test_unknown_licenses_are_excluded_without_download(
    license_id: str,
) -> None:
    calls: list[str] = []
    with _http(license_id=license_id, recorded=calls) as client:
        result = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client
        )
    assert result.manifest.artifacts == ()
    assert len(result.skipped) == 1
    assert "license" in result.skipped[0].reason.lower()
    assert ARCHIVE_URL not in calls
    assert result.unresolved_node_types == ("FooNode",)


def test_no_license_metadata_is_not_treated_as_approved() -> None:
    with _http(no_license=True) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client
        )
    assert not draft.manifest.artifacts
    assert draft.skipped


def test_mismatched_license_file_in_archive_is_excluded() -> None:
    with _http(zip_bytes=_zip(license_bytes=b"changed license")) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client
        )
    assert not draft.manifest.artifacts
    assert "License bytes differ" in draft.skipped[0].reason


def test_forged_repo_identity_is_rejected() -> None:
    with _http(repo_identity="other-owner/other-repository") as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client
        )
    assert not draft.manifest.artifacts
    assert "identity" in draft.skipped[0].reason


def test_tiny_download_limit_excludes_oversized_archive() -> None:
    with _http() as client:
        with pytest.raises(ValueError, match="max_archive_mib"):
            dependency_drafter.draft_missing_node_manifest(
                _resolution(), client=client, max_archive_mib=0,
            )
    with _http(zip_bytes=b"z" * (1024 * 1024 + 1)) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client, max_archive_mib=1,
        )
    assert not draft.manifest.artifacts
    assert "download limit" in draft.skipped[0].reason


def test_unsafe_or_not_a_python_node_zip_is_excluded() -> None:
    with _http(zip_bytes=_zip(init_name="../bad.py")) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(), client=client,
        )
    assert not draft.manifest.artifacts
    assert "Unsafe" in draft.skipped[0].reason


def test_ambiguous_provider_never_triggers_upstream_download() -> None:
    candidates = (
        NodeCandidate(repository_url=REPO_URL, matched_node_types=("FooNode",)),
        NodeCandidate(
            repository_url="https://github.com/other-owner/other-nodes",
            matched_node_types=("FooNode",),
        ),
    )
    calls: list[str] = []
    with _http(recorded=calls) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(candidates=candidates), client=client,
        )
    assert calls == []
    assert not draft.manifest.artifacts
    assert len(draft.skipped) == 2
    assert draft.unresolved_node_types == ("FooNode",)


def test_forged_candidate_url_never_downloaded() -> None:
    candidate = NodeCandidate(
        repository_url="https://github.com/example-owner@example.evil/steal",
        matched_node_types=("FooNode",),
    )
    calls: list[str] = []
    with _http(recorded=calls) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            _resolution(candidates=(candidate,)), client=client,
        )
    assert calls == []
    assert not draft.manifest.artifacts
    assert draft.skipped


def test_repository_limit_reports_remaining_unresolved_without_downloading() -> None:
    choices = (
        NodeCandidate(repository_url=REPO_URL, matched_node_types=("FooNode",)),
        NodeCandidate(
            repository_url="https://github.com/another-owner/another-repo",
            matched_node_types=("BarNode",),
        ),
    )
    base = _resolution(candidates=choices).model_copy(
        update={"missing_node_types": ("BarNode", "FooNode")}
    )
    calls: list[str] = []
    with _http(recorded=calls) as client:
        draft = dependency_drafter.draft_missing_node_manifest(
            base, client=client, max_repositories=1
        )
    assert len(draft.manifest.artifacts) == 1
    assert draft.unresolved_node_types == ("BarNode",)
    assert any("limit" in r.reason for r in draft.skipped)
    assert calls.count(ARCHIVE_URL) == 1


def test_cli_draft_writes_only_verified_manifest_without_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.comfy.workflow_audit import WorkflowAudit

    monkeypatch.setattr(
        "artifex.cli.audit_workflows",
        lambda settings: _fake_audit(),
    )
    monkeypatch.setattr(
        "artifex.cli.resolve_missing_dependencies",
        lambda audit, **kwargs: _resolution(),
    )
    async def fake_audit(settings: Any) -> WorkflowAudit:
        return _fake_audit()

    def _fake_audit() -> WorkflowAudit:
        return WorkflowAudit(
            ready=False, comfyui_url="http://comfy.test:8188",
            custom_node_folders=(), entries=(), guidance=(),
        )

    monkeypatch.setattr("artifex.cli.audit_workflows", fake_audit)
    with _http() as client:
        monkeypatch.setattr(
            "artifex.cli.draft_missing_node_manifest",
            lambda resolution, **kwargs: dependency_drafter.draft_missing_node_manifest(
                resolution, client=client,
                max_repositories=kwargs["max_repositories"],
                max_archive_mib=kwargs["max_archive_mib"],
            ),
        )
        output = tmp_path / "approved.json"
        result = CliRunner().invoke(
            app, [
                "onboard", "deps-draft", "--output", str(output),
                "--max-repositories", "2",
            ],
        )
        assert result.exit_code == 0, result.output
        stored = DependencyManifest.model_validate_json(output.read_bytes())
        assert len(stored.artifacts) == 1
        assert '"output": "' in result.output
        before = output.read_bytes()
        again = CliRunner().invoke(
            app, ["onboard", "deps-draft", "--output", str(output)],
        )
        assert again.exit_code == 1
        assert output.read_bytes() == before


def test_cli_no_safe_candidates_exits_nonzero_and_does_not_create_empty_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.comfy.workflow_audit import WorkflowAudit

    async def fake_audit(settings: Any) -> WorkflowAudit:
        return WorkflowAudit(
            ready=False, comfyui_url="http://comfy.test",
            custom_node_folders=(), entries=(), guidance=(),
        )

    monkeypatch.setattr("artifex.cli.audit_workflows", fake_audit)
    monkeypatch.setattr(
        "artifex.cli.resolve_missing_dependencies",
        lambda audit, **kwargs: _resolution(),
    )
    with _http(no_license=True) as client:
        monkeypatch.setattr(
            "artifex.cli.draft_missing_node_manifest",
            lambda resolution, **kwargs: dependency_drafter.draft_missing_node_manifest(
                resolution, client=client,
            ),
        )
        output = tmp_path / "missing.json"
        result = CliRunner().invoke(
            app, ["onboard", "deps-draft", "--output", str(output)]
        )
    assert result.exit_code == 1
    assert not output.exists()
    assert '"skipped"' in result.output
