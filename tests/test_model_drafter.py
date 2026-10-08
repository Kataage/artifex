from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy import model_drafter
from artifex.comfy.dependency_installer import DependencyManifest
from artifex.comfy.workflow_audit import (
    MissingWorkflowAsset,
    WorkflowAudit,
    WorkflowAuditEntry,
)

SHA = "a" * 40
LFS_SHA = "b" * 64
REPO = "illustration/official-model"
OTHER = "community/copy"
FILENAME = "refiner.safetensors"
NESTED = "weights/refiner.safetensors"


def _audit(
    *,
    name: str = FILENAME,
    label: str = "checkpoint",
    unknown: bool = False,
    additional: tuple[MissingWorkflowAsset, ...] = (),
) -> WorkflowAudit:
    item = MissingWorkflowAsset(
        label=label,
        node_class="CheckpointLoaderSimple",
        input_name="ckpt_name",
        requested=name,
        available=("other_model.safetensors",),
    )
    return WorkflowAudit(
        ready=False, comfyui_url="http://comfy.test:8188", custom_node_folders=(),
        entries=(
            WorkflowAuditEntry(
                template_id="illust_main_v1", source_sha256="0" * 64,
                ready=False, required_node_count=8, required_asset_count=1,
                missing_node_types=(),
                missing_assets=() if unknown else (item, *additional),
                unverifiable_assets=(item,) if unknown else (),
            ),
        ),
        guidance=(),
    )


def _hub(
    *,
    repositories: tuple[str, ...] = (REPO,),
    paths: tuple[str, ...] = (NESTED,),
    license_id: str = "creativeml-openrail-m",
    lfs: dict[str, Any] | None = None,
    gated: bool = False,
    identity: str | None = None,
    http_calls: list[str] | None = None,
    model_meta: dict[str, Any] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if http_calls is not None:
            http_calls.append(request.method + " " + url)
        for repo in repositories:
            api = "https://huggingface.co/api/models/" + repo
            if request.method == "GET" and url == api:
                return httpx.Response(
                    200, json=model_meta or {
                        "id": identity or repo,
                        "sha": SHA,
                        "gated": gated,
                        "cardData": {"license": license_id},
                        "siblings": [
                            {"rfilename": "README.md"},
                            *({"rfilename": path} for path in paths),
                        ],
                    },
                )
            if request.method == "POST" and url == api + "/paths-info/" + SHA:
                requested = request.read()
                assert b"refiner.safetensors" in requested
                return httpx.Response(
                    200,
                    json=[
                        {
                            "type": "file",
                            "path": paths[0],
                            "size": 1024,
                            "lfs": lfs if lfs is not None else {
                                "size": 1024, "sha256": LFS_SHA,
                            },
                        },
                    ],
                )
        raise AssertionError("unexpected external request: " + url)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_exact_hf_lfs_file_creates_pinned_install_ready_manifest() -> None:
    calls: list[str] = []
    with _hub(http_calls=calls) as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        )
    assert not result.unresolved
    assert len(result.manifest.artifacts) == 1
    file = result.manifest.artifacts[0]
    assert file.kind == "model"
    assert file.model_folder == "checkpoints"
    assert file.name == FILENAME
    assert file.url == (
        "https://huggingface.co/" + REPO + "/resolve/" + SHA + "/" + NESTED
    )
    assert file.sha256 == LFS_SHA and file.size_bytes == 1024
    assert file.license_id == "creativeml-openrail-m"
    assert file.license_url == (
        "https://huggingface.co/" + REPO + "/blob/" + SHA + "/README.md"
    )
    assert result.evidence[0].source_file_path == NESTED
    assert len(calls) == 2
    assert all("/resolve/" not in uri for uri in calls)
    assert calls[1].startswith("POST https://huggingface.co/api/models/")


def test_duplicate_exact_filename_from_two_shortlisted_publishers_is_ambiguous() -> None:
    with _hub(repositories=(REPO, OTHER)) as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO, OTHER), client=client,
        )
    assert not result.manifest.artifacts
    assert result.unresolved[0].reason.startswith("Ambiguous")
    assert len(result.unresolved[0].candidates) == 2


def test_multiple_matching_basename_paths_within_repo_is_ambiguous() -> None:
    calls: list[str] = []
    with _hub(paths=("a/refiner.safetensors", "b/refiner.safetensors"), http_calls=calls) as client:
        result = model_drafter.draft_hf_models(_audit(), repositories=(REPO,), client=client)
    assert not result.manifest.artifacts
    assert len(calls) == 1
    assert "Multiple files" in result.unresolved[0].candidates[0]


@pytest.mark.parametrize("license_id", ["other", "none", "unknown", "NOASSERTION"])
def test_unrecognized_license_is_rejected_before_weight_metadata(
    license_id: str,
) -> None:
    calls: list[str] = []
    with _hub(license_id=license_id, http_calls=calls) as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert len(calls) == 1
    assert "license" in result.unresolved[0].candidates[0].lower()


def test_gated_repo_does_not_draft_or_fetch_weight() -> None:
    with _hub(gated=True) as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert "Gated" in result.unresolved[0].candidates[0]


def test_repository_identity_mismatch_cannot_be_used_as_official_source() -> None:
    with _hub(identity="attacker/impersonation") as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert "identity mismatch" in result.unresolved[0].candidates[0]


@pytest.mark.parametrize(
    "lfs",
    [
        {},
        {"size": 1024, "sha256": "bad-sha"},
        {"size": 999, "sha256": LFS_SHA},
        {"size": 0, "sha256": LFS_SHA},
        {"size": 32 * 1024**3 + 1, "sha256": LFS_SHA},
    ],
)
def test_missing_conflicting_or_oversized_lfs_metadata_is_rejected(
    lfs: dict[str, Any],
) -> None:
    with _hub(lfs=lfs) as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert "LFS" in result.unresolved[0].candidates[0]


def test_unknown_runtime_assets_are_reported_not_auto_downloaded() -> None:
    calls: list[str] = []
    with _hub(http_calls=calls) as client:
        result = model_drafter.draft_hf_models(
            _audit(unknown=True), repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert any("confirm" in issue.reason for issue in result.unresolved)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("label", "folder"),
    [
        ("checkpoint", "checkpoints"),
        ("vae", "vae"),
        ("upscale_model", "upscale_models"),
        ("lora", "loras"),
        ("static_lora", "loras"),
    ],
)
def test_role_to_model_folder_mapping_is_explicit(
    label: str, folder: str,
) -> None:
    with _hub() as client:
        result = model_drafter.draft_hf_models(
            _audit(label=label), repositories=(REPO,), client=client,
        )
    assert result.manifest.artifacts[0].model_folder == folder


def test_model_with_nested_requested_runtime_name_is_not_guessed() -> None:
    with _hub() as client:
        result = model_drafter.draft_hf_models(
            _audit(name="subdir/refiner.safetensors"),
            repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert result.unresolved and "Unsupported" in result.unresolved[0].reason


@pytest.mark.parametrize(
    "repo",
    [
        "evil.example.com/repo",
        "https://huggingface.co/owner/repo",
        "../repo",
        "user/repo/name",
        "user@attacker/repo",
    ],
)
def test_model_repo_identifier_must_be_exact_safe_hub_id(repo: str) -> None:
    with pytest.raises(ValueError, match="owner/repository"):
        model_drafter.draft_hf_models(
            _audit(), repositories=(repo,),
        )


def test_no_operator_shortlist_fails_before_any_http() -> None:
    with pytest.raises(ValueError, match="shortlisted"):
        model_drafter.draft_hf_models(_audit(), repositories=())


def test_readme_is_required_for_pinned_license_review() -> None:
    model = {
        "id": REPO, "sha": SHA, "cardData": {"license": "mit"},
        "gated": False, "siblings": [{"rfilename": NESTED}],
    }
    with _hub(model_meta=model) as client:
        result = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        )
    assert not result.manifest.artifacts
    assert "README.md" in result.unresolved[0].candidates[0]


def test_installer_model_url_accepts_nested_hf_paths_and_rejects_traversal() -> None:
    with _hub() as client:
        model = model_drafter.draft_hf_models(
            _audit(), repositories=(REPO,), client=client,
        ).manifest.artifacts[0]
    payload = model.model_dump()
    payload["url"] = payload["url"].replace(
        "/weights/refiner.safetensors", "/../refiner.safetensors"
    )
    with pytest.raises(ValidationError):
        model.__class__.model_validate(payload)
    payload["url"] = model.url + "?download=true"
    with pytest.raises(ValidationError):
        model.__class__.model_validate(payload)


def test_cli_models_draft_creates_new_json_and_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_audit(settings: Any) -> WorkflowAudit:
        return _audit()

    monkeypatch.setattr("artifex.cli.audit_workflows", fake_audit)
    with _hub() as client:
        monkeypatch.setattr(
            "artifex.cli.draft_hf_models",
            lambda audit, **kwargs: model_drafter.draft_hf_models(
                audit, repositories=kwargs["repositories"], client=client
            ),
        )
        output = tmp_path / "models.json"
        result = CliRunner().invoke(
            app, [
                "onboard", "models-draft", "--output", str(output),
                "--repo", REPO,
            ],
        )
        assert result.exit_code == 0, result.output
        stored = DependencyManifest.model_validate_json(output.read_bytes())
        assert len(stored.artifacts) == 1
        before = output.read_bytes()
        second = CliRunner().invoke(
            app, [
                "onboard", "models-draft", "--output", str(output), "--repo", REPO,
            ],
        )
        assert second.exit_code == 1
        assert output.read_bytes() == before


def test_cli_models_draft_no_verified_match_exits_one_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_audit(settings: Any) -> WorkflowAudit:
        return _audit(unknown=True)

    monkeypatch.setattr("artifex.cli.audit_workflows", fake_audit)
    with _hub() as client:
        monkeypatch.setattr(
            "artifex.cli.draft_hf_models",
            lambda audit, **kwargs: model_drafter.draft_hf_models(
                audit, repositories=kwargs["repositories"], client=client
            ),
        )
        path = tmp_path / "nothing.json"
        result = CliRunner().invoke(
            app, ["onboard", "models-draft", "--output", str(path), "--repo", REPO]
        )
    assert result.exit_code == 1
    assert not path.exists()
    assert '"unresolved":' in result.output
