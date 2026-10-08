from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy.model_drafter import draft_hf_models
from artifex.comfy.model_sources import (
    ApprovedModelSource,
    ModelSourceRegistry,
    read_model_source_registry,
    register_model_source,
    save_model_source_registry,
    unregister_model_source,
)
from artifex.comfy.workflow_audit import (
    MissingWorkflowAsset,
    WorkflowAudit,
    WorkflowAuditEntry,
)

GOOD = "publisher/approved-model"
UNRELATED = "copycat/same-filename"
HASH = "a" * 64
OTHER_HASH = "b" * 64
REV = "c" * 40


def _source(
    role: str = "checkpoint",
    name: str = "example.safetensors",
    repo: str = GOOD,
    *,
    file_path: str | None = "models/example.safetensors",
    hash: str | None = None,
) -> ApprovedModelSource:
    return ApprovedModelSource.model_validate(
        {
            "role": role,
            "model_name": name,
            "repository": repo,
            "file_path": file_path,
            "expected_sha256": hash,
            "rationale": "Operator checked original model release card",
        }
    )


def _audit(*, second: bool = False, unknown: bool = False) -> WorkflowAudit:
    first = MissingWorkflowAsset(
        label="checkpoint", node_class="CheckpointLoaderSimple",
        input_name="ckpt_name", requested="example.safetensors",
    )
    another = MissingWorkflowAsset(
        label="vae", node_class="VAELoader",
        input_name="vae_name", requested="other.safetensors",
    )
    return WorkflowAudit(
        ready=False, comfyui_url="http://comfy.test:8188",
        custom_node_folders=(),
        guidance=(),
        entries=(
            WorkflowAuditEntry(
                template_id="illust_main_v1", source_sha256="f" * 64,
                ready=False, required_node_count=2, required_asset_count=2,
                missing_node_types=(),
                missing_assets=(
                    (another,) if second else ()
                ) if unknown else ((first, another) if second else (first,)),
                unverifiable_assets=(first,) if unknown else (),
            ),
        ),
    )


def _hub(
    paths: tuple[str, ...] = ("models/example.safetensors",),
    *,
    repos: tuple[str, ...] = (GOOD,),
    sha: str = HASH,
    requests: list[str] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if requests is not None:
            requests.append(request.method + " " + url)
        for repo in repos:
            api = "https://huggingface.co/api/models/" + repo
            if url == api and request.method == "GET":
                return httpx.Response(
                    200, json={
                        "id": repo, "sha": REV, "gated": False,
                        "cardData": {"license": "mit"},
                        "siblings": [{"rfilename": "README.md"}]
                        + [{"rfilename": p} for p in paths],
                    }
                )
            if url == api + "/paths-info/" + REV and request.method == "POST":
                selected = request.read()
                path = next(p for p in paths if p.encode() in selected)
                return httpx.Response(
                    200, json=[
                        {
                            "type": "file", "path": path, "size": 512,
                            "lfs": {"size": 512, "sha256": sha},
                        },
                    ],
                )
        raise AssertionError("Unexpected HTTP request: " + url)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_source_registry_round_trip_and_atomic_add_remove(tmp_path: Path) -> None:
    store = tmp_path / "config" / "model-sources.json"
    assert read_model_source_registry(store, if_missing_empty=True).sources == ()
    binding = _source(hash=HASH)
    registry = register_model_source(ModelSourceRegistry(), binding)
    save_model_source_registry(store, registry)
    loaded = read_model_source_registry(store)
    assert loaded.sources == (binding,)
    assert loaded.schema_version == 1
    with pytest.raises(FileExistsError, match="already exists"):
        register_model_source(loaded, _source(repo=UNRELATED))
    with pytest.raises(ValueError, match="No matching"):
        unregister_model_source(
            loaded, role="checkpoint", model_name="example.safetensors",
            repository=UNRELATED,
        )
    removed = unregister_model_source(
        loaded, role="checkpoint", model_name="example.safetensors",
        repository=GOOD,
    )
    save_model_source_registry(store, removed)
    assert not read_model_source_registry(store).sources
    assert not list(store.parent.glob(".artifex-source-*"))


@pytest.mark.parametrize(
    "payload",
    [
        {"model_name": "../model.safetensors"},
        {"model_name": "sub/model.safetensors"},
        {"repository": "https://huggingface.co/a/b"},
        {"repository": "../../evil"},
        {"file_path": "../models/example.safetensors"},
        {"file_path": "weights/another.safetensors"},
        {"expected_sha256": "xyz"},
        {"rationale": "x"},
        {"role": "detector_model"},
    ],
)
def test_reject_unsafe_or_unknown_registration_fields(payload: dict[str, Any]) -> None:
    fields = _source().model_dump()
    fields.update(payload)
    with pytest.raises(ValidationError):
        ApprovedModelSource.model_validate(fields)


def test_duplicate_registration_by_role_and_case_is_refused() -> None:
    first = _source()
    copy = _source(name="Example.safetensors", file_path=None)
    with pytest.raises(ValidationError, match="Only one explicitly"):
        ModelSourceRegistry(sources=(first, copy))


def test_registry_symlinks_are_never_read_or_overwritten(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    save_model_source_registry(real, ModelSourceRegistry())
    link = tmp_path / "model-sources.json"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink privilege unavailable")
    with pytest.raises(ValueError, match="symlinked"):
        read_model_source_registry(link)
    with pytest.raises(ValueError, match="symlinked"):
        save_model_source_registry(link, ModelSourceRegistry())
    assert real.is_file()


def test_registered_publisher_only_is_queried_even_if_copycat_has_same_name() -> None:
    calls: list[str] = []
    with _hub(repos=(GOOD, UNRELATED), requests=calls) as client:
        draft = draft_hf_models(
            _audit(),
            repositories=(),
            bindings=(_source(),),
            client=client,
        )
    assert draft.repositories_checked == (GOOD,)
    assert len(draft.manifest.artifacts) == 1
    assert draft.evidence[0].source_repository == GOOD
    assert all(UNRELATED not in call for call in calls)


def test_explicit_registered_path_resolves_same_named_paths_without_guessing() -> None:
    paths = ("models/example.safetensors", "old/example.safetensors")
    with _hub(paths=paths) as client:
        draft = draft_hf_models(
            _audit(), repositories=(), bindings=(_source(),), client=client
        )
    assert draft.manifest.artifacts[0].url.endswith(
        "/" + REV + "/models/example.safetensors"
    )


def test_registered_hash_mismatch_blocks_manifest() -> None:
    with _hub(sha=OTHER_HASH) as client:
        draft = draft_hf_models(
            _audit(), repositories=(),
            bindings=(_source(hash=HASH),), client=client,
        )
    assert not draft.manifest.artifacts
    assert "differs from registered expected hash" in draft.unresolved[0].candidates[0]


def test_registered_source_file_path_missing_does_not_fallback_to_other_file() -> None:
    with _hub(paths=("archive/example.safetensors",)) as client:
        draft = draft_hf_models(
            _audit(), repositories=(), bindings=(_source(),), client=client
        )
    assert not draft.manifest.artifacts
    assert "Registered exact model source path is absent" in draft.unresolved[0].candidates[0]


def test_unregistered_missing_models_are_reported_not_crossmatched() -> None:
    with _hub() as client:
        draft = draft_hf_models(
            _audit(second=True), repositories=(), bindings=(_source(),), client=client
        )
    assert len(draft.manifest.artifacts) == 1
    assert len(draft.unresolved) == 1
    assert draft.unresolved[0].requested == "other.safetensors"
    assert "No operator-approved" in draft.unresolved[0].reason


def test_no_runtime_missing_asset_leads_to_zero_external_requests() -> None:
    requests: list[str] = []
    with _hub(requests=requests) as client:
        draft = draft_hf_models(
            _audit(unknown=True), repositories=(),
            bindings=(_source(),), client=client,
        )
    assert not draft.manifest.artifacts
    assert not requests
    assert "confirm" in draft.unresolved[0].reason


def test_cli_source_add_list_remove_explicit_approval_and_no_overwrite(
    tmp_path: Path,
) -> None:
    store = tmp_path / "config" / "model-sources.json"
    runner = CliRunner()
    args = [
        "--role", "checkpoint", "--name", "example.safetensors",
        "--repo", GOOD, "--rationale", "Checked model card and publisher",
        "--source-file", "models/example.safetensors", "--registry", str(store),
    ]
    denied = runner.invoke(app, ["onboard", "model-sources-add", *args])
    assert denied.exit_code == 1
    assert not store.exists()
    added = runner.invoke(
        app, ["onboard", "model-sources-add", *args, "--approve"]
    )
    assert added.exit_code == 0, added.output
    listed = runner.invoke(
        app, ["onboard", "model-sources-list", "--registry", str(store)]
    )
    assert listed.exit_code == 0
    assert '"repository": "publisher/approved-model"' in listed.output
    duplicate = runner.invoke(
        app, ["onboard", "model-sources-add", *args, "--approve"]
    )
    assert duplicate.exit_code == 1
    denied_removal = runner.invoke(
        app, [
            "onboard", "model-sources-remove", "--role", "checkpoint",
            "--name", "example.safetensors", "--repo", GOOD,
            "--registry", str(store),
        ],
    )
    assert denied_removal.exit_code == 1
    removed = runner.invoke(
        app, [
            "onboard", "model-sources-remove", "--role", "checkpoint",
            "--name", "example.safetensors", "--repo", GOOD,
            "--registry", str(store), "--confirm",
        ],
    )
    assert removed.exit_code == 0, removed.output
    assert not read_model_source_registry(store).sources


def test_cli_models_draft_defaults_to_saved_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_audit(settings: Any) -> WorkflowAudit:
        return _audit()

    monkeypatch.setattr("artifex.cli.audit_workflows", fake_audit)
    root = tmp_path / "config"
    root.mkdir()
    store = root / "model-sources.json"
    save_model_source_registry(
        store, ModelSourceRegistry(sources=(_source(),))
    )
    with _hub() as client:
        monkeypatch.setattr(
            "artifex.cli.draft_hf_models",
            lambda audit, **kwargs: draft_hf_models(audit, client=client, **kwargs),
        )
        output = tmp_path / "models.json"
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            app, ["onboard", "models-draft", "--output", str(output)]
        )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert GOOD in output.read_text(encoding="utf-8")
    assert read_model_source_registry(store).sources


def test_cli_manual_repo_mode_still_works_and_cannot_mix_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_audit(settings: Any) -> WorkflowAudit:
        return _audit()

    monkeypatch.setattr("artifex.cli.audit_workflows", fake_audit)
    with _hub() as client:
        monkeypatch.setattr(
            "artifex.cli.draft_hf_models",
            lambda audit, **kwargs: draft_hf_models(audit, client=client, **kwargs),
        )
        result = CliRunner().invoke(
            app, [
                "onboard", "models-draft",
                "--repo", GOOD, "--output", str(tmp_path / "manual.json"),
            ],
        )
    assert result.exit_code == 0, result.output
    invalid = CliRunner().invoke(
        app, [
            "onboard", "models-draft",
            "--repo", GOOD, "--sources", str(tmp_path / "absent.json"),
        ],
    )
    assert invalid.exit_code == 1
    assert "either --repo shortlist or --sources registry" in invalid.output
