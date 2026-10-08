from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy import dependency_installer, dependency_resolver
from artifex.comfy.workflow_audit import (
    MissingWorkflowAsset,
    WorkflowAudit,
    WorkflowAuditEntry,
)

MANAGER_SHA = "a" * 40
NODE_SHA = "b" * 40
MODEL_SHA = "c" * 40
REPO = "example-owner/example-node"
NODE_URL = f"https://codeload.github.com/{REPO}/zip/{NODE_SHA}"
MODEL_URL = (
    f"https://huggingface.co/example-owner/model/resolve/{MODEL_SHA}/demo.safetensors"
)
SOURCE = (
    f"https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/{MANAGER_SHA}/"
)


def _audit() -> WorkflowAudit:
    asset = MissingWorkflowAsset(
        label="checkpoint", node_class="CheckpointLoaderSimple",
        input_name="ckpt_name", requested="demo.safetensors",
        available=("other.safetensors",),
    )
    return WorkflowAudit(
        ready=False, comfyui_url="http://render.test",
        custom_node_folders=(),
        guidance=("missing dependencies",),
        entries=(
            WorkflowAuditEntry(
                template_id="illust_main_v1", source_sha256="f" * 64,
                ready=False, required_node_count=10,
                required_asset_count=1,
                missing_node_types=("SpecialNode", "UnknownNode"),
                missing_assets=(asset,), unverifiable_assets=(),
            ),
        ),
    )


def _manager_http(requests: list[str], *,
                  with_index: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/commits/main"):
            return httpx.Response(200, json={"sha": MANAGER_SHA})
        if request.url.path.endswith("extension-node-map.json"):
            assert with_index
            return httpx.Response(
                200,
                json={
                    f"https://github.com/{REPO}": [
                        ["SpecialNode", "OtherNode"],
                        {"description": "Demo test nodes"},
                    ],
                    "https://attacker.invalid/repo": [["UnknownNode"], {}],
                    "https://github.com/not-listed/nope": [["UnknownNode"], {}],
                },
            )
        if request.url.path.endswith("custom-node-list.json"):
            return httpx.Response(
                200,
                json={
                    "custom_nodes": [
                        {
                            "name": "demo",
                            "files": [f"https://github.com/{REPO}"],
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resolver_pins_registry_commit_and_crosschecks_source_list() -> None:
    requests: list[str] = []
    with _manager_http(requests) as client:
        plan = dependency_resolver.resolve_missing_dependencies(
            _audit(), client=client
        )
    assert plan.manager_commit == MANAGER_SHA
    assert len(plan.candidates) == 1
    assert plan.candidates[0].repository_url == f"https://github.com/{REPO}"
    assert plan.candidates[0].matched_node_types == ("SpecialNode",)
    assert not plan.candidates[0].installation_ready
    assert plan.unresolved_node_types == ("UnknownNode",)
    assert plan.missing_model_choices == ("checkpoint:demo.safetensors",)
    assert requests == [
        dependency_resolver._MANAGER_COMMIT,
        SOURCE + "extension-node-map.json",
        SOURCE + "custom-node-list.json",
    ]


def test_resolver_explicit_commit_never_fetches_latest() -> None:
    requests: list[str] = []
    with _manager_http(requests) as client:
        plan = dependency_resolver.resolve_missing_dependencies(
            _audit(), client=client, manager_commit=MANAGER_SHA
        )
    assert plan.manager_commit == MANAGER_SHA
    assert all("/commits/main" not in value for value in requests)


def test_resolver_without_missing_nodes_does_not_download_indexes() -> None:
    audit = WorkflowAudit(
        ready=True, comfyui_url="http://render.test",
        custom_node_folders=(), entries=(), guidance=(),
    )
    requests: list[str] = []
    with _manager_http(requests) as client:
        plan = dependency_resolver.resolve_missing_dependencies(audit, client=client)
    assert not plan.candidates
    assert requests == [dependency_resolver._MANAGER_COMMIT]


def test_resolver_disallows_unpinned_reference_without_http() -> None:
    with _manager_http([]) as client, pytest.raises(ValueError, match="40-character"):
        dependency_resolver.resolve_missing_dependencies(
            _audit(), client=client, manager_commit="../../master"
        )


def _model_bytes() -> bytes:
    return b"SAFE_TEST_MODEL" * 20


def _node_zip(path: str = "node-bbbb/__init__.py") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path, "NODE_CLASS_MAPPINGS = {}")
    return out.getvalue()


def _manifest(model: bytes, node: bytes | None = None) -> dependency_installer.DependencyManifest:
    raw: list[dict[str, Any]] = [
        {
            "id": "demo-model",
            "kind": "model", "name": "demo.safetensors",
            "model_folder": "checkpoints",
            "url": MODEL_URL, "size_bytes": len(model),
            "sha256": hashlib.sha256(model).hexdigest(),
            "license_id": "MIT",
            "license_url": "https://example.org/mit",
        }
    ]
    if node is not None:
        raw.append(
            {
                "id": "demo-node",
                "kind": "custom_node", "name": "example-node",
                "repository": REPO, "commit": NODE_SHA,
                "url": NODE_URL, "size_bytes": len(node),
                "sha256": hashlib.sha256(node).hexdigest(),
                "license_id": "Apache-2.0",
                "license_url": "https://example.org/license",
            }
        )
    return dependency_installer.DependencyManifest.model_validate(
        {"schema_version": 1, "artifacts": raw}
    )


def _comfy(tmp_path: Path) -> Path:
    folder = tmp_path / "isolated" / "ComfyUI"
    folder.mkdir(parents=True)
    (folder / "main.py").write_text("# source", encoding="utf-8")
    (folder.parent / ".artifex-comfy-install.json").write_text(
        '{"source":"Comfy-Org/ComfyUI"}', encoding="utf-8"
    )
    return folder


def _download_http(
    model: bytes,
    node: bytes | None = None,
    calls: list[str] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        if str(request.url) == MODEL_URL:
            return httpx.Response(200, content=model)
        if str(request.url) == NODE_URL and node is not None:
            return httpx.Response(200, content=node)
        raise AssertionError(f"Unexpected request {request.url}")
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_model_installer_needs_approval_and_only_targets_owned_comfy(tmp_path: Path) -> None:
    data = _model_bytes()
    manifest = _manifest(data)
    root = _comfy(tmp_path)
    with _download_http(data) as client, pytest.raises(ValueError, match="accept-licenses"):
        dependency_installer.install_manifest(manifest, root, client=client)
    other = tmp_path / "unowned" / "ComfyUI"
    other.mkdir(parents=True)
    (other / "main.py").touch()
    with _download_http(data) as client, pytest.raises(ValueError, match="isolated"):
        dependency_installer.install_manifest(
            manifest, other, accept_licenses=True, client=client
        )
    assert not (root / "models").exists()


def test_checksum_verified_model_install_and_no_overwrite(tmp_path: Path) -> None:
    data = _model_bytes()
    manifest = _manifest(data)
    root = _comfy(tmp_path)
    calls: list[str] = []
    with _download_http(data, calls=calls) as client:
        result = dependency_installer.install_manifest(
            manifest, root, accept_licenses=True, client=client
        )
        assert not result.restart_required
        dest = root / "models" / "checkpoints" / "demo.safetensors"
        assert dest.read_bytes() == data
        with pytest.raises(FileExistsError, match="overwrite"):
            dependency_installer.install_manifest(
                manifest, root, accept_licenses=True, client=client
            )
    assert calls == [MODEL_URL]


def test_custom_node_installer_needs_code_approval_and_does_not_run_scripts(
    tmp_path: Path,
) -> None:
    model, node = _model_bytes(), _node_zip()
    root = _comfy(tmp_path)
    manifest = _manifest(model, node)
    with _download_http(model, node) as client:
        with pytest.raises(ValueError, match="allow-custom-code"):
            dependency_installer.install_manifest(
                manifest, root, accept_licenses=True, client=client
            )
        result = dependency_installer.install_manifest(
            manifest, root, accept_licenses=True,
            allow_custom_code=True, client=client,
        )
    assert len(result.installed) == 2
    assert result.restart_required
    assert (root / "custom_nodes" / "example-node" / "__init__.py").is_file()


def test_download_sha_failure_leaves_model_and_node_targets_untouched(
    tmp_path: Path,
) -> None:
    data = _model_bytes()
    root = _comfy(tmp_path)
    manifest = _manifest(data)
    with _download_http(b"tampered") as client, pytest.raises(ValueError, match="size/SHA256"):
        dependency_installer.install_manifest(
            manifest, root, accept_licenses=True, client=client
        )
    assert not (root / "models" / "checkpoints" / "demo.safetensors").exists()


def test_zip_slip_is_rejected_before_any_artifacts_installed(tmp_path: Path) -> None:
    root = _comfy(tmp_path)
    node = _node_zip("node-bbbb/../../escape.py")
    model = _model_bytes()
    manifest = _manifest(model, node)
    with _download_http(model, node) as client, pytest.raises(ValueError, match="Unsafe"):
        dependency_installer.install_manifest(
            manifest, root, accept_licenses=True,
            allow_custom_code=True, client=client,
        )
    assert not (root / "models" / "checkpoints" / "demo.safetensors").exists()
    assert not (root / "escape.py").exists()


def test_manifest_path_validation_and_symlink_prevention(tmp_path: Path) -> None:
    model = _model_bytes()
    raw = _manifest(model).model_dump(mode="json")
    raw["artifacts"][0]["name"] = "../escape.py"
    with pytest.raises(ValidationError):
        dependency_installer.DependencyManifest.model_validate(raw)
    raw = _manifest(model).model_dump(mode="json")
    raw["artifacts"][0]["url"] = "https://attacker.invalid/payload"
    with pytest.raises(ValidationError):
        dependency_installer.DependencyManifest.model_validate(raw)
    raw = _manifest(model).model_dump(mode="json")
    raw["artifacts"][0]["sha256"] = "0" * 64
    root = _comfy(tmp_path)
    # Rejected at verification, with no final model file.
    with _download_http(model) as client, pytest.raises(ValueError):
        dependency_installer.install_manifest(
            dependency_installer.DependencyManifest.model_validate(raw),
            root, accept_licenses=True, client=client,
        )


def test_manifest_no_duplicate_ids_or_destinations() -> None:
    first = _manifest(_model_bytes()).model_dump(mode="json")
    first["artifacts"].append(dict(first["artifacts"][0]))
    with pytest.raises(ValidationError, match="unique"):
        dependency_installer.DependencyManifest.model_validate(first)
    second = _manifest(_model_bytes()).model_dump(mode="json")
    dup = dict(second["artifacts"][0])
    dup["id"] = "other-id"
    second["artifacts"].append(dup)
    with pytest.raises(ValidationError, match="destinations"):
        dependency_installer.DependencyManifest.model_validate(second)


def test_cli_install_is_dry_run_by_default(tmp_path: Path) -> None:
    manifest = _manifest(_model_bytes())
    path = tmp_path / "approved.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(
        app, [
            "onboard", "deps-install", "--manifest", str(path),
            "--comfy-root", str(tmp_path / "does-not-exist"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"dry_run": true' in result.output
    assert "demo-model" in result.output
    assert not (tmp_path / "does-not-exist").exists()


def test_cli_install_calls_apply_only_after_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(_model_bytes())
    path = tmp_path / "approved.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake_install(manifest: Any, root: Path, **kwargs: Any) -> dependency_installer.DependencyInstallResult:
        seen.update(kwargs)
        return dependency_installer.DependencyInstallResult(
            comfyui_directory=root, installed=(), restart_required=False
        )

    monkeypatch.setattr("artifex.cli.install_manifest", fake_install)
    result = CliRunner().invoke(
        app, [
            "onboard", "deps-install", "--manifest", str(path),
            "--comfy-root", str(tmp_path), "--apply", "--accept-licenses",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["accept_licenses"] is True
    assert seen["allow_custom_code"] is False
