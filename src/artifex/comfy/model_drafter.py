from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.comfy.dependency_installer import DependencyManifest, PinnedDependency
from artifex.comfy.dependency_resolver import DependencyResolution
from artifex.comfy.workflow_audit import WorkflowAudit

_HF_API = "https://huggingface.co/api/models/"
_HF_BASE = "https://huggingface.co/"
_REPOSITORY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}"
)
_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_DIGEST = re.compile(r"[0-9a-fA-F]{64}")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_LICENSE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]{1,99}")
_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".pt", ".pth", ".onnx"})
_FOLDER_BY_LABEL = {
    "checkpoint": "checkpoints",
    "refiner": "checkpoints",
    "vae": "vae",
    "upscale_model": "upscale_models",
    "lora": "loras",
    "static_lora": "loras",
}
_MAX_SOURCES = 12
_MAX_SIBLINGS = 12000
_MAX_SIZE = 32 * 1024**3


class ModelSourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_folder: str
    source_repository: str
    source_file_path: str
    source_commit: str
    expected_sha256: str
    expected_size_bytes: int
    license_id: str
    license_url: str
    detail: str = "Hugging Face LFS metadata; file payload not downloaded"


class UnresolvedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    requested: str
    reason: str
    candidates: tuple[str, ...] = ()


class ModelManifestDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: DependencyManifest
    evidence: tuple[ModelSourceEvidence, ...]
    unresolved: tuple[UnresolvedModel, ...]
    repositories_checked: tuple[str, ...]
    warnings: tuple[str, ...] = (
        "A repository appearing in an operator-supplied shortlist is not proof "
        "of original authorship or redistribution permission.",
        "Hub license metadata does not constitute a legal review; confirm "
        "the repository card and author terms before --accept-licenses.",
        "Weights are not downloaded while drafting. The separate installer "
        "verifies every byte against the recorded LFS SHA-256 on --apply.",
    )


class _HubFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    path: str
    filename: str
    revision: str
    size: int = Field(gt=0, le=_MAX_SIZE)
    sha256: str
    license_id: str
    license_url: str


def _clean_repo(repo: str) -> str:
    if not _REPOSITORY.fullmatch(repo) or ".." in repo.split("/"):
        raise ValueError("Model source must be a literal owner/repository ID")
    return repo


def _clean_file_path(path: str) -> bool:
    segments = path.split("/")
    return bool(
        1 <= len(segments) <= 8
        and all(_SEGMENT.fullmatch(segment) and segment not in {".", ".."} for segment in segments)
    )


def _validated_license(meta: dict[str, Any]) -> str:
    card = meta.get("cardData")
    license_id = card.get("license") if isinstance(card, dict) else None
    if not isinstance(license_id, str) or not _LICENSE.fullmatch(license_id):
        raise ValueError("Source repo has no verifiable model card license ID")
    if license_id.casefold() in {"unknown", "other", "none", "noassertion"}:
        raise ValueError("Model card license is not specific enough for a draft")
    return license_id


def _read_repo(http: httpx.Client, repo: str) -> tuple[str, str, dict[str, tuple[str, ...]]]:
    """Read bounded HF metadata; collect paths by exact filename, not fuzzy search."""
    result = http.get(_HF_API + repo)
    result.raise_for_status()
    meta: Any = result.json()
    if not isinstance(meta, dict):
        raise TypeError("Hugging Face model metadata is not an object")
    if str(meta.get("id", "")).casefold() != repo.casefold():
        raise ValueError("Hugging Face model identity mismatch")
    if meta.get("gated") not in (False, None):
        raise ValueError("Gated or restricted model repository was not allowed")
    revision = meta.get("sha")
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise ValueError("No immutable 40-character Hub commit in metadata")
    license_id = _validated_license(meta)
    siblings = meta.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > _MAX_SIBLINGS:
        raise ValueError("Hugging Face repository file index is absent or too large")
    names: dict[str, list[str]] = defaultdict(list)
    for entry in siblings:
        if not isinstance(entry, dict):
            continue
        file_path = entry.get("rfilename")
        if not isinstance(file_path, str) or not _clean_file_path(file_path):
            continue
        basename = file_path.rsplit("/", 1)[-1]
        names[basename].append(file_path)
    if "README.md" not in names or "README.md" not in names["README.md"]:
        raise ValueError("Source repository has no model card README.md")
    return revision.lower(), license_id, {key: tuple(value) for key, value in names.items()}


def _file_info(
    http: httpx.Client, repo: str, revision: str, file_path: str,
    license_id: str,
) -> _HubFile:
    result = http.post(
        _HF_API + repo + "/paths-info/" + revision,
        json={"paths": [file_path]},
        headers={"Accept": "application/json"},
    )
    result.raise_for_status()
    raw: Any = result.json()
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("Pinned Hub file path information was not uniquely returned")
    data = raw[0]
    lfs = data.get("lfs")
    if data.get("path") != file_path or data.get("type") != "file":
        raise ValueError("Hub path info disagrees with chosen exact filename")
    if not isinstance(lfs, dict):
        raise ValueError("No LFS SHA-256 metadata; refusing unverifiable model file")
    sha256, lfs_size = lfs.get("sha256"), lfs.get("size")
    actual_size = data.get("size")
    if (
        not isinstance(sha256, str) or not _DIGEST.fullmatch(sha256)
        or not isinstance(lfs_size, int) or not isinstance(actual_size, int)
        or lfs_size != actual_size or not 0 < lfs_size <= _MAX_SIZE
    ):
        raise ValueError("Hub LFS digest/size missing, conflicting or out of bounds")
    return _HubFile(
        repo=repo,
        path=file_path,
        filename=file_path.rsplit("/", 1)[-1],
        revision=revision,
        size=lfs_size,
        sha256=sha256.lower(),
        license_id=license_id,
        license_url=f"{_HF_BASE}{repo}/blob/{revision}/README.md",
    )


def _needed(
    audit: WorkflowAudit,
) -> tuple[set[tuple[str, str]], list[UnresolvedModel]]:
    missing: set[tuple[str, str]] = set()
    warnings: list[UnresolvedModel] = []
    seen: set[tuple[str, str]] = set()
    for entry in audit.entries:
        for asset in (*entry.missing_assets, *entry.unverifiable_assets):
            label, requested = asset.label, asset.requested
            key = (label, requested)
            if key in seen:
                continue
            seen.add(key)
            filename = PurePosixPath(requested).name
            if (
                label not in _FOLDER_BY_LABEL or not _SEGMENT.fullmatch(requested)
                or filename != requested or filename in {".", ".."} or len(filename) > 100
                or PurePosixPath(filename).suffix.casefold() not in _EXTENSIONS
            ):
                warnings.append(
                    UnresolvedModel(
                        label=label, requested=requested,
                        reason="Unsupported model role, path, filename or file extension",
                    )
                )
            elif asset in entry.unverifiable_assets:
                warnings.append(
                    UnresolvedModel(
                        label=label, requested=requested,
                        reason="Runtime loader did not enumerate this model; confirm "
                        "whether it is actually missing before downloading",
                    )
                )
            else:
                missing.add(key)
    return missing, warnings


def draft_hf_models(
    audit: WorkflowAudit, *, repositories: tuple[str, ...],
    client: httpx.Client | None = None,
) -> ModelManifestDraft:
    """Draft only exact, unique files from explicitly shortlisted HF publishers.

    No speculative filename-to-publisher search. Hub LFS SHA256 and pinned
    revision are used without downloading multi-GB model weights.
    """
    if not repositories or len(repositories) > _MAX_SOURCES:
        raise ValueError("Provide between 1 and 12 shortlisted Hugging Face repositories")
    clean = tuple(_clean_repo(repo) for repo in repositories)
    if len({value.casefold() for value in clean}) != len(clean):
        raise ValueError("Duplicate model publisher repository in shortlist")
    pending, unresolved = _needed(audit)
    owned = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=20, read=30, write=20, pool=20),
        follow_redirects=False, trust_env=False,
    )
    found: dict[tuple[str, str], list[_HubFile]] = defaultdict(list)
    failures: dict[str, str] = {}
    try:
        for repo in clean:
            try:
                revision, license_id, index = _read_repo(http, repo)
                for _, name in sorted(pending):
                    paths = index.get(name, ())
                    if len(paths) > 1:
                        failures[repo + ":" + name] = (
                            "Multiple files in the same repository share this basename"
                        )
                    elif len(paths) == 1:
                        try:
                            found_name = _file_info(
                                http, repo, revision, paths[0], license_id
                            )
                            for key in pending:
                                if key[1] == name:
                                    found[key].append(found_name)
                        except (ValueError, TypeError, httpx.HTTPError) as exc:
                            failures[repo + ":" + name] = str(exc)
            except (ValueError, TypeError, httpx.HTTPError) as exc:
                failures[repo] = str(exc)
    finally:
        if owned:
            http.close()
    artifacts: list[PinnedDependency] = []
    proof: list[ModelSourceEvidence] = []
    for label, requested in sorted(pending):
        candidates = found.get((label, requested), [])
        if len(candidates) != 1:
            cause = (
                "Ambiguous: multiple shortlisted repositories contain the exact filename"
                if len(candidates) > 1 else
                "No unique SHA-256 verified candidate in shortlisted repositories"
            )
            unresolved.append(
                UnresolvedModel(
                    label=label, requested=requested, reason=cause,
                    candidates=tuple(
                        [f"{file.repo}@{file.revision}:{file.path}" for file in candidates]
                        + [f"{key}: {msg}" for key, msg in sorted(failures.items())]
                    )[:32],
                )
            )
            continue
        file = candidates[0]
        model_folder = _FOLDER_BY_LABEL[label]
        item = PinnedDependency(
            id=(
                f"model-{model_folder}-{file.sha256[:12]}-"
                + hashlib.sha256(requested.encode("utf-8")).hexdigest()[:8]
            ),
            kind="model",
            name=requested,
            model_folder=model_folder,
            url=f"{_HF_BASE}{file.repo}/resolve/{file.revision}/"
                + "/".join(quote(part, safe="") for part in file.path.split("/")),
            sha256=file.sha256,
            size_bytes=file.size,
            license_id=file.license_id,
            license_url=file.license_url,
        )
        artifacts.append(item)
        proof.append(
            ModelSourceEvidence(
                model_name=requested, model_folder=model_folder,
                source_repository=file.repo, source_file_path=file.path,
                source_commit=file.revision, expected_sha256=file.sha256,
                expected_size_bytes=file.size, license_id=file.license_id,
                license_url=file.license_url,
            )
        )
    return ModelManifestDraft(
        manifest=DependencyManifest(artifacts=tuple(artifacts)),
        evidence=tuple(proof),
        unresolved=tuple(unresolved),
        repositories_checked=clean,
    )
