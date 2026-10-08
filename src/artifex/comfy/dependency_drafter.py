from __future__ import annotations

import base64
import binascii
import hashlib
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.comfy.dependency_installer import (
    DependencyManifest,
    PinnedDependency,
    _unpack_node,
)
from artifex.comfy.dependency_resolver import DependencyResolution, NodeCandidate

_GITHUB_API = "https://api.github.com/repos/"
_GITHUB_CDN = "https://codeload.github.com/"
_FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_SPDX = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,99}")
_MAX_REPOS = 12
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_LICENSE_LIMIT = 1024 * 1024


class DraftRejection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_url: str
    matched_node_types: tuple[str, ...]
    reason: str


class DraftArtifactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    repository_url: str
    matched_node_types: tuple[str, ...]
    commit: str
    archive_sha256: str
    archive_bytes: int
    license_id: str
    license_url: str
    license_file_sha256: str


class DependencyDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_commit: str
    manifest: DependencyManifest
    evidence: tuple[DraftArtifactEvidence, ...]
    skipped: tuple[DraftRejection, ...]
    missing_model_choices: tuple[str, ...]
    unverifiable_model_choices: tuple[str, ...]
    unresolved_node_types: tuple[str, ...]
    notes: tuple[str, ...] = (
        "License identifiers are GitHub's classifications of committed license "
        "files, not legal approval; check upstream terms before --accept-licenses.",
        "Only unambiguous registered node providers are drafted. "
        "No arbitrary third-party code is installed or executed.",
        "Missing checkpoint/VAE/upscale files are not inferred from filename alone.",
    )


def _get_json(http: httpx.Client, url: str) -> dict[str, Any]:
    response = http.get(url, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    raw: Any = response.json()
    if not isinstance(raw, dict):
        raise TypeError("Official GitHub API returned a non-object payload")
    return raw


def _repo(candidate: NodeCandidate) -> tuple[str, str]:
    parsed = urlsplit(candidate.repository_url)
    path = parsed.path.strip("/")
    components = path.split("/")
    if (
        parsed.scheme != "https" or parsed.hostname != "github.com"
        or parsed.port is not None or parsed.username or parsed.password
        or parsed.query or parsed.fragment or len(components) != 2
        or any(not _ID.fullmatch(part) or part in {".", ".."} for part in components)
    ):
        raise ValueError("Candidate is not an exact GitHub owner/repository URL")
    return components[0], components[1]


def _pin_default_branch(http: httpx.Client, owner: str, repo: str) -> str:
    api = _GITHUB_API + owner + "/" + repo
    data = _get_json(http, api)
    if data.get("full_name", "").casefold() != f"{owner}/{repo}".casefold():
        raise ValueError("GitHub API repository identity does not match candidate")
    default_branch = data.get("default_branch")
    if not isinstance(default_branch, str) or not (
        0 < len(default_branch) <= 100
    ):
        raise ValueError("GitHub repository has no valid default branch")
    commit = _get_json(
        http, api + "/commits/" + quote(default_branch, safe="")
    ).get("sha")
    if not isinstance(commit, str) or not _FULL_SHA.fullmatch(commit):
        raise ValueError("GitHub did not provide a full immutable commit")
    return commit.lower()


def _license_at_commit(
    http: httpx.Client, owner: str, repo: str, commit: str
) -> tuple[str, str, bytes, str]:
    api = _GITHUB_API + owner + "/" + repo
    data = _get_json(http, api + "/license?ref=" + commit)
    license_info = data.get("license")
    if not isinstance(license_info, dict):
        raise ValueError("Pinned commit has no GitHub-recognized license")
    spdx = license_info.get("spdx_id")
    if (
        not isinstance(spdx, str)
        or not _SPDX.fullmatch(spdx)
        or spdx.upper() in {"NOASSERTION", "NONE", "OTHER"}
    ):
        raise ValueError("Pinned commit has no conclusive SPDX license identifier")
    path = data.get("path")
    if not isinstance(path, str) or len(path) > 220:
        raise ValueError("GitHub license file path is invalid")
    safe_path = PurePosixPath(path)
    if (
        safe_path.is_absolute() or not safe_path.parts or ".." in safe_path.parts
        or any(":" in item or "\\" in item for item in safe_path.parts)
    ):
        raise ValueError("GitHub license path is unsafe")
    if data.get("encoding") != "base64":
        raise ValueError("GitHub did not include verifiable license bytes")
    content = data.get("content")
    if not isinstance(content, str) or len(content) > (_LICENSE_LIMIT * 2):
        raise ValueError("GitHub license file is missing or oversized")
    try:
        decoded = base64.b64decode(content, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("GitHub license file content is not base64") from exc
    if not decoded or len(decoded) > _LICENSE_LIMIT:
        raise ValueError("GitHub license file content size is invalid")
    url = (
        f"https://github.com/{owner}/{repo}/blob/{commit}/"
        + quote(path, safe="/")
    )
    return spdx, path, decoded, url


def _archive_hash_and_verify(
    http: httpx.Client, url: str, expected_license_path: str,
    expected_license_bytes: bytes, staging: Path, *,
    max_archive_bytes: int,
) -> tuple[str, int, str]:
    archive = staging / "source.zip"
    digest = hashlib.sha256()
    length = 0
    with http.stream("GET", url) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Pinned source archive did not return a complete HTTP 200")
        with archive.open("xb") as output:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                length += len(chunk)
                if length > max_archive_bytes:
                    raise ValueError("Pinned custom-node ZIP exceeds download limit")
                digest.update(chunk)
                output.write(chunk)
    if length <= 0:
        raise ValueError("Pinned custom-node archive is empty")
    folder = staging / "unpacked"
    folder.mkdir()
    # Reuse the exact extraction and Python package checks enforced by the
    # actual installer. The downloaded archive is never imported/executed.
    _unpack_node(archive, folder)
    license_file = folder.joinpath(*PurePosixPath(expected_license_path).parts)
    if not license_file.is_file() or license_file.is_symlink():
        raise ValueError("GitHub-identified license is absent from pinned source ZIP")
    license_bytes = license_file.read_bytes()
    if license_bytes != expected_license_bytes:
        raise ValueError("License bytes differ between GitHub API and pinned ZIP")
    return digest.hexdigest(), length, hashlib.sha256(license_bytes).hexdigest()


def _try_candidate(
    http: httpx.Client, candidate: NodeCandidate, *,
    max_archive_bytes: int,
) -> tuple[PinnedDependency, DraftArtifactEvidence]:
    owner, repo = _repo(candidate)
    commit = _pin_default_branch(http, owner, repo)
    spdx, license_path, license_bytes, license_url = _license_at_commit(
        http, owner, repo, commit
    )
    archive_url = f"{_GITHUB_CDN}{owner}/{repo}/zip/{commit}"
    with tempfile.TemporaryDirectory(prefix="artifex-dependency-draft-") as temp:
        archive_sha256, archive_size, license_sha = _archive_hash_and_verify(
            http, archive_url, license_path, license_bytes,
            Path(temp), max_archive_bytes=max_archive_bytes,
        )
    item = PinnedDependency(
        id=f"node-{repo}",
        kind="custom_node",
        name=repo,
        repository=f"{owner}/{repo}",
        commit=commit,
        url=archive_url,
        size_bytes=archive_size,
        sha256=archive_sha256,
        license_id=spdx,
        license_url=license_url,
    )
    evidence = DraftArtifactEvidence(
        id=item.id,
        repository_url=candidate.repository_url,
        matched_node_types=candidate.matched_node_types,
        commit=commit,
        archive_sha256=archive_sha256,
        archive_bytes=archive_size,
        license_id=spdx,
        license_url=license_url,
        license_file_sha256=license_sha,
    )
    return item, evidence


def draft_missing_node_manifest(
    resolution: DependencyResolution,
    *, max_repositories: int = 3,
    max_archive_mib: int = 128,
    client: httpx.Client | None = None,
) -> DependencyDraft:
    """Make a verified *draft*, never run the discovered third-party Python.

    Only candidates with uniquely mapped missing classes are evaluated.
    A missing, ambiguous, or unverified license, commit or archive excludes
    that candidate completely. No model weights are guessed from filenames.
    """
    if not 1 <= max_repositories <= _MAX_REPOS:
        raise ValueError("max_repositories must be between 1 and 12")
    if not 1 <= max_archive_mib <= 512:
        raise ValueError("max_archive_mib must be between 1 and 512")
    counts = Counter(
        node for candidate in resolution.candidates
        for node in candidate.matched_node_types
    )
    approved: list[NodeCandidate] = []
    skipped: list[DraftRejection] = []
    for candidate in resolution.candidates:
        if not candidate.matched_node_types or any(
            counts[node] != 1 for node in candidate.matched_node_types
        ):
            skipped.append(
                DraftRejection(
                    repository_url=candidate.repository_url,
                    matched_node_types=candidate.matched_node_types,
                    reason="Ambiguous node-to-repository mapping; manual review required",
                )
            )
        elif len(approved) >= max_repositories:
            skipped.append(
                DraftRejection(
                    repository_url=candidate.repository_url,
                    matched_node_types=candidate.matched_node_types,
                    reason="Candidate exceeds explicit per-run repository limit",
                )
            )
        else:
            approved.append(candidate)
    own = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=20, read=120, write=30, pool=30),
        follow_redirects=True, trust_env=False,
    )
    items: list[PinnedDependency] = []
    evidence: list[DraftArtifactEvidence] = []
    try:
        for candidate in approved:
            try:
                item, proof = _try_candidate(
                    http, candidate,
                    max_archive_bytes=max_archive_mib * 1024 * 1024,
                )
                if any(
                    prev.id.casefold() == item.id.casefold()
                    or prev.name.casefold() == item.name.casefold()
                    for prev in items
                ):
                    raise ValueError("Destination collision among verified candidates")
                items.append(item)
                evidence.append(proof)
            except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
                skipped.append(
                    DraftRejection(
                        repository_url=candidate.repository_url,
                        matched_node_types=candidate.matched_node_types,
                        reason=f"Could not verify pinned source/license/archive: {exc}",
                    )
                )
    finally:
        if own:
            http.close()
    resolved = {
        node for proof in evidence for node in proof.matched_node_types
    }
    unresolved = set(resolution.missing_node_types).difference(resolved)
    return DependencyDraft(
        registry_commit=resolution.manager_commit,
        manifest=DependencyManifest(artifacts=tuple(items)),
        evidence=tuple(evidence),
        skipped=tuple(skipped),
        missing_model_choices=resolution.missing_model_choices,
        unverifiable_model_choices=resolution.unverifiable_model_choices,
        unresolved_node_types=tuple(sorted(unresolved)),
    )
