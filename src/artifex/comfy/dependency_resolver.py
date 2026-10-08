from __future__ import annotations

import re
from collections import defaultdict
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.comfy.workflow_audit import WorkflowAudit

_MANAGER_BASE = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/"
_MANAGER_COMMIT = "https://api.github.com/repos/Comfy-Org/ComfyUI-Manager/commits/main"
_LIMIT = 10 * 1024 * 1024
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GITHUB_PATH = re.compile(r"/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?")


class NodeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_url: str
    matched_node_types: tuple[str, ...]
    description: str = ""
    confidence: str = "registry_mapping_only"
    installation_ready: bool = False


class DependencyResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manager_commit: str
    source: str
    missing_node_types: tuple[str, ...]
    candidates: tuple[NodeCandidate, ...]
    unresolved_node_types: tuple[str, ...]
    missing_model_choices: tuple[str, ...]
    unverifiable_model_choices: tuple[str, ...]
    ready: bool = False
    warnings: tuple[str, ...] = (
        "Third-party index entries are suggestions, not verified licenses, "
        "compatible versions, safe install scripts, or SHA-256-pinned releases.",
    )


def _github_repository(value: str) -> str | None:
    url = urlsplit(value)
    if url.scheme != "https" or url.hostname not in {"github.com", "www.github.com"}:
        return None
    if url.username or url.password or url.query or url.fragment or url.port:
        return None
    match = _GITHUB_PATH.fullmatch(url.path)
    if match is None:
        return None
    owner, name = match.groups()
    if name.endswith(".git"):
        name = name[:-4]
    if not owner or not name or owner == "." or name == ".":
        return None
    return f"https://github.com/{owner}/{name}"


def _safe_json_download(http: httpx.Client, url: str) -> Any:
    with http.stream("GET", url) as response:
        response.raise_for_status()
        size = 0
        parts: list[bytes] = []
        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
            size += len(chunk)
            if size > _LIMIT:
                raise ValueError("Pinned ComfyUI Manager index exceeds 10 MiB")
            parts.append(chunk)
    import json

    try:
        return json.loads(b"".join(parts))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("ComfyUI Manager index is not valid JSON") from exc


def _manager_commit(http: httpx.Client) -> str:
    response = http.get(_MANAGER_COMMIT)
    response.raise_for_status()
    data: Any = response.json()
    if not isinstance(data, dict):
        raise ValueError("GitHub commit response was not an object")
    sha = data.get("sha")
    if not isinstance(sha, str) or not _COMMIT.fullmatch(sha):
        raise ValueError("GitHub did not provide an immutable manager commit")
    return sha


def _candidate_registry(
    node_map: Any, known_sources: Any, required: set[str],
) -> tuple[tuple[NodeCandidate, ...], tuple[str, ...]]:
    if not isinstance(node_map, dict):
        raise ValueError("ComfyUI Manager extension-node-map is not an object")
    # Listing file is an independent cross-check that this repository is in
    # the same pinned Manager index. Node class names alone are not evidence.
    if not isinstance(known_sources, dict) or not isinstance(
        known_sources.get("custom_nodes"), list
    ):
        raise ValueError("ComfyUI Manager custom-node-list has invalid schema")
    allow: set[str] = set()
    for item in known_sources["custom_nodes"]:
        if not isinstance(item, dict) or not isinstance(item.get("files"), list):
            continue
        for value in item["files"]:
            if isinstance(value, str):
                parsed = _github_repository(value)
                if parsed:
                    allow.add(parsed.casefold())
    grouped: dict[str, set[str]] = defaultdict(set)
    descriptions: dict[str, str] = {}
    for extension, value in node_map.items():
        if not isinstance(extension, str) or not isinstance(value, list):
            continue
        repository = _github_repository(extension)
        if repository is None or repository.casefold() not in allow:
            continue
        if not value or not isinstance(value[0], list):
            continue
        mapped = {node for node in value[0] if isinstance(node, str) and node in required}
        if not mapped:
            continue
        grouped[repository].update(mapped)
        meta = value[1] if len(value) > 1 else None
        if isinstance(meta, dict) and isinstance(meta.get("description"), str):
            descriptions[repository] = meta["description"][:200]
    ordered = sorted(grouped)
    candidates = tuple(
        NodeCandidate(
            repository_url=repository,
            matched_node_types=tuple(sorted(grouped[repository])),
            description=descriptions.get(repository, ""),
        )
        for repository in ordered[:150]
    )
    found = {name for item in candidates for name in item.matched_node_types}
    return candidates, tuple(sorted(required.difference(found)))


def resolve_missing_dependencies(
    audit: WorkflowAudit, *, manager_commit: str | None = None,
    client: httpx.Client | None = None,
) -> DependencyResolution:
    """Use one immutable Manager commit and corroborated repository mapping.

    The registry is *discovery only*. No suggested repository is auto-executed
    or installed and no third-party metadata is accepted as license approval.
    """
    missing = {
        node for entry in audit.entries for node in entry.missing_node_types
    }
    absent = {
        f"{asset.label}:{asset.requested}"
        for entry in audit.entries for asset in entry.missing_assets
    }
    unknown = {
        f"{asset.label}:{asset.requested}"
        for entry in audit.entries for asset in entry.unverifiable_assets
    }
    owned = client is None
    http = client or httpx.Client(
        follow_redirects=True, timeout=httpx.Timeout(30.0), trust_env=False
    )
    try:
        pinned = manager_commit or _manager_commit(http)
        pinned = pinned.lower()
        if not _COMMIT.fullmatch(pinned):
            raise ValueError("Manager reference must be a full 40-character commit SHA")
        candidates: tuple[NodeCandidate, ...] = ()
        unresolved: tuple[str, ...] = ()
        if missing:
            base = _MANAGER_BASE + pinned + "/"
            node_map = _safe_json_download(http, base + "extension-node-map.json")
            sources = _safe_json_download(http, base + "custom-node-list.json")
            candidates, unresolved = _candidate_registry(node_map, sources, missing)
        return DependencyResolution(
            manager_commit=pinned,
            source=_MANAGER_BASE + pinned + "/",
            missing_node_types=tuple(sorted(missing)),
            candidates=candidates,
            unresolved_node_types=unresolved,
            missing_model_choices=tuple(sorted(absent)),
            unverifiable_model_choices=tuple(sorted(unknown)),
        )
    finally:
        if owned:
            http.close()
