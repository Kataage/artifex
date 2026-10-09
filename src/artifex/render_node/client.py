from __future__ import annotations

import os

import httpx

from artifex.config.models import RenderNodeConfig
from artifex.render_node.models import (
    RemoteOwnerReadinessEvidence,
    RemoteRendererOwnerAudit,
    RemoteSurvivalTrace,
    RenderNodeAttestation,
)
from artifex.render_node.startup_inspection import RemoteRendererSafetyInspection


def fetch_render_attestation(
    node_id: str,
    config: RenderNodeConfig,
    *,
    timeout_seconds: float = 120.0,
    fresh: bool = False,
    client: httpx.Client | None = None,
) -> RenderNodeAttestation:
    if not config.attestation_url:
        raise ValueError(f"render node {node_id} has no attestation_url configured")
    headers: dict[str, str] = {}
    if config.attestation_token_env:
        token = os.environ.get(config.attestation_token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    url = config.attestation_url.rstrip("/") + "/v1/attestation"
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds))
    try:
        response = http.get(
            url,
            headers=headers,
            params={"fresh": "1"} if fresh else None,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            http.close()
    attestation = RenderNodeAttestation.model_validate(payload)
    if attestation.node_id != node_id:
        raise ValueError(
            f"render node attestation id mismatch: {attestation.node_id} != {node_id}"
        )
    return attestation

def fetch_renderer_owner_audit(
    node_id: str,
    config: RenderNodeConfig,
    *,
    timeout_seconds: float = 35.0,
    client: httpx.Client | None = None,
) -> RemoteRendererOwnerAudit:
    """Read actual current PC-B ownership over existing authenticated attestation.

    Caller must *never* treat the result as restart permission or a GPU
    production qualification. The request is read-only and non-cached.
    """
    if not config.attestation_url:
        raise ValueError(f"render node {node_id} has no attestation_url configured")
    if not config.attestation_token_env:
        raise ValueError("remote owner audit requires a configured attestation token")
    token = os.environ.get(config.attestation_token_env)
    if not token:
        raise ValueError("remote owner audit token is missing")
    url = config.attestation_url.rstrip("/") + "/v1/owner-audit"
    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )
    try:
        response = http.get(
            url, headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds, follow_redirects=False,
        )
        response.raise_for_status()
        if len(response.content) > 64 * 1024:
            raise ValueError("remote owner audit exceeds size limit")
        observed = RemoteRendererOwnerAudit.model_validate(response.json())
    finally:
        if owns_client:
            http.close()
    if observed.node_id != node_id:
        raise ValueError("remote owner audit node ID does not match configured node")
    return observed


def fetch_renderer_safety_inspection(
    node_id: str,
    config: RenderNodeConfig,
    *,
    timeout_seconds: float = 90.0,
    client: httpx.Client | None = None,
) -> RemoteRendererSafetyInspection:
    """Fetch uncached PC-B three-port safety over the existing Bearer channel.

    This is strictly observational; even `owned_observed` never authorizes
    startup, reattachment, GPU submission or production qualification.
    """
    if not config.attestation_url:
        raise ValueError(f"render node {node_id} has no attestation_url configured")
    if not config.attestation_token_env:
        raise ValueError("remote safety requires a configured attestation token")
    token = os.environ.get(config.attestation_token_env)
    if not token:
        raise ValueError("remote safety attestation token is missing")
    url = config.attestation_url.rstrip("/") + "/v1/renderer-safety"
    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )
    try:
        response = http.get(
            url, headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds, follow_redirects=False,
        )
        response.raise_for_status()
        if len(response.content) > 64 * 1024:
            raise ValueError("remote safety response exceeds size limit")
        observation = RemoteRendererSafetyInspection.model_validate(response.json())
    finally:
        if owns_client:
            http.close()
    if observation.node_id != node_id:
        raise ValueError("remote safety node ID mismatch")
    if observation.inspection.renderer_node_id != node_id:
        raise ValueError("remote safety inspection renderer ID mismatch")
    return observation


def fetch_remote_owner_readiness(
    node_id: str,
    config: RenderNodeConfig,
    *,
    timeout_seconds: float = 35.0,
    client: httpx.Client | None = None,
) -> RemoteOwnerReadinessEvidence:
    """Read memory-cached PC-B no-GPU evidence; never ask PC-B to run a probe."""
    if not config.attestation_url:
        raise ValueError("render node has no attestation_url configured")
    if not config.attestation_token_env:
        raise ValueError("remote owner evidence requires a configured token")
    token = os.environ.get(config.attestation_token_env)
    if not token:
        raise ValueError("remote owner evidence Bearer token is missing")
    url = config.attestation_url.rstrip("/") + "/v1/owner-readiness-evidence"
    owns = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )
    try:
        response = http.get(
            url, headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds, follow_redirects=False,
        )
        response.raise_for_status()
        if len(response.content) > 128 * 1024:
            raise ValueError("remote owner evidence exceeds response size bound")
        evidence = RemoteOwnerReadinessEvidence.model_validate(response.json())
    finally:
        if owns:
            http.close()
    if evidence.node_id != node_id:
        raise ValueError("remote owner evidence node ID mismatch")
    return evidence



def fetch_remote_survival_trace(
    node_id: str,
    config: RenderNodeConfig,
    *,
    timeout_seconds: float = 35.0,
    client: httpx.Client | None = None,
) -> RemoteSurvivalTrace:
    """Obtain the newest saved PC-B report, never triggering a local probe."""
    if not config.attestation_url:
        raise ValueError("render node has no attestation_url configured")
    if not config.attestation_token_env:
        raise ValueError("remote survival requires configured Bearer token")
    token = os.environ.get(config.attestation_token_env)
    if not token:
        raise ValueError("remote survival Bearer token is missing")
    url = config.attestation_url.rstrip("/") + "/v1/survival-evidence"
    owns = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False, trust_env=False,
    )
    try:
        # Do not follow redirects or accept unbounded HTTP payloads.
        with http.stream(
            "GET", url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds, follow_redirects=False,
        ) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for part in response.iter_bytes():
                total += len(part)
                if total > 13 * 1024 * 1024:
                    raise ValueError("remote survival response exceeds size limit")
                chunks.append(part)
        received = RemoteSurvivalTrace.model_validate_json(b"".join(chunks))
    finally:
        if owns:
            http.close()
    if received.node_id != node_id:
        raise ValueError("remote survival node ID mismatch")
    return received
