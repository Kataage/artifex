from __future__ import annotations

import os

import httpx

from artifex.config.models import RenderNodeConfig
from artifex.render_node.models import RenderNodeAttestation


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
