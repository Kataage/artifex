from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from artifex.comfy.admission import ComfySubmissionFence
from artifex.comfy.errors import (
    ComfyErrorKind,
    ComfyUIError,
    ComfyUIExecutionError,
    ComfyUIProtocolError,
    ComfyUITimeoutError,
)
from artifex.comfy.models import (
    ComfyExecutionResult,
    ComfyHealth,
    ComfyOutput,
    QueueReceipt,
    WorkflowPatchRequest,
    WorkflowRequirements,
    WorkflowRequirementStatus,
)
from artifex.comfy.templates import WorkflowTemplateLike
from artifex.config.models import ComfyUiConfig


class ComfyUIClient:
    def __init__(
        self,
        config: ComfyUiConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._gateway_token: str | None = None
        if config.gateway_token_env is not None:
            self._gateway_token = os.environ.get(config.gateway_token_env)
            if not self._gateway_token:
                raise ValueError(
                    f"ComfyUI gateway token environment variable is missing: "
                    f"{config.gateway_token_env}"
                )
        self._submission_fence = ComfySubmissionFence(config.submission_fence_path)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    async def health(self) -> ComfyHealth:
        try:
            body = await self._request_json("GET", "/system_stats")
        except ComfyUIError as exc:
            return ComfyHealth(available=False, detail=str(exc))

        system = body.get("system")
        devices_raw = body.get("devices", ())
        version: str | None = None
        devices: list[str] = []

        if isinstance(system, Mapping):
            raw_version = system.get("comfyui_version")
            if raw_version is not None:
                version = str(raw_version)
        if isinstance(devices_raw, list):
            for device in devices_raw:
                if isinstance(device, Mapping) and device.get("name") is not None:
                    devices.append(str(device["name"]))

        return ComfyHealth(
            available=True,
            version=version,
            devices=tuple(devices),
        )

    async def submit(
        self,
        graph: dict[str, dict[str, Any]],
        *,
        client_id: str | None = None,
    ) -> QueueReceipt:
        payload: dict[str, Any] = {"prompt": graph}
        if client_id is not None:
            payload["client_id"] = client_id

        # The lock covers the COMPLETE submit, including HTTP retries. An
        # administrator's seal cannot overtake an already-started /prompt.
        async with self._submission_fence.admit():
            body = await self._request_json("POST", "/prompt", json=payload)

        raw_error = body.get("error")
        node_errors = body.get("node_errors")
        if raw_error or (isinstance(node_errors, Mapping) and node_errors):
            detail = raw_error if raw_error else node_errors
            raise ComfyUIError(
                ComfyErrorKind.INVALID_WORKFLOW,
                f"ComfyUI rejected workflow: {detail}",
                retryable=False,
            )

        prompt_id = body.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyUIProtocolError("ComfyUI /prompt response missing prompt_id")

        queue_number_raw = body.get("number")
        queue_number = (
            float(queue_number_raw)
            if isinstance(queue_number_raw, (int, float))
            else None
        )
        return QueueReceipt(prompt_id=prompt_id, queue_number=queue_number)

    async def execute(
        self,
        template: WorkflowTemplateLike,
        patch: WorkflowPatchRequest,
        *,
        client_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ComfyExecutionResult:
        graph = template.patch(patch)
        receipt = await self.submit(graph, client_id=client_id)
        return await self.wait_for_completion(
            receipt.prompt_id,
            timeout_seconds=timeout_seconds,
        )

    async def get_history(
        self,
        prompt_id: str,
    ) -> ComfyExecutionResult | None:
        body = await self._request_json("GET", f"/history/{prompt_id}")
        raw_entry = body.get(prompt_id)
        if raw_entry is None:
            return None
        if not isinstance(raw_entry, Mapping):
            raise ComfyUIProtocolError(
                f"history entry for {prompt_id} must be an object"
            )

        raw_status = raw_entry.get("status", {})
        if not isinstance(raw_status, Mapping):
            raise ComfyUIProtocolError(
                f"history status for {prompt_id} must be an object"
            )
        status = str(raw_status.get("status_str", "unknown"))
        completed = bool(raw_status.get("completed", False))

        if status.casefold() in {"error", "failed"}:
            raise ComfyUIExecutionError(
                self._execution_error_message(prompt_id, raw_status)
            )

        outputs = self._discover_outputs(raw_entry.get("outputs", {}))
        return ComfyExecutionResult(
            prompt_id=prompt_id,
            completed=completed,
            status=status,
            outputs=outputs,
            raw_status=dict(raw_status),
        )

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ComfyExecutionResult:
        timeout = (
            self._config.execution_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        deadline = time.monotonic() + timeout
        while True:
            result = await self.get_history(prompt_id)
            if result is not None and result.completed:
                return result
            if time.monotonic() >= deadline:
                raise ComfyUITimeoutError(
                    f"ComfyUI execution timed out: {prompt_id}"
                )
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def cancel(self, prompt_id: str) -> None:
        # Pending work is removed from the queue; the targeted interrupt handles
        # the same prompt if it is already running.
        await self._request_no_content(
            "POST",
            "/queue",
            json={"delete": [prompt_id]},
        )
        await self._request_no_content(
            "POST",
            "/interrupt",
            json={"prompt_id": prompt_id},
        )

    async def queue_snapshot(self) -> dict[str, Any]:
        return await self._request_json("GET", "/queue")

    async def object_info(self) -> dict[str, Any]:
        return await self._request_json("GET", "/object_info")

    async def download_output(
        self,
        output: ComfyOutput,
        destination_dir: Path,
    ) -> Path:
        """Stream an output through /view, publishing it only after full delivery."""
        root = destination_dir.expanduser().resolve(strict=False)
        relative = Path(output.subfolder.replace("\\", "/")) / output.filename
        target = (root / relative).resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ComfyUIProtocolError(
                f"ComfyUI output escapes download directory: {relative}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        params = {
            "filename": output.filename,
            "subfolder": output.subfolder,
            "type": output.output_type,
        }
        for attempt in range(self._config.request_attempts):
            try:
                total = 0
                async with self._client.stream("GET", "/view", params=params) as response:
                    response.raise_for_status()
                    raw_size = response.headers.get("Content-Length")
                    expected_size = int(raw_size) if raw_size is not None else None
                    if expected_size is not None and expected_size < 0:
                        raise ComfyUIProtocolError("negative ComfyUI output length")
                    with temporary.open("wb") as stream:
                        async for chunk in response.aiter_bytes():
                            stream.write(chunk)
                            total += len(chunk)
                if total == 0:
                    raise ComfyUIProtocolError("ComfyUI returned an empty image")
                if expected_size is not None and total != expected_size:
                    raise ComfyUIProtocolError(
                        f"ComfyUI output truncated: {total} != {expected_size}"
                    )
                temporary.replace(target)
                return target
            except (
                httpx.HTTPError,
                ValueError,
                ComfyUIProtocolError,
            ) as exc:
                temporary.unlink(missing_ok=True)
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code < 500
                ):
                    raise ComfyUIProtocolError(
                        f"ComfyUI /view rejected image: {exc}"
                    ) from exc
                if attempt + 1 >= self._config.request_attempts:
                    raise ComfyUIError(
                        ComfyErrorKind.CONNECTION,
                        f"ComfyUI /view download failed after retries: {exc}",
                        retryable=True,
                    ) from exc
                await asyncio.sleep(
                    self._config.reconnect_backoff_seconds * (attempt + 1)
                )
        raise AssertionError("unreachable ComfyUI download retry state")

    async def validate_requirements(
        self,
        requirements: WorkflowRequirements,
    ) -> WorkflowRequirementStatus:
        info = await self.object_info()
        missing_node_types = tuple(
            sorted(
                node_type
                for node_type in requirements.node_types
                if node_type not in info
            )
        )
        missing_assets: list[str] = []
        for asset in requirements.assets:
            node_info = info.get(asset.node_class)
            if not isinstance(node_info, Mapping):
                continue
            options = self._input_choices(node_info, asset.input_name)
            if not options:
                continue
            if not any(
                self._asset_name_matches(asset.value, available)
                for available in options
            ):
                missing_assets.append(f"{asset.label}:{asset.value}")

        missing_assets_tuple = tuple(sorted(dict.fromkeys(missing_assets)))
        ready = not missing_node_types and not missing_assets_tuple
        detail_parts: list[str] = []
        if missing_node_types:
            detail_parts.append(
                "missing nodes=" + ", ".join(missing_node_types)
            )
        if missing_assets_tuple:
            detail_parts.append(
                "missing assets=" + ", ".join(missing_assets_tuple)
            )
        if not detail_parts:
            detail_parts.append(
                f"{len(requirements.node_types)} node types and "
                f"{len(requirements.assets)} assets validated"
            )
        return WorkflowRequirementStatus(
            ready=ready,
            missing_node_types=missing_node_types,
            missing_assets=missing_assets_tuple,
            detail="; ".join(detail_parts),
        )

    async def free_memory(
        self,
        *,
        unload_models: bool = True,
        free_memory: bool = True,
    ) -> None:
        await self._request_no_content(
            "POST",
            "/free",
            json={
                "unload_models": unload_models,
                "free_memory": free_memory,
            },
        )

    @staticmethod
    def _input_choices(
        node_info: Mapping[str, Any],
        input_name: str,
    ) -> tuple[str, ...]:
        raw_input = node_info.get("input")
        if not isinstance(raw_input, Mapping):
            return ()
        schema: object | None = None
        for section in ("required", "optional"):
            values = raw_input.get(section)
            if isinstance(values, Mapping) and input_name in values:
                schema = values[input_name]
                break
        if not isinstance(schema, list | tuple) or not schema:
            return ()
        first = schema[0]
        if not isinstance(first, list | tuple):
            return ()
        return tuple(
            str(value)
            for value in first
            if isinstance(value, str)
        )

    @staticmethod
    def _asset_name_matches(configured: str, available: str) -> bool:
        def normalize(value: str) -> str:
            return Path(value.replace("\\", "/")).stem.casefold()

        return (
            configured.casefold() == available.casefold()
            or normalize(configured) == normalize(available)
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._request(method, path, json=json)
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIProtocolError(
                f"ComfyUI returned non-JSON response for {path}"
            ) from exc
        if not isinstance(body, dict):
            raise ComfyUIProtocolError(
                f"ComfyUI returned non-object JSON for {path}"
            )
        return body

    async def _request_no_content(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> None:
        await self._request(method, path, json=json)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        # /prompt is a non-idempotent GPU enqueue. If the response is lost,
        # the server may already have accepted the work; silently repeating
        # the POST would create expensive duplicate renders.
        ambiguous_submit = method.upper() == "POST" and path == "/prompt"
        attempts = 1 if ambiguous_submit else self._config.request_attempts

        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                    headers=(
                        {"Authorization": f"Bearer {self._gateway_token}"}
                        if self._gateway_token is not None else None
                    ),
                )
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"ComfyUI server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    if path == "/prompt":
                        # Preserve the structured validation body for submit().
                        return response
                    response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                await asyncio.sleep(
                    self._config.reconnect_backoff_seconds * (attempt + 1)
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                await asyncio.sleep(
                    self._config.reconnect_backoff_seconds * (attempt + 1)
                )

        assert last_error is not None
        raise ComfyUIError(
            ComfyErrorKind.CONNECTION,
            (
                "ComfyUI /prompt outcome is unknown; do not automatically "
                "resubmit potentially accepted GPU work"
                if ambiguous_submit else
                f"ComfyUI request failed after retries: {method} {path}: {last_error}"
            ),
            retryable=not ambiguous_submit,
        ) from last_error

    @staticmethod
    def _discover_outputs(raw_outputs: object) -> tuple[ComfyOutput, ...]:
        if not isinstance(raw_outputs, Mapping):
            raise ComfyUIProtocolError("ComfyUI history outputs must be an object")

        outputs: list[ComfyOutput] = []
        for raw_node_id, raw_node in raw_outputs.items():
            if not isinstance(raw_node, Mapping):
                continue
            images = raw_node.get("images", ())
            if not isinstance(images, list):
                continue
            for image in images:
                if not isinstance(image, Mapping):
                    continue
                filename = image.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                outputs.append(
                    ComfyOutput(
                        node_id=str(raw_node_id),
                        filename=filename,
                        subfolder=str(image.get("subfolder", "")),
                        output_type=str(image.get("type", "output")),
                    )
                )
        return tuple(outputs)

    @staticmethod
    def _execution_error_message(
        prompt_id: str,
        status: Mapping[str, Any],
    ) -> str:
        messages = status.get("messages")
        if isinstance(messages, list) and messages:
            return f"ComfyUI execution failed for {prompt_id}: {messages[-1]}"
        return f"ComfyUI execution failed for {prompt_id}"
