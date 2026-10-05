from __future__ import annotations

from enum import StrEnum


class ComfyErrorKind(StrEnum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    INVALID_WORKFLOW = "invalid_workflow"
    EXECUTION = "execution"
    PROTOCOL = "protocol"
    CANCELLED = "cancelled"


class ComfyUIError(RuntimeError):
    def __init__(
        self,
        kind: ComfyErrorKind,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class ComfyUITimeoutError(ComfyUIError):
    def __init__(self, message: str) -> None:
        super().__init__(ComfyErrorKind.TIMEOUT, message, retryable=True)


class ComfyUIProtocolError(ComfyUIError):
    def __init__(self, message: str) -> None:
        super().__init__(ComfyErrorKind.PROTOCOL, message, retryable=False)


class ComfyUIExecutionError(ComfyUIError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(ComfyErrorKind.EXECUTION, message, retryable=retryable)
