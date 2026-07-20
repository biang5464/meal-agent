from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolErrorCode(str, Enum):
    TIMEOUT = "TIMEOUT"
    NETWORK = "NETWORK"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    PARSE_ERROR = "PARSE_ERROR"
    INTERNAL = "INTERNAL"


@dataclass
class ToolError:
    code: ToolErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMeta:
    tool_name: str
    elapsed_ms: int = 0
    attempts: int = 1
    fallback_used: bool = False
    source: str = ""
    trace_id: str = ""


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: ToolError | None = None
    meta: ToolMeta | None = None


def tool_success(
    data: Any,
    *,
    tool_name: str,
    elapsed_ms: int = 0,
    attempts: int = 1,
    source: str = "",
    trace_id: str = "",
) -> ToolResult:
    return ToolResult(
        ok=True,
        data=data,
        error=None,
        meta=ToolMeta(
            tool_name=tool_name,
            elapsed_ms=elapsed_ms,
            attempts=attempts,
            fallback_used=False,
            source=source,
            trace_id=trace_id,
        )
    )


def tool_failure(
    code: ToolErrorCode,
    message: str,
    *,
    tool_name: str,
    retryable: bool = False,
    fallback_data: Any = None,
    fallback_used: bool = False,
    elapsed_ms: int = 0,
    attempts: int = 1,
    source: str = "",
    trace_id: str = "",
    details: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        ok=False,
        data=fallback_data,
        error=ToolError(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        ),
        meta=ToolMeta(
            tool_name=tool_name,
            elapsed_ms=elapsed_ms,
            attempts=attempts,
            fallback_used=fallback_used,
            source=source,
            trace_id=trace_id,
        )
    )
