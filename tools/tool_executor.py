from __future__ import annotations

import asyncio
import time
import logging
from typing import TYPE_CHECKING, Callable, Any, TypeVar

if TYPE_CHECKING:
    from core.tool_protocol import ToolResult

logger = logging.getLogger(__name__)
T = TypeVar("T")

# 哨兵：区分"调用方未传 fallback_data"与"调用方显式传 None 作为 fallback"
_FALLBACK_UNSET = object()


class ToolExecutor:
    """
    统一工具执行器。
    封装 Timeout / Retry（指数退避）/ Fallback / Logging。
    所有外部工具调用通过此执行器统一处理。
    """

    async def run(
        self,
        tool: Callable,
        *args,
        timeout: float = 8.0,
        retries: int = 2,
        fallback: Any = None,
        tool_name: str = "",
        retry_on: tuple = (asyncio.TimeoutError, ConnectionError, OSError),
        **kwargs,
    ) -> Any:
        """
        执行工具调用，自动处理超时/重试/降级。

        参数：
        - tool: 可调用对象（async 或 sync）
        - timeout: 单次超时秒数
        - retries: 最大重试次数（不含首次）
        - fallback: 所有重试失败后的返回值
        - tool_name: 用于日志标识
        - retry_on: 触发重试的异常类型（不重试4xx类业务错误）
        """
        name = tool_name or getattr(tool, "__name__", "unknown_tool")
        last_exception = None

        for attempt in range(retries + 1):
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(tool):
                    result = await asyncio.wait_for(
                        tool(*args, **kwargs),
                        timeout=timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(tool, *args, **kwargs),
                        timeout=timeout,
                    )
                elapsed = time.time() - start
                logger.info(f"[tool] {name} 成功 attempt={attempt + 1} elapsed={elapsed:.2f}s")
                return result

            except asyncio.TimeoutError as e:
                elapsed = time.time() - start
                logger.warning(f"[tool] {name} 超时 attempt={attempt + 1} elapsed={elapsed:.2f}s")
                last_exception = e
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # 指数退避 0.5s → 1s → 2s
                continue

            except retry_on as e:
                elapsed = time.time() - start
                logger.warning(f"[tool] {name} 错误 attempt={attempt + 1} error={e} elapsed={elapsed:.2f}s")
                last_exception = e
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                continue

            except Exception as e:
                # 非重试类错误（业务错误）直接返回 fallback
                elapsed = time.time() - start
                logger.error(f"[tool] {name} 不可重试错误 error={e} elapsed={elapsed:.2f}s")
                return fallback

        logger.error(f"[tool] {name} 所有重试失败，使用 fallback")
        return fallback

    async def execute(
        self,
        tool: Callable,
        *args,
        policy_name: str,
        tool_name: str = "",
        fallback_data: Any = _FALLBACK_UNSET,
        source: str = "",
        trace_id: str = "",
        **kwargs,
    ) -> ToolResult:
        """
        新协议执行器：返回 ToolResult，不返回裸值。

        - 通过 policy_name 从 TOOL_POLICIES 获取超时 / 重试 / 退避策略
        - 非幂等策略（idempotent=False）强制 retries=0
        - 错误映射：
            TimeoutError          → TIMEOUT,        retryable=True
            ConnectionError/OSError → NETWORK,       retryable=True
            ValueError/TypeError  → INVALID_INPUT,  retryable=False（立即返回）
            其他                  → INTERNAL,        retryable=False（立即返回）
        """
        from core.tool_policy import get_tool_policy
        from core.tool_protocol import ToolErrorCode, tool_failure, tool_success

        policy = get_tool_policy(policy_name)
        name = tool_name or getattr(tool, "__name__", "unknown_tool")

        # 非幂等策略强制不重试
        effective_retries = 0 if not policy.idempotent else policy.retries

        _has_fallback = fallback_data is not _FALLBACK_UNSET
        _fallback_value = None if not _has_fallback else fallback_data

        total_start = time.perf_counter()
        last_code = ToolErrorCode.INTERNAL
        last_msg = "unknown error"
        last_retryable = False

        for attempt in range(effective_retries + 1):
            try:
                if asyncio.iscoroutinefunction(tool):
                    result = await asyncio.wait_for(
                        tool(*args, **kwargs),
                        timeout=policy.timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(tool, *args, **kwargs),
                        timeout=policy.timeout,
                    )
                elapsed_ms = int((time.perf_counter() - total_start) * 1000)
                logger.info(f"[execute] {name} ok attempt={attempt + 1} elapsed={elapsed_ms}ms")
                return tool_success(
                    result,
                    tool_name=name,
                    elapsed_ms=elapsed_ms,
                    attempts=attempt + 1,
                    source=source,
                    trace_id=trace_id,
                )

            except asyncio.TimeoutError:
                # 必须先于 OSError 捕获（Python 3.11+ TimeoutError 是 OSError 子类）
                last_code = ToolErrorCode.TIMEOUT
                last_msg = f"超时（>{policy.timeout}s）"
                last_retryable = True
                logger.warning(f"[execute] {name} TIMEOUT attempt={attempt + 1}")

            except (ConnectionError, OSError) as exc:
                last_code = ToolErrorCode.NETWORK
                last_msg = str(exc)
                last_retryable = True
                logger.warning(f"[execute] {name} NETWORK attempt={attempt + 1} error={exc}")

            except (ValueError, TypeError) as exc:
                elapsed_ms = int((time.perf_counter() - total_start) * 1000)
                logger.error(f"[execute] {name} INVALID_INPUT error={exc}")
                return tool_failure(
                    ToolErrorCode.INVALID_INPUT,
                    str(exc),
                    tool_name=name,
                    retryable=False,
                    fallback_data=_fallback_value,
                    fallback_used=_has_fallback,
                    elapsed_ms=elapsed_ms,
                    attempts=attempt + 1,
                    source=source,
                    trace_id=trace_id,
                )

            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - total_start) * 1000)
                logger.error(f"[execute] {name} INTERNAL error={exc}")
                return tool_failure(
                    ToolErrorCode.INTERNAL,
                    str(exc),
                    tool_name=name,
                    retryable=False,
                    fallback_data=_fallback_value,
                    fallback_used=_has_fallback,
                    elapsed_ms=elapsed_ms,
                    attempts=attempt + 1,
                    source=source,
                    trace_id=trace_id,
                )

            # 可重试错误：退避后继续下一次尝试
            if attempt < effective_retries:
                await asyncio.sleep(policy.backoff_base * (2 ** attempt))

        # 所有重试耗尽
        elapsed_ms = int((time.perf_counter() - total_start) * 1000)
        logger.error(f"[execute] {name} 所有重试失败 fallback_used={_has_fallback}")
        return tool_failure(
            last_code,
            last_msg,
            tool_name=name,
            retryable=last_retryable,
            fallback_data=_fallback_value,
            fallback_used=_has_fallback,
            elapsed_ms=elapsed_ms,
            attempts=effective_retries + 1,
            source=source,
            trace_id=trace_id,
        )


# 全局单例
tool_executor = ToolExecutor()
