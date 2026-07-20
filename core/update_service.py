"""
core/update_service.py
Phase 6B：用户画像写入服务层，遵循 ToolResult 协议。

所有 impl 函数：同步，内部创建并关闭 Session，由 ToolExecutor 经 asyncio.to_thread 包装。
所有 result 函数：异步，调用 tool_executor.execute() 返回 ToolResult。
"""
from __future__ import annotations

import json
from datetime import datetime

from core.tool_protocol import ToolResult


# ── 用户画像 Upsert ──────────────────────────────────────────────────────────

def _update_user_profile_impl(user_id: str, field: str, value: str) -> dict:
    from core.database import SessionLocal
    from models.user import (
        UserProfileORM,
        orm_to_dict,
        UPDATABLE_FIELDS,
        LIST_FIELDS,
        VALID_BUDGETS,
        encode_list,
    )

    if field not in UPDATABLE_FIELDS:
        raise ValueError(f"不支持的字段: '{field}'，可选字段：{sorted(UPDATABLE_FIELDS)}")
    if field == "budget" and value not in VALID_BUDGETS:
        raise ValueError(f"budget 只能是 {VALID_BUDGETS} 之一，收到 '{value}'")

    with SessionLocal() as session:
        row = session.get(UserProfileORM, user_id)
        if row is None:
            row = UserProfileORM(user_id=user_id)
            session.add(row)
            session.flush()

        if field in LIST_FIELDS:
            parsed = json.loads(value) if isinstance(value, str) else value
            if not isinstance(parsed, list):
                raise ValueError(f"列表字段需要 JSON 数组，收到 {type(parsed)}")
            setattr(row, field, encode_list([str(x) for x in parsed]))
        else:
            setattr(row, field, value)

        row.updated_at = datetime.utcnow()
        session.commit()
        return {"database_updated": True, **orm_to_dict(row)}


async def update_user_profile_result(user_id: str, field: str, value: str) -> ToolResult:
    """用户画像字段 Upsert（含创建），返回 ToolResult。database_write 策略，不重试。

    - ok=True,  data={"database_updated": True, ...} → DB commit 成功
    - ok=False                                        → DB 故障或输入验证失败
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _update_user_profile_impl,
        user_id,
        field,
        value,
        policy_name="database_write",
        tool_name="update_user_profile",
        source="sqlite",
    )


# ── Redis 画像缓存失效 ────────────────────────────────────────────────────────

def _invalidate_cache_impl(user_id: str) -> dict:
    from core.cache import _client as redis_client
    key = f"user_profile:{user_id}"
    redis_client.delete(key)
    return {"cache_invalidated": True}


async def invalidate_cache_result(user_id: str) -> ToolResult:
    """删除 Redis 用户画像缓存，返回 ToolResult。redis_write 策略，不重试。

    - ok=True,  data={"cache_invalidated": True}  → DEL 成功
    - ok=False, data={"cache_invalidated": False} → Redis 故障（TTL 过期后最终一致）
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _invalidate_cache_impl,
        user_id,
        policy_name="redis_write",
        tool_name="invalidate_profile_cache",
        fallback_data={"cache_invalidated": False},
        source="redis",
    )
