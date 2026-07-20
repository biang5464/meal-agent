import json
import redis
from datetime import datetime

from core.tool_protocol import ToolMeta, ToolResult

_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 配置常量（以后改这里）
PROFILE_TTL = 300        # 用户画像缓存 5分钟
CHAT_HISTORY_TTL = 7200  # 对话历史 2小时
CHAT_HISTORY_LIMIT = 10  # 最多保留10轮（每轮=user+assistant共2条）

# ---------- 用户画像缓存 ----------

def get_profile_cache(user_id: str) -> dict | None:
    """从 Redis 读用户画像，未命中返回 None"""
    key = f"user_profile:{user_id}"
    data = _client.get(key)
    if data is None:
        return None
    return json.loads(data)

def set_profile_cache(user_id: str, profile: dict) -> None:
    """写入 Redis，TTL 5分钟"""
    key = f"user_profile:{user_id}"
    _client.set(key, json.dumps(profile, ensure_ascii=False), ex=PROFILE_TTL)

def invalidate_profile_cache(user_id: str) -> None:
    """画像更新后删除缓存，让下次重新从 MySQL 加载"""
    key = f"user_profile:{user_id}"
    _client.delete(key)

# ---------- 对话历史 ----------

def get_chat_history(user_id: str) -> list[dict]:
    """先查 Redis；Redis 为空则从 MySQL 取最近20条并回填 Redis。"""
    key = f"chat_history:{user_id}"
    messages = _client.lrange(key, -(CHAT_HISTORY_LIMIT * 2), -1)
    if messages:
        return [json.loads(m) for m in messages]

    # Redis 为空，从 MySQL 补
    from models.chat_history import ChatHistoryORM
    from core.database import SessionLocal
    with SessionLocal() as session:
        rows = (
            session.query(ChatHistoryORM)
            .filter_by(user_id=user_id)
            .order_by(ChatHistoryORM.created_at.desc(), ChatHistoryORM.id.desc())
            .limit(CHAT_HISTORY_LIMIT * 2)
            .all()
        )
        rows.reverse()  # 恢复正序
        history = [
            {
                "role": r.role,
                "content": r.content,
                "timestamp": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]
        # 写回 Redis
        for msg in history:
            _client.rpush(key, json.dumps(msg, ensure_ascii=False))
        if history:
            _client.expire(key, CHAT_HISTORY_TTL)
        return history


def append_chat_message(user_id: str, role: str, content: str) -> None:
    """同时写 Redis 和 MySQL；Redis 超出上限自动裁剪并刷新 TTL。"""
    key = f"chat_history:{user_id}"
    message = json.dumps({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }, ensure_ascii=False)

    # Redis 写入
    _client.rpush(key, message)
    _client.ltrim(key, -(CHAT_HISTORY_LIMIT * 2), -1)
    _client.expire(key, CHAT_HISTORY_TTL)

    # MySQL 双写
    from models.chat_history import ChatHistoryORM
    from core.database import SessionLocal
    with SessionLocal() as session:
        session.add(ChatHistoryORM(user_id=user_id, role=role, content=content))
        session.commit()


def clear_chat_history(user_id: str) -> None:
    """清空某用户的对话历史（Redis + MySQL）"""
    key = f"chat_history:{user_id}"
    _client.delete(key)

    from models.chat_history import ChatHistoryORM
    from core.database import SessionLocal
    with SessionLocal() as session:
        session.query(ChatHistoryORM).filter_by(user_id=user_id).delete()
        session.commit()


# ── ToolResult 协议接口（Phase 6A）────────────────────────────────────────────

async def get_profile_cache_result(user_id: str) -> ToolResult:
    """Redis 画像缓存读取，返回 ToolResult。

    - ok=True, data=profile_dict, source="redis"       → 缓存命中
    - ok=True, data=None,         source="redis_miss"  → 缓存未命中（key 不存在）
    - ok=False, data=None, fallback_used=True           → Redis 故障
    """
    from tools.tool_executor import tool_executor
    result = await tool_executor.execute(
        get_profile_cache,
        user_id,
        policy_name="redis_read",
        tool_name="get_profile_cache",
        fallback_data=None,
        source="redis",
    )
    # 命中时 data 为 dict；未命中时函数返回 None，ToolExecutor 视为成功
    if result.ok and result.data is None and result.meta:
        result.meta.source = "redis_miss"
    return result


async def get_chat_history_result(user_id: str) -> ToolResult:
    """读取对话历史，返回 ToolResult。

    内部包含 Redis→SQLite fallback：Redis miss 时自动从 SQLite 补回并写回 Redis。
    使用 database_read 策略（3.0s）以容纳 SQLite fallback 路径。

    - ok=True, data=[...], source="chat_cache"          → 正常读取
    - ok=False, data=[], fallback_used=True             → Redis 连接失败
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        get_chat_history,
        user_id,
        policy_name="database_read",
        tool_name="get_chat_history",
        fallback_data=[],
        source="chat_cache",
    )


# ── ToolResult 写入协议接口（Phase 6B）─────────────────────────────────────────

def _set_profile_cache_impl(user_id: str, profile: dict) -> dict:
    set_profile_cache(user_id, profile)
    return {"cache_written": True}


async def set_profile_cache_result(user_id: str, profile: dict) -> ToolResult:
    """写入 Redis 用户画像缓存，返回 ToolResult。redis_write 策略，不重试。

    - ok=True, data={"cache_written": True}   → 写入成功
    - ok=False, data={"cache_written": False} → Redis 故障（可降级）
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _set_profile_cache_impl,
        user_id,
        profile,
        policy_name="redis_write",
        tool_name="set_profile_cache",
        fallback_data={"cache_written": False},
        source="redis",
    )


def _append_chat_message_impl(user_id: str, role: str, content: str) -> None:
    append_chat_message(user_id, role, content)


async def append_chat_message_result(user_id: str, role: str, content: str) -> ToolResult:
    """追加聊天消息（Redis + SQLite 双写），返回 ToolResult。database_write 策略，不重试。

    - ok=True  → 写入成功
    - ok=False → Redis 或 DB 故障（记录日志，不阻断主流程）
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _append_chat_message_impl,
        user_id,
        role,
        content,
        policy_name="database_write",
        tool_name="append_chat_message",
        source="chat_history",
    )
