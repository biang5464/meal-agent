"""用户记忆工具：读写用户画像，供 Agent 在对话中调用。"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from models.user import (
    Base,
    UserProfileORM,
    MEAL_HISTORY_MAX,
    UPDATABLE_FIELDS,
    LIST_FIELDS,
    VALID_BUDGETS,
    decode_list,
    encode_list,
    orm_to_dict,
)

load_dotenv()

_Session: sessionmaker | None = None


def init_db(db_url: str | None = None) -> None:
    """应用启动时调用。

    db_url 参数：
    - None           → 使用 MySQL（从环境变量读取，生产/tests/ 目录测试）
    - 裸文件路径      → 如 "./data/users.db"，使用 SQLite（兼容旧测试）
    - SQLite URL     → 如 "sqlite:///..." 或 "sqlite:///:memory:"
    """
    global _Session

    if db_url is None:
        # MySQL 模式
        import models.chat_history   # noqa: F401 — 注册 ChatHistoryORM
        import models.maintenance    # noqa: F401 — 注册 ORM
        import models.price_history  # noqa: F401 — 注册 PriceHistoryORM
        from core.database import engine as _engine, SessionLocal
        Base.metadata.create_all(_engine)
        _Session = SessionLocal
        return

    # SQLite 兼容模式（旧测试）
    if not db_url.startswith("sqlite"):
        os.makedirs(os.path.dirname(os.path.abspath(db_url)), exist_ok=True)
        db_url = f"sqlite:///{db_url}"

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    # SQLite 字段补丁（category_preferences 可能不存在于旧库）
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE user_profiles ADD COLUMN category_preferences TEXT"))
            conn.commit()
        except Exception:
            pass

    _Session = sessionmaker(bind=engine)


def _session() -> Session:
    if _Session is None:
        raise RuntimeError("DB not initialized. Call init_db() first.")
    return _Session()


def _get_or_create(session: Session, user_id: str) -> UserProfileORM:
    """查询用户画像，不存在则创建默认画像并持久化。"""
    row = session.get(UserProfileORM, user_id)
    if row is None:
        row = UserProfileORM(user_id=user_id)
        session.add(row)
        session.flush()
    return row


# ---------- LangChain Tools ----------

@tool
def get_user_profile(user_id: str) -> dict:
    """
    获取指定用户的饮食偏好画像。用户不存在时返回空画像，不创建数据库记录。

    Args:
        user_id: 用户唯一标识

    Returns:
        包含 user_id / likes / dislikes / budget / diet_restriction / meal_history 的字典。
        用户不存在时返回空画像（不写库）。
    """
    with _session() as session:
        row = session.get(UserProfileORM, user_id)
        if row is None:
            return {
                "user_id": user_id,
                "likes": [],
                "dislikes": [],
                "budget": "mid",
                "diet_restriction": "",
                "meal_history": [],
                "category_preferences": {},
            }
        return orm_to_dict(row)


@tool
def update_user_profile(user_id: str, field: str, value: str) -> str:
    """
    更新用户画像中的单个字段。用户不存在时自动创建。

    支持的字段：
      - likes          (list) 传 JSON 数组字符串，如 '["辣", "川菜"]'，整体替换
      - dislikes       (list) 同上
      - budget         (str)  取值 low / mid / high
      - diet_restriction (str) 如"素食"、"清真"，传空字符串表示无限制

    Args:
        user_id: 用户唯一标识
        field:   要更新的字段名
        value:   新值字符串

    Returns:
        操作结果描述。
    """
    if field not in UPDATABLE_FIELDS:
        return f"错误：不支持更新字段 '{field}'，可选字段：{sorted(UPDATABLE_FIELDS)}"

    if field == "budget" and value not in VALID_BUDGETS:
        return f"错误：budget 只能是 {VALID_BUDGETS} 之一，收到 '{value}'"

    with _session() as session:
        row = _get_or_create(session, user_id)

        if field in LIST_FIELDS:
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    return "错误：列表字段需要传 JSON 数组字符串，如 '[\"辣\", \"川菜\"]'"
                setattr(row, field, encode_list([str(x) for x in parsed]))
            except json.JSONDecodeError:
                return f"错误：value 不是合法的 JSON 字符串：{value}"
        else:
            setattr(row, field, value)

        row.updated_at = datetime.utcnow()
        session.commit()

    return f"已更新 {user_id} 的 {field}"


@tool
def append_meal_history(user_id: str, meal_name: str) -> str:
    """
    向用户的用餐历史追加一条记录，超过 20 条时自动移除最旧的一条。

    Args:
        user_id:   用户唯一标识
        meal_name: 本次餐食名称，如 "番茄炒蛋"

    Returns:
        操作结果描述，包含当前历史条数。
    """
    with _session() as session:
        row = _get_or_create(session, user_id)
        history = decode_list(row.meal_history)
        history.append(meal_name)
        if len(history) > MEAL_HISTORY_MAX:
            history = history[-MEAL_HISTORY_MAX:]
        row.meal_history = encode_list(history)
        row.updated_at = datetime.utcnow()
        session.commit()

    return f"已记录 \"{meal_name}\"，当前历史共 {len(history)} 条"


# 导出给 router.py 使用
MEMORY_TOOLS = [get_user_profile, update_user_profile, append_meal_history]


# ── 对话记忆（ChromaDB） ──────────────────────────────────────────────────────

import chromadb
from datetime import timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

_PERMANENT_KEYWORDS = [
    "糖尿病", "高血压", "痛风", "过敏", "术后", "手术后",
    "肾病", "心脏病", "高血脂", "乳糖不耐", "麸质过敏"
]

def _get_chroma_path() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

_MEMORY_COLLECTION = "conversation_memory"


def init_conversation_memory_chroma():
    """初始化 conversation_memory collection"""
    client = chromadb.PersistentClient(path=_get_chroma_path())
    client.get_or_create_collection(name=_MEMORY_COLLECTION)
    print("conversation_memory collection 初始化完成")


def _is_permanent(summary: str) -> bool:
    """判断摘要是否涉及疾病类饮食要求"""
    return any(kw in summary for kw in _PERMANENT_KEYWORDS)


def _summarize_impl(messages: list) -> object:
    """同步 LLM 调用，由 ToolExecutor 在 asyncio.to_thread 中运行，避免阻塞事件循环。"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        max_retries=0,
        request_timeout=65,
    )
    return llm.invoke(messages)


async def save_conversation_memory(
    user_id: str,
    user_input: str,
    result: str,
    intent: str,
):
    """
    异步提取摘要并存入 ChromaDB。
    在对话结束后用 asyncio.create_task 调用，不阻塞主流程。
    """
    try:
        from tools.tool_executor import tool_executor
        messages = [
            SystemMessage(content=(
                "你是一个对话摘要助手。"
                "用1-2句中文概括以下对话的核心内容，"
                "重点提取：用户的饮食限制、健康状况、偏好、推荐结果。"
                "不要输出任何解释，只输出摘要本身。"
            )),
            HumanMessage(content=(
                f"用户输入：{user_input}\n"
                f"系统回复：{result[:500]}"
            )),
        ]
        tool_result = await tool_executor.execute(
            _summarize_impl,
            messages,
            policy_name="llm_generation",
            tool_name="summarize_memory",
            fallback_data=None,
        )
        if not tool_result.ok or tool_result.data is None:
            print(f"[memory] 摘要 LLM 失败，跳过存储: {tool_result.error}")
            return
        summary = tool_result.data.content.strip()

        memory_type = "permanent" if _is_permanent(summary) else "temporary"
        now = datetime.now()
        expires_at = (
            None if memory_type == "permanent"
            else (now + timedelta(days=90)).strftime("%Y-%m-%d")
        )

        client = chromadb.PersistentClient(path=_get_chroma_path())
        collection = client.get_or_create_collection(name=_MEMORY_COLLECTION)

        doc_id = f"{user_id}_{now.strftime('%Y%m%d%H%M%S%f')}"
        collection.add(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{
                "user_id": user_id,
                "intent": intent,
                "memory_type": memory_type,
                "created_at": now.strftime("%Y-%m-%d"),
                "expires_at": expires_at or "never",
            }],
        )
        print(f"[memory] 存储摘要成功 user={user_id} type={memory_type}: {summary[:50]}...")

    except Exception as e:
        print(f"[memory] 摘要存储失败（不影响主流程）: {e}")


def search_conversation_memory(user_id: str, query: str, n_results: int = 3) -> list[dict]:
    """
    检索用户相关历史摘要，综合语义相似度 + Recency + Importance 重排序。
    返回格式：[{"summary": ..., "intent": ..., "date": ..., "memory_type": ...}]
    """
    try:
        from datetime import date
        client = chromadb.PersistentClient(path=_get_chroma_path())
        collection = client.get_or_create_collection(name=_MEMORY_COLLECTION)

        # 多取一些，后面重排序
        fetch_n = min(n_results * 3, 20)
        results = collection.query(
            query_texts=[query],
            n_results=fetch_n,
            where={"user_id": user_id},
        )

        today = date.today()
        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            # Recency 权重
            created_at_str = meta.get("created_at", str(today))
            try:
                created_date = date.fromisoformat(created_at_str)
                days_ago = (today - created_date).days
            except Exception:
                days_ago = 30
            recency_score = 1 / (1 + days_ago * 0.1)

            # Importance 权重
            importance_score = 2.0 if meta.get("memory_type") == "permanent" else 1.0

            # 综合得分（distance 越小越好，转换为相似度）
            similarity = max(0, 1 - distance)
            final_score = similarity * recency_score * importance_score

            output.append({
                "summary": doc,
                "intent": meta.get("intent", ""),
                "date": meta.get("created_at", ""),
                "memory_type": meta.get("memory_type", "temporary"),
                "final_score": final_score,
            })

        # 按综合得分降序排列，取 top n_results
        output.sort(key=lambda x: x["final_score"], reverse=True)
        return output[:n_results]

    except Exception as e:
        print(f"[memory] 检索历史失败: {e}")
        return []


def cleanup_expired_conversation_memory():
    """
    清理过期的 temporary 记录（由 maintenance_agent 每天凌晨调用）。
    """
    try:
        client = chromadb.PersistentClient(path=_get_chroma_path())
        collection = client.get_or_create_collection(name=_MEMORY_COLLECTION)

        today = datetime.now().strftime("%Y-%m-%d")
        all_items = collection.get(where={"memory_type": "temporary"})

        expired_ids = []
        for i, meta in enumerate(all_items["metadatas"]):
            expires_at = meta.get("expires_at", "never")
            if expires_at != "never" and expires_at < today:
                expired_ids.append(all_items["ids"][i])

        if expired_ids:
            collection.delete(ids=expired_ids)
            print(f"[memory cleanup] 已清理 {len(expired_ids)} 条过期记录")
        else:
            print("[memory cleanup] 无过期记录")

    except Exception as e:
        print(f"[memory cleanup] 清理失败: {e}")


# ── 统一协议服务层（Phase 5）────────────────────────────────────────────────────


def _search_conversation_memory_impl(user_id: str, query: str, n_results: int) -> list[dict]:
    """严格检索对话记忆；collection 未就绪或查询失败时抛出异常。"""
    from datetime import date

    try:
        client = chromadb.PersistentClient(path=_get_chroma_path())
        collection = client.get_collection(name=_MEMORY_COLLECTION)
    except Exception as exc:
        raise ConnectionError(f"conversation_memory collection 未就绪: {exc}") from exc

    count = collection.count()
    if count == 0:
        return []

    fetch_n = min(n_results * 3, 20)
    results = collection.query(
        query_texts=[query],
        n_results=min(fetch_n, count),
        where={"user_id": user_id},
    )

    today = date.today()
    output = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        created_at_str = meta.get("created_at", str(today))
        try:
            created_date = date.fromisoformat(created_at_str)
            days_ago = (today - created_date).days
        except Exception:
            days_ago = 30
        recency_score = 1 / (1 + days_ago * 0.1)

        importance_score = 2.0 if meta.get("memory_type") == "permanent" else 1.0
        similarity = max(0, 1 - distance)
        final_score = similarity * recency_score * importance_score

        output.append({
            "summary": doc,
            "intent": meta.get("intent", ""),
            "date": meta.get("created_at", ""),
            "memory_type": meta.get("memory_type", "temporary"),
            "final_score": final_score,
        })

    output.sort(key=lambda x: x["final_score"], reverse=True)
    return output[:n_results]


async def search_conversation_memory_result(user_id: str, query: str, n_results: int = 3):
    """
    通过 ToolExecutor.execute() 检索对话记忆，返回 ToolResult。

    - ok=True, data=[memories]：检索成功，有历史
    - ok=True, data=[]：检索成功，无历史（新用户 / 新 topic）
    - ok=False, TIMEOUT：检索超时
    - ok=False, DEPENDENCY_UNAVAILABLE：collection 未就绪或 ChromaDB 故障
    """
    from core.tool_protocol import ToolError, ToolErrorCode, ToolResult
    from tools.tool_executor import tool_executor

    result = await tool_executor.execute(
        _search_conversation_memory_impl,
        user_id,
        query,
        n_results=n_results,
        policy_name="chroma_search",
        tool_name="search_conversation_memory",
        fallback_data=[],
        source="conversation_memory_chroma",
    )

    if not result.ok and result.error and result.error.code != ToolErrorCode.TIMEOUT:
        return ToolResult(
            ok=False,
            data=result.data,
            error=ToolError(
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                message="对话记忆知识库暂时不可用",
                retryable=True,
                details={"component": "collection", "collection": "conversation_memory"},
            ),
            meta=result.meta,
        )
    return result
