import json
import logging
import os
import re
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

from agents.session_context import (
    TopicCache, ContextSlot, get_domain,
    extract_excluded_by_rule, needs_llm_extraction,
    NEGATIONS, MAX_CONTEXTS
)
from agents.state import AgentState

REDIS_SESSION_TTL = 7200

PRONOUNS = ["它", "这个", "那个", "这款", "那款", "这种", "那种", "继续", "刚才"]

# ── 约束提取常量 ──────────────────────────────────────────────────────────────

_ALLOWED_CONSTRAINT_KEYS = frozenset({
    "budget", "servings", "time_limit_minutes", "cuisine", "flavor",
    "dietary_requirement", "brand", "color", "storage", "size", "price_range",
})
_INT_CONSTRAINT_KEYS = frozenset({"servings", "time_limit_minutes"})
_INT_CONSTRAINT_RANGES: dict = {"servings": (1, 99), "time_limit_minutes": (1, 480)}
_CTRL_TRANS_CM = str.maketrans("", "", "\x00\r\n\t\x0b\x0c")
_INVALID_VALUE = object()  # sentinel: 无效约束值，丢弃

# 删除信号：独立短语（可单独触发）和需要约束目标配合的动词
_DELETION_PHRASES = frozenset(["不限", "不限制", "不要限制", "去掉限制"])
_DELETION_VERBS = frozenset(["取消", "去掉", "移除", "删除", "重置"])

# 约束目标别名表：用于判断删除动词是否针对约束
_CONSTRAINT_TARGET_ALIASES: dict[str, tuple] = {
    "budget":             ("预算", "金额", "价格上限", "花费", "费用"),
    "servings":           ("人数", "几个人", "份数", "几份", "多少人"),
    "time_limit_minutes": ("时间", "时长", "分钟", "小时", "快手", "限时"),
    "cuisine":            ("菜系",),
    "flavor":             ("口味", "辣度", "甜度", "咸度", "风味"),
    "dietary_requirement": ("饮食要求", "饮食限制", "素食", "清真", "无麸质", "忌口"),
    "brand":              ("品牌",),
    "color":              ("颜色",),
    "storage":            ("存储", "容量"),
    "size":               ("尺寸", "大小"),
    "price_range":        ("价格范围", "价格区间", "价格限制"),
}
# 所有约束目标别名的扁平集合，供快速查找
_ALL_CONSTRAINT_TARGETS: frozenset = frozenset(
    alias for aliases in _CONSTRAINT_TARGET_ALIASES.values() for alias in aliases
)


def _has_deletion_constraint_signal(user_input: str) -> bool:
    """True 当且仅当：独立删除短语，或 删除动词 + 约束目标 同时出现。"""
    for phrase in _DELETION_PHRASES:
        if phrase in user_input:
            return True
    if not any(verb in user_input for verb in _DELETION_VERBS):
        return False
    return any(target in user_input for target in _ALL_CONSTRAINT_TARGETS)

_RULE_CONSTRAINT_SIGNAL_PATS = [
    re.compile(r"(?<![A-Za-z])(low|mid|high)(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"预算"),
    re.compile(r"\d+\s*个?人"),
    re.compile(r"\d+\s*分钟"),
]
_LLM_ONLY_CONSTRAINT_SIGNAL_PATS = [
    re.compile(r"菜系|[川粤闽湘鲁苏浙徽]菜|日本料理|意大利料理|泰国料理|泰国菜|法国菜"),
    re.compile(r"口味"),
    re.compile(r"素食|纯素|清真|无麸质|低糖|低卡|减肥|生酮"),
    re.compile(r"品牌"),
    re.compile(r"\d+\s*(?:GB|TB)", re.IGNORECASE),
    re.compile(r"\d+\s*(?:寸|英寸|inch)", re.IGNORECASE),
    re.compile(r"价格区间|价格范围"),
    re.compile(r"澳元|AUD", re.IGNORECASE),
    re.compile(r"颜色"),
]

_CONSTRAINT_EXTRACT_PROMPT = """\
从用户输入中提取明确的查询约束，只输出合法 JSON 对象，不带 Markdown 代码块，不带其他文字。

只允许以下键（其余忽略，不允许嵌套值）：
  budget（str：low/mid/high 或原始金额，如"100澳元以内"）
  servings（int：正整数人数，范围 1-99）
  time_limit_minutes（int：正整数分钟，范围 1-480）
  cuisine（str：菜系名称）
  flavor（str：仅提取用户明确想要的口味；否定性表达不提取）
  dietary_requirement（str：饮食限制）
  brand（str：品牌名称）
  color（str：颜色）
  storage（str：存储容量）
  size（str：尺寸）
  price_range（str：价格区间）

特殊规则：
- 用户明确要删除某约束时，该键值输出 null
- 否定口味（不要辣、不想要甜）不提取为 flavor；它们属于排除项
- 无约束时输出 {}

示例：
"给4个人做" → {"servings": 4}
"30分钟以内" → {"time_limit_minutes": 30}
"取消人数限制" → {"servings": null}
"今天吃什么" → {}
"预算low，川菜" → {"budget": "low", "cuisine": "川菜"}
"不要辣" → {}
"100澳元以内" → {"budget": "100澳元以内"}
"""


def _has_non_negation_content(user_input: str) -> bool:
    """检查否定表达之外是否有实质性内容（暗示可能存在新实体）。

    移除各否定词及其后跟的至多 15 个非标点字符（含可选收尾标点），
    若残余内容超过 2 个字符则认为存在新实体信号。
    """
    text = user_input
    for neg in NEGATIONS:
        text = re.sub(re.escape(neg) + r"[^，。？！,!?]{0,15}[，。？！,!?]?", "", text)
    residual = re.sub(r"[，。？！,!?、\s]+", "", text).strip()
    return len(residual) > 2


# ── Embedding 工具函数 ──────────────────────────────────────────────────────

def _cosine_similarity(text_a: str, text_b: str) -> float:
    """用 bge-small-zh 计算两段文本的余弦相似度"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("BAAI/bge-small-zh")
        embs = model.encode([text_a, text_b])
        a, b = embs[0], embs[1]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception:
        return 0.0


# ── 保留旧函数供现有测试兼容 ──────────────────────────────────────────────────

def is_same_topic(
    current_entity: str,
    last_entity: str,
    current_intent: str,
    last_intent: str,
    embedding_fn=None,
) -> bool:
    """
    三级 Topic 判断（保留供测试兼容）。
    Level1: Entity 是否一致
    Level2: Embedding 相似度 > 0.8
    Level3: Intent domain 是否一致
    """
    if current_entity and last_entity:
        if (current_entity.lower() in last_entity.lower() or
                last_entity.lower() in current_entity.lower()):
            return True
    if embedding_fn and current_entity and last_entity:
        try:
            if embedding_fn(current_entity, last_entity) > 0.8:
                return True
        except Exception:
            pass
    if current_intent and last_intent:
        return get_domain(current_intent) == get_domain(last_intent)
    return False


# ── Redis 操作 ──────────────────────────────────────────────────────────────

async def load_topic_cache(user_id: str, redis_client) -> TopicCache:
    try:
        key = f"topic_cache:{user_id}"
        data = redis_client.get(key)
        if data:
            return TopicCache.from_dict(json.loads(data))
    except Exception as e:
        print(f"[context_manager] 加载失败: {e}")
    return TopicCache.empty()


async def save_topic_cache(user_id: str, cache: TopicCache, redis_client):
    try:
        key = f"topic_cache:{user_id}"
        redis_client.setex(
            key, REDIS_SESSION_TTL,
            json.dumps(cache.to_dict(), ensure_ascii=False)
        )
    except Exception as e:
        print(f"[context_manager] 保存失败: {e}")


# ── Phase 6B：TopicCache 写入 ToolResult 协议 ──────────────────────────────────

def _save_topic_cache_impl(user_id: str, cache_dict: dict) -> None:
    from core.cache import _client as _redis_client
    key = f"topic_cache:{user_id}"
    _redis_client.setex(key, REDIS_SESSION_TTL, json.dumps(cache_dict, ensure_ascii=False))


async def save_topic_cache_result(user_id: str, cache: TopicCache):
    """保存话题缓存，返回 ToolResult。redis_write 策略，不重试。统一使用 core.cache._client 单例。

    - ok=True  → 保存成功（TTL=7200）
    - ok=False → Redis 故障（主业务流程继续）
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _save_topic_cache_impl,
        user_id,
        cache.to_dict(),
        policy_name="redis_write",
        tool_name="save_topic_cache",
        source="topic_cache",
    )


# ── Phase 6A：ToolResult 协议接口 ─────────────────────────────────────────────

def _load_topic_cache_impl(user_id: str, redis_client) -> TopicCache:
    """同步 impl：Redis 命中返回 TopicCache，miss 返回 TopicCache.empty()，连接错误冒泡。"""
    key = f"topic_cache:{user_id}"
    data = redis_client.get(key)  # ConnectionError/OSError → ToolExecutor 捕获为 NETWORK
    if data is None:
        return TopicCache.empty()  # 正常 miss，非错误
    return TopicCache.from_dict(json.loads(data))


async def load_topic_cache_result(user_id: str, redis_client):
    """读取话题缓存，返回 ToolResult。

    - ok=True,  data=TopicCache(...)        → Redis 命中
    - ok=True,  data=TopicCache.empty()    → Redis miss（key 不存在）
    - ok=False, data=TopicCache.empty(), fallback_used=True → Redis 故障
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _load_topic_cache_impl,
        user_id,
        redis_client,
        policy_name="redis_read",
        tool_name="load_topic_cache",
        fallback_data=TopicCache.empty(),
        source="topic_cache",
    )


# ── Entity 提取 ─────────────────────────────────────────────────────────────

async def _extract_entity_impl(messages: list):
    """Async LLM call; ToolExecutor wraps this with asyncio.wait_for for timeout."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        max_retries=0,
        request_timeout=65,
    )
    return await llm.ainvoke(messages)


async def extract_entity(user_input: str, last_entity: str = "") -> str:
    if any(p in user_input for p in PRONOUNS) and last_entity:
        return last_entity
    # 仅排除附属条件（否定词外无新实体信号）：保留 last_entity，防止 LLM 把排除项误识为新实体
    if (any(neg in user_input for neg in NEGATIONS)
            and last_entity
            and not _has_non_negation_content(user_input)):
        return last_entity
    from langchain_core.messages import SystemMessage, HumanMessage
    from tools.tool_executor import tool_executor
    messages = [
        SystemMessage(content=(
            "从用户输入里提取核心查询实体（产品名/食物名/疾病名等）。"
            "只输出实体本身，无则输出NONE。"
            "例如：'MacBook Air M3价格' → 'MacBook Air M3'"
            "例如：'今天吃什么' → 'NONE'"
            "例如：'头孢能喝酒吗' → '头孢'"
        )),
        HumanMessage(content=user_input[:200])
    ]
    result = await tool_executor.execute(
        _extract_entity_impl,
        messages,
        policy_name="llm_classification",
        tool_name="extract_entity",
        fallback_data=None,
    )
    if result.ok and result.data is not None:
        content = result.data.content.strip()
        return "" if content == "NONE" else content
    return last_entity


# ── Excluded 提取 ───────────────────────────────────────────────────────────

async def _extract_excluded_impl(messages: list):
    """Async LLM call; ToolExecutor wraps this with asyncio.wait_for for timeout."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        max_retries=0,
        request_timeout=65,
    )
    return await llm.ainvoke(messages)


async def extract_excluded_with_llm(user_input: str) -> list[str]:
    from langchain_core.messages import SystemMessage, HumanMessage
    from tools.tool_executor import tool_executor
    messages = [
        SystemMessage(content=(
            "从用户输入里提取用户明确不想要的条件。"
            "只输出排除条件列表，用逗号分隔，无则输出NONE。"
            "例如：'不是保护壳' → '保护壳'"
            "例如：'我更偏向不是那种特别重的' → '重'"
        )),
        HumanMessage(content=user_input[:200])
    ]
    result = await tool_executor.execute(
        _extract_excluded_impl,
        messages,
        policy_name="llm_classification",
        tool_name="extract_excluded",
        fallback_data=None,
    )
    if result.ok and result.data is not None:
        content = result.data.content.strip()
        if content == "NONE":
            return []
        return [x.strip() for x in content.split(",") if x.strip()]
    return []


# ── Context 恢复逻辑（旧版，保留供兼容）────────────────────────────────────

async def resolve_context(
    cache: TopicCache,
    user_input: str,
    current_entity: str,
    current_intent: str,
) -> tuple:
    """
    恢复或新建 context。
    返回 (slot, is_existing)：
      is_existing=True  → 复用已有 context
      is_existing=False → 新建 context
    """
    # Step1：引用词 → 恢复 current_context
    if any(p in user_input for p in PRONOUNS):
        current = cache.get_current()
        if current:
            return current, True

    # Step2：有 entity → 找已有 context
    if current_entity:
        found = cache.find_by_entity(current_entity, _cosine_similarity)
        if found:
            return found, True

    # Step3：Embedding 找最相似 context
    if user_input:
        found = cache.find_by_embedding(user_input, _cosine_similarity)
        if found:
            return found, True

    return None, False


# ── 消歧输入构建 ────────────────────────────────────────────────────────────

def _build_resolved_input(user_input: str, candidate: dict) -> str:
    """
    构建消歧后的输入。
    - pronoun + entity：把引用词替换为实体名
    - entity_match / embedding_match + entity 不在输入中：补充实体名
    """
    entity = candidate.get("entity", "")
    match_type = candidate.get("match_type", "new")

    if match_type == "pronoun" and entity:
        PRONOUNS_REPLACE = ["它", "这个", "那个", "这款", "那款", "这种", "那种"]
        result = user_input
        for p in PRONOUNS_REPLACE:
            result = result.replace(p, entity)
        return result

    if match_type in ("entity_match", "embedding_match") and entity:
        if entity.lower() not in user_input.lower():
            return f"{entity} {user_input}"

    return user_input


# ── 约束提取函数 ────────────────────────────────────────────────────────────

def _has_constraint_signal(user_input: str) -> bool:
    if _has_deletion_constraint_signal(user_input):
        return True
    for pat in _RULE_CONSTRAINT_SIGNAL_PATS:
        if pat.search(user_input):
            return True
    for pat in _LLM_ONLY_CONSTRAINT_SIGNAL_PATS:
        if pat.search(user_input):
            return True
    return False


def _needs_llm_constraint_extraction(user_input: str) -> bool:
    if _has_deletion_constraint_signal(user_input):
        return True
    for pat in _LLM_ONLY_CONSTRAINT_SIGNAL_PATS:
        if pat.search(user_input):
            return True
    return False


def _extract_constraints_by_rule(user_input: str) -> dict:
    result: dict = {}
    # Use ASCII lookaround: \b doesn't work at Chinese/ASCII boundary (Chinese chars are \w in Python)
    m = re.search(r"(?<![A-Za-z])(low|mid|high)(?![A-Za-z])", user_input, re.IGNORECASE)
    if m:
        result["budget"] = m.group(1).lower()
    m = re.search(r"(\d+)\s*个?人", user_input)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 99:
            result["servings"] = n
    m = re.search(r"(\d+)\s*分钟以?内", user_input)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 480:
            result["time_limit_minutes"] = n
    return result


def _clean_constraint_value(key: str, value):
    if value is None:
        return None
    if key in _INT_CONSTRAINT_KEYS:
        try:
            n = int(value)
        except (ValueError, TypeError):
            return _INVALID_VALUE
        lo, hi = _INT_CONSTRAINT_RANGES[key]
        if not (lo <= n <= hi):
            return _INVALID_VALUE
        return n
    else:
        if not isinstance(value, (str, int, float, bool)):
            return _INVALID_VALUE
        s = str(value).translate(_CTRL_TRANS_CM).strip()[:50]
        return s if s else _INVALID_VALUE


def _parse_constraints_json(raw: str) -> dict:
    try:
        text = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip().strip("`").strip()
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {}
        result = {}
        for k, v in list(obj.items())[:20]:
            if not isinstance(k, str) or k not in _ALLOWED_CONSTRAINT_KEYS:
                continue
            if isinstance(v, (dict, list)):
                continue
            cleaned = _clean_constraint_value(k, v)
            if cleaned is not _INVALID_VALUE:
                result[k] = cleaned
        return result
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def _merge_rule_llm(rule_result: dict, llm_result: dict) -> dict:
    """Merge rule+LLM: delete(None) > rule > LLM for same key."""
    merged = dict(llm_result)
    for k, v in rule_result.items():
        if k in merged and merged[k] is None:
            pass  # LLM delete wins, keep None
        else:
            merged[k] = v  # rule overrides LLM
    return merged


def _merge_constraints(existing: dict, turn: dict) -> dict:
    """Pure function. turn[k]=None → delete k. Result never contains None."""
    result = dict(existing)
    for k, v in turn.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result


async def _extract_constraints_impl(messages: list):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        max_retries=0,
        request_timeout=65,
    )
    return await llm.ainvoke(messages)


async def _extract_constraints_with_llm(user_input: str) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage
    from tools.tool_executor import tool_executor
    messages = [
        SystemMessage(content=_CONSTRAINT_EXTRACT_PROMPT),
        HumanMessage(content=user_input[:200]),
    ]
    result = await tool_executor.execute(
        _extract_constraints_impl,
        messages,
        policy_name="llm_classification",
        tool_name="extract_constraints",
        fallback_data=None,
    )
    if result.ok and result.data is not None:
        return _parse_constraints_json(result.data.content.strip())
    return {}


async def extract_constraints(user_input: str) -> dict:
    if not _has_constraint_signal(user_input):
        return {}
    rule_result = _extract_constraints_by_rule(user_input)
    if _needs_llm_constraint_extraction(user_input):
        llm_result = await _extract_constraints_with_llm(user_input)
        pre_validated = _merge_rule_llm(rule_result, llm_result)
    else:
        pre_validated = dict(rule_result)
    result: dict = {}
    for k, v in pre_validated.items():
        if k not in _ALLOWED_CONSTRAINT_KEYS:
            continue
        cleaned = _clean_constraint_value(k, v)
        if cleaned is not _INVALID_VALUE:
            result[k] = cleaned
    return result


# ── 阶段一：只读候选解析（在 supervisor 之前运行）──────────────────────────

async def resolve_context_candidate(state: AgentState, redis_client) -> dict:
    """
    只读操作，不修改 Redis。
    提取实体、excluded，做三步候选匹配，返回候选供 supervisor 参考。
    """
    user_id = state.get("user_id", "")
    user_input = state.get("user_input", "")

    # 加载 TopicCache（只读，ToolResult 协议）
    cache_result = await load_topic_cache_result(user_id, redis_client)
    if not cache_result.ok:
        logger.warning("[context_manager] TopicCache 读取降级: %s", cache_result.error)
    cache = cache_result.data  # TopicCache 或 TopicCache.empty()

    # 元数据：TopicCache 读取状态（hit/miss/degraded）
    if not cache_result.ok:
        cache_read_status = "degraded"
    elif cache.contexts:
        cache_read_status = "hit"
    else:
        cache_read_status = "miss"

    # 获取上一轮 entity
    current_slot = cache.get_current()
    last_entity = current_slot.entity if current_slot else ""

    # 执行路径标志（在调用 extract_entity 之前确定，供 entity_source 精确标记）
    has_pronoun = any(p in user_input for p in PRONOUNS) and bool(last_entity)
    has_negation_only = (
        not has_pronoun
        and bool(last_entity)
        and any(neg in user_input for neg in NEGATIONS)
        and not _has_non_negation_content(user_input)
    )

    # 提取当前 entity（pronoun 和 negation_only 在内部短路，不调用 LLM）
    current_entity = await extract_entity(user_input, last_entity)

    # entity_source：基于真实执行路径（pronoun/negation_carry/llm/llm_fallback/slot_carry/none）
    if has_pronoun:
        entity_source = "pronoun"
    elif has_negation_only:
        entity_source = "negation_carry"
    elif current_entity and current_entity != last_entity:
        entity_source = "llm"
    elif not current_entity:
        entity_source = "none"
    else:
        # current_entity == last_entity：LLM 失败回退，或成功但结果恰好同上轮
        entity_source = "llm_fallback"

    # 提取本轮 excluded（规则+LLM兜底）
    rule_excluded = extract_excluded_by_rule(user_input)
    if needs_llm_extraction(user_input, rule_excluded):
        turn_excluded = await extract_excluded_with_llm(user_input)
    else:
        turn_excluded = rule_excluded

    # 提取本轮约束（Phase 9D）
    turn_constraints = await extract_constraints(user_input)

    # 三步匹配找候选 context（只读，不修改 cache）
    match_type = "new"
    match_score = 0.0
    matched_slot = None

    # Step1：引用词 → 恢复 current_context
    if any(p in user_input for p in PRONOUNS):
        matched_slot = cache.get_current()
        if matched_slot:
            match_type = "pronoun"
            match_score = 1.0

    # Step2：entity 精确/包含匹配
    if not matched_slot and current_entity:
        found = cache.find_by_entity(current_entity, _cosine_similarity)
        if found:
            matched_slot = found
            match_type = "entity_match"
            match_score = 0.9

    # Step3：整句 Embedding 匹配
    if not matched_slot and user_input:
        found = cache.find_by_embedding(user_input, _cosine_similarity)
        if found:
            matched_slot = found
            match_type = "embedding_match"
            match_score = 0.6

    # 最终实体（current_entity 为空时由 matched_slot 补充，来源标为 slot_carry）
    if current_entity:
        final_entity = current_entity
    else:
        slot_entity = matched_slot.entity if matched_slot else ""
        if slot_entity:
            final_entity = slot_entity
            entity_source = "slot_carry"
        else:
            final_entity = ""

    # 构建候选（不写 Redis，不修改 cache）
    candidate = {
        "ctx_id": matched_slot.ctx_id if matched_slot else "",
        "match_type": match_type,
        "match_score": match_score,
        "entity": final_entity,
        "entity_source": entity_source,        # pronoun / negation_carry / llm / llm_fallback / slot_carry / none
        "cache_read_status": cache_read_status, # hit / miss / degraded
        "domain": matched_slot.domain if matched_slot else "",
        "existing_excluded": list(matched_slot.excluded) if matched_slot else [],
        "existing_constraints": dict(matched_slot.constraints) if matched_slot else {},
        "turn_excluded": turn_excluded,
        "turn_constraints": turn_constraints,
    }

    return {
        **state,
        "context_candidate": candidate,
        "resolved_entity": candidate["entity"],
        "resolved_input": _build_resolved_input(user_input, candidate),
        "turn_excluded": turn_excluded,
        "turn_constraints": turn_constraints,
    }


# ── 阶段二：上下文提交（在 supervisor 之后运行）──────────────────────────────

# 非 Context 意图：不持久化 TopicCache（UPDATE_AND_MEAL 属 food context，不在此列）
_NON_CONTEXT_INTENTS = frozenset({"CHAT", "MEMORY_UPDATE"})


def _stable_unique(items: list) -> list:
    """稳定去重，保持首次出现顺序，不原地修改入参。"""
    seen: set = set()
    result: list = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _build_skipped_session_context(
    candidate: dict,
    current_intent: str,
    turn_excluded: list,
    turn_constraints: dict,
    context_status: str,
    missing_slots: list,
    is_pending: bool,
) -> dict:
    """跳过持久化时构建临时 session_context（不写 Redis，ctx_id 留空）。"""
    existing_excluded = list(candidate.get("existing_excluded", []) or [])
    existing_constraints = dict(candidate.get("existing_constraints", {}) or {})
    merged_constraints = _merge_constraints(existing_constraints, turn_constraints)
    merged_excluded = _stable_unique(existing_excluded + list(turn_excluded))
    return {
        "domain": get_domain(current_intent),
        "entity": candidate.get("entity", ""),
        "topic_turns": 0,
        "constraints": merged_constraints,
        "excluded": merged_excluded,
        "last_intent": current_intent,
        "ctx_id": "",
        "total_contexts": 0,
        "context_status": context_status,
        "missing_slots": list(missing_slots) if is_pending else [],
    }


async def commit_context(state: AgentState, redis_client) -> dict:
    """
    在 supervisor 输出真实 intent 后运行。
    按持久化策略校验候选 context，决定复用、新建或跳过，写回 Redis。
    """
    user_id = state.get("user_id", "")
    current_intent = state.get("intent", "")
    confidence = state.get("confidence", "HIGH")
    missing_slots = state.get("missing_slots", [])
    candidate = state.get("context_candidate", {})
    turn_excluded = list(state.get("turn_excluded", []))
    turn_constraints = dict(state.get("turn_constraints", {}))

    current_domain = get_domain(current_intent)

    # 置信度与 pending 状态
    if confidence == "LOW":
        is_pending = True
        context_status = "pending"
    elif confidence == "MEDIUM" and missing_slots:
        is_pending = True
        context_status = "pending"
    else:
        is_pending = False
        context_status = "active"

    # 策略一：非 Context 意图（CHAT/MEMORY_UPDATE）一律跳过持久化，不读 Redis
    if current_intent in _NON_CONTEXT_INTENTS:
        sc = _build_skipped_session_context(
            candidate, current_intent, turn_excluded, turn_constraints,
            context_status, missing_slots, is_pending,
        )
        sc["context_persisted"] = False
        sc["persistence_status"] = "skipped"
        return {**state, "session_context": sc, "context_status": context_status}

    # 策略二：LOW 置信度一律跳过，不读 Redis（clarify 路由必然触发，不提前污染 Context）
    if confidence == "LOW":
        sc = _build_skipped_session_context(
            candidate, current_intent, turn_excluded, turn_constraints,
            context_status, missing_slots, is_pending,
        )
        sc["context_persisted"] = False
        sc["persistence_status"] = "skipped"
        return {**state, "session_context": sc, "context_status": context_status}

    # 早期 ghost 检测：match_type="new" + 完全空 Context → 跳过，无需读 Redis
    # 只含 None 的 turn_constraints（删除标记）等同于空（新 Slot 无需记录删除）
    _effective_constraints = {k: v for k, v in turn_constraints.items() if v is not None}
    if (candidate.get("match_type", "new") == "new"
            and not candidate.get("entity", "")
            and not turn_excluded
            and not _effective_constraints):
        sc = _build_skipped_session_context(
            candidate, current_intent, turn_excluded, turn_constraints,
            context_status, missing_slots, is_pending,
        )
        sc["context_persisted"] = False
        sc["persistence_status"] = "skipped"
        return {**state, "session_context": sc, "context_status": context_status}

    # 加载 TopicCache（ToolResult 协议，redis_read 策略）
    cache_result = await load_topic_cache_result(user_id, redis_client)
    if not cache_result.ok:
        logger.warning("[context_manager] commit: TopicCache 读取降级: %s", cache_result.error)
        existing_excluded = list(candidate.get("existing_excluded", []) or [])
        existing_constraints = dict(candidate.get("existing_constraints", {}) or {})
        merged_constraints = _merge_constraints(existing_constraints, turn_constraints)
        sc = {
            "domain": get_domain(current_intent),
            "entity": candidate.get("entity", ""),
            "topic_turns": 1,
            "constraints": merged_constraints,
            "excluded": _stable_unique(existing_excluded + list(turn_excluded)),
            "last_intent": current_intent,
            "ctx_id": "",
            "total_contexts": 0,
            "context_status": context_status,
            "missing_slots": list(missing_slots) if is_pending else [],
            "context_persisted": False,
            "persistence_status": "read_degraded",
        }
        return {**state, "session_context": sc, "context_status": context_status}

    cache = cache_result.data

    candidate_domain = candidate.get("domain", "")
    match_type = candidate.get("match_type", "new")
    ctx_id = candidate.get("ctx_id", "")

    # 所有 match type 统一 domain 校验（pronoun 不豁免，candidate domain 空视为不匹配）
    should_reuse = (
        match_type in ("pronoun", "entity_match", "embedding_match")
        and bool(candidate_domain)
        and current_domain == candidate_domain
        and bool(ctx_id)
        and ctx_id in cache.contexts
    )

    # Domain mismatch + 历史继承来源：跨 domain 不创建 Slot，entity 丢弃，只保留当前轮数据
    _HISTORICAL_ENTITY_SOURCES = frozenset({"pronoun", "negation_carry", "slot_carry", "llm_fallback"})
    entity_source = candidate.get("entity_source", "")
    if (match_type in ("pronoun", "entity_match", "embedding_match")
            and not should_reuse
            and entity_source in _HISTORICAL_ENTITY_SOURCES):
        sc = {
            "domain": get_domain(current_intent),
            "entity": "",
            "topic_turns": 0,
            "constraints": _merge_constraints({}, turn_constraints),
            "excluded": _stable_unique(list(turn_excluded)),
            "last_intent": current_intent,
            "ctx_id": "",
            "total_contexts": 0,
            "context_status": context_status,
            "missing_slots": list(missing_slots) if is_pending else [],
            "context_persisted": False,
            "persistence_status": "skipped",
        }
        return {**state, "session_context": sc, "context_status": context_status}

    # 有意义 Context 判断（防止 ghost slot 消耗 LRU 位；只计非删除的有效约束）
    _is_meaningful = (
        bool(candidate.get("entity", ""))
        or bool(turn_excluded)
        or bool(_effective_constraints)
        or should_reuse
    )

    if not _is_meaningful:
        sc = _build_skipped_session_context(
            candidate, current_intent, turn_excluded, turn_constraints,
            context_status, missing_slots, is_pending,
        )
        sc["context_persisted"] = False
        sc["persistence_status"] = "skipped"
        return {**state, "session_context": sc, "context_status": context_status}

    if should_reuse:
        cache.access(ctx_id)
        slot = cache.contexts[ctx_id]
        slot.topic_turns += 1
        slot.last_intent = current_intent
        slot.excluded = _stable_unique(slot.excluded + list(turn_excluded))
        slot.constraints = _merge_constraints(slot.constraints, turn_constraints)
        new_entity = candidate.get("entity", "")
        if new_entity and len(new_entity) > len(slot.entity):
            slot.entity = new_entity
    else:
        new_slot = ContextSlot(
            ctx_id=f"ctx_{uuid.uuid4().hex[:8]}",
            domain=current_domain,
            entity=candidate.get("entity", ""),
            topic_turns=1,
            constraints=_merge_constraints({}, turn_constraints),
            excluded=_stable_unique(list(turn_excluded)),
            last_intent=current_intent,
            last_active=datetime.now().isoformat(),
        )
        cache.add(new_slot)
        slot = new_slot

    # 防御：slot.constraints 绝不含 None（_merge_constraints 已保证，此处作最终护栏）
    slot.constraints = {k: v for k, v in slot.constraints.items() if v is not None}

    # 写回 Redis
    cache_save = await save_topic_cache_result(user_id, cache)
    if cache_save.ok:
        context_persisted = True
        persistence_status = "saved"
    else:
        logger.warning("[context_manager] commit: TopicCache 保存失败: %s", cache_save.error)
        context_persisted = False
        persistence_status = "write_degraded"

    session_context = {
        "domain": slot.domain,
        "entity": slot.entity,
        "topic_turns": slot.topic_turns,
        "constraints": slot.constraints,
        "excluded": slot.excluded,
        "last_intent": slot.last_intent,
        "ctx_id": slot.ctx_id,
        "total_contexts": len(cache.contexts),
        "context_status": context_status,
        "missing_slots": list(missing_slots) if is_pending else [],
        "context_persisted": context_persisted,
        "persistence_status": persistence_status,
    }

    return {**state, "session_context": session_context, "context_status": context_status}


# ── 兼容别名（保留供旧测试使用）────────────────────────────────────────────────

async def run_context_manager(state: AgentState, redis_client) -> dict:
    """兼容别名，MVP 阶段保留，后续可移除。"""
    state = await resolve_context_candidate(state, redis_client)
    return await commit_context(state, redis_client)
