from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
import re
import uuid

DOMAINS = {
    "electronics": ["ELECTRONICS_PRICE"],
    "food": ["MEAL", "PRICE", "UPDATE_AND_MEAL", "HEALTH_CYCLE"],
    "health": ["NUTRITION", "FOOD_SAFETY"],
    "system": ["MEMORY_UPDATE", "CHAT"],
}

NEGATIONS = ["不是", "不要", "排除", "除了", "别", "不考虑", "去掉", "不需要", "不想要"]


def get_domain(intent: str) -> str:
    for domain, intents in DOMAINS.items():
        if intent in intents:
            return domain
    return "system"


def extract_excluded_by_rule(user_input: str) -> list[str]:
    """
    规则层：从用户输入里提取排除条件。
    覆盖80%场景，失败时返回空列表交给LLM处理。
    """
    excluded = []
    for neg in NEGATIONS:
        pattern = f"{neg}(.{{1,20}}?)(?:[，。？！,!?]|$)"
        matches = re.findall(pattern, user_input)
        for m in matches:
            m = m.strip()
            if m and len(m) <= 15:
                excluded.append(m)
    return excluded


def needs_llm_extraction(user_input: str, rule_result: list) -> bool:
    has_negation = any(neg in user_input for neg in NEGATIONS)
    return has_negation and not rule_result


@dataclass
class SessionContext:
    domain: str = ""
    entity: str = ""
    topic_turns: int = 0
    constraints: dict = field(default_factory=dict)
    excluded: list = field(default_factory=list)
    last_intent: str = ""
    last_updated: str = ""
    turn_count: int = 0

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "entity": self.entity,
            "topic_turns": self.topic_turns,
            "constraints": self.constraints,
            "excluded": self.excluded,
            "last_intent": self.last_intent,
            "last_updated": self.last_updated,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionContext":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def empty(cls) -> "SessionContext":
        return cls()


# ── Multi-Topic Cache ────────────────────────────────────────────────────────

MAX_CONTEXTS = 5                    # LRU 最多保存5个 context
ENTITY_SIMILARITY_THRESHOLD = 0.85  # entity 复用阈值
EMBEDDING_MATCH_THRESHOLD = 0.5     # Embedding 匹配阈值


@dataclass
class ContextSlot:
    """单个话题的上下文槽"""
    ctx_id: str = ""
    domain: str = ""
    entity: str = ""
    topic_turns: int = 0
    constraints: dict = field(default_factory=dict)
    excluded: list = field(default_factory=list)
    last_intent: str = ""
    last_active: str = ""  # ISO格式时间，用于LRU排序

    def to_dict(self) -> dict:
        return {
            "ctx_id": self.ctx_id,
            "domain": self.domain,
            "entity": self.entity,
            "topic_turns": self.topic_turns,
            "constraints": self.constraints,
            "excluded": self.excluded,
            "last_intent": self.last_intent,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextSlot":
        return cls(**{k: v for k, v in data.items()
                     if k in cls.__dataclass_fields__})


@dataclass
class TopicCache:
    """Multi-Topic Cache，LRU管理最多5个 context"""
    current_context_id: str = ""
    # 注意：运行时使用 OrderedDict 以支持 move_to_end
    contexts: dict = field(default_factory=OrderedDict)

    def to_dict(self) -> dict:
        return {
            "current_context_id": self.current_context_id,
            "contexts": {k: v.to_dict() for k, v in self.contexts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopicCache":
        contexts = OrderedDict()
        for k, v in data.get("contexts", {}).items():
            contexts[k] = ContextSlot.from_dict(v)
        return cls(
            current_context_id=data.get("current_context_id", ""),
            contexts=contexts,
        )

    @classmethod
    def empty(cls) -> "TopicCache":
        return cls()

    def get_current(self) -> "ContextSlot | None":
        """获取当前活跃的 context"""
        if self.current_context_id and self.current_context_id in self.contexts:
            return self.contexts[self.current_context_id]
        return None

    def access(self, ctx_id: str):
        """访问一个 context，更新 LRU 顺序"""
        if ctx_id in self.contexts:
            self.contexts.move_to_end(ctx_id)
            self.contexts[ctx_id].last_active = datetime.now().isoformat()
            self.current_context_id = ctx_id

    def add(self, slot: "ContextSlot"):
        """新增 context，超过上限时淘汰最久未访问的"""
        if len(self.contexts) >= MAX_CONTEXTS:
            oldest_id, _ = next(iter(self.contexts.items()))
            del self.contexts[oldest_id]
            print(f"[topic_cache] LRU 淘汰: {oldest_id}")
        self.contexts[slot.ctx_id] = slot
        self.contexts.move_to_end(slot.ctx_id)
        self.current_context_id = slot.ctx_id

    def find_by_entity(self, entity: str, embedding_fn=None) -> "ContextSlot | None":
        """
        按 entity 找已有 context。
        先精确/包含匹配，再 embedding 相似度匹配。
        """
        if not entity:
            return None
        entity_lower = entity.lower()
        for slot in self.contexts.values():
            if slot.entity and (
                entity_lower in slot.entity.lower() or
                slot.entity.lower() in entity_lower
            ):
                return slot
        if embedding_fn:
            best_slot = None
            best_score = 0.0
            for slot in self.contexts.values():
                if not slot.entity:
                    continue
                try:
                    score = embedding_fn(entity, slot.entity)
                    if score > best_score:
                        best_score = score
                        best_slot = slot
                except Exception:
                    continue
            if best_score >= ENTITY_SIMILARITY_THRESHOLD:
                return best_slot
        return None

    def find_by_embedding(self, query: str, embedding_fn) -> "ContextSlot | None":
        """
        用 query 找语义最相似的 context。
        相似度低于阈值时返回 None（触发新建）。
        """
        if not embedding_fn or not self.contexts:
            return None
        best_slot = None
        best_score = 0.0
        for slot in self.contexts.values():
            if not slot.entity:
                continue
            try:
                score = embedding_fn(query, slot.entity)
                if score > best_score:
                    best_score = score
                    best_slot = slot
            except Exception:
                continue
        if best_score >= EMBEDDING_MATCH_THRESHOLD:
            return best_slot
        return None
