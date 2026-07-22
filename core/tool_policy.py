from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPolicy:
    timeout: float
    retries: int
    backoff_base: float = 0.5
    idempotent: bool = True


TOOL_POLICIES: dict[str, ToolPolicy] = {
    "redis_read": ToolPolicy(timeout=0.3, retries=1),
    "redis_write": ToolPolicy(timeout=0.3, retries=0, idempotent=False),
    "rate_limit_write": ToolPolicy(timeout=0.5, retries=0, idempotent=False),
    "chroma_search": ToolPolicy(timeout=2.0, retries=1),
    "database_read": ToolPolicy(timeout=3.0, retries=1),
    "database_write": ToolPolicy(timeout=3.0, retries=0, idempotent=False),
    "external_http": ToolPolicy(timeout=8.0, retries=2),
    "llm_classification": ToolPolicy(timeout=20.0, retries=1),
    "llm_generation": ToolPolicy(timeout=60.0, retries=0),
}


def get_tool_policy(name: str) -> ToolPolicy:
    if name not in TOOL_POLICIES:
        raise ValueError(
            f"未知的 ToolPolicy: '{name}'。"
            f"可用策略：{sorted(TOOL_POLICIES.keys())}"
        )
    return TOOL_POLICIES[name]
