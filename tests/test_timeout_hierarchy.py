"""Phase 8 test: 超时层级数值正确性 + main.py 使用集中配置。"""

from __future__ import annotations

import pathlib

import pytest

from tools.timeout_config import TimeoutConfig


# ── 1. 数值层级验证 ──────────────────────────────────────────────────────────────

def test_llm_less_than_all_agent_deadlines():
    """单次 LLM 调用超时必须小于所有 Agent Deadline。"""
    assert TimeoutConfig.LLM < TimeoutConfig.MEAL_AGENT_DEADLINE
    assert TimeoutConfig.LLM < TimeoutConfig.PRICE_AGENT_DEADLINE
    assert TimeoutConfig.LLM < TimeoutConfig.UPDATE_AGENT_DEADLINE


def test_agent_deadlines_less_than_graph():
    """所有 Agent Deadline 必须小于 Graph Deadline。"""
    assert TimeoutConfig.MEAL_AGENT_DEADLINE < TimeoutConfig.GRAPH
    assert TimeoutConfig.PRICE_AGENT_DEADLINE < TimeoutConfig.GRAPH
    assert TimeoutConfig.UPDATE_AGENT_DEADLINE < TimeoutConfig.GRAPH


def test_graph_le_sse_idle():
    """Graph Deadline 必须不大于 SSE 空闲超时（watchdog 先于 SSE 触发）。"""
    assert TimeoutConfig.GRAPH <= TimeoutConfig.SSE_IDLE


def test_complete_ordering():
    """验证完整的超时层级递增关系。"""
    # 单次外部依赖 < Agent Deadline < Graph <= SSE
    assert TimeoutConfig.REDIS < TimeoutConfig.LLM
    assert TimeoutConfig.LLM < TimeoutConfig.PRICE_AGENT_DEADLINE
    assert TimeoutConfig.PRICE_AGENT_DEADLINE <= TimeoutConfig.MEAL_AGENT_DEADLINE
    assert TimeoutConfig.MEAL_AGENT_DEADLINE < TimeoutConfig.GRAPH
    assert TimeoutConfig.GRAPH <= TimeoutConfig.SSE_IDLE


# ── 2. main.py 使用集中配置验证（源码检查） ────────────────────────────────────

def test_main_uses_sse_idle_config():
    """main.py 的 queue.get() 超时必须使用 TimeoutConfig.SSE_IDLE，不得硬编码。"""
    src = (pathlib.Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
    assert "TimeoutConfig.SSE_IDLE" in src, \
        "main.py 中 queue.get() 超时必须使用 TimeoutConfig.SSE_IDLE"


def test_main_uses_graph_config_not_hardcoded():
    """main.py 的 Graph watchdog 超时必须使用 TimeoutConfig.GRAPH，不得硬编码 35。"""
    src = (pathlib.Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
    assert "TimeoutConfig.GRAPH" in src, \
        "main.py 的 watchdog 超时必须使用 TimeoutConfig.GRAPH"
    # 35.0 是旧硬编码值，不应再出现在 wait_for 调用中
    assert "timeout=35" not in src, \
        "main.py 中不应存在硬编码的 timeout=35"
