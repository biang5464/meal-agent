"""
tests/test_nutrition.py

验证营养知识库 RAG 系统：
  1. chunk_size     — 切块后每个 chunk ≤ CHUNK_SIZE 字符
  2. metadata       — 每个 chunk 有 source、chunk_index 字段
  3. doc_loaded     — load_nutrition_documents 后 collection count > 0
  4. search_struct  — search_nutrition 返回 {"content", "source", "distance"} 结构
  5. search_called  — nutrition_agent 调用了 search_nutrition
  6. agent_profile  — nutrition_agent 将用户饮食限制拼入检索 query
  7. supervisor_diabetes — supervisor 将"糖尿病"问题路由到 NUTRITION
  8. supervisor_gout     — supervisor 将"痛风"问题路由到 NUTRITION
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.nutrition as nut_mod
from tools.nutrition import (
    CHUNK_SIZE,
    init_nutrition_chroma,
    load_nutrition_documents,
    search_nutrition,
)
from agents.nutrition_agent import nutrition_agent
from agents.supervisor_agent import supervisor_agent
from core.tool_protocol import ToolMeta, ToolResult

_DOC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "nutrition"))
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")


# ── fixture：模块级加载文档一次 ────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def nutrition_loaded():
    init_nutrition_chroma(persist_dir=_PERSIST_DIR)
    load_nutrition_documents(doc_dir=_DOC_DIR, persist_dir=_PERSIST_DIR)
    yield


# ── 工具函数：构造最小 AgentState ──────────────────────────────────────────────

def _make_state(**overrides) -> dict:
    base = {
        "user_id": "test",
        "user_input": "",
        "user_brief": {},
        "intent": "",
        "result": "",
        "messages": [],
        "needs_meal": False,
        "chat_history": [],
    }
    base.update(overrides)
    return base


# ── 测试 1：chunk 长度不超过 CHUNK_SIZE ───────────────────────────────────────

def test_chunk_size():
    col = nut_mod._nutrition_collection
    assert col is not None
    items = col.get(include=["documents"])
    for doc in items["documents"]:
        assert len(doc) <= CHUNK_SIZE, f"chunk 超过 {CHUNK_SIZE} 字符: len={len(doc)}"
    print(f"\n[1] PASS — {len(items['documents'])} 个 chunk 均 ≤ {CHUNK_SIZE} 字符")


# ── 测试 2：chunk metadata 包含 source 和 chunk_index ──────────────────────────

def test_chunk_metadata():
    col = nut_mod._nutrition_collection
    assert col is not None
    items = col.get(include=["metadatas"])
    for meta in items["metadatas"]:
        assert "source" in meta, f"缺少 source: {meta}"
        assert "chunk_index" in meta, f"缺少 chunk_index: {meta}"
    print(f"\n[2] PASS — 所有 {len(items['metadatas'])} 个 chunk metadata 包含 source/chunk_index")


# ── 测试 3：load_nutrition_documents 后 count > 0 ─────────────────────────────

def test_doc_loaded():
    col = nut_mod._nutrition_collection
    assert col is not None
    count = col.count()
    assert count > 0, "文档未加载，collection 为空"
    print(f"\n[3] PASS — nutrition collection 共 {count} 个 chunk")


# ── 测试 4：search_nutrition 返回正确结构 ─────────────────────────────────────

def test_search_result_structure():
    results = search_nutrition("糖尿病饮食建议", n_results=2)
    assert isinstance(results, list)
    assert len(results) >= 1, "应至少返回 1 条结果"
    for r in results:
        assert "content" in r and r["content"], f"缺少 content: {r}"
        assert "source" in r, f"缺少 source: {r}"
        assert "distance" in r, f"缺少 distance: {r}"
        assert isinstance(r["distance"], float)
    print(f"\n[4] PASS — search_nutrition 返回 {len(results)} 条，结构正确")


# ── 测试 5：nutrition_agent 调用了 search_nutrition_result ──────────────────

@pytest.mark.asyncio
async def test_agent_calls_search_nutrition():
    async def _fake_astream(*args, **kwargs):
        chunk = MagicMock()
        chunk.content = "建议少吃高嘌呤食物。"
        yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = _fake_astream

    mock_rag = ToolResult(
        ok=True,
        data=[{"content": "低嘌呤饮食", "source": "gout_diet", "distance": 0.1}],
        meta=ToolMeta(tool_name="search_nutrition", elapsed_ms=100, attempts=1),
    )

    with patch("agents.nutrition_agent.search_nutrition_result",
               new=AsyncMock(return_value=mock_rag)) as mock_search, \
         patch("agents.nutrition_agent.get_llm", return_value=mock_llm):
        result_state = await nutrition_agent(
            _make_state(user_input="痛风可以吃什么")
        )

    mock_search.assert_called_once()
    assert result_state["result"]
    print(f"\n[5] PASS — nutrition_agent 调用了 search_nutrition_result，result 非空")


# ── 测试 6：nutrition_agent 将饮食限制拼入检索 query ─────────────────────────

@pytest.mark.asyncio
async def test_agent_includes_restrictions_in_query():
    async def _fake_astream(*args, **kwargs):
        chunk = MagicMock()
        chunk.content = "建议控制血糖。"
        yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = _fake_astream

    captured: list[str] = []

    async def fake_search_result(query, **kwargs):
        captured.append(query)
        return ToolResult(
            ok=True,
            data=[{"content": "控糖原则", "source": "diabetes_diet", "distance": 0.15}],
            meta=ToolMeta(tool_name="search_nutrition", elapsed_ms=100, attempts=1),
        )

    with patch("agents.nutrition_agent.search_nutrition_result",
               side_effect=fake_search_result), \
         patch("agents.nutrition_agent.get_llm", return_value=mock_llm):
        await nutrition_agent(
            _make_state(
                user_input="能吃什么",
                user_brief={"active_restrictions": ["糖尿病"]},
            )
        )

    assert captured, "search_nutrition_result 未被调用"
    assert "糖尿病" in captured[0], f"query 未包含 active_restrictions: {captured[0]!r}"
    print(f"\n[6] PASS — query 包含饮食限制: {captured[0]!r}")


# ── 测试 7：supervisor 将"糖尿病"问题路由到 NUTRITION ────────────────────────

@pytest.mark.asyncio
async def test_supervisor_routes_diabetes_to_nutrition():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="INTENT: NUTRITION\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
    ))

    with patch("agents.supervisor_agent.get_llm", return_value=mock_llm):
        result = await supervisor_agent(
            _make_state(user_input="糖尿病患者能吃白米饭吗")
        )

    assert result["intent"] == "NUTRITION", f"期望 NUTRITION，实际 {result['intent']!r}"
    print(f"\n[7] PASS — 糖尿病问题 → intent=NUTRITION")


# ── 测试 8：supervisor 将"痛风"问题路由到 NUTRITION ──────────────────────────

@pytest.mark.asyncio
async def test_supervisor_routes_gout_to_nutrition():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="INTENT: NUTRITION\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
    ))

    with patch("agents.supervisor_agent.get_llm", return_value=mock_llm):
        result = await supervisor_agent(
            _make_state(user_input="痛风发作期间哪些食物要忌口")
        )

    assert result["intent"] == "NUTRITION", f"期望 NUTRITION，实际 {result['intent']!r}"
    print(f"\n[8] PASS — 痛风问题 → intent=NUTRITION")
