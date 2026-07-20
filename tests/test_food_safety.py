"""
tests/test_food_safety.py

验证食品安全知识库 RAG 系统：
  1. chunk_size          — 切块后每个 chunk ≤ CHUNK_SIZE 字符
  2. chunk_metadata      — 每个 chunk 有 source、chunk_index 字段
  3. search_returns      — search_food_safety 返回非空结果
  4. search_structure    — 返回结果包含 content/source/distance
  5. cooking_safety_doc  — cooking_safety.txt 正确加载（source 字段可检索到）
  6. supervisor_cooking  — "四季豆没煮熟能吃吗" 路由到 FOOD_SAFETY
  7. supervisor_storage  — "木耳泡多久会有毒" 路由到 FOOD_SAFETY
  8. supervisor_nutrition— "糖尿病能吃燕麦吗" 路由到 NUTRITION 不是 FOOD_SAFETY
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.nutrition as nut_mod
from tools.nutrition import (
    CHUNK_SIZE,
    init_food_safety_chroma,
    load_food_safety_documents,
    search_food_safety,
)
from agents.supervisor_agent import supervisor_agent

_DOC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "food_safety")
)
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")


# ── fixture：模块级加载文档一次 ────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def food_safety_loaded():
    init_food_safety_chroma(persist_dir=_PERSIST_DIR)
    load_food_safety_documents(doc_dir=_DOC_DIR, persist_dir=_PERSIST_DIR)
    yield


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

def test_food_safety_chunk_size():
    col = nut_mod._food_safety_collection
    assert col is not None
    items = col.get(include=["documents"])
    assert items["documents"], "collection 为空，文档未加载"
    for doc in items["documents"]:
        assert len(doc) <= CHUNK_SIZE, f"chunk 超过 {CHUNK_SIZE} 字符: len={len(doc)}"
    print(f"\n[1] PASS — {len(items['documents'])} 个 chunk 均 ≤ {CHUNK_SIZE} 字符")


# ── 测试 2：chunk metadata 包含 source 和 chunk_index ──────────────────────────

def test_food_safety_chunk_metadata():
    col = nut_mod._food_safety_collection
    assert col is not None
    items = col.get(include=["metadatas"])
    for meta in items["metadatas"]:
        assert "source" in meta, f"缺少 source: {meta}"
        assert "chunk_index" in meta, f"缺少 chunk_index: {meta}"
    print(f"\n[2] PASS — 所有 {len(items['metadatas'])} 个 chunk metadata 包含 source/chunk_index")


# ── 测试 3：search_food_safety 返回非空结果 ────────────────────────────────────

def test_search_food_safety_returns_results():
    results = search_food_safety("四季豆中毒", n_results=2)
    assert isinstance(results, list)
    assert len(results) >= 1, "应至少返回 1 条结果"
    print(f"\n[3] PASS — search_food_safety 返回 {len(results)} 条结果")


# ── 测试 4：search_food_safety 返回正确结构 ────────────────────────────────────

def test_search_food_safety_structure():
    results = search_food_safety("木耳泡发安全", n_results=2)
    for r in results:
        assert "content" in r and r["content"], f"缺少 content: {r}"
        assert "source" in r, f"缺少 source: {r}"
        assert "distance" in r, f"缺少 distance: {r}"
        assert isinstance(r["distance"], float)
    print(f"\n[4] PASS — 结果结构正确，{len(results)} 条")


# ── 测试 5：cooking_safety.txt 正确加载 ───────────────────────────────────────

def test_cooking_safety_doc_loaded():
    col = nut_mod._food_safety_collection
    assert col is not None
    items = col.get(include=["metadatas"])
    sources = {m["source"] for m in items["metadatas"]}
    assert "cooking_safety" in sources, f"cooking_safety 未加载，已有来源: {sources}"
    print(f"\n[5] PASS — cooking_safety 已加载，所有来源: {sources}")


# ── 测试 6：supervisor 将"四季豆没煮熟能吃吗"路由到 FOOD_SAFETY ─────────────

@pytest.mark.asyncio
async def test_supervisor_routes_food_safety_cooking():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="INTENT: FOOD_SAFETY\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
    ))

    with patch("agents.supervisor_agent.get_llm", return_value=mock_llm):
        result = await supervisor_agent(
            _make_state(user_input="四季豆没煮熟能吃吗")
        )

    assert result["intent"] == "FOOD_SAFETY", f"期望 FOOD_SAFETY，实际 {result['intent']!r}"
    print(f"\n[6] PASS — 四季豆问题 → intent=FOOD_SAFETY")


# ── 测试 7：supervisor 将"木耳泡多久会有毒"路由到 FOOD_SAFETY ────────────────

@pytest.mark.asyncio
async def test_supervisor_routes_food_safety_storage():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="INTENT: FOOD_SAFETY\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
    ))

    with patch("agents.supervisor_agent.get_llm", return_value=mock_llm):
        result = await supervisor_agent(
            _make_state(user_input="木耳泡多久会有毒")
        )

    assert result["intent"] == "FOOD_SAFETY", f"期望 FOOD_SAFETY，实际 {result['intent']!r}"
    print(f"\n[7] PASS — 木耳问题 → intent=FOOD_SAFETY")


# ── 测试 8：supervisor 将"糖尿病能吃燕麦吗"路由到 NUTRITION ─────────────────

@pytest.mark.asyncio
async def test_supervisor_distinguishes_nutrition():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="INTENT: NUTRITION\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
    ))

    with patch("agents.supervisor_agent.get_llm", return_value=mock_llm):
        result = await supervisor_agent(
            _make_state(user_input="糖尿病能吃燕麦吗")
        )

    assert result["intent"] == "NUTRITION", f"期望 NUTRITION，实际 {result['intent']!r}"
    assert result["intent"] != "FOOD_SAFETY", "不应路由到 FOOD_SAFETY"
    print(f"\n[8] PASS — 糖尿病营养问题 → intent=NUTRITION（不是 FOOD_SAFETY）")
