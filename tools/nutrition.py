"""营养知识库工具：基于 ChromaDB 向量库做 RAG 语义检索。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from tools.timeout_config import TimeoutConfig

logger = logging.getLogger(__name__)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

NUTRITION_COLLECTION = "nutrition"
FOOD_SAFETY_COLLECTION = "food_safety"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
_EMBEDDING_MODEL = "BAAI/bge-small-zh"

_nutrition_collection: chromadb.Collection | None = None
_nutrition_client: chromadb.ClientAPI | None = None

_food_safety_collection: chromadb.Collection | None = None
_food_safety_client: chromadb.ClientAPI | None = None


def _make_ef() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)


def init_nutrition_chroma(persist_dir: str | None = None) -> chromadb.Collection:
    """连接或创建 nutrition ChromaDB collection，返回 collection 对象。"""
    global _nutrition_collection, _nutrition_client

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    os.makedirs(persist_dir, exist_ok=True)
    _nutrition_client = chromadb.PersistentClient(path=persist_dir)

    # 先用无 ef 的 get_collection 探测，避免 ChromaDB 0.6+ ef 类型冲突
    try:
        probe = _nutrition_client.get_collection(name=NUTRITION_COLLECTION)
        col_meta = probe.metadata or {}
        if col_meta.get("embedding_model") != _EMBEDDING_MODEL:
            _nutrition_client.delete_collection(NUTRITION_COLLECTION)
    except Exception:
        pass

    _nutrition_collection = _nutrition_client.get_or_create_collection(
        name=NUTRITION_COLLECTION,
        embedding_function=_make_ef(),
        metadata={"hnsw:space": "cosine", "embedding_model": _EMBEDDING_MODEL},
    )
    return _nutrition_collection


def load_nutrition_documents(doc_dir: str, persist_dir: str | None = None) -> int:
    """
    加载 doc_dir 下所有 .txt 文件，切块后写入 nutrition collection。
    先清空再重载（幂等）。返回写入的 chunk 数量。
    """
    col = _nutrition_collection if _nutrition_collection is not None else init_nutrition_chroma(persist_dir)

    existing = col.get(include=[])
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    ids, documents, metadatas = [], [], []
    for path in sorted(Path(doc_dir).glob("*.txt")):
        source = path.stem
        text = path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{source}_{i}")
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})

    if ids:
        col.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(ids)


def search_nutrition(
    query: str,
    n_results: int = 4,
    persist_dir: str | None = None,
) -> list[dict]:
    """
    语义检索营养知识库，返回最相关的 chunk 列表。

    Returns:
        [{"content": str, "source": str, "distance": float}, ...]
    """
    col = _nutrition_collection if _nutrition_collection is not None else init_nutrition_chroma(persist_dir)

    count = col.count()
    if count == 0:
        return []

    try:
        _start = time.time()
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        _elapsed = time.time() - _start
        if _elapsed > TimeoutConfig.CHROMA:
            print(f"[chroma] search_nutrition 慢查询: {_elapsed:.2f}s")
    except Exception as e:
        print(f"[chroma] search_nutrition 失败: {e}")
        return []

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "content": doc,
            "source": meta.get("source", ""),
            "distance": round(dist, 4),
        })

    return chunks


# ── 食品安全知识库 ─────────────────────────────────────────────────────────────

def init_food_safety_chroma(persist_dir: str | None = None) -> chromadb.Collection:
    """连接或创建 food_safety ChromaDB collection，返回 collection 对象。"""
    global _food_safety_collection, _food_safety_client

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    os.makedirs(persist_dir, exist_ok=True)
    _food_safety_client = chromadb.PersistentClient(path=persist_dir)

    try:
        probe = _food_safety_client.get_collection(name=FOOD_SAFETY_COLLECTION)
        col_meta = probe.metadata or {}
        if col_meta.get("embedding_model") != _EMBEDDING_MODEL:
            _food_safety_client.delete_collection(FOOD_SAFETY_COLLECTION)
    except Exception:
        pass

    _food_safety_collection = _food_safety_client.get_or_create_collection(
        name=FOOD_SAFETY_COLLECTION,
        embedding_function=_make_ef(),
        metadata={"hnsw:space": "cosine", "embedding_model": _EMBEDDING_MODEL},
    )
    return _food_safety_collection


def load_food_safety_documents(doc_dir: str, persist_dir: str | None = None) -> int:
    """
    加载 doc_dir 下所有 .txt 文件，切块后写入 food_safety collection。
    先清空再重载（幂等）。返回写入的 chunk 数量。
    """
    col = _food_safety_collection if _food_safety_collection is not None else init_food_safety_chroma(persist_dir)

    existing = col.get(include=[])
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    _SOURCE_ALIAS = {
        "ingredient_combinations": "食材配伍禁忌与药食相克规范",
    }

    ids, documents, metadatas = [], [], []
    for path in sorted(Path(doc_dir).glob("*.txt")):
        source = _SOURCE_ALIAS.get(path.stem, path.stem)
        text = path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{source}_{i}")
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})

    if ids:
        col.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(ids)


def search_food_safety(
    query: str,
    n_results: int = 4,
    persist_dir: str | None = None,
) -> list[dict]:
    """
    语义检索食品安全知识库，返回最相关的 chunk 列表。

    Returns:
        [{"content": str, "source": str, "distance": float}, ...]
    """
    col = _food_safety_collection if _food_safety_collection is not None else init_food_safety_chroma(persist_dir)

    count = col.count()
    if count == 0:
        return []

    try:
        _start = time.time()
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        _elapsed = time.time() - _start
        if _elapsed > TimeoutConfig.CHROMA:
            print(f"[chroma] search_food_safety 慢查询: {_elapsed:.2f}s")
    except Exception as e:
        print(f"[chroma] search_food_safety 失败: {e}")
        return []

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "content": doc,
            "source": meta.get("source", ""),
            "distance": round(dist, 4),
        })

    return chunks


# ── 统一协议服务层（Phase 5）────────────────────────────────────────────────────


def _search_nutrition_impl(query: str, n_results: int) -> list[dict]:
    """严格检索营养知识库；collection 未就绪或查询失败时抛出异常。"""
    if _nutrition_collection is None:
        raise ConnectionError("nutrition collection 未初始化")
    col = _nutrition_collection
    count = col.count()
    if count == 0:
        return []
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        chunks.append({"content": doc, "source": meta.get("source", ""), "distance": round(dist, 4)})
    return chunks


async def search_nutrition_result(query: str, n_results: int = 4):
    """
    通过 ToolExecutor.execute() 检索营养知识库，返回 ToolResult。

    - ok=True, data=[chunks]：检索成功，有结果
    - ok=True, data=[]：检索成功，无匹配内容
    - ok=False, TIMEOUT：检索超时
    - ok=False, DEPENDENCY_UNAVAILABLE：collection 未就绪或 ChromaDB 故障
    """
    from core.tool_protocol import ToolError, ToolErrorCode, ToolResult
    from tools.tool_executor import tool_executor

    result = await tool_executor.execute(
        _search_nutrition_impl,
        query,
        n_results=n_results,
        policy_name="chroma_search",
        tool_name="search_nutrition",
        fallback_data=[],
        source="nutrition_chroma",
    )

    if not result.ok and result.error and result.error.code != ToolErrorCode.TIMEOUT:
        return ToolResult(
            ok=False,
            data=result.data,
            error=ToolError(
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                message="营养知识库暂时不可用",
                retryable=True,
                details={"component": "collection", "collection": "nutrition"},
            ),
            meta=result.meta,
        )
    return result


def _search_food_safety_impl(query: str, n_results: int) -> list[dict]:
    """严格检索食品安全知识库；collection 未就绪或查询失败时抛出异常。"""
    if _food_safety_collection is None:
        raise ConnectionError("food_safety collection 未初始化")
    col = _food_safety_collection
    count = col.count()
    if count == 0:
        return []
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        chunks.append({"content": doc, "source": meta.get("source", ""), "distance": round(dist, 4)})
    return chunks


async def search_food_safety_result(query: str, n_results: int = 4):
    """
    通过 ToolExecutor.execute() 检索食品安全知识库，返回 ToolResult。

    - ok=True, data=[chunks]：检索成功，有结果
    - ok=True, data=[]：检索成功，无匹配内容
    - ok=False, TIMEOUT：检索超时
    - ok=False, DEPENDENCY_UNAVAILABLE：collection 未就绪或 ChromaDB 故障
    """
    from core.tool_protocol import ToolError, ToolErrorCode, ToolResult
    from tools.tool_executor import tool_executor

    result = await tool_executor.execute(
        _search_food_safety_impl,
        query,
        n_results=n_results,
        policy_name="chroma_search",
        tool_name="search_food_safety",
        fallback_data=[],
        source="food_safety_chroma",
    )

    if not result.ok and result.error and result.error.code != ToolErrorCode.TIMEOUT:
        return ToolResult(
            ok=False,
            data=result.data,
            error=ToolError(
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                message="食品安全知识库暂时不可用",
                retryable=True,
                details={"component": "collection", "collection": "food_safety"},
            ),
            meta=result.meta,
        )
    return result
