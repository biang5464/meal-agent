"""菜谱检索工具：基于 ChromaDB 向量库做语义检索。"""

from __future__ import annotations

import json
import os
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.tools import tool

_EMBEDDING_MODEL = "BAAI/bge-small-zh"


def _make_ef() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)

_collection: chromadb.Collection | None = None
_client = None  # chromadb.ClientAPI

# ---------- 预置菜谱数据 ----------

_SEED_RECIPES = [
    {
        "name": "宫保鸡丁",
        "ingredients": ["鸡胸肉", "花生", "干辣椒", "葱", "姜", "蒜", "生抽", "醋", "糖"],
        "cuisine": "川菜",
        "flavor": "辣",
        "steps": "鸡肉切丁腌制，热油爆香干辣椒和花椒，下鸡丁翻炒至变色，加花生和调味汁大火收汁。",
        "budget_tier": "unknown",
        "category": "荤菜",
    },
    {
        "name": "番茄炒蛋",
        "ingredients": ["番茄", "鸡蛋", "盐", "糖", "葱"],
        "cuisine": "家常菜",
        "flavor": "鲜甜",
        "steps": "鸡蛋打散炒熟盛出，番茄切块下锅翻炒出汁，加回鸡蛋，调盐和少许糖提鲜。",
        "budget_tier": "unknown",
        "category": "半荤素",
    },
    {
        "name": "清蒸鲈鱼",
        "ingredients": ["鲈鱼", "姜", "葱", "生抽", "料酒", "蒸鱼豉油"],
        "cuisine": "粤菜",
        "flavor": "清淡",
        "steps": "鱼身划刀，姜片垫底，蒸锅上汽后蒸8分钟，取出浇热油和蒸鱼豉油，撒葱丝。",
        "budget_tier": "unknown",
        "category": "海鲜",
    },
    {
        "name": "麻婆豆腐",
        "ingredients": ["嫩豆腐", "猪肉末", "豆瓣酱", "花椒", "干辣椒", "葱", "蒜", "生抽"],
        "cuisine": "川菜",
        "flavor": "麻辣",
        "steps": "热油炒肉末，加豆瓣酱炒红油，加水烧开放豆腐，勾芡后撒花椒粉和葱花。",
        "budget_tier": "unknown",
        "category": "豆腐",
    },
    {
        "name": "蒜蓉西兰花",
        "ingredients": ["西兰花", "大蒜", "盐", "生抽", "食用油"],
        "cuisine": "家常菜",
        "flavor": "清淡",
        "steps": "西兰花焯水，热油爆香蒜末，下西兰花大火翻炒，加盐和生抽调味出锅。",
        "budget_tier": "unknown",
        "category": "蔬菜",
    },
    {
        "name": "红烧肉",
        "ingredients": ["五花肉", "生抽", "老抽", "冰糖", "料酒", "姜", "八角", "桂皮"],
        "cuisine": "家常菜",
        "flavor": "咸甜",
        "steps": "五花肉焯水，锅中炒糖色，放肉翻炒上色，加调料和热水小火焖40分钟，大火收汁。",
        "budget_tier": "unknown",
        "category": "肉类",
    },
    {
        "name": "白灼虾",
        "ingredients": ["鲜虾", "姜", "葱", "盐", "生抽", "香油"],
        "cuisine": "粤菜",
        "flavor": "鲜甜",
        "steps": "虾洗净，锅中水烧开加姜葱盐，放入鲜虾汆熟捞出，配蘸汁（生抽+香油）上桌。",
        "budget_tier": "unknown",
        "category": "海鲜",
    },
    {
        "name": "土豆丝炒肉",
        "ingredients": ["土豆", "猪里脊", "青椒", "红椒", "醋", "盐", "生抽"],
        "cuisine": "家常菜",
        "flavor": "酸辣",
        "steps": "土豆丝泡水去淀粉，里脊切丝腌制，热锅先炒肉丝，再下土豆丝和青红椒，加醋和盐翻炒。",
        "budget_tier": "unknown",
        "category": "半荤素",
    },
    {
        "name": "冬瓜排骨汤",
        "ingredients": ["排骨", "冬瓜", "姜", "盐", "葱"],
        "cuisine": "家常菜",
        "flavor": "清淡",
        "steps": "排骨焯水，加姜片冷水下锅，大火烧开转小火煮40分钟，加冬瓜块再煮15分钟，盐调味。",
        "budget_tier": "unknown",
        "category": "汤",
    },
    {
        "name": "干煸四季豆",
        "ingredients": ["四季豆", "猪肉末", "干辣椒", "蒜", "生抽", "盐"],
        "cuisine": "川菜",
        "flavor": "辣",
        "steps": "四季豆下锅干煸至表皮微皱，盛出；热油炒肉末，加蒜末和干辣椒，放回四季豆翻炒调味。",
        "budget_tier": "unknown",
        "category": "蔬菜",
    },
]


def _recipe_document(r: dict) -> str:
    """将菜谱结构化字段拼接为自然语言文本，用于向量化。"""
    return (
        f"菜名：{r['name']}。"
        f"菜系：{r['cuisine']}。"
        f"口味：{r['flavor']}。"
        f"食材：{'、'.join(r['ingredients'])}。"
        f"做法：{r['steps']}"
    )


# ---------- 初始化 ----------

def init_chroma(persist_dir: str | None = None, collection_name: str = "recipes") -> None:
    """
    连接或创建 ChromaDB collection。
    若已有数据缺少 budget_tier 字段，则清空重建以保证 where 过滤可用。

    persist_dir 未传则读取 CHROMA_PERSIST_DIR 环境变量。
    """
    global _collection, _client

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    os.makedirs(persist_dir, exist_ok=True)

    _client = chromadb.PersistentClient(path=persist_dir)

    # ChromaDB 0.6+ 在 get_or_create_collection 时校验 ef 类型，旧集合若用 default ef
    # 直接传新 ef 就会抛 ValueError。先用无 ef 的 get_collection 探测，按需删除再创建。
    try:
        probe = _client.get_collection(name=collection_name)
        col_meta = probe.metadata or {}
        needs_rebuild = col_meta.get("embedding_model") != _EMBEDDING_MODEL
        if not needs_rebuild and probe.count() > 0:
            items = probe.get(limit=1, include=["metadatas"])
            needs_rebuild = bool(
                items["metadatas"] and (
                    "budget_tier" not in items["metadatas"][0]
                    or "category" not in items["metadatas"][0]
                )
            )
        if needs_rebuild:
            print("[chroma] 检测到数据需重建（embedding 模型变更或缺少字段），清空重建...")
            _client.delete_collection(collection_name)
    except Exception:
        pass  # 集合不存在，忽略

    _collection = _client.get_or_create_collection(
        name=collection_name,
        embedding_function=_make_ef(),
        metadata={"hnsw:space": "cosine", "embedding_model": _EMBEDDING_MODEL},
    )

    if _collection.count() == 0:
        _seed_recipes()


def _seed_recipes() -> None:
    col = _ensure_collection()
    ids, documents, metadatas = [], [], []
    for r in _SEED_RECIPES:
        ids.append(f"recipe_{r['name']}")
        documents.append(_recipe_document(r))
        metadatas.append({
            "name": r["name"],
            "cuisine": r["cuisine"],
            "flavor": r["flavor"],
            "ingredients": json.dumps(r["ingredients"], ensure_ascii=False),
            "steps": r["steps"],
            "budget_tier": r.get("budget_tier", "unknown"),
            "category": r.get("category", ""),
        })
    col.add(ids=ids, documents=documents, metadatas=metadatas)


def _ensure_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError("ChromaDB not initialized. Call init_chroma() first.")
    return _collection


def _meta_to_dict(meta: dict) -> dict:
    """将 ChromaDB metadata 反序列化为结构化字典。"""
    return {
        "name": meta.get("name", ""),
        "cuisine": meta.get("cuisine", ""),
        "flavor": meta.get("flavor", ""),
        "ingredients": json.loads(meta.get("ingredients", "[]")),
        "steps": meta.get("steps", ""),
        "budget_tier": meta.get("budget_tier", "unknown"),
        "category": meta.get("category", ""),
    }


# ---------- LangChain Tools ----------

def _search_recipes_impl(
    query: str,
    user_likes: list[str],
    user_dislikes: list[str],
    budget_tier: str,
    n_results: int = 3,
) -> list[dict]:
    """严格检索菜谱；collection 未就绪或查询失败时抛出异常。"""
    if _collection is None:
        raise ConnectionError("recipe collection 未初始化")
    col = _collection

    enriched = query
    if user_likes:
        enriched += " 喜欢：" + "、".join(user_likes)
    if user_dislikes:
        enriched += " 不要：" + "、".join(user_dislikes)

    where_filter = (
        {"budget_tier": {"$eq": budget_tier}}
        if budget_tier in ("low", "mid", "high")
        else None
    )

    if where_filter:
        matching_ids = col.get(where=where_filter, include=[])["ids"]
        actual_n = min(n_results, len(matching_ids))
        if actual_n == 0:
            return []
    else:
        actual_n = min(n_results, col.count())

    if actual_n == 0:
        return []

    results = col.query(query_texts=[enriched], n_results=actual_n, where=where_filter)

    recipes = []
    for meta, dist in zip(
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        recipe = _meta_to_dict(meta)
        if any(d in " ".join(recipe["ingredients"]) for d in user_dislikes):
            continue
        recipe["relevance_score"] = round(1 - dist, 4)
        recipes.append(recipe)

    return recipes[:n_results]


async def search_recipes_result(
    query: str,
    user_likes: list[str] | None = None,
    user_dislikes: list[str] | None = None,
    budget_tier: str = "",
    n_results: int = 5,
):
    """
    通过 ToolExecutor.execute() 检索菜谱知识库，返回 ToolResult。

    - ok=True, data=[recipes]：检索成功，有结果
    - ok=True, data=[]：检索成功，无匹配菜谱
    - ok=False, TIMEOUT：检索超时
    - ok=False, DEPENDENCY_UNAVAILABLE：collection 未就绪或 ChromaDB 故障
    """
    from core.tool_protocol import ToolError, ToolErrorCode, ToolResult
    from tools.tool_executor import tool_executor

    result = await tool_executor.execute(
        _search_recipes_impl,
        query,
        user_likes=user_likes or [],
        user_dislikes=user_dislikes or [],
        budget_tier=budget_tier,
        n_results=n_results,
        policy_name="chroma_search",
        tool_name="search_recipes",
        fallback_data=[],
        source="recipe_chroma",
    )

    if not result.ok and result.error and result.error.code != ToolErrorCode.TIMEOUT:
        return ToolResult(
            ok=False,
            data=result.data,
            error=ToolError(
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                message="菜谱知识库暂时不可用",
                retryable=True,
                details={"component": "collection", "collection": "recipes"},
            ),
            meta=result.meta,
        )
    return result


@tool
async def search_recipes(
    query: str,
    user_likes: List[str] = [],
    user_dislikes: List[str] = [],
    budget_tier: str = "",
) -> List[dict]:
    """
    根据用户口味偏好检索匹配的菜谱，返回 Top3 结果。

    Args:
        query:         饮食需求描述，如"下饭的辣菜"
        user_likes:    用户喜欢的口味/食材/菜系列表
        user_dislikes: 用户忌口列表
        budget_tier:   预算档位过滤，"low"/"mid"/"high"/"" 表示不过滤

    Returns:
        最多 3 条匹配菜谱，按相关度排序，包含 name/cuisine/flavor/ingredients/steps/budget_tier。
    """
    result = await search_recipes_result(
        query,
        user_likes=user_likes,
        user_dislikes=user_dislikes,
        budget_tier=budget_tier,
        n_results=3,
    )
    return result.data if result.ok else []


@tool
def get_recipe_by_name(name: str) -> dict:
    """
    按菜名精确查找菜谱详情。

    Args:
        name: 菜名，如"宫保鸡丁"

    Returns:
        菜谱详情字典；未找到时返回 {"error": "not found", "name": name}。
    """
    col = _ensure_collection()

    results = col.get(
        where={"name": {"$eq": name}},
        include=["metadatas"],
    )

    metadatas = results.get("metadatas", [])
    if not metadatas:
        return {"error": "not found", "name": name}

    return _meta_to_dict(metadatas[0])


@tool
def add_recipe(
    name: str,
    ingredients: List[str],
    cuisine: str,
    flavor: str,
    steps: str,
) -> str:
    """
    向知识库中添加新菜谱。若同名菜谱已存在则覆盖更新。

    Args:
        name:        菜名
        ingredients: 食材列表，如 ["鸡蛋", "番茄"]
        cuisine:     菜系，如"川菜"
        flavor:      口味标签，如"辣"
        steps:       做法描述

    Returns:
        操作结果描述。
    """
    col = _ensure_collection()

    recipe = {"name": name, "ingredients": ingredients, "cuisine": cuisine,
              "flavor": flavor, "steps": steps}
    doc_id = f"recipe_{name}"
    document = _recipe_document(recipe)
    metadata = {
        "name": name,
        "cuisine": cuisine,
        "flavor": flavor,
        "ingredients": json.dumps(ingredients, ensure_ascii=False),
        "steps": steps,
        "budget_tier": "unknown",
    }

    existing = col.get(ids=[doc_id], include=[])
    if existing["ids"]:
        col.update(ids=[doc_id], documents=[document], metadatas=[metadata])
        return f"已更新菜谱：{name}"

    col.add(ids=[doc_id], documents=[document], metadatas=[metadata])
    return f"已添加菜谱：{name}（当前共 {col.count()} 条）"


def update_recipe_budget_tier(recipe_name: str, budget_tier: str) -> bool:
    """更新单道菜的 budget_tier metadata，返回是否成功。"""
    try:
        col = _ensure_collection()
        results = col.get(
            where={"name": {"$eq": recipe_name}},
            include=["metadatas"],
        )
        if not results["ids"]:
            return False
        meta = dict(results["metadatas"][0])
        meta["budget_tier"] = budget_tier
        col.update(ids=[results["ids"][0]], metadatas=[meta])
        return True
    except Exception as e:
        print(f"[chroma] 更新 {recipe_name} budget_tier 失败: {e}")
        return False


def get_all_recipes() -> list[dict]:
    """获取 ChromaDB 里所有菜谱（含 budget_tier）。"""
    col = _ensure_collection()
    results = col.get(include=["metadatas"])
    return [_meta_to_dict(m) for m in results["metadatas"]]


# 导出给 router.py 使用
RECIPE_TOOLS = [search_recipes, get_recipe_by_name, add_recipe]
