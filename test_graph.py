"""
测试 compiled_graph 完整流转，重点验证 UPDATE_AND_MEAL 双节点路径。

运行方式:
    .venv\\Scripts\\python.exe -m pytest test_graph.py -v -s
"""

import pytest

from dotenv import load_dotenv
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def init_services():
    from models.maintenance import HealthCycleORM, PreferenceHistoryORM, ReminderORM  # noqa: F401
    from tools.memory import init_db
    from tools.recipe import init_chroma
    init_db("./data/test_users.db")
    init_chroma("./data/test_chroma")
    yield


def _initial_state(user_input: str, user_id: str = "graph_test_user") -> dict:
    return {
        "user_id": user_id,
        "user_input": user_input,
        "user_brief": {},
        "intent": "",
        "result": "",
        "messages": [],
        "needs_meal": False,
    }


# ── 辅助：记录每个节点执行顺序 ──────────────────────────────────────────────────

class NodeTracer:
    """包装 compiled_graph.ainvoke，通过 stream_mode='updates' 捕获每步 state 变化。"""

    def __init__(self):
        self.steps: list[dict] = []   # 每步 {node: str, keys_changed: list}

    async def run(self, initial_state: dict) -> dict:
        from agents.graph import compiled_graph
        self.steps = []
        final = None
        async for step in compiled_graph.astream(initial_state, stream_mode="updates"):
            # step 是 {node_name: {changed_keys...}} 的 dict
            for node_name, patch in step.items():
                changed = list(patch.keys())
                self.steps.append({"node": node_name, "keys": changed, "patch": patch})
                print(f"  [graph step] {node_name} changed keys: {changed}")
                if "result" in patch and patch["result"]:
                    print(f"    result preview: {patch['result'][:120]!r}")
                if "intent" in patch:
                    print(f"    intent: {patch['intent']!r}  needs_meal: {patch.get('needs_meal')}")
            final = step
        # 最终 state 通过 ainvoke 拿
        return await compiled_graph.ainvoke(initial_state)

    def node_order(self) -> list[str]:
        return [s["node"] for s in self.steps]


# ── Test 1: MEAL 场景 ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_meal_flow():
    """MEAL 场景：memory → supervisor → meal → END"""
    from agents.graph import compiled_graph

    state = _initial_state("帮我推荐今天吃什么", user_id="graph_meal_user")
    tracer = NodeTracer()

    print("\n[graph/MEAL] 开始追踪节点执行顺序...")
    final = await tracer.run(state)

    order = tracer.node_order()
    print(f"[graph/MEAL] 节点顺序: {order}")
    print(f"[graph/MEAL] 最终 intent: {final['intent']!r}")
    print(f"[graph/MEAL] result 前200字: {final['result'][:200]!r}")

    assert "memory" in order, "memory 节点未执行"
    assert "supervisor" in order, "supervisor 节点未执行"
    assert "meal" in order, "meal 节点未执行"
    assert "price" not in order, "MEAL 场景不应触发 price 节点"
    assert "update" not in order, "MEAL 场景不应触发 update 节点"
    assert final["intent"] == "MEAL"
    assert len(final["result"]) > 10
    print("[graph/MEAL] [OK] 路由正确，节点顺序: memory→supervisor→meal")


# ── Test 2: UPDATE_AND_MEAL 场景（核心双节点验证） ────────────────────────────

@pytest.mark.asyncio
async def test_graph_update_and_meal_flow():
    """UPDATE_AND_MEAL 场景：memory → supervisor → update → meal → END"""
    state = _initial_state("今天不想吃香菜，帮我推荐一个菜", user_id="graph_uam_user")
    tracer = NodeTracer()

    print("\n[graph/UPDATE_AND_MEAL] 开始追踪节点执行顺序...")
    final = await tracer.run(state)

    order = tracer.node_order()
    print(f"[graph/UPDATE_AND_MEAL] 节点顺序: {order}")
    print(f"[graph/UPDATE_AND_MEAL] 最终 intent: {final['intent']!r}  needs_meal: {final['needs_meal']}")
    print(f"[graph/UPDATE_AND_MEAL] result 前200字: {final['result'][:200]!r}")

    assert "memory" in order, "memory 节点未执行"
    assert "supervisor" in order, "supervisor 节点未执行"
    assert "update" in order, "update 节点未执行"
    assert "meal" in order, "meal 节点未执行（needs_meal=True 时应继续到 meal）"

    update_idx = order.index("update")
    meal_idx = order.index("meal")
    assert update_idx < meal_idx, f"update 应在 meal 之前执行，实际顺序: {order}"

    assert final["intent"] == "UPDATE_AND_MEAL"
    assert len(final["result"]) > 10
    print(f"[graph/UPDATE_AND_MEAL] [OK] 双节点路由正确: update(idx={update_idx}) → meal(idx={meal_idx})")


# ── Test 3: PRICE 场景 ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_price_flow():
    """PRICE 场景：memory → supervisor → price → END"""
    state = _initial_state("chicken breast 多少钱", user_id="graph_price_user")
    tracer = NodeTracer()

    print("\n[graph/PRICE] 开始追踪节点执行顺序...")
    final = await tracer.run(state)

    order = tracer.node_order()
    print(f"[graph/PRICE] 节点顺序: {order}")
    print(f"[graph/PRICE] 最终 intent: {final['intent']!r}")
    print(f"[graph/PRICE] result 前200字: {final['result'][:200]!r}")

    assert "price" in order, "price 节点未执行"
    assert "meal" not in order, "PRICE 场景不应触发 meal 节点"
    assert final["intent"] == "PRICE"
    assert len(final["result"]) > 10
    print("[graph/PRICE] [OK] 路由正确: memory→supervisor→price")
