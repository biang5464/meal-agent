class TimeoutConfig:
    """统一超时配置，所有工具调用从这里取值。

    超时层级（严格递增）：
        单次外部调用 < Agent Deadline < Graph Deadline <= SSE_IDLE
        60s           < 90/120s       < 200s            <= 210s
    """
    # ── 单次外部依赖超时（与 ToolPolicy 对应，仅参考，不修改 ToolPolicy） ──
    REDIS: float = 0.3    # Redis 正常几十ms，超过300ms说明有问题
    CHROMA: float = 2.0   # ChromaDB 本地向量检索
    MYSQL: float = 3.0    # MySQL 查询
    HTTP: float = 8.0     # 外部 HTTP API（Woolworths/JB Hi-Fi/The Good Guys）
    LLM: float = 60.0     # DeepSeek LLM 单次 generation

    # ── ReAct Agent 总 Deadline ─────────────────────────────────────────────
    # 覆盖整个多轮 ReAct 过程（多次 LLM + Tool 调用），独立于单次 request_timeout
    MEAL_AGENT_DEADLINE: float = 120.0   # 可能包含多轮推理及多个 Tool 调用
    PRICE_AGENT_DEADLINE: float = 90.0   # 通常 1–2 轮价格查询
    UPDATE_AGENT_DEADLINE: float = 90.0  # 用户画像更新及相关推理

    # ── Graph 与 SSE ──────────────────────────────────────────────────────────
    GRAPH: float = 200.0    # 整个 LangGraph 执行总 Deadline（覆盖最长 Agent Deadline + 节点开销）
    SSE_IDLE: float = 210.0 # SSE queue.get() 每次空闲超时；略大于 GRAPH 确保 watchdog 先于 SSE 触发
