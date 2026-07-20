"""共享状态结构：所有 Agent 节点读写同一个 AgentState 实例。"""

from typing import TypedDict


class AgentState(TypedDict):
    user_id: str
    user_input: str
    user_brief: dict          # memory_agent 填充
    intent: str               # supervisor_agent 填充
    result: str               # 末端 Agent 填充
    messages: list            # 多轮对话历史（LangChain Message 格式）
    needs_meal: bool          # UPDATE_AND_MEAL 场景：update 完成后继续走 meal
    chat_history: list[dict]  # Redis/MySQL 持久化的对话历史，graph 入口处注入
    conversation_memory: str  # memory_agent 检索的历史摘要，supervisor 用于上下文
    confidence: str           # HIGH / MEDIUM / LOW，supervisor 填充
    missing_slots: list       # 缺失槽位列表，supervisor 填充
    top2: str                 # 第二可能意图，supervisor 填充
    last_turn: dict           # 上一轮对话的 user_input / intent / result_preview
    session_context: dict     # Context Manager 维护的会话状态
    context_candidate: dict   # 阶段一只读候选（不写 Redis）
    resolved_entity: str      # 恢复后的真实实体，供 supervisor 使用
    resolved_input: str       # 消歧后的输入，供专用 agent 使用
    turn_excluded: list       # 本轮新提取的排除项
    turn_constraints: dict    # 本轮新提取的约束
    context_status: str       # active / pending
