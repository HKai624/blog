"""查询重写器：多轮对话意图提取

v1.1 重构：
- 删除所有 startswith("[系统提示]") 检测逻辑
- 图片/文档上下文通过 state 独立字段传递，rewriter 只关注 query 本身
- 简化重写逻辑：单轮直接透传，多轮用历史重写
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.1,
)


def rewrite_query(state: dict, current_query: str = "") -> dict:
    """根据对话历史重写查询为独立、清晰的单句

    v1.1: 不再检测 query 中的 [系统提示] 前缀。
    query 始终是用户原始输入，图片/文档上下文由 solver 从独立字段读取。
    """
    messages = state.get("messages", [])
    if not messages:
        return {"rewritten_query": current_query or state.get("query", "")}

    last_msg = messages[-1]
    user_query = current_query or (last_msg.content if hasattr(last_msg, "content") else str(last_msg))

    # 单轮对话：直接透传
    if len(messages) <= 1:
        return {"rewritten_query": user_query}

    # 多轮对话：用历史重写
    history = []
    for msg in messages[:-1]:
        role = "用户" if getattr(msg, "type", "human") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        history.append(f"{role}: {content}")

    context = "\n".join(history[-6:])

    prompt = f"""你是一个查询重写助手。根据对话历史和用户的最新问题，重写一个完整的、清晰的、独立的查询。
只输出重写后的查询，不要添加任何解释。

对话历史：
{context}

用户最新问题：{user_query}

重写后的查询："""

    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
    return {"rewritten_query": rewritten}
