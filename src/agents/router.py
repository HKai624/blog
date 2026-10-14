"""智能路由器：分析用户意图，分派到 RAG / Tool / Chat 节点

路由策略（优化版）：
1. 显式工具请求（天气、待办、邮件、通知） → tool
2. 知识库就绪 + 信息类问题 → rag（默认优先 RAG，不再限制特定关键词）
3. 简单问候 / 闲聊 → chat
4. 知识库未就绪 + 非工具请求 → chat

核心理念：只要知识库有内容，任何信息类问题都先尝试 RAG 检索，
不再要求用户必须说出具体文件名或制度关键词。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

# ── v1.2: Lazy LLM 客户端 — 避免 import 时连接失败导致整个模块崩溃 ──
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
        )
    return _llm


def route_query(state: dict) -> dict:
    query = state.get("rewritten_query", "") or state.get("query", "")
    knowledge_ready = state.get("knowledge_ready", False)

    available_routes = ["chat"]
    route_descriptions = ["直接回答：通用对话、问候、闲聊等"]

    if knowledge_ready:
        available_routes.append("rag")
        route_descriptions.append("rag：查询知识库中的文档内容（公司制度、流程、FAQ、上传的各类文件等）")

    available_routes.append("tool")
    route_descriptions.append("tool：查天气、创建待办、发邮件、发通知等需要调用外部工具的操作")

    routes_text = "\n".join([f"- {d}" for d in route_descriptions])
    routes_list = ", ".join(available_routes)

    prompt = f"""你是一个智能路由器。分析用户查询，将其分类到以下类别之一。

可用类别：
{routes_text}

分类规则：
- 如果用户需要查天气、创建待办、发送邮件、发送通知等具体工具操作，选择 tool
- 如果用户是简单打招呼、感谢、告别等纯社交对话，选择 chat
- 其他所有信息类问题（查询文档内容、询问制度流程、文件内容总结、阅读材料等），{"优先选择 rag" if knowledge_ready else "只能选择 chat"}
- 注意：用户不需要说出具体文件名，只要是在询问文档/知识/信息相关的内容，就应该走 rag

用户查询：{query}

请只输出一个类别名称（{routes_list}），不要输出其他内容。"""

    response = _get_llm().invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()

    if decision not in ["rag", "tool", "chat"]:
        if "rag" in decision or "文档" in decision or "知识" in decision or "制度" in decision or "文件" in decision:
            decision = "rag"
        elif "tool" in decision or "工具" in decision or "天气" in decision or "待办" in decision or "邮件" in decision or "通知" in decision:
            decision = "tool"
        else:
            decision = "chat"

    return {"route_decision": decision}
