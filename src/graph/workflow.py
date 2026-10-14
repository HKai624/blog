"""LangGraph 工作流：Agent 管线 + 全局异常处理

节点流程：
vision → document → rewriter → router → [retriever → solver] or [solver]

增强能力：
1. 全局异常捕获，节点异常不中断服务
2. 工具调用失败自动重试（最多2次）+ 友好降级
3. 大模型超时/报错降级：直接返回检索结果
4. 未知意图兜底节点，拒绝危险操作
5. 异常信息日志打印

v1.1 重构：
- AgentState 新增 injected_document / image_description 显式字段，
  删除 [系统提示] 字符串 hack，节点间契约改为 TypedDict 字段
"""

from typing import TypedDict, Annotated, Sequence, Optional
import operator
import functools
import traceback
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from src.agents.query_rewriter import rewrite_query
from src.agents.router import route_query
from src.agents.retriever import retrieve_docs
from src.agents.solver import solve, solve_stream
from src.rag.retriever_engine import knowledge_is_ready
from src.memory.checkpoint_store import get_checkpointer
from src.vision_adapter import handle_message_with_image
from src.agents.document_reader import document_node as _document_node

thought_logger = logging.getLogger("agent_thought")
error_logger = logging.getLogger("agent_error")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    image_url: Optional[str]
    image_urls: Optional[list]              # v1.2: 多图片支持
    image_description: Optional[str]       # 视觉节点输出的图片描述文本
    document_text: Optional[str]            # [兼容保留] 旧字段，新代码请用 injected_document
    document_filename: Optional[str]
    # ── v1.1: 显式文档/图片注入字段（替代 [系统提示] 字符串 hack） ──
    injected_document: Optional[str]        # 文档阅读器注入的文档全文
    injected_document_filename: Optional[str]  # 注入的文档文件名
    rewritten_query: str
    route_decision: str
    retrieved_docs: list
    retrieved_context: str
    final_answer: str
    tool_calls: list
    tool_results: list
    knowledge_ready: bool
    # ── 异常追踪字段 ──
    errors: list              # 累积的异常信息
    fallback_reason: str       # 降级原因


# ── 危险意图黑名单 ───────────────────────────────────────────────
DANGEROUS_PATTERNS = [
    "删除所有", "rm -rf", "DROP TABLE", "shutdown", "格式化",
    "执行命令", "sudo", "系统命令", "提权", "越权",
    "密码", "token", "key", "secret", "API密钥",
    "伪造", "绕过", "hack", "exploit", "注入",
]

# ── 节点异常包装器 ────────────────────────────────────────────────
def _safe_node(node_fn, node_name: str, fallback_value: dict = None):
    """全局异常捕获装饰器：包装每个 LangGraph 节点

    - 异常不向上传播，返回 fallback_value 兜底
    - 完整记录异常堆栈到 agent_error 日志
    - 在 state["errors"] 中累积错误信息

    Args:
        node_fn: 原始节点函数
        node_name: 节点名称（用于日志）
        fallback_value: 异常时的兜底返回值
    """
    @functools.wraps(node_fn)
    def wrapper(state: AgentState) -> dict:
        try:
            return node_fn(state)
        except Exception as e:
            error_msg = f"[{node_name}] 节点异常: {e}"
            error_logger.error(f"{error_msg}\n{traceback.format_exc()}")
            thought_logger.warning(error_msg)

            # 累积错误信息
            errors = state.get("errors", []) or []
            errors.append({
                "node": node_name,
                "error": str(e),
                "type": type(e).__name__,
            })

            result = dict(fallback_value or {})
            result["errors"] = errors
            return result

    return wrapper


def vision_node(state: AgentState) -> dict:
    """视觉节点：调用智谱 GLM-4V 分析图片，将描述存入 image_description 字段

    v1.1: image_description 独立存储，不再拼入 query 字符串。
    后续节点通过 state["image_description"] 获取图片内容。

    v1.2: 支持多图片 — image_urls 列表中的每张图片独立分析后拼接。
    """
    image_url = state.get("image_url", "") or ""
    image_urls = state.get("image_urls", []) or []

    # 合并单图片和多图片字段
    all_images = list(image_urls)
    if image_url and image_url not in all_images:
        all_images.insert(0, image_url)

    if not all_images:
        thought_logger.info("vision_node: 无图片，跳过视觉分析")
        return {"image_description": ""}

    query = state.get("query", "")
    thought_logger.info(f"vision_node: 检测到 {len(all_images)} 张图片，调用智谱视觉分析")

    descriptions = []
    for i, img_url in enumerate(all_images):
        try:
            desc = handle_message_with_image(query, img_url)
            if len(all_images) > 1:
                descriptions.append(f"【图片 {i + 1}】\n{desc}")
            else:
                descriptions.append(desc)
        except Exception as e:
            thought_logger.error(f"vision_node: 图片 {i + 1} 分析失败: {e}")
            descriptions.append(f"【图片 {i + 1}】[识别失败: {str(e)[:100]}]")

    combined = "\n\n---\n\n".join(descriptions) if descriptions else ""

    return {
        "query": combined if combined else query,
        "image_description": combined,
    }


def rewriter_node(state: AgentState) -> dict:
    state["knowledge_ready"] = knowledge_is_ready()
    enhanced_query = state.get("query", "")
    result = rewrite_query(dict(state), current_query=enhanced_query)
    return {"rewritten_query": result.get("rewritten_query", "")}


def router_node(state: AgentState) -> dict:
    query = state.get("rewritten_query", "") or state.get("query", "")

    # ── 危险意图检测 ──
    query_lower = query.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in query_lower:
            thought_logger.warning(f"router_node: 检测到危险意图 → '{pattern}' → 重定向到 fallback")
            return {
                "route_decision": "fallback",
                "fallback_reason": f"检测到危险关键词: {pattern}",
            }

    result = route_query(dict(state))
    return {"route_decision": result.get("route_decision", "chat")}


def retriever_node(state: AgentState) -> dict:
    """检索节点：带超时保护"""
    result = retrieve_docs(dict(state))
    return {
        "retrieved_docs": result.get("retrieved_docs", []),
        "retrieved_context": result.get("retrieved_context", ""),
    }


def solver_node(state: AgentState) -> dict:
    """解答节点：使用 solve_stream 以支持 graph.astream_events() 逐 token 输出

    v1.1: solve_stream 内部的 stream_llm.stream() 会自动触发
    LangGraph 的 on_chat_model_stream 事件，前端可获得实时逐字输出。
    solve_stream._last_metadata 存储工具调用等结构化数据。
    """
    try:
        raw_answer = ""
        for token in solve_stream(dict(state)):
            raw_answer += token
        metadata = getattr(solve_stream, '_last_metadata', {})
    except Exception as e:
        error_logger.error(f"solver_node: LLM 调用异常: {e}\n{traceback.format_exc()}")
        # 降级：如果 RAG 路由且有检索结果，直接返回检索内容
        route = state.get("route_decision", "chat")
        context = state.get("retrieved_context", "")
        if route == "rag" and context:
            return {
                "final_answer": (
                    "⚠️ AI 模型暂时不可用，以下为知识库直接检索结果：\n\n"
                    f"{context}\n\n"
                    "---\n💡 请稍后重试。"
                ),
                "tool_calls": [],
                "tool_results": [],
                "messages": [AIMessage(content=context)],
            }
        return {
            "final_answer": "抱歉，处理您的请求时遇到错误，请稍后重试。",
            "tool_calls": [],
            "tool_results": [],
            "messages": [AIMessage(content="抱歉，处理您的请求时遇到错误，请稍后重试。")],
        }

    answer = raw_answer.strip()
    return {
        "final_answer": answer,
        "tool_calls": metadata.get("tool_calls", []),
        "tool_results": metadata.get("tool_results", []),
        "messages": [AIMessage(content=answer)],
    }


def fallback_node(state: AgentState) -> dict:
    """未知意图兜底节点：拒绝执行危险操作，返回安全提示"""
    reason = state.get("fallback_reason", "未知意图")
    thought_logger.info(f"fallback_node: 进入兜底节点，原因: {reason}")

    return {
        "final_answer": (
            "⚠️ **安全提示**：无法处理该请求。\n\n"
            f"原因：{reason}\n\n"
            "WorkMate Agent 仅支持以下操作：\n"
            "- 📚 查询企业内部知识库（制度、流程、FAQ）\n"
            "- 🌤️ 查询天气\n"
            "- ✅ 管理待办事项\n"
            "- 📧 发送邮件通知\n"
            "- 📄 读取上传的文档文件\n\n"
            "如需其他帮助，请联系管理员。"
        ),
        "route_decision": "fallback",
        "messages": [AIMessage(content="无法处理该请求，请参考上述支持的操作范围。")],
    }


def route_after_router(state: AgentState) -> str:
    """路由器后的条件分支

    v1.2: 上传文档后仍走 RAG 检索知识库，两个来源共存。
    solver 负责在 System Prompt 中清晰区分「上传文档」与「知识库」。
    """
    decision = state.get("route_decision", "chat")

    # 危险意图 → 兜底
    if decision == "fallback":
        return "fallback"

    if decision == "rag" and state.get("knowledge_ready", False):
        has_injected = bool(state.get("injected_document", "")) or bool(state.get("document_text", ""))
        if has_injected:
            thought_logger.info("route_after_router: 文档已注入 + RAG 知识库 → 双源检索")
        return "retriever"
    return "solver"


def document_graph_node(state: AgentState) -> dict:
    """Graph 包装器：调用 document_reader 模块的 document_node

    v1.1: 文档内容通过 injected_document 字段传递，不再拼入 query 字符串。
    """
    result = _document_node(dict(state))
    # 确保 injected_document 独立于 query 传递
    if result.get("injected_document") and not result.get("document_text"):
        result["document_text"] = result["injected_document"]  # 兼容旧代码
    return result


def build_agent_graph():
    """构建 Agent 管线图（含全局异常处理节点）"""
    builder = StateGraph(AgentState)

    # ── 注册节点（用安全包装器） ──
    builder.add_node("vision", _safe_node(vision_node, "vision", {"image_description": ""}))
    builder.add_node("document", _safe_node(document_graph_node, "document", {"query": ""}))
    builder.add_node("rewriter", _safe_node(rewriter_node, "rewriter", {"rewritten_query": ""}))
    builder.add_node("router", _safe_node(router_node, "router", {"route_decision": "chat"}))
    builder.add_node("retriever", _safe_node(retriever_node, "retriever", {"retrieved_docs": [], "retrieved_context": ""}))
    builder.add_node("solver", _safe_node(solver_node, "solver", {"final_answer": "服务异常，请稍后重试。"}))
    builder.add_node("fallback", fallback_node)

    # ── 流程：vision → document → rewriter → router → [retriever → solver] or [fallback] or [solver] ──
    builder.set_entry_point("vision")
    builder.add_edge("vision", "document")
    builder.add_edge("document", "rewriter")
    builder.add_edge("rewriter", "router")
    builder.add_conditional_edges("router", route_after_router, {
        "retriever": "retriever",
        "solver": "solver",
        "fallback": "fallback",
    })
    builder.add_edge("retriever", "solver")
    builder.add_edge("solver", END)
    builder.add_edge("fallback", END)

    return builder


def compile_graph():
    """编译 LangGraph 工作流（带持久化），用于同步 invoke 调用"""
    builder = build_agent_graph()
    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


def compile_graph_for_streaming():
    """编译 LangGraph 工作流（无 checkpointer），用于 astream_events 流式调用

    SqliteSaver 不支持 async 方法，因此 astream_events 需要一个精简版图。
    状态由调用方通过 initial_state 手动传入，完成后通过 update_state 写回。
    """
    builder = build_agent_graph()
    return builder.compile()
