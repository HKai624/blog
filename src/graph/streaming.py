"""SSE 流式适配器：将 LangGraph 原生 astream_events 转换为前端 SSE 格式

v1.1 重构：
- 删除手写的 Vision→Document→Rewriter→Router→Retriever→Solver 流水线
- 直接使用 graph.astream_events(version="v2") 获取事件
- on_chain_start/end → 步骤进度事件
- on_chat_model_stream → 逐 token 流式输出
- 新增节点只需在 workflow.py 中注册，此处自动捕获
"""

import json
import logging
import os
from typing import AsyncGenerator, Optional
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.workflow import compile_graph, compile_graph_for_streaming
from src.rag.retriever_engine import knowledge_is_ready
from src.agents.document_reader import read_document_file
from src.config import UPLOAD_DIR

thought_logger = logging.getLogger("agent_thought")

# ── 节点名 → 前端 step 名映射 ──
STEP_NAME_MAP = {
    "vision": "vision",
    "document": "document",
    "rewriter": "rewrite",
    "router": "route",
    "retriever": "retrieve",
    "solver": "solver",
}

# LangGraph 图中的所有节点名（用于过滤事件）
GRAPH_NODES = {
    "vision", "document", "rewriter", "router", "retriever", "solver", "fallback",
}


def _load_display_names() -> dict:
    """加载 UUID 存储名 → 用户上传原始文件名的映射"""
    import json as _json
    names_path = os.path.join(UPLOAD_DIR, "_filenames.json")
    if os.path.exists(names_path):
        try:
            with open(names_path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return {}


async def run_agent_stream(
    query: str,
    session_id: str,
    image_urls: Optional[list] = None,
    document_filenames: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """流式执行 Agent 管线，yield SSE 格式的事件字符串。

    v1.1: 使用 LangGraph 原生 graph.astream_events(version="v2")。
    v1.2: 支持多图片/多文档 — image_urls 和 document_filenames 改为 list。
    """
    # 带 checkpointer 的图：用于读取/写入持久化状态
    checkpoint_graph = compile_graph()
    # 无 checkpointer 的图：用于 astream_events（SqliteSaver 不支持 async）
    stream_graph = compile_graph_for_streaming()

    config = {"configurable": {"thread_id": session_id}}

    # ── 加载历史消息 ──
    snapshot = checkpoint_graph.get_state(config)
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))

    # ── 读取所有文档文件并拼接 ──
    doc_text_parts = []
    doc_filenames = document_filenames or []
    # 加载 UUID → 原始文件名映射，确保回答中显示用户上传时的真实文件名
    display_names = _load_display_names()
    for dfn in doc_filenames:
        doc_path = os.path.join(UPLOAD_DIR, dfn)
        if os.path.exists(doc_path):
            doc_result = read_document_file(doc_path)
            text = doc_result.get("text", "")
            if text:
                display_name = display_names.get(dfn, dfn)
                doc_text_parts.append(f"=== 文档: {display_name} ===\n{text}")
    doc_text = "\n\n".join(doc_text_parts)

    # ── 构建初始状态 ──
    all_image_urls = image_urls or []
    # 解析后的原始文件名列表
    resolved_filenames = [display_names.get(dfn, dfn) for dfn in doc_filenames]
    initial_state = {
        "messages": existing_messages + [HumanMessage(content=query)],
        "query": query,
        "image_url": all_image_urls[0] if all_image_urls else "",
        "image_urls": all_image_urls,
        "document_text": doc_text,
        "document_filename": ", ".join(resolved_filenames) if resolved_filenames else "",
        "knowledge_ready": knowledge_is_ready(),
    }

    emitted_steps = set()
    full_answer_parts = []

    try:
        async for event in stream_graph.astream_events(initial_state, config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            # ── 节点开始 ──
            if kind == "on_chain_start" and name in GRAPH_NODES:
                step_name = STEP_NAME_MAP.get(name, name)
                emitted_steps.add(name)
                yield _sse("step", json.dumps(
                    {"step": step_name, "status": "running"}, ensure_ascii=False
                ))

            # ── 节点完成 ──
            elif kind == "on_chain_end" and name in GRAPH_NODES:
                step_name = STEP_NAME_MAP.get(name, name)
                data = event.get("data", {})
                output = data.get("output", {}) if isinstance(data, dict) else {}

                yield _sse("step", json.dumps(
                    _extract_step_result(name, step_name, output), ensure_ascii=False
                ))

            # ── Token 流式输出（只捕获 solver 节点，过滤 router 的输出） ──
            elif kind == "on_chat_model_stream":
                # v1.1: 通过 metadata.langgraph_node 过滤，
                # router 也会调 LLM 输出 "chat"/"rag"/"tool"，不能让用户看到
                meta = event.get("metadata", {})
                parent_node = meta.get("langgraph_node", "")
                if parent_node != "solver":
                    continue

                chunk = event.get("data", {}).get("chunk", None)
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token_text = chunk.content
                    if isinstance(token_text, str) and token_text:
                        full_answer_parts.append(token_text)
                        yield _sse("token", json.dumps(
                            {"content": token_text}, ensure_ascii=False
                        ))

        # ── 管线完成 ──
        full_answer = "".join(full_answer_parts)

        # 持久化状态（通过带 checkpointer 的图写入）
        if full_answer or len(existing_messages) > 0:
            try:
                checkpoint_graph.update_state(config, {
                    "messages": existing_messages + [
                        HumanMessage(content=query),
                        AIMessage(content=full_answer),
                    ],
                    "final_answer": full_answer,
                    "query": query,
                })
            except Exception:
                pass

        yield _sse("done", json.dumps({
            "answer": full_answer,
            "steps": sorted(emitted_steps),
        }, ensure_ascii=False))

    except Exception as e:
        thought_logger.error(f"astream_events 异常: {e}")
        yield _sse("error", json.dumps({"error": str(e)}, ensure_ascii=False))


def _extract_step_result(node_name: str, step_name: str, output: dict) -> dict:
    """从节点输出中提取前端展示所需的数据"""
    if node_name == "vision":
        img_desc = output.get("image_description", "")
        return {"step": step_name, "status": "done" if img_desc else "skipped",
                "description": (img_desc[:100] + "..." if len(img_desc) > 100 else img_desc)}

    elif node_name == "document":
        inj_doc = output.get("injected_document", "") or output.get("document_text", "")
        return {"step": step_name, "status": "done" if inj_doc else "skipped"}

    elif node_name == "rewriter":
        rewritten = output.get("rewritten_query", "")
        display = rewritten[:200] + ("..." if len(rewritten) > 200 else "")
        return {"step": step_name, "query": display}

    elif node_name == "router":
        decision = output.get("route_decision", "chat")
        return {"step": step_name, "decision": decision}

    elif node_name == "retriever":
        docs = output.get("retrieved_docs", [])
        return {"step": step_name, "count": len(docs)}

    elif node_name == "solver":
        return {"step": step_name, "status": "done"}

    elif node_name == "fallback":
        return {"step": step_name, "status": "done"}

    return {"step": step_name, "status": "done"}


def _sse(event: str, data: str) -> str:
    """构建 SSE 格式的事件字符串"""
    return f"event: {event}\ndata: {data}\n\n"
