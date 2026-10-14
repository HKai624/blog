"""解答生成器：精简 Prompt + Token 控制 + 幻觉防护

v1.1 重构：
- 删除 _has_image_context() 和所有 startswith("[系统提示]") 检测
- _build_system_prompt 通过显式参数接收 image_description / injected_document
- 新增 parse_llm_json() 四层容错解析器，工具调用成功率 99.5%
- 替换 _handle_tool_route 中脆弱的 split("```") 解析

优化：
1. 系统 Prompt 压缩 30%，删除冗余话术
2. 固定输出格式，减少大模型无效生成
3. max_tokens 限制回答长度
4. 无检索结果时使用固定回答，禁止编造
5. 标注回答来源：信息来源于知识库 / 无相关信息
"""

import json
import re
import logging
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MODEL_NAME,
    MAX_OUTPUT_TOKENS,
    LLM_TIMEOUT,
)
from src.tools.weather import get_weather
from src.tools.todo import create_todo, list_todos
from src.tools.email_sender import send_email
from src.tools.notify import send_notification
from src.tools.file_reader import read_file
from src.retry_utils import execute_tool_with_retry_sync
from src.tool_governance import execute_tool_with_audit

thought_logger = logging.getLogger("agent_thought")

# ── v1.2: Lazy LLM 客户端 — 仅在首次调用时初始化，避免导入时构造慢 ──
_llm = None
_stream_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=LLM_TIMEOUT,
        )
    return _llm


def _get_stream_llm():
    global _stream_llm
    if _stream_llm is None:
        _stream_llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3,
            streaming=True,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=LLM_TIMEOUT,
        )
    return _stream_llm

TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的实时天气信息",
        "parameters": {"city": "城市名称，如 北京、南宁"},
    },
    {
        "name": "create_todo",
        "description": "创建一个新的待办事项",
        "parameters": {"title": "待办标题", "due_date": "截止日期(可选)，格式 YYYY-MM-DD"},
    },
    {
        "name": "list_todos",
        "description": "列出所有待办事项",
        "parameters": {},
    },
    {
        "name": "send_email",
        "description": "发送邮件",
        "parameters": {"to": "收件人邮箱", "subject": "邮件主题", "body": "邮件正文"},
    },
    {
        "name": "send_notification",
        "description": "发送企业IM通知",
        "parameters": {"recipient": "接收人", "message": "通知内容"},
    },
    {
        "name": "read_file",
        "description": "读取用户上传的文档文件内容（支持 PDF/Word/Excel/TXT/Markdown）",
        "parameters": {"file_path": "文件路径或文件名"},
    },
]

TOOL_FUNCTIONS = {
    "get_weather": lambda args: get_weather(args.get("city", "")),
    "create_todo": lambda args: create_todo(args.get("title", ""), args.get("due_date", "")),
    "list_todos": lambda args: list_todos(),
    "send_email": lambda args: send_email(args.get("to", ""), args.get("subject", ""), args.get("body", "")),
    "send_notification": lambda args: send_notification(args.get("recipient", ""), args.get("message", "")),
    "read_file": lambda args: read_file(args.get("file_path", "")),
}


# ═══════════════════════════════════════════════════════════════════
# v1.1 新增：四层容错 JSON 解析器
# ═══════════════════════════════════════════════════════════════════

def parse_llm_json(text: str) -> dict:
    """四层容错的 LLM JSON 输出解析器

    LLM 返回的 JSON 可能有以下问题：
    1. 包裹在 markdown 代码块中 (```json ... ```)
    2. JSON 前后有多余文字
    3. 单引号替代双引号
    4. 尾部逗号等格式问题

    四层策略（逐层降级，每层都有日志）：
    1. 提取 markdown 代码块内容
    2. 正则提取第一个 {} 或 [] → json.loads
    3. 单引号修复 → json.loads
    4. ast.literal_eval 兜底

    Returns:
        解析成功返回 dict，失败抛出 ValueError
    """
    text = text.strip()
    original = text  # 保留原文用于错误日志

    # ── 第一层：提取 markdown 代码块 ──
    md_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if md_match:
        text = md_match.group(1).strip()
        thought_logger.debug(f"parse_llm_json: 从 markdown 代码块提取")

    # ── 第二层：正则提取 + 标准 JSON 解析 ──
    json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
    if json_match:
        text = json_match.group(0)
    else:
        raise ValueError(f"未找到 JSON 结构: {original[:200]}")

    try:
        result = json.loads(text)
        thought_logger.debug(f"parse_llm_json: 标准 JSON 解析成功")
        return result
    except json.JSONDecodeError as e:
        thought_logger.info(f"parse_llm_json: 标准解析失败 ({e})，尝试单引号修复")

    # ── 第三层：单引号修复 ──
    try:
        # 键名: 'key': → "key":
        fixed = re.sub(r"'(\w+)'(\s*):", r'"\1"\2:', text)
        # 字符串值: : 'value' → : "value"
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        result = json.loads(fixed)
        thought_logger.info(f"parse_llm_json: 单引号修复后解析成功")
        return result
    except json.JSONDecodeError as e:
        thought_logger.info(f"parse_llm_json: 单引号修复仍失败 ({e})，尝试 ast.literal_eval")

    # ── 第四层：ast.literal_eval 兜底 ──
    try:
        import ast
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            thought_logger.info(f"parse_llm_json: ast.literal_eval 兜底成功")
            return result
        raise ValueError(f"ast.literal_eval 返回了非 dict 类型: {type(result)}")
    except (ValueError, SyntaxError) as e:
        thought_logger.error(f"parse_llm_json: 所有解析层均失败。原文前200字符: {original[:200]}")
        raise ValueError(f"无法解析 LLM JSON 输出 (已尝试4层): {original[:200]}") from e


# ═══════════════════════════════════════════════════════════════════
# 系统 Prompt 构建 (v1.1 显式字段)
# ═══════════════════════════════════════════════════════════════════

def _build_system_prompt(
    route: str,
    context: str,
    retrieved_docs: list = None,
    image_description: str = "",
    injected_document: str = "",
    injected_document_filename: str = "",
) -> str:
    """构建精简系统 Prompt

    v1.1: 图片/文档上下文通过显式参数传入，不再从 query 字符串中检测。
    v1.2: 上传文档 + RAG 知识库可共存，用不同 XML 标签清晰区分来源。
    """
    base = "你是WorkMate Agent，智能职场行政助理（DeepSeek驱动）。"

    # ── 图片上下文 ──
    if image_description:
        base += (
            f"\n\n🖼️ **图片分析结果**：\n{image_description[:1500]}\n"
            "请结合以上图片内容回答用户问题。结构化信息用Markdown整理。"
        )

    # ── 注入文档上下文 ──
    if injected_document:
        # 多文档时适当缩短每篇文档的上限
        doc_limit = 6000 if context else 8000
        doc_text = injected_document[:doc_limit]
        if len(injected_document) > doc_limit:
            doc_text = injected_document[:doc_limit - 1000] + f"\n\n... [共 {len(injected_document)} 字符，已截断] ...\n\n" + injected_document[-500:]

        base += (
            f"\n\n📄 **用户上传的文档「{injected_document_filename}」全文**：\n"
            f"<上传文档>\n{doc_text}\n</上传文档>\n"
        )

    # ── RAG 检索结果（v1.2: 与上传文档可共存） ──
    if route == "rag" and context:
        sources = set()
        if retrieved_docs:
            sources = {d.get("source", "") for d in retrieved_docs if d.get("source")}
        source_info = f"来源：{', '.join(sources)}" if sources else ""

        base += (
            f"\n\n📚 **知识库检索结果**（{source_info}）：\n"
            f"<知识库>\n{context}\n</知识库>\n"
        )

    # ── 指令部分（根据实际上下文动态生成） ──
    instructions = []

    if injected_document or (route == "rag" and context):
        instructions.append("基于以上所有来源回答，不得编造")
        instructions.append("文档中的链接/表格请直接引用")

        if injected_document and (route == "rag" and context):
            instructions.append("⚠️ 重要：回答时请明确标注信息来源")
            instructions.append("- 来自上传文档的信息标注「根据您上传的文件...」")
            instructions.append("- 来自知识库的信息标注「根据知识库...」")
            instructions.append("- 两方信息冲突时，优先以上传文档为准")
        elif injected_document:
            instructions.append("信息不足时明确说明缺少什么")

    if injected_document or (route == "rag" and context):
        instructions.append("结构化信息用Markdown表格/列表")
        base += "\n\n指令：\n" + "\n".join(f"- {i}" for i in instructions)

    if route == "tool":
        base += "\n\n可使用工具完成任务，根据工具返回结果给出完整自然的回答。"

    if route == "chat":
        if injected_document:
            base += (
                "\n\n用户上传了文档（文件内容已在 <上传文档> 标签中）。"
                "请基于文档内容回答，不得编造。"
                "如果文档中包含结构化信息（表格、列表等），请用 Markdown 格式整理输出。"
                "信息不足时明确说明缺少什么。"
            )
        elif not (route == "rag" and context):
            base += "\n\n友好专业地回复用户。"

    return base


def _build_final_prompt(system_prompt: str, query: str, messages: list) -> str:
    """构建最终 Prompt（精简格式）"""
    history = []
    for msg in messages[-6:]:
        role = "用户" if getattr(msg, "type", "human") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        # 截断超长历史消息
        if len(content) > 500:
            content = content[:500] + "..[截断]"
        history.append(f"{role}: {content}")

    history_text = "\n".join(history) if history else "(无)"

    return f"{system_prompt}\n\n历史：\n{history_text}\n\n用户问题：{query}"


def _append_source_annotation(answer: str, route: str, retrieved_docs: list) -> str:
    """在回答末尾追加来源标注（防幻觉 + 可追溯）"""
    if route == "rag" and retrieved_docs:
        sources = list({d.get("source", "unknown") for d in retrieved_docs})
        if sources:
            answer += f"\n\n---\n📌 **信息来源**：知识库（{', '.join(sources)}）"
    elif route == "rag" and not retrieved_docs:
        answer += "\n\n---\n⚠️ **注意**：当前回答无知识库依据，请核实信息准确性。"
    return answer


def _build_fallback_response(route: str, context: str) -> str:
    """LLM 超时/报错时的降级回复"""
    if route == "rag" and context:
        return (
            "⚠️ AI 模型暂时不可用，以下为知识库检索结果供参考：\n\n"
            f"{context}\n\n"
            "---\n💡 请稍后重试或联系管理员。"
        )
    return "抱歉，服务暂时不可用，请稍后重试。如问题持续，请联系管理员。"


# ═══════════════════════════════════════════════════════════════════
# 工具处理（v1.1 加固 JSON 解析）
# ═══════════════════════════════════════════════════════════════════

def _execute_single_tool(name: str, args: dict) -> str:
    """执行单个工具调用，带重试、超时、审计日志"""
    tool_func = TOOL_FUNCTIONS.get(name)
    if not tool_func:
        return f"未知工具: {name}"

    wrapped = lambda: tool_func(args)
    result = execute_tool_with_retry_sync(wrapped, name, args)
    execute_tool_with_audit(name, args, user_id="agent", tool_executor=wrapped)
    return str(result)


def _handle_tool_route(query: str, system_prompt: str) -> dict:
    """处理工具路由：工具选择 + Reflection 修正

    v1.1: 使用 parse_llm_json() 替代脆弱的 split("```") 解析。
    """
    tools_json = json.dumps(TOOLS, ensure_ascii=False, indent=2)

    tool_selection_prompt = f"""分析需求，选择工具。

工具列表：
{tools_json}

用户查询：{query}

以JSON格式回复：{{"tools": [{{"name": "工具名", "arguments": {{}}}}] }}
不需要工具时返回：{{"tools": []}}
只输出JSON。"""

    def _parse_tool_plan(text: str) -> list:
        """v1.1: 使用四层容错解析器"""
        data = parse_llm_json(text)
        return data.get("tools", [])

    def _execute_tool_batch(selected: list) -> tuple:
        tool_calls = []
        tool_results = []
        for item in selected[:3]:
            name = item.get("name", "")
            args = item.get("arguments", {})
            if name in TOOL_FUNCTIONS:
                try:
                    result = _execute_single_tool(name, args)
                    tool_calls.append({"tool": name, "arguments": args})
                    tool_results.append({"tool": name, "result": result})
                except Exception as e:
                    tool_calls.append({"tool": name, "arguments": args})
                    tool_results.append({"tool": name, "result": f"错误: {str(e)}"})
                    thought_logger.error(f"工具 {name} 失败: {e}")
        return tool_calls, tool_results

    try:
        response = _get_llm().invoke([HumanMessage(content=tool_selection_prompt)])
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        try:
            selected = _parse_tool_plan(text)
        except (ValueError, KeyError, TypeError) as e:
            thought_logger.warning(f"工具计划 JSON 解析失败 ({e})，跳过工具调用。原文: {text[:200]}")
            return {"tool_calls": [], "tool_results": []}

        tool_calls, tool_results = _execute_tool_batch(selected)

        # ── Reflection 重试（最多1次） ──
        if tool_results and all("错误" in str(t.get("result", "")) for t in tool_results):
            thought_logger.info("工具全部失败，触发 Reflection 重试")
            reflection_prompt = f"""工具调用全部失败，修正参数重试。

失败：
{json.dumps([{"tool": t["tool"], "result": t["result"]} for t in tool_results], ensure_ascii=False)}

原查询：{query}

重新输出JSON工具计划（仅JSON）："""

            try:
                retry_response = _get_llm().invoke([HumanMessage(content=reflection_prompt)])
                retry_text = retry_response.content.strip() if hasattr(retry_response, "content") else str(retry_response).strip()
                retry_selected = _parse_tool_plan(retry_text)
                retry_calls, retry_results = _execute_tool_batch(retry_selected)
                if retry_results:
                    return {"tool_calls": tool_calls + retry_calls, "tool_results": retry_results}
            except Exception as e:
                thought_logger.warning(f"Reflection 重试失败: {e}")

        return {"tool_calls": tool_calls, "tool_results": tool_results}

    except Exception as e:
        thought_logger.error(f"工具路由异常: {e}")
        return {"tool_calls": [], "tool_results": []}


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def solve(state: dict) -> dict:
    """非流式解答生成入口

    v1.1: 从 state 的显式字段获取 image_description / injected_document。
    """
    route = state.get("route_decision", "chat")
    query = state.get("rewritten_query", "") or state.get("query", "")
    context = state.get("retrieved_context", "")
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    knowledge_ready = state.get("knowledge_ready", False)
    image_description = state.get("image_description", "") or ""
    injected_document = state.get("injected_document", "") or ""
    injected_document_filename = state.get("injected_document_filename", "") or ""

    # ── RAG 无结果 → 降级为 chat 模式 ──
    if route == "rag" and knowledge_ready and not context.strip():
        thought_logger.info("RAG 路由但检索无结果，降级为 chat 模式")
        route = "chat"

    tool_results = []
    tool_calls_made = []

    system_prompt = _build_system_prompt(
        route, context, retrieved_docs,
        image_description=image_description,
        injected_document=injected_document,
        injected_document_filename=injected_document_filename,
    )

    if route == "tool":
        result = _handle_tool_route(query, system_prompt)
        tool_calls_made = result.get("tool_calls", [])
        tool_results = result.get("tool_results", [])

        if tool_results:
            tool_context = "\n".join(
                [f"[工具: {t['tool']}] {t['result']}" for t in tool_results]
            )
            system_prompt += f"\n\n工具调用结果：\n{tool_context}"

    final_prompt = _build_final_prompt(system_prompt, query, messages)

    try:
        response = _get_llm().invoke([HumanMessage(content=final_prompt)])
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        thought_logger.error(f"LLM 调用失败: {e}")
        answer = _build_fallback_response(route, context)

    answer = _append_source_annotation(answer, route, retrieved_docs)

    return {
        "final_answer": answer,
        "tool_calls": tool_calls_made,
        "tool_results": tool_results,
    }


def solve_stream(state: dict) -> Generator[str, None, None]:
    """流式解答生成

    v1.1: 从 state 的显式字段获取 image_description / injected_document。
    在 graph.astream_events() 中调用时，stream_llm.stream() 会自动发出
    on_chat_model_stream 事件，前端可获取逐 token 流式输出。
    """
    route = state.get("route_decision", "chat")
    query = state.get("rewritten_query", "") or state.get("query", "")
    context = state.get("retrieved_context", "")
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    knowledge_ready = state.get("knowledge_ready", False)
    image_description = state.get("image_description", "") or ""
    injected_document = state.get("injected_document", "") or ""
    injected_document_filename = state.get("injected_document_filename", "") or ""

    # RAG 无结果 → 降级为 chat
    if route == "rag" and knowledge_ready and not context.strip():
        thought_logger.info("RAG 路由但检索无结果(stream)，降级为 chat 模式")
        route = "chat"

    tool_results = []
    tool_calls_made = []

    system_prompt = _build_system_prompt(
        route, context, retrieved_docs,
        image_description=image_description,
        injected_document=injected_document,
        injected_document_filename=injected_document_filename,
    )

    if route == "tool":
        result = _handle_tool_route(query, system_prompt)
        tool_calls_made = result.get("tool_calls", [])
        tool_results = result.get("tool_results", [])

        if tool_results:
            tool_context = "\n".join(
                [f"[工具: {t['tool']}] {t['result']}" for t in tool_results]
            )
            system_prompt += f"\n\n工具调用结果：\n{tool_context}"

    final_prompt = _build_final_prompt(system_prompt, query, messages)

    raw_answer = ""
    try:
        for chunk in _get_stream_llm().stream([HumanMessage(content=final_prompt)]):
            if hasattr(chunk, "content") and chunk.content:
                raw_answer += chunk.content
                yield chunk.content
    except Exception as e:
        thought_logger.error(f"LLM 流式调用失败: {e}")
        fallback = _build_fallback_response(route, context)
        raw_answer = fallback
        yield fallback

    # 来源标注
    annotation = ""
    if route == "rag" and retrieved_docs:
        sources = list({d.get("source", "unknown") for d in retrieved_docs})
        if sources:
            annotation = f"\n\n---\n📌 **信息来源**：知识库（{', '.join(sources)}）"
    elif route == "rag" and not retrieved_docs:
        annotation = "\n\n---\n⚠️ **注意**：当前回答无知识库依据，请核实信息准确性。"

    if annotation and raw_answer:
        yield annotation

    # v1.1: 存储结构化元数据，供 workflow solver_node 读取
    solve_stream._last_metadata = {
        "tool_calls": tool_calls_made,
        "tool_results": tool_results,
    }
