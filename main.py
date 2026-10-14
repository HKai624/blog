import os
import sys
import uuid
import json
import asyncio
import time

from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.graph.workflow import compile_graph
from src.graph.streaming import run_agent_stream

# v1.2: RAG 模块延迟导入（chromadb import 慢，避免阻塞 FastAPI 启动）
_rag_imports = {}


def _lazy_import(name: str):
    """延迟导入 RAG 模块，首次调用时才触发 chromadb import"""
    if name not in _rag_imports:
        if name == "knowledge_is_ready":
            from src.rag.retriever_engine import knowledge_is_ready as _fn
            _rag_imports[name] = _fn
        elif name == "build_index":
            from src.rag.index_builder import build_index as _fn
            _rag_imports[name] = _fn
        elif name == "add_to_index":
            from src.rag.index_builder import add_to_index as _fn
            _rag_imports[name] = _fn
        elif name == "remove_from_index":
            from src.rag.index_builder import remove_from_index as _fn
            _rag_imports[name] = _fn
        elif name == "load_document":
            from src.rag.document_loader import load_document as _fn
            _rag_imports[name] = _fn
    return _rag_imports[name]
from src.memory.checkpoint_store import session_store
from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MODEL_NAME,
    FEISHU_APP_ID,
    FEISHU_APP_SECRET,
    SERVER_HOST,
    SERVER_PORT,
    RATE_LIMIT_PER_USER,
    PUBLIC_BASE_URL,
    UPLOAD_DIR,
)
from src.feishu.ws_client import FeishuWSClient
from src.logging_setup import setup_logging
from src.session_manager import get_user_session, USER_SESSIONS, cleanup_expired_sessions
from src.metrics import metrics
from src.thought_recorder import ThoughtRecorder
from src.retry_utils import call_llm_with_fallback
from src.pre_launch import run_pre_launch_checks
import httpx
import logging

load_dotenv()

# ── 日志初始化 ────────────────────────────────────────────────
setup_logging()

app = FastAPI(title="WorkMate Agent", version="1.0")

# ── 静态文件服务（图片上传目录）────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ── 上传文件原始名称映射（UUID 存储名 → 用户上传时的真实文件名）──
_UPLOAD_NAMES_PATH = os.path.join(UPLOAD_DIR, "_filenames.json")


def _load_upload_names() -> dict:
    """加载上传文件的 UUID → 原始名称映射"""
    if os.path.exists(_UPLOAD_NAMES_PATH):
        try:
            with open(_UPLOAD_NAMES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_upload_names(mapping: dict):
    """持久化上传文件名称映射"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(_UPLOAD_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)


def _resolve_upload_name(uuid_name: str) -> str:
    """将 UUID 存储名解析为用户上传时的原始文件名"""
    if not uuid_name:
        return uuid_name
    mapping = _load_upload_names()
    return mapping.get(uuid_name, uuid_name)


# v1.2: Lazy graph — 延迟到首次调用时编译，避免启动时阻塞
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = compile_graph()
    return _graph


# ── 会话锁：确保同一会话串行执行，避免上下文串台 ──────────────
session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# ── 飞书最近上传文件追踪（chat_id → local_path）────────────────
_recent_feishu_files: dict[str, str] = {}

# ── 简易频率限制 ──────────────────────────────────────────────
rate_limit_store: dict[str, list[float]] = defaultdict(list)

# ── 异步任务状态追踪 ──────────────────────────────────────────
# task_id → {"status": "pending"|"processing"|"done"|"error", "progress": str, "result": dict}
task_store: dict[str, dict] = {}
task_store_lock = asyncio.Lock()


def check_rate_limit(user_id: str, limit: int = RATE_LIMIT_PER_USER) -> bool:
    now = time.time()
    window = 60.0
    rate_limit_store[user_id] = [
        t for t in rate_limit_store[user_id] if now - t < window
    ]
    if len(rate_limit_store[user_id]) >= limit:
        return False
    rate_limit_store[user_id].append(now)
    return True


# ── 简单认证 ──────────────────────────────────────────────────
# 账号密码从环境变量读取（见 .env.example），不要写死在代码里
AUTH_USERS = {
    "admin": {"password": os.getenv("ADMIN_PASSWORD", "admin"), "role": "admin"},
    "hk": {
        "password": os.getenv("HK_PASSWORD", ""),
        "role": "feishu_user",
        "feishu_user_id": os.getenv("HK_FEISHU_USER_ID", ""),
    },
}

auth_tokens: dict[str, dict] = {}


# ── 飞书 API 辅助 ──────────────────────────────────────────────
_tenant_token: dict = {"token": "", "expires_at": 0}
_token_lock = asyncio.Lock()


async def get_tenant_access_token() -> str:
    """获取/刷新飞书 tenant_access_token，带缓存"""
    async with _token_lock:
        now = time.time()
        if _tenant_token["token"] and now < _tenant_token["expires_at"] - 60:
            return _tenant_token["token"]

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
                timeout=10,
            )
            data = resp.json()
            if data.get("code") != 0:
                raise Exception(f"获取 token 失败: {data}")
            _tenant_token["token"] = data["tenant_access_token"]
            _tenant_token["expires_at"] = now + data.get("expire", 7200)
            logging.getLogger("agent_thought").info("飞书 tenant_access_token 已刷新")
            return _tenant_token["token"]


def _strip_markdown_for_feishu(text: str) -> str:
    """洗掉 Markdown 标记，输出适合飞书文本消息的纯文本"""
    import re

    lines = text.split("\n")
    result: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    pending = False

    for line in lines:
        stripped = line.strip()

        # 代码块（跳过）
        if stripped.startswith("```"):
            continue

        # 表格分隔行
        if re.match(r"^\|[-:\s|]+\|$", stripped):
            continue

        # 表格行
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            table_rows.append(cells)
            in_table = True
            continue

        # 表格结束，输出为键值对
        if in_table:
            in_table = False
            if table_rows:
                headers = table_rows[0] if len(table_rows) > 0 else []
                for row in table_rows[1:]:
                    for i, cell in enumerate(row):
                        if i < len(headers):
                            result.append(f"  • {headers[i]}：{cell}")
                table_rows = []
            pending = True

        # 标题 → 【标题】
        m = re.match(r"^#{1,4}\s+(.*)", stripped)
        if m:
            if pending:
                result.append("")
                pending = False
            result.append(f"【{m.group(1)}】")
            pending = True
            continue

        # 水平线
        if re.match(r"^---+$", stripped):
            if pending:
                result.append("")
                pending = False
            result.append("—" * 20)
            pending = True
            continue

        # 无序列表
        m = re.match(r"^[-*]\s+(.*)", stripped)
        if m:
            if pending:
                result.append("")
                pending = False
            result.append(f"  • {m.group(1)}")
            continue

        # 有序列表
        m = re.match(r"^\d+\.\s+(.*)", stripped)
        if m:
            if pending:
                result.append("")
                pending = False
            result.append(f"  {m.group(1)}")
            continue

        # 普通行：去掉行内 ** 和 ` 标记
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

        if pending:
            result.append("")
            pending = False
        result.append(cleaned)

    if in_table and table_rows:
        headers = table_rows[0] if len(table_rows) > 0 else []
        for row in table_rows[1:]:
            for i, cell in enumerate(row):
                if i < len(headers):
                    result.append(f"  • {headers[i]}：{cell}")

    text = "\n".join(result)
    # 全局兜底：去残留的 ** * 标记
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # 去多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def send_feishu_message(chat_id: str, text: str, chat_type: str = ""):
    """通过飞书 API 发送消息，自动洗掉 Markdown 标记

    Args:
        chat_id: 会话 ID (oc_xxx 或 ou_xxx)
        text: 消息文本
        chat_type: 会话类型 (\"p2p\" 或 \"group\")，用于决定 receive_id_type
    """
    logger = logging.getLogger("agent_thought")
    token = await get_tenant_access_token()

    # p2p 私聊的 chat_id 是 ou_xxx (open_id)，需要 receive_id_type=open_id
    # group 群聊的 chat_id 是 oc_xxx，需要 receive_id_type=chat_id
    if chat_type == "p2p" or chat_id.startswith("ou_"):
        receive_id_type = "open_id"
    else:
        receive_id_type = "chat_id"

    url = (
        "https://open.feishu.cn/open-apis/im/v1/messages"
        f"?receive_id_type={receive_id_type}"
    )

    cleaned = _strip_markdown_for_feishu(text)
    body = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": cleaned}),
    }
    logger.info(f"发送飞书消息: chat_id={chat_id}, len={len(text)}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=15,
        )
        data = resp.json()
        logger.info(f"飞书发送响应: status={resp.status_code}, body={json.dumps(data, ensure_ascii=False)}")

        if resp.status_code >= 400 or data.get("code") != 0:
            logger.error(
                f"飞书消息发送失败! status={resp.status_code}, code={data.get('code')}, "
                f"msg={data.get('msg')}, chat_id={chat_id}"
            )
        return data


async def download_feishu_file(message_id: str, file_key: str, file_name: str) -> str:
    """下载飞书文件并保存到知识库目录，返回本地路径"""
    token = await get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=file"
    headers = {"Authorization": f"Bearer {token}"}

    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    safe_name = file_name or f"feishu_file_{int(time.time())}.txt"
    local_path = os.path.join(knowledge_dir, safe_name)

    logger = logging.getLogger("agent_thought")
    logger.info(f"下载飞书文件: file_name={file_name}, message_id={message_id}, file_key={file_key}")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"飞书文件已保存: {local_path} ({len(resp.content)} bytes)")
            return local_path
        else:
            # 解析飞书错误响应
            try:
                err_data = resp.json()
                err_code = err_data.get("code", "unknown")
                err_msg = err_data.get("msg", resp.text[:200])
            except Exception:
                err_code = str(resp.status_code)
                err_msg = resp.text[:200]
            logger.error(f"下载飞书文件失败: status={resp.status_code}, code={err_code}, msg={err_msg}")
            raise Exception(f"HTTP {resp.status_code}, 错误码 {err_code}: {err_msg}")


async def download_feishu_image(message_id: str, image_key: str) -> str:
    """下载飞书图片并保存到上传目录，返回公网可访问 URL"""
    token = await get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{image_key}?type=image"
    headers = {"Authorization": f"Bearer {token}"}

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"feishu_img_{image_key}.jpg"
    local_path = os.path.join(UPLOAD_DIR, safe_name)

    logger = logging.getLogger("agent_thought")
    logger.info(f"下载飞书图片: image_key={image_key} → {local_path}")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            public_url = f"{PUBLIC_BASE_URL}/uploads/{safe_name}"
            logger.info(f"飞书图片已保存: {local_path} ({len(resp.content)} bytes)")
            return public_url
        else:
            logger.error(f"下载飞书图片失败: status={resp.status_code}, body={resp.text[:200]}")
            raise Exception(f"下载图片失败: {resp.status_code}")


def _generate_file_summary(file_path: str, file_name: str) -> str:
    """读取文档文件，用 LLM 生成 1-2 句话的内容摘要（供飞书回复使用）"""
    try:
        text = _lazy_import("load_document")(file_path)
    except Exception as e:
        logging.getLogger("agent_thought").warning(f"文档加载失败，跳过摘要: {e}")
        return ""

    if not text or not text.strip():
        return ""

    # 截取前 2000 字符作为摘要素材
    preview = text[:2000] if len(text) > 2000 else text

    try:
        summary_llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3,
        )
        prompt = (
            f"你是一个文档分析助手。请用1-2句话概括以下文档的主要内容，"
            f"让读者快速了解这份文件是什么、有什么用。"
            f"如果文档中包含链接、表格等结构化信息，也请简要提及。\n\n"
            f"文件名：{file_name}\n"
            f"文档内容（节选）：\n{preview}\n\n"
            f"请直接输出概括（不要加\"这份文档\"开头），控制在80字以内："
        )
        response = summary_llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
        # 清理可能的引号包裹
        summary = summary.strip('"').strip("'").strip("「").strip("」")
        return summary
    except Exception as e:
        logging.getLogger("agent_thought").warning(f"文件摘要生成失败: {e}")
        return ""


# ── 统一消息处理（Webhook 回调入口） ──────────────────────────
async def process_message(event_data):
    """统一的消息处理入口，集成会话治理、指标、思考记录
    支持两种输入：飞书 SDK 的 EventContext 对象、Webhook 的 dict
    """
    import logging
    thought_logger = logging.getLogger("agent_thought")

    # 解析 Webhook JSON 事件
    msg_type = ""
    file_key = ""
    file_name = ""
    image_key = ""
    if not isinstance(event_data, dict):
        thought_logger.error(f"未知的消息格式: {type(event_data)}")
        return {"code": -1, "msg": "unknown message format"}

    event = event_data.get("event", {})
    sender = event.get("sender", {}).get("sender_id", {})
    user_id = sender.get("user_id") or sender.get("open_id", "unknown")
    message = event.get("message", {})
    chat_id = message.get("chat_id", user_id)
    msg_content = message.get("content", "{}")
    msg_type = message.get("msg_type", "") or message.get("message_type", "")
    file_key = message.get("file_key", "")
    file_name = message.get("file_name", "")
    image_key = message.get("image_key", "")

    # 从 content JSON 中回退提取 file_key / file_name / image_key
    # 飞书 v2 事件中，这些字段嵌套在 content JSON 内，而非 message 顶层
    if not file_key:
        try:
            content_json = json.loads(msg_content)
            file_key = content_json.get("file_key", "")
            if not file_name:
                file_name = content_json.get("file_name", "")
        except (json.JSONDecodeError, TypeError):
            pass

    if not image_key:
        try:
            content_json = json.loads(msg_content)
            image_key = content_json.get("image_key", "")
        except (json.JSONDecodeError, TypeError):
            pass

    # 文件消息 → 下载并入库 RAG 知识库
    file_text_from_caption = ""
    if msg_type == "file" and file_key:
        message_id = message.message_id if hasattr(message, "message_id") else message.get("message_id", "")
        thought_logger.info(f"飞书文件消息: message_id={message_id}, file_key={file_key}, file_name={file_name}, chat_id={chat_id}")
        try:
            local_path = await download_feishu_file(message_id, file_key, file_name)
            _recent_feishu_files[chat_id] = local_path
            doc_count = _lazy_import("build_index")()
            thought_logger.info(f"飞书文件已入库: {local_path}, 索引块数: {doc_count}")

            # 生成智能摘要回复
            summary = _generate_file_summary(local_path, file_name)
            if summary:
                reply = (
                    f"已收到文件「{file_name}」\n\n"
                    f"📄 {summary}\n\n"
                    f"💡 文件已自动添加到知识库，有什么想了解的请随时提问。"
                )
            else:
                reply = (
                    f"已收到文件「{file_name}」，已自动添加到知识库。"
                    f"有什么想了解的请随时提问。"
                )
            await send_feishu_message(chat_id, reply)
        except Exception as e:
            thought_logger.error(f"飞书文件入库失败: {e}, message_id={message_id}, file_key={file_key}")
            await send_feishu_message(chat_id, f"文件接收失败：{e}")
            return {"code": -1, "msg": "file_download_failed"}

        # 检查是否有附带文本（用户发文件时可能同时输入了问题）
        try:
            content_json = json.loads(msg_content)
            file_text_from_caption = content_json.get("text", "").strip()
        except (json.JSONDecodeError, TypeError):
            pass

        # 如果没有附带文本，回复摘要后返回
        if not file_text_from_caption:
            return {"code": 0, "msg": "file_processed"}
        # 有附带文本：继续走后续管线，text 设为 caption 内容

    # 图片消息 → 下载并通过智谱视觉分析
    image_url_for_graph = ""
    if msg_type == "image" and image_key:
        message_id = message.message_id if hasattr(message, "message_id") else message.get("message_id", "")
        try:
            image_url_for_graph = await download_feishu_image(message_id, image_key)
            thought_logger.info(f"飞书图片已转为URL: {image_url_for_graph}")
        except Exception as e:
            thought_logger.error(f"飞书图片下载失败: {e}")
            await send_feishu_message(chat_id, "图片接收失败，请重试。")
            return {"code": -1, "msg": "image_download_failed"}

        # 图片无附带文本时：调用视觉分析并回复智能摘要
        try:
            caption_text = json.loads(msg_content).get("text", "").strip()
        except (json.JSONDecodeError, TypeError):
            caption_text = ""
        if not caption_text:
            try:
                from src.vision_adapter import analyze_image
                img_desc = analyze_image(image_url_for_graph)
                img_summary = img_desc[:200] if len(img_desc) > 200 else img_desc
                reply = (
                    f"已收到图片\n\n"
                    f"🖼️ {img_summary}\n\n"
                    f"💡 有什么想了解的请随时提问。"
                )
                await send_feishu_message(chat_id, reply)
                return {"code": 0, "msg": "image_processed"}
            except Exception as e:
                thought_logger.warning(f"图片摘要生成失败: {e}")
                await send_feishu_message(chat_id, "已收到图片，有什么想了解的请随时提问。")
                return {"code": 0, "msg": "image_processed"}

    # 提取文本：文件消息的附带文本优先，否则从 content 解析
    if file_text_from_caption:
        text = file_text_from_caption
    else:
        try:
            text = json.loads(msg_content).get("text", "")
        except (json.JSONDecodeError, TypeError):
            text = str(msg_content)

    # 纯图片消息（无文本）使用默认提示
    if image_url_for_graph and not text.strip():
        text = "请帮我分析这张图片的内容"

    if not text.strip() and not image_url_for_graph:
        return {"code": 0, "msg": "empty message"}

    # 飞书会话 ID（用于在 Web 端展示）
    feishu_session_id = f"feishu_{user_id}"

    # 需要注册到 session_store 才能在 Web 端看到
    if not session_store.get(feishu_session_id):
        session_store.create(feishu_session_id, f"飞书-{user_id[:10]}")
    session_store.touch(feishu_session_id)

    # 频率限制
    if not check_rate_limit(user_id):
        return {"code": -1, "msg": "请求过于频繁，请稍后再试"}

    # 获取会话锁，同会话串行执行
    async with session_locks[chat_id]:
        task_id = f"task_{int(time.time())}_{user_id}"
        recorder = ThoughtRecorder(user_id, task_id)
        recorder.record_thought("input", text)

        # 获取用户会话（自动处理过期）
        session = get_user_session(user_id)

        # 添加用户消息 → 自动走场景 1/2/3
        await session.add_message("user", text)
        recorder.record_thought(
            "session_update",
            f"历史轮数: {len(session.history)}, 摘要长度: {len(session.long_summary)}",
        )

        # 使用 LangGraph Agent 管线处理（和 Web 端一致）
        config = {"configurable": {"thread_id": feishu_session_id}}
        user_msg = HumanMessage(content=text)

        # 加载最近上传的飞书文件内容，注入到图状态
        doc_text = ""
        doc_filename = ""
        recent_file_path = _recent_feishu_files.get(chat_id, "")
        if recent_file_path and os.path.exists(recent_file_path):
            try:
                doc_text = _lazy_import("load_document")(recent_file_path)
                doc_filename = os.path.basename(recent_file_path)
                thought_logger.info(f"注入最近文件: {doc_filename} ({len(doc_text)} 字符)")
            except Exception as e:
                thought_logger.warning(f"加载最近文件失败: {e}")

        try:
            result = await asyncio.to_thread(
                lambda: _get_graph().invoke(
                    {
                        "messages": [user_msg],
                        "query": text,
                        "image_url": image_url_for_graph or "",
                        "document_text": doc_text,
                        "document_filename": doc_filename,
                        # v1.1: 显式注入字段
                        "injected_document": doc_text,
                        "injected_document_filename": doc_filename,
                    },
                    config,
                )
            )
            ai_reply = result.get("final_answer", "")
            recorder.record_thought("agent_route", result.get("route_decision", ""))
            recorder.record_thought("agent_rewrite", result.get("rewritten_query", ""))
            recorder.record_thought("agent_retrieve", f"检索到 {len(result.get('retrieved_docs', []))} 篇文档")
        except Exception as e:
            ai_reply = f"抱歉，处理请求时遇到错误：{str(e)}"
            recorder.record_thought("error", str(e))
            metrics.record_task(False, {"total": 1, "success": 0})
            await send_feishu_message(chat_id, ai_reply)
            return {"code": -1, "msg": ai_reply}

        recorder.record_thought("ai_response", ai_reply)

        # 添加 AI 回复
        await session.add_message("assistant", ai_reply)

        # 记录指标
        context_tokens = (len(text) + len(ai_reply)) // 4
        metrics.record_task(True, {"total": 1, "success": 1}, context_tokens)
        metrics.update_session_stats(USER_SESSIONS)

        # 保存复盘
        recorder.save_session_review()

        # 发送飞书回复
        try:
            await send_feishu_message(chat_id, ai_reply)
        except Exception as e:
            logging.getLogger("agent_thought").error(f"飞书消息发送失败: {e}")

        return {"code": 0, "reply": ai_reply}


# ── 工具函数 ──────────────────────────────────────────────────
def _parse_comma_list(value: str) -> list[str]:
    """将逗号分隔的字符串解析为列表，过滤空值"""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# ── 请求模型 ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    image_url: str = ""
    document_filename: str = ""


class SessionCreate(BaseModel):
    title: str = ""


# ── Static files ──────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/favicon.ico")
async def favicon():
    """返回空响应避免 404 噪音"""
    return Response(status_code=204)


# ── System status ─────────────────────────────────────────────
@app.get("/api/status")
async def status():
    return {
        "llm_connected": bool(DEEPSEEK_API_KEY),
        "model": "deepseek-chat",
        "knowledge_ready": _lazy_import("knowledge_is_ready")(),
        "sessions_count": len(session_store.list_all()),
        "version": "1.0",
    }


# ── Knowledge base upload ─────────────────────────────────────
ALLOWED_KB_EXTENSIONS = {".txt", ".pdf", ".docx", ".xlsx", ".md"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or ".txt")[1].lower()
    if not file.filename or ext not in ALLOWED_KB_EXTENSIONS:
        raise HTTPException(400, f"只支持 {', '.join(sorted(ALLOWED_KB_EXTENSIONS))} 文件")

    content = await file.read()
    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    file_path = os.path.join(knowledge_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # 优先增量更新，失败时回退到全量构建
        count = _lazy_import("add_to_index")(file_path)
        if count == 0:
            logger = logging.getLogger("agent_thought")
            logger.info(f"增量索引未添加新块，尝试全量构建")
            count = _lazy_import("build_index")()
        return {"msg": "知识库已就绪", "chunks": count}
    except Exception as e:
        # 增量失败时回退到全量重建
        logging.getLogger("agent_thought").warning(f"增量索引失败 ({e})，回退到全量构建")
        try:
            count = _lazy_import("build_index")()
            return {"msg": "知识库已就绪（全量重建）", "chunks": count}
        except Exception as e2:
            raise HTTPException(500, f"索引构建失败: {e2}")


# ── Knowledge base management ──────────────────────────────────
@app.get("/api/knowledge")
async def get_knowledge_status():
    """获取知识库状态和文件列表"""
    import glob as _glob
    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)

    files = []
    for fpath in _glob.glob(os.path.join(knowledge_dir, "*")):
        if os.path.isfile(fpath):
            fname = os.path.basename(fpath)
            fsize = os.path.getsize(fpath)
            files.append({
                "name": fname,
                "size": fsize,
                "size_display": f"{fsize / 1024:.1f} KB" if fsize < 1024 * 1024 else f"{fsize / 1024 / 1024:.1f} MB",
            })

    from src.rag.index_builder import get_collection
    col = get_collection()
    chunk_count = col.count() if col else 0

    return {
        "ready": _lazy_import("knowledge_is_ready")(),
        "chunks": chunk_count,
        "files": files,
        "file_count": len(files),
        "backend": os.getenv("VECTOR_STORE_BACKEND", "chroma"),
    }


@app.delete("/api/knowledge")
async def delete_knowledge_base():
    """删除整个知识库（向量索引 + 源文件）"""
    import shutil
    import glob as _glob

    deleted_info = {"files_removed": 0, "vector_store_cleared": False, "errors": []}

    # 1. 删除 ChromaDB 向量索引
    try:
        from src.rag.vector_store import get_vector_store
        vs = get_vector_store()
        if vs:
            vs.delete_all()
            deleted_info["vector_store_cleared"] = True
    except Exception as e:
        deleted_info["errors"].append(f"向量库清理失败: {e}")

    # 2. 清空知识库目录下的文件
    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    if os.path.exists(knowledge_dir):
        for fpath in _glob.glob(os.path.join(knowledge_dir, "*")):
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    deleted_info["files_removed"] += 1
            except Exception as e:
                deleted_info["errors"].append(f"删除文件 {os.path.basename(fpath)} 失败: {e}")

    logging.getLogger("agent_thought").info(
        f"知识库已清空: {deleted_info['files_removed']} 文件, "
        f"向量库: {deleted_info['vector_store_cleared']}"
    )

    return {
        "msg": "知识库已清空",
        "details": deleted_info,
    }


@app.delete("/api/knowledge/files/{filename}")
async def delete_knowledge_file(filename: str):
    """删除知识库中的单个文件并重建索引"""
    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    file_path = os.path.join(knowledge_dir, filename)

    # 安全检查：防止路径穿越
    real_path = os.path.realpath(file_path)
    real_knowledge = os.path.realpath(knowledge_dir)
    if not real_path.startswith(real_knowledge):
        raise HTTPException(403, "非法的文件路径")

    if not os.path.exists(file_path):
        raise HTTPException(404, f"文件不存在: {filename}")

    try:
        os.remove(file_path)
        # 优先从索引中增量删除
        removed = _lazy_import("remove_from_index")(filename)
        if removed == 0:
            # 没有向量索引记录，回退到全量重建
            count = _lazy_import("build_index")()
        else:
            count = removed
        logging.getLogger("agent_thought").info(f"知识库文件已删除: {filename}，移除索引: {count} 块")
        return {"msg": f"已删除 {filename}，索引已更新", "chunks": count}
    except Exception as e:
        raise HTTPException(500, f"删除失败: {e}")


# ── 异步文档上传（后台处理 + 状态轮询） ──────────────────────────
@app.post("/api/upload-async")
async def upload_async(file: UploadFile = File(...)):
    """异步上传文档：立即返回任务 ID，后台处理，前端通过 /api/task/{id} 轮询状态"""
    ext = os.path.splitext(file.filename or ".txt")[1].lower()
    if not file.filename or ext not in ALLOWED_KB_EXTENSIONS:
        raise HTTPException(400, f"只支持 {', '.join(sorted(ALLOWED_KB_EXTENSIONS))} 文件")

    task_id = uuid.uuid4().hex[:12]

    # 保存文件
    knowledge_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    content = await file.read()
    file_path = os.path.join(knowledge_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # 注册任务
    async with task_store_lock:
        task_store[task_id] = {
            "status": "pending",
            "progress": "文件已保存，等待处理",
            "filename": file.filename,
            "result": None,
            "created_at": time.time(),
        }

    # 启动后台处理
    asyncio.create_task(_process_document_async(task_id, file_path, file.filename))

    return {
        "task_id": task_id,
        "status": "pending",
        "msg": "上传成功，正在处理中...",
    }


async def _process_document_async(task_id: str, file_path: str, filename: str):
    """后台异步处理文档：解析 + 索引"""
    logger = logging.getLogger("agent_thought")

    async def _update_status(status: str, progress: str, result: dict = None):
        async with task_store_lock:
            if task_id in task_store:
                task_store[task_id]["status"] = status
                task_store[task_id]["progress"] = progress
                if result:
                    task_store[task_id]["result"] = result

    try:
        await _update_status("processing", "正在解析文档...")

        # 在线程池中运行同步的文档解析（避免阻塞事件循环）
        text = await asyncio.to_thread(_lazy_import("load_document"), file_path)

        if not text or not text.strip():
            await _update_status("error", "文档解析失败：内容为空或解析超时",
                                 {"error": "文档解析失败，请检查文件是否损坏"})
            return

        await _update_status("processing", f"文档解析完成 ({len(text)} 字符)，正在构建索引...")

        # 增量索引
        count = await asyncio.to_thread(_lazy_import("add_to_index"), file_path)
        if count == 0:
            await _update_status("processing", "增量索引未生效，正在全量重建...")
            count = await asyncio.to_thread(_lazy_import("build_index"))

        await _update_status("done", "处理完成", {
            "chunks": count,
            "text_length": len(text),
            "preview": text[:300] if len(text) > 300 else text,
        })

        logger.info(f"异步文档处理完成: {filename} → {count} 块")

        # 生成摘要（可选）
        summary = _generate_file_summary(file_path, filename)
        if summary:
            async with task_store_lock:
                if task_id in task_store and task_store[task_id].get("result"):
                    task_store[task_id]["result"]["summary"] = summary

    except Exception as e:
        logger.error(f"异步文档处理失败 {filename}: {e}")
        await _update_status("error", f"处理失败: {str(e)[:100]}",
                             {"error": f"文档处理失败: {str(e)[:200]}"})


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """轮询异步任务状态"""
    async with task_store_lock:
        task = task_store.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在或已过期")
    return task


# ── 清理过期任务（启动时和定期执行） ──────────────────────────
async def cleanup_expired_tasks():
    """清理超过 30 分钟的已完成/错误任务"""
    async with task_store_lock:
        now = time.time()
        expired = [
            tid for tid, t in task_store.items()
            if now - t.get("created_at", 0) > 1800
        ]
        for tid in expired:
            del task_store[tid]
    if expired:
        logging.getLogger("agent_thought").info(f"清理了 {len(expired)} 个过期任务")


# ── Image upload (for vision pipeline) ──────────────────────────
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    allowed_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    ext = os.path.splitext(file.filename or ".jpg")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"不支持的图片格式: {ext}，支持 {', '.join(allowed_ext)}")

    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # ── 记录原始文件名映射 ──
    names = _load_upload_names()
    names[safe_name] = file.filename or safe_name
    _save_upload_names(names)

    public_url = f"{PUBLIC_BASE_URL}/uploads/{safe_name}"
    logging.getLogger("agent_thought").info(f"图片已上传: {safe_name} → {public_url}")
    return {"url": public_url, "filename": safe_name, "original_name": file.filename}


# ── Document upload (for document pipeline) ─────────────────────
DOC_ALLOWED_EXT = {".txt", ".pdf", ".docx", ".xlsx", ".md"}

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """上传文档文件（PDF/Word/Excel/TXT），返回解析预览"""
    ext = os.path.splitext(file.filename or ".txt")[1].lower()
    if ext not in DOC_ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文档格式: {ext}，支持 {', '.join(sorted(DOC_ALLOWED_EXT))}")

    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # ── 记录原始文件名映射（UUID → 用户上传的真实文件名）──
    names = _load_upload_names()
    names[safe_name] = file.filename or safe_name
    _save_upload_names(names)

    # ── 同时保存到知识库目录并增量索引（让 RAG 也能查到） ──
    kb_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    os.makedirs(kb_dir, exist_ok=True)
    kb_path = os.path.join(kb_dir, file.filename or safe_name)
    with open(kb_path, "wb") as f:
        f.write(content)
    try:
        chunk_count = _lazy_import("add_to_index")(kb_path)
        if chunk_count == 0:
            chunk_count = _lazy_import("build_index")()
        logging.getLogger("agent_thought").info(
            f"文档已同步到知识库: {file.filename} → {chunk_count} 块"
        )
    except Exception as e:
        logging.getLogger("agent_thought").warning(f"知识库索引更新跳过: {e}")

    # 解析文档获取文本预览（带超时保护）
    text = ""
    preview = ""
    try:
        text = _lazy_import("load_document")(file_path)
        preview = text[:500] if len(text) > 500 else text
    except Exception as e:
        logging.getLogger("agent_thought").warning(f"文档预览生成失败: {e}")
        preview = "(文档解析超时或失败，请稍后在知识库中查看)"

    logging.getLogger("agent_thought").info(f"文档已上传: {safe_name} → {len(text) if text else 0} 字符")
    return {
        "url": f"{PUBLIC_BASE_URL}/uploads/{safe_name}",
        "filename": safe_name,
        "original_name": file.filename,
        "preview": preview,
        "text_length": len(text) if text else 0,
    }


# ── Non-streaming chat ────────────────────────────────────────
@app.post("/api/ask")
async def ask(req: AskRequest):
    # 提供了文档或图片时，允许绕过知识库检查（文档已自动入库，或走直接注入流程）
    has_document = bool(req.document_filename) or bool(req.image_url)
    if not _lazy_import("knowledge_is_ready")() and not has_document:
        raise HTTPException(400, "知识库未就绪，请先上传文档")

    config = {"configurable": {"thread_id": req.session_id}}

    snapshot = _get_graph().get_state(config)
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))

    is_first = len(existing_messages) == 0
    user_msg = HumanMessage(content=req.question.strip())

    # 读取文档（如果有）
    doc_text = ""
    if req.document_filename:
        doc_path = os.path.join(UPLOAD_DIR, req.document_filename)
        if os.path.exists(doc_path):
            try:
                doc_text = _lazy_import("load_document")(doc_path)
            except Exception:
                doc_text = ""

    # 接入 SessionManager
    web_session = get_user_session(req.session_id)

    try:
        result = _get_graph().invoke(
            {
                "messages": [user_msg],
                "query": req.question.strip(),
                "image_url": req.image_url or "",
                "document_text": doc_text,
                "document_filename": _resolve_upload_name(req.document_filename or ""),
                # v1.1: 显式注入字段，替代 [系统提示] 字符串 hack
                "injected_document": doc_text,
                "injected_document_filename": _resolve_upload_name(req.document_filename or ""),
            },
            config,
        )

        answer = result.get("final_answer", "抱歉，处理出现错误。")

        session_store.touch(
            req.session_id,
            message_count=len(existing_messages) + 2,
        )
        if is_first:
            session_store.set_title_from_first_message(req.session_id, req.question)

        web_session.add_message_sync("user", req.question.strip())

        return {
            "answer": answer,
            "route_decision": result.get("route_decision", ""),
            "rewritten_query": result.get("rewritten_query", ""),
            "retrieved_count": len(result.get("retrieved_docs", [])),
            "tool_calls": [t.get("tool", "") for t in result.get("tool_calls", [])],
        }

    except Exception as e:
        raise HTTPException(500, f"处理请求失败: {e}")


# ── Streaming chat (SSE) ──────────────────────────────────────
@app.get("/api/ask/stream")
async def ask_stream(
    question: str,
    session_id: str = "default",
    image_url: str = Query(""),
    image_urls: str = Query(""),
    document_filename: str = Query(""),
    document_filenames: str = Query(""),
):
    # 提供了文档或图片时，允许绕过知识库检查
    has_document = bool(document_filename) or bool(document_filenames) or bool(image_url) or bool(image_urls)
    if not _lazy_import("knowledge_is_ready")() and not has_document:
        raise HTTPException(400, "知识库未就绪，请先上传文档")

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    if not session_store.get(session_id):
        session_store.create(session_id)

    # 接入 SessionManager：治理 Web 会话（过期检测 + 指标统计）
    web_session = get_user_session(session_id)

    snapshot = _get_graph().get_state({"configurable": {"thread_id": session_id}})
    existing_messages = []
    if snapshot and snapshot.values:
        existing_messages = list(snapshot.values.get("messages", []))
    is_first = len(existing_messages) == 0
    if is_first:
        session_store.set_title_from_first_message(session_id, question)

    # ── 合并多图片/多文档参数 ──
    _image_urls = _parse_comma_list(image_urls)
    if image_url and image_url not in _image_urls:
        _image_urls.insert(0, image_url)
    _doc_filenames = _parse_comma_list(document_filenames)
    if document_filename and document_filename not in _doc_filenames:
        _doc_filenames.insert(0, document_filename)

    async def event_generator():
        try:
            async for sse_chunk in run_agent_stream(
                question.strip(),
                session_id,
                image_urls=_image_urls if _image_urls else None,
                document_filenames=_doc_filenames if _doc_filenames else None,
            ):
                try:
                    yield sse_chunk.encode("utf-8")
                except ConnectionResetError:
                    return
        except ConnectionResetError:
            pass
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n".encode("utf-8")

        msg_count = len(existing_messages) + 2
        session_store.touch(session_id, message_count=msg_count)
        web_session.add_message_sync("user", question.strip())

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session management ────────────────────────────────────────
@app.get("/api/sessions")
async def list_sessions():
    cleanup_expired_sessions()
    return {"sessions": session_store.list_all()}


@app.post("/api/sessions")
async def create_session(req: SessionCreate):
    sid = uuid.uuid4().hex
    session_store.create(sid, req.title or "新对话")
    return session_store.get(sid)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_store.delete(session_id)
    return {"msg": "已删除"}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    snapshot = _get_graph().get_state(config)
    messages = []
    if snapshot and snapshot.values:
        for msg in snapshot.values.get("messages", []):
            role = "user" if getattr(msg, "type", "human") == "human" else "assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            messages.append({"role": role, "content": content})
    return {"session_id": session_id, "messages": messages}


# ── 认证 ──────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def auth_login(req: AuthRequest):
    user = AUTH_USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(401, "用户名或密码错误")

    token = uuid.uuid4().hex
    auth_tokens[token] = {
        "username": req.username,
        "role": user["role"],
        "feishu_user_id": user.get("feishu_user_id", ""),
        "created_at": time.time(),
    }
    return {
        "token": token,
        "username": req.username,
        "role": user["role"],
    }


@app.get("/api/auth/me")
async def auth_me(token: str = Query(...)):
    info = auth_tokens.get(token)
    if not info:
        raise HTTPException(401, "未登录或 token 已过期")
    return info


# ── 飞书回调 ──────────────────────────────────────────────────
# 已切换为 WebSocket 长连接模式，此端点仅作健康检查
@app.get("/feishu/callback")
async def feishu_verify():
    return {"status": "ok", "mode": "websocket", "message": "飞书长连接已启用，HTTP 回调已停用"}

@app.post("/feishu/callback")
async def feishu_callback():
    return {"status": "ok", "mode": "websocket", "message": "飞书长连接已启用，HTTP 回调已停用"}


# ── 指标查询接口 ──────────────────────────────────────────────
@app.get("/api/metrics")
async def get_metrics():
    metrics.update_session_stats(USER_SESSIONS)
    metrics.save_metrics_to_file(force=True)
    return {
        "uptime_seconds": round(time.time() - metrics.start_time, 2),
        "total_tasks": metrics.total_tasks,
        "success_tasks": metrics.success_tasks,
        "tool_calls": metrics.tool_calls,
        "tool_success": metrics.tool_success,
        "session_stats": metrics.session_stats,
    }


# ── 健康检查 ──────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}




# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print(f"WorkMate Agent v1.0 启动中...")
    print(f"LLM: DeepSeek (deepseek-chat)")
    print(f"知识库: {'已就绪' if _lazy_import("knowledge_is_ready")() else '未建库'}")
    print(f"访问: http://{SERVER_HOST}:{SERVER_PORT}")

    # 上线前检查
    run_pre_launch_checks()

    # 定期清理过期任务（每 10 分钟）
    async def _periodic_task_cleanup():
        while True:
            await asyncio.sleep(600)
            await cleanup_expired_tasks()

    asyncio.create_task(_periodic_task_cleanup())

    # 启动飞书 WebSocket 长连接（无需公网 URL，无需 cpolar）
    if FEISHU_APP_ID and FEISHU_APP_SECRET:
        feishu_ws = FeishuWSClient(
            app_id=FEISHU_APP_ID,
            app_secret=FEISHU_APP_SECRET,
            on_event=process_message,
        )
        feishu_ws.start_background()
        print(f"飞书长连接: 已启动")
    else:
        print(f"飞书长连接: 未配置 APP_ID/APP_SECRET，跳过")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
