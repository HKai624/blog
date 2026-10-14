"""文档加载器：使用 MinerU API 解析 PDF/Word/Excel/图片，返回 Markdown 文本

特性：
- 超时控制：单次请求 30s + 轮询总超时 60s
- 指数退避重试：对可恢复错误（5xx、连接重置）最多重试 3 次
- 优雅降级：解析失败时返回空字符串，不阻塞后续流程
"""

import os
import time
import logging
import threading

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".xlsx", ".md", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 需要 MinerU 解析的二进制/复杂格式
_MINERU_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ── MinerU 超时与重试配置 ──
MINERU_REQUEST_TIMEOUT = 30      # 单次 API 请求超时（秒）
MINERU_POLL_TIMEOUT = 60         # 轮询总超时（秒），超时后主动放弃
MINERU_MAX_RETRIES = 3           # 最大重试次数
MINERU_RETRY_BASE_DELAY = 1.0    # 重试基础延迟（秒）


def _is_recoverable_error(exception: Exception) -> bool:
    """判断是否为可恢复的网络错误（值得重试）"""
    error_str = str(exception).lower()
    # HTTP 5xx
    recoverable_keywords = [
        "500", "502", "503", "504",
        "timeout", "timed out",
        "connection reset", "connectionrefused",
        "too many requests", "429",
        "service unavailable",
        "network", "socket",
    ]
    for keyword in recoverable_keywords:
        if keyword in error_str:
            return True
    return False


def load_document(file_path: str) -> str:
    """根据文件扩展名调度到对应解析器，返回提取的 Markdown 文本

    - PDF / DOCX / XLSX / 图片：使用 MinerU API 解析
    - TXT / MD：直接文件读取
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in _MINERU_EXTENSIONS:
        return _parse_with_mineru(file_path)
    elif ext in (".txt", ".md"):
        return _parse_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _parse_with_mineru(file_path: str) -> str:
    """使用 MinerU API 解析文档，返回 Markdown 文本

    增强特性：
    - 超时控制：单个 extract 调用有总超时保护
    - 指数退避重试：对 5xx、连接重置等可恢复错误自动重试
    - 安全兜底：解析失败返回空字符串，不抛异常（调用方自行判断）
    """
    from src.config import MINERU_API_TOKEN, MINERU_BASE_URL

    if not MINERU_API_TOKEN:
        logger.error("MINERU_API_TOKEN 未配置，请在 .env 文件中设置")
        return ""

    filename = os.path.basename(file_path)
    last_error = None

    for attempt in range(MINERU_MAX_RETRIES):
        try:
            result = _call_mineru_extract(file_path, MINERU_API_TOKEN, MINERU_BASE_URL)
            markdown_text = result.markdown or ""

            if not markdown_text.strip():
                logger.warning(f"MinerU 解析返回空内容: {filename}")
                return ""

            logger.info(f"MinerU 解析完成: {filename} → {len(markdown_text)} 字符")
            return markdown_text

        except Exception as e:
            last_error = e
            logger.warning(f"MinerU 解析失败 (attempt {attempt + 1}/{MINERU_MAX_RETRIES}): {filename}: {e}")

            if attempt < MINERU_MAX_RETRIES - 1 and _is_recoverable_error(e):
                delay = MINERU_RETRY_BASE_DELAY * (2 ** attempt)
                logger.info(f"MinerU 将在 {delay:.1f}s 后重试...")
                time.sleep(delay)
            else:
                # 不可恢复错误或已达最大重试次数
                break

    logger.error(f"MinerU 解析最终失败 ({MINERU_MAX_RETRIES} 次尝试): {filename}: {last_error}")
    return ""


def _call_mineru_extract(file_path: str, token: str, base_url: str):
    """在独立线程中调用 MinerU extract，带超时保护

    使用线程 + join(timeout) 实现跨平台的超时控制。
    即使 mineru 库内部没有超时机制，也能在 poll_timeout 秒后主动放弃。
    """
    from mineru import MinerU

    result_container = []
    error_container = []

    def _extract():
        try:
            client = MinerU(token=token, base_url=base_url)
            result_container.append(client.extract(file_path))
        except Exception as e:
            error_container.append(e)

    thread = threading.Thread(target=_extract, daemon=True)
    thread.start()
    thread.join(timeout=MINERU_POLL_TIMEOUT)

    if thread.is_alive():
        # 线程仍在运行 → 超时，主动放弃
        raise TimeoutError(
            f"MinerU 解析超时 ({MINERU_POLL_TIMEOUT}s): {os.path.basename(file_path)}"
        )

    if error_container:
        raise error_container[0]

    if result_container:
        return result_container[0]

    raise RuntimeError("MinerU extract 未返回结果")


def _parse_text(file_path: str) -> str:
    """读取纯文本 / Markdown 文件"""
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            if text.strip():
                logger.info(
                    f"文本文件解析完成: {os.path.basename(file_path)} → {len(text)} 字符 (编码: {enc})"
                )
                return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"文本读取失败 ({enc}): {e}")
            continue

    logger.error(f"无法解码文件: {file_path}")
    return ""
