"""Document 节点：处理上传的文档文件（通过 MinerU API 解析），将内容注入 Agent 上下文

v1.1 重构：
- 文档全文通过 injected_document 字段显式传递，不再拼入 query 字符串
- 删除了 [系统提示] 字符串 hack，下游节点通过 state["injected_document"] 获取
"""

import os
import logging

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 8000  # 最大注入上下文长度


def read_document_file(file_path: str) -> dict:
    """读取文档文件，返回解析结果

    Args:
        file_path: 文档文件路径

    Returns:
        {"text": str, "filename": str, "file_type": str, "size": int}
    """
    from src.rag.document_loader import load_document

    if not os.path.exists(file_path):
        return {"text": "", "filename": os.path.basename(file_path), "file_type": "unknown", "size": 0}

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    file_type_map = {".pdf": "PDF", ".docx": "Word", ".xlsx": "Excel", ".txt": "TXT", ".md": "Markdown"}
    file_type = file_type_map.get(ext, ext.upper())

    size = os.path.getsize(file_path)

    try:
        text = load_document(file_path)
    except Exception as e:
        logger.error(f"文档读取失败 {filename}: {e}")
        return {"text": "", "filename": filename, "file_type": file_type, "size": size, "error": str(e)}

    # 长文本智能截断
    if len(text) > MAX_CONTENT_LENGTH:
        head = text[:6000]
        tail = text[-1500:]
        text = f"{head}\n\n... [中间内容已省略，共 {len(text)} 字符] ...\n\n{tail}"

    logger.info(f"文档读取完成: {filename} ({file_type}) → {len(text)} 字符")
    return {"text": text, "filename": filename, "file_type": file_type, "size": size}


def document_node(state: dict) -> dict:
    """Graph 节点：处理上传的文档文件，将内容存入独立的 injected_document 字段

    v1.1: 不再将文档内容拼入 query 字符串。
    - 文档全文 → state["injected_document"]
    - 文件名     → state["injected_document_filename"]
    - 用户的 query 保持原样不污染

    下游 solver 在构建 system prompt 时，通过 <文档内容> 标签独立注入。
    """
    document_text = state.get("document_text", "")
    document_filename = state.get("document_filename", "")

    if not document_text:
        return {}

    query = state.get("query", "")

    logger.info(
        f"Document 节点: 文档 {document_filename} 已注入 ({len(document_text)} 字符), "
        f"query 保持原样不污染"
    )

    return {
        "query": query,
        "injected_document": document_text,
        "injected_document_filename": document_filename,
        "document_text": document_text,
        "document_filename": document_filename,
    }
