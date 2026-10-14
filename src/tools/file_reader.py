"""Agent 工具：读取已上传的文档文件（通过 MinerU API 解析）"""

import os
import logging

logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """Agent 可调用的文件读取工具

    读取已上传到 data/uploads/ 目录的文档文件，返回文本内容。
    使用 MinerU API 解析 PDF / Word / Excel / 图片等格式。
    """
    from src.rag.document_loader import load_document, SUPPORTED_EXTENSIONS

    # 允许直接文件名（相对于 uploads 目录）
    upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
    if not os.path.isabs(file_path):
        file_path = os.path.join(upload_dir, file_path)

    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"不支持的文件格式: {ext}（支持: {', '.join(SUPPORTED_EXTENSIONS)}）"

    try:
        text = load_document(file_path)
        filename = os.path.basename(file_path)

        # 截断过长内容
        if len(text) > 4000:
            text = text[:3000] + f"\n\n... [共 {len(text)} 字符，已截断显示前 3000 字符] ..."

        return f"[文件: {filename}]\n{text}"
    except Exception as e:
        logger.error(f"read_file 工具失败: {e}")
        return f"读取文件失败: {e}"
