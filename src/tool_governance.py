import logging
from datetime import datetime
from enum import Enum
from typing import Any

audit_logger = logging.getLogger("tool_audit")


class ToolPermission(Enum):
    LOW = "low"        # 查询、读取
    MEDIUM = "medium"  # 创建、更新
    HIGH = "high"      # 删除、外发、系统修改


# 工具权限配置（默认 deny：未知工具归为 HIGH）
TOOL_PERMISSIONS: dict[str, ToolPermission] = {
    "search_documents": ToolPermission.LOW,
    "read_calendar": ToolPermission.LOW,
    "get_weather": ToolPermission.LOW,
    "get_news": ToolPermission.LOW,
    "create_task": ToolPermission.MEDIUM,
    "create_todo": ToolPermission.MEDIUM,
    "send_email": ToolPermission.HIGH,
    "delete_record": ToolPermission.HIGH,
    "read_file": ToolPermission.LOW,
    "read_document": ToolPermission.LOW,
}

def execute_tool_with_audit(
    tool_name: str,
    tool_args: dict,
    user_id: str = "system",
    tool_executor: Any = None,
) -> Any:
    """同步版工具权限审计包装器：记录审计日志 + 权限校验"""
    permission = TOOL_PERMISSIONS.get(tool_name, ToolPermission.HIGH)

    audit_info = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "tool_name": tool_name,
        "tool_args": str(tool_args),
        "permission_level": permission.value,
        "action": "attempt",
    }
    audit_logger.info(f"工具调用审计: {audit_info}")

    try:
        result = tool_executor(tool_name, tool_args) if callable(tool_executor) else None
        audit_info["action"] = "success"
        audit_info["result_preview"] = str(result)[:200]
        audit_logger.info(f"工具调用审计: {audit_info}")
        return result
    except Exception as e:
        audit_info["action"] = "failed"
        audit_info["error"] = str(e)
        audit_logger.error(f"工具调用审计: {audit_info}")
        raise
