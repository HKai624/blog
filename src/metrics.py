import json
import os
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil 未安装，系统资源监控功能将被禁用")


class AgentMetrics:
    def __init__(self):
        self.total_tasks = 0
        self.success_tasks = 0
        self.tool_calls = 0
        self.tool_success = 0

        self.session_stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "expired_sessions": 0,
            "avg_session_length": 0.0,
            "total_context_tokens": 0,
        }

        self.rag_calls = 0
        self.rag_recall_success = 0
        self.rag_total_latency_ms = 0.0
        self.rag_human_verified = 0
        self.rag_human_accurate = 0
        # RAG 评估指标
        self.rag_eval_recall_at_3 = 0.0
        self.rag_eval_precision_at_3 = 0.0
        self.rag_eval_mrr = 0.0

        self.start_time = time.time()
        self._is_dirty = False

    def record_task(
        self,
        is_success: bool,
        tool_info: Optional[dict] = None,
        context_tokens: int = 0,
    ):
        tool_info = tool_info or {"total": 0, "success": 0}
        self.total_tasks += 1
        if is_success:
            self.success_tasks += 1
        self.tool_calls += tool_info.get("total", 0)
        self.tool_success += tool_info.get("success", 0)
        self.session_stats["total_context_tokens"] += context_tokens
        self._is_dirty = True

    def record_rag_query(self, is_recall_success: bool, latency_ms: float):
        self.rag_calls += 1
        if is_recall_success:
            self.rag_recall_success += 1
        self.rag_total_latency_ms += latency_ms
        self._is_dirty = True

    def record_rag_human_verification(self, is_accurate: bool):
        self.rag_human_verified += 1
        if is_accurate:
            self.rag_human_accurate += 1
        self._is_dirty = True

    def record_rag_eval(self, recall_at_3: float, precision_at_3: float, mrr: float):
        self.rag_eval_recall_at_3 = recall_at_3
        self.rag_eval_precision_at_3 = precision_at_3
        self.rag_eval_mrr = mrr
        self._is_dirty = True

    def update_session_stats(self, session_manager_dict: dict):
        self.session_stats["total_sessions"] = len(session_manager_dict)
        active = 0
        total_length = 0

        for sess in session_manager_dict.values():
            if hasattr(sess, "is_expired") and not sess.is_expired():
                active += 1
            if hasattr(sess, "history"):
                total_length += len(sess.history)

        self.session_stats["active_sessions"] = active
        self.session_stats["expired_sessions"] = (
            self.session_stats["total_sessions"] - active
        )
        if self.session_stats["total_sessions"] > 0:
            self.session_stats["avg_session_length"] = round(
                total_length / self.session_stats["total_sessions"], 1
            )
        self._is_dirty = True

    def save_metrics_to_file(
        self, filepath: str = "metrics.json", force: bool = False
    ):
        if not self._is_dirty and not force:
            return

        recall_rate = (
            f"{(self.rag_recall_success / self.rag_calls * 100):.2f}%"
            if self.rag_calls else "0%"
        )
        avg_latency = (
            f"{(self.rag_total_latency_ms / self.rag_calls):.2f}ms"
            if self.rag_calls else "0ms"
        )
        accuracy_rate = (
            f"{(self.rag_human_accurate / self.rag_human_verified * 100):.2f}%"
            if self.rag_human_verified else "待人工校验"
        )

        system_res = {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0}
        if psutil:
            disk_path = "C:\\"
            system_res = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(disk_path).percent,
            }

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "system_resources": system_res,
            "task_metrics": {
                "task_completion_rate": (
                    f"{(self.success_tasks / self.total_tasks * 100):.2f}%"
                    if self.total_tasks else "0%"
                ),
                "total_tasks": self.total_tasks,
                "success_tasks": self.success_tasks,
            },
            "tool_metrics": {
                "tool_accuracy_rate": (
                    f"{(self.tool_success / self.tool_calls * 100):.2f}%"
                    if self.tool_calls else "0%"
                ),
                "total_tool_calls": self.tool_calls,
                "successful_tool_calls": self.tool_success,
            },
            "rag_metrics": {
                "retrieval_recall_rate": recall_rate,
                "avg_response_latency": avg_latency,
                "answer_accuracy_rate": accuracy_rate,
                "total_rag_calls": self.rag_calls,
                "total_human_reviews": self.rag_human_verified,
                "eval_recall@3": self.rag_eval_recall_at_3,
                "eval_precision@3": self.rag_eval_precision_at_3,
                "eval_mrr": self.rag_eval_mrr,
            },
            "session_metrics": self.session_stats,
            "cost_metrics": {
                "estimated_total_tokens": self.session_stats["total_context_tokens"],
                "estimated_cost": f"${self.session_stats['total_context_tokens'] * 0.00001:.4f}",
            },
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            self._is_dirty = False
        except IOError as e:
            logger.error(f"指标数据持久化失败: {e}")


metrics = AgentMetrics()
