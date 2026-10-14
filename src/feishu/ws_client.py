"""飞书 WebSocket 长连接客户端（基于 lark_oapi SDK）

使用飞书开放平台的长连接模式接收事件，无需公网 URL 和内网穿透。
SDK 自动处理鉴权、保活、重连。
"""

import asyncio
import json
import logging
import threading
import time
from typing import Callable, Awaitable, Any

import lark_oapi as lark
from lark_oapi.api.im.v1.processor import P2ImMessageReceiveV1

logger = logging.getLogger("feishu_ws")


class FeishuWSClient:
    """飞书 WebSocket 长连接客户端

    在独立线程中运行 lark_oapi SDK 的 ws.Client，
    通过同步 handler + asyncio.Queue 将事件桥接到 FastAPI 事件循环。
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        on_event: Callable[[dict], Awaitable[Any]],
    ):
        self._app_id = app_id
        self._app_secret = app_secret
        self._on_event = on_event
        self._running = False
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    def start_background(self) -> None:
        """在后台线程启动 WebSocket 长连接（非阻塞）"""
        if self._running:
            return
        self._running = True
        self._ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self._ws_thread.start()
        logger.info("飞书 WebSocket 后台线程已启动")

    def stop(self) -> None:
        """停止 WebSocket 长连接"""
        self._running = False
        logger.info("飞书 WebSocket 客户端已标记停止")

    def _run_ws_loop(self) -> None:
        """在独立线程中运行 SDK 的 ws.Client"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # 修复 SDK 内部使用模块级 loop 的问题
        import lark_oapi.ws.client as _ws_client
        _ws_client.loop = self._loop

        # 构建事件处理器
        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_im_message)
            .build()
        )

        ws_client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        self._ready.set()

        while self._running:
            try:
                logger.info("飞书 WebSocket 正在连接...")
                ws_client.start()  # 阻塞直到连接断开或异常
            except Exception as e:
                logger.warning(f"飞书 WebSocket 异常: {e}")
            if self._running:
                delay = 5
                logger.info(f"飞书 WebSocket {delay}s 后重连...")
                time.sleep(delay)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop.close()
        logger.info("飞书 WebSocket 事件循环已关闭")

    def _handle_im_message(self, data: P2ImMessageReceiveV1) -> None:
        """SDK 回调（在 SDK 的事件循环内执行），将事件转发给 process_message"""
        try:
            event_json = lark.JSON.marshal(data)
            event_dict: dict = json.loads(event_json) if event_json else {}
            msg = event_dict.get("event", {}).get("message", {})
            msg_type = msg.get("msg_type") or msg.get("message_type", "?")
            chat_id = msg.get("chat_id", "?")
            content = msg.get("content", "")
            text_preview = ""
            if content:
                try:
                    text_preview = json.loads(content).get("text", "")[:100]
                except Exception:
                    text_preview = content[:100]
            print(f"[飞书消息] type={msg_type} chat={chat_id} text={text_preview}")

            # 用 wrapper 包装 on_event，捕获所有异常避免静默失败
            async def _safe_handler():
                try:
                    result = await self._on_event(event_dict)
                    print(f"[飞书回复] done: {result}")
                except Exception as e:
                    logger.error(f"飞书 process_message 异常: {e}", exc_info=True)
                    print(f"[飞书错误] process_message 异常: {e}")

            asyncio.ensure_future(_safe_handler())
        except Exception as e:
            logger.error(f"飞书事件解析异常: {e}", exc_info=True)
            print(f"[飞书错误] 事件解析失败: {e}")
