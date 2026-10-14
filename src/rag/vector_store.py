"""向量存储抽象层：统一接口，支持 ChromaDB / Milvus 一键切换

设计原则：
- 保留原有 ChromaDB 代码不动（在 index_builder.py 中）
- 新增 Milvus 适配层，通过环境变量 VECTOR_STORE_BACKEND 切换
- 统一接口：build(), search(), delete(), count()

向量库对比：
┌────────────────┬────────────────────┬────────────────────┐
│ 特性           │ ChromaDB           │ Milvus             │
├────────────────┼────────────────────┼────────────────────┤
│ 定位           │ 轻量嵌入式          │ 企业级分布式        │
│ 部署           │ 本地文件，零配置     │ 需独立服务 (Docker) │
│ 适合场景       │ 小项目、PoC、开发   │ 海量数据、生产环境  │
│ 索引类型       │ HNSW (内存)         │ IVF/HNSW/ANNOY 等  │
│ 扩展性         │ 单机                │ 水平扩展            │
│ 一致性         │ 最终一致            │ 强一致              │
│ 过滤           │ metadata 过滤       │ 标量字段过滤        │
│ 性能(10万+)    │ 一般                │ 优秀                │
│ 持久化         │ SQLite 文件         │ MinIO/S3 等         │
└────────────────┴────────────────────┴────────────────────┘

使用方式（一键切换）：
  export VECTOR_STORE_BACKEND=chroma   # 默认，使用 ChromaDB
  export VECTOR_STORE_BACKEND=milvus   # 切换到 Milvus
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from src.config import VECTOR_STORE_BACKEND, MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def build_index(self, documents: list[str], metadatas: list[dict], ids: list[str]) -> int:
        """构建/重建索引

        Args:
            documents: 文本内容列表
            metadatas: 元数据列表 [{"source": str, "chunk": int}, ...]
            ids: 唯一 ID 列表

        Returns:
            索引的文档块数量
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """向量相似度检索

        Args:
            query: 查询文本
            top_k: 返回数量
            where: 元数据过滤条件

        Returns:
            [{"content": str, "source": str, "distance": float, "id": str}, ...]
        """
        ...

    @abstractmethod
    def delete_all(self) -> bool:
        """删除所有知识库数据"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回文档块总数"""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """检查向量库是否可用"""
        ...


# ── 工厂函数：一键切换向量库 ──────────────────────────────────────
_vector_store_instance: Optional[VectorStoreBase] = None


def get_vector_store() -> Optional[VectorStoreBase]:
    """获取当前配置的向量存储实例（单例）"""
    global _vector_store_instance

    if _vector_store_instance is not None:
        return _vector_store_instance

    backend = VECTOR_STORE_BACKEND.lower()

    if backend == "milvus":
        try:
            from src.rag.vector_milvus import MilvusStore
            _vector_store_instance = MilvusStore(
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                collection_name=MILVUS_COLLECTION,
            )
            logger.info("向量库后端: Milvus（企业级分布式）")
        except ImportError as e:
            logger.warning(f"Milvus 不可用 ({e})，降级为 ChromaDB")
            backend = "chroma"
        except Exception as e:
            logger.warning(f"Milvus 连接失败 ({e})，降级为 ChromaDB")
            backend = "chroma"

    if backend == "chroma":
        from src.rag.vector_chroma import ChromaStore
        _vector_store_instance = ChromaStore()
        logger.info("向量库后端: ChromaDB（轻量本地）")

    return _vector_store_instance


def reset_vector_store():
    """重置向量存储实例（切换后端时调用）"""
    global _vector_store_instance
    _vector_store_instance = None
