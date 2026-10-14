"""ChromaDB 向量存储实现：轻量级、本地文件、适合小项目

ChromaDB 特点：
- 嵌入式部署，无需独立服务
- 数据持久化到 SQLite 文件
- HNSW 索引（内存中）
- 支持 metadata 过滤
- 适合 < 10 万文档块的场景

本模块是对 index_builder.py 中 ChromaDB 代码的封装适配层。
原有 ChromaDB 直接调用代码保留在 index_builder.py 中不变。
"""

import logging
from typing import Optional
from src.rag.index_builder import (
    get_collection,
    build_index as _chroma_build_index,
    get_chroma_client,
)
from src.rag.vector_store import VectorStoreBase

logger = logging.getLogger(__name__)


class ChromaStore(VectorStoreBase):
    """ChromaDB 向量存储适配器

    封装 index_builder.py 中的原生 ChromaDB 操作，
    提供统一的 VectorStoreBase 接口。
    """

    def __init__(self):
        self._collection_name = "workmate_knowledge"

    def build_index(self, documents: list[str], metadatas: list[dict], ids: list[str]) -> int:
        """构建/重建索引（委托给 index_builder 原生逻辑）"""
        return _chroma_build_index()

    def search(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """向量相似度检索"""
        collection = get_collection()
        if collection is None or collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where,
        )

        docs = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = {}
                if results.get("metadatas") and results["metadatas"][0]:
                    meta = results["metadatas"][0][i] or {}
                dist = None
                if results.get("distances") and results["distances"][0]:
                    dist = results["distances"][0][i]
                docs.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "distance": dist,
                    "id": f"{meta.get('source', 'unknown')}_{meta.get('chunk', i)}",
                })

        return docs

    def delete_all(self) -> bool:
        """删除 ChromaDB 集合（清空知识库）"""
        try:
            client = get_chroma_client()
            client.delete_collection(self._collection_name)
            logger.info(f"ChromaDB 集合 '{self._collection_name}' 已删除")
            return True
        except Exception as e:
            # 集合不存在也算成功
            if "does not exist" in str(e).lower():
                return True
            logger.error(f"删除 ChromaDB 集合失败: {e}")
            return False

    def count(self) -> int:
        """返回文档块总数"""
        collection = get_collection()
        if collection is None:
            return 0
        return collection.count()

    def is_ready(self) -> bool:
        """检查 ChromaDB 是否可用"""
        try:
            collection = get_collection()
            return collection is not None and collection.count() > 0
        except Exception:
            return False
