"""Milvus 向量存储实现：企业级、分布式、适合海量数据

Milvus vs ChromaDB 关键区别：
┌──────────────────────┬──────────────────────┬──────────────────────┐
│                      │ ChromaDB             │ Milvus               │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ 部署方式             │ 嵌入式 (pip install)  │ 需独立服务 (Docker)  │
│ 安装命令             │ pip install chromadb │ pip install pymilvus │
│ 启动服务             │ 无需                  │ docker-compose up -d │
│ 连接方式             │ 本地文件路径          │ host:port 网络连接   │
│ 数据存储             │ ~/data/chroma_db/    │ MinIO/S3 + etcd      │
│ 索引                 │ HNSW (自动)           │ IVF_FLAT/HNSW (可选) │
│ 数据分片             │ 不支持                │ 支持 (shards)        │
│ 副本                 │ 不支持                │ 支持 (replicas)      │
│ 一致性               │ 最终一致              │ 可配 (强/最终)       │
│ 适合数据量           │ < 10 万块             │ 10 万 ~ 亿级         │
│ 性能(大规模)         │ 一般 (单机瓶颈)       │ 优秀 (分布式并行)    │
│ 标量过滤             │ metadata dict         │ 显式 schema 字段     │
│ 动态 schema           │ 支持                  │ 需预定义             │
│ 适用场景             │ 开发/小团队/PoC       │ 生产/企业/大数据     │
└──────────────────────┴──────────────────────┴──────────────────────┘

使用前提：
  export VECTOR_STORE_BACKEND=milvus
  export MILVUS_HOST=localhost
  export MILVUS_PORT=19530

Milvus 服务部署 (Docker):
  wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml
  docker compose -f milvus-standalone-docker-compose.yml up -d

注意：本模块需要安装 pymilvus。如果未安装，vector_store.py 工厂会自动降级到 ChromaDB。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── 条件导入：pymilvus 未安装时给出明确提示 ─────────────────────────
try:
    from pymilvus import (
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )
    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False
    logger.warning(
        "pymilvus 未安装。Milvus 后端不可用，将自动降级为 ChromaDB。\n"
        "安装: pip install pymilvus"
    )

from src.rag.vector_store import VectorStoreBase


class MilvusStore(VectorStoreBase):
    """Milvus 向量存储适配器

    封装 pymilvus 操作，提供统一的 VectorStoreBase 接口。
    与 ChromaDB 的主要区别：
    1. 需要预定义 Collection schema（字段 + 索引）
    2. 连接是显式的，需要 host:port
    3. 数据插入是批量异步的，需要 flush()
    4. 索引创建是显式的，需指定索引类型和参数
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "workmate_knowledge",
    ):
        if not _MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus 未安装。请执行: pip install pymilvus\n"
                "或切换回 ChromaDB: export VECTOR_STORE_BACKEND=chroma"
            )

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._dim = 2048  # 与智谱 embedding-3 维度一致
        self._collection: Optional[Collection] = None

        # 建立连接
        self._connect()

    def _connect(self):
        """建立 Milvus 连接，创建/加载 Collection"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            logger.info(f"Milvus 已连接: {self.host}:{self.port}")

            # 如果 Collection 不存在则创建
            if not utility.has_collection(self.collection_name):
                self._create_collection()

            self._collection = Collection(self.collection_name)
            self._collection.load()
            logger.info(
                f"Milvus Collection '{self.collection_name}' "
                f"已加载 ({self._collection.num_entities} 条)"
            )
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def _create_collection(self):
        """创建 Milvus Collection 并定义 Schema

        与 ChromaDB 的区别：
        - Milvus 需要显式定义每个字段的类型和维度
        - ChromaDB 是动态 schema，自动适配
        """
        # 主键
        pk_field = FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=512,
        )
        # 文本内容
        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
        )
        # 向量字段（与 ChromaDB 不同：Milvus 的向量是独立类型 FieldSchema）
        vector_field = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim,
        )
        # 元数据
        source_field = FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=256,
        )
        chunk_field = FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
        )

        schema = CollectionSchema(
            fields=[pk_field, text_field, vector_field, source_field, chunk_field],
            description="WorkMate 知识库向量存储",
        )

        self._collection = Collection(self.collection_name, schema=schema)

        # 创建 IVF_FLAT 索引（与 ChromaDB HNSW 的区别：更适合大规模数据）
        index_params = {
            "metric_type": "COSINE",  # 与 ChromaDB 的 cosine 一致
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index("embedding", index_params)
        logger.info(f"Milvus Collection '{self.collection_name}' 已创建 (dim={self._dim})")

    def build_index(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> int:
        """构建索引：批量插入文档和向量

        与 ChromaDB 的区别：
        - Milvus 需要显式生成 embedding 后再插入
        - ChromaDB 通过 EmbeddingFunction 自动处理
        """
        if not documents:
            return 0

        # 删除旧数据
        self.delete_all()

        # 生成 embedding（复用智谱 API）
        from src.rag.index_builder import _get_zhipu_embedding

        try:
            embeddings = _get_zhipu_embedding(documents)
        except Exception as e:
            logger.error(f"Milvus 生成 embedding 失败: {e}")
            return 0

        # 准备插入数据
        sources = [m.get("source", "unknown") for m in metadatas]
        chunks = [m.get("chunk", i) for i, m in enumerate(metadatas)]

        insert_data = [ids, documents, embeddings, sources, chunks]

        try:
            mr = self._collection.insert(insert_data)
            self._collection.flush()
            count = mr.insert_count
            logger.info(f"Milvus 索引构建完成: {count} 条")
            return count
        except Exception as e:
            logger.error(f"Milvus 插入数据失败: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """向量相似度检索

        与 ChromaDB 的区别：
        - Milvus 需要手动生成 query embedding
        - Milvus 返回格式不同，需做字段映射
        - Milvus 支持更丰富的标量过滤表达式
        """
        if self._collection is None:
            return []

        # 生成查询向量
        from src.rag.index_builder import _get_zhipu_embedding

        try:
            query_vec = _get_zhipu_embedding([query])[0]
        except Exception as e:
            logger.error(f"Milvus 生成查询向量失败: {e}")
            return []

        # 搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        # 构建过滤表达式（Milvus 使用类 SQL 语法，与 ChromaDB 的 dict 过滤不同）
        expr = None
        if where:
            conditions = []
            for k, v in where.items():
                if isinstance(v, str):
                    conditions.append(f'{k} == "{v}"')
                else:
                    conditions.append(f"{k} == {v}")
            expr = " and ".join(conditions) if conditions else None

        try:
            results = self._collection.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "source", "chunk_index"],
            )

            docs = []
            if results and len(results) > 0:
                for hit in results[0]:
                    docs.append({
                        "content": hit.entity.get("text", ""),
                        "source": hit.entity.get("source", "unknown"),
                        "distance": hit.distance,
                        "id": f"{hit.entity.get('source', 'unknown')}_{hit.entity.get('chunk_index', hit.id)}",
                    })

            return docs

        except Exception as e:
            logger.error(f"Milvus 搜索失败: {e}")
            return []

    def delete_all(self) -> bool:
        """删除所有数据（Milvus 用 drop + recreate，比 ChromaDB 更重量）"""
        if self._collection is None:
            return False

        try:
            # Milvus 不支持清空 Collection，需要删除后重建
            utility.drop_collection(self.collection_name)
            self._create_collection()
            self._collection.load()
            logger.info(f"Milvus Collection '{self.collection_name}' 已清空重建")
            return True
        except Exception as e:
            logger.error(f"Milvus 清空失败: {e}")
            return False

    def count(self) -> int:
        """返回实体总数"""
        if self._collection is None:
            return 0
        try:
            return self._collection.num_entities
        except Exception:
            return 0

    def is_ready(self) -> bool:
        """检查 Milvus 是否可用"""
        try:
            return (
                _MILVUS_AVAILABLE
                and self._collection is not None
                and self._collection.num_entities > 0
            )
        except Exception:
            return False
