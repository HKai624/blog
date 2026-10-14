import os
import hashlib
import struct
import logging
import time
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.config import CHROMA_PERSIST_DIR, KNOWLEDGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.rag.document_loader import load_document, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

COLLECTION_NAME = "workmate_knowledge"

# 智谱 embedding-3 维度: 2048; 降级方案 (字符 n-gram): 384
EMBEDDING_DIM = 2048
EMBEDDING_DIM_FALLBACK = 384


# ── 全局缓存的嵌入函数（保证创建和查询使用同一维度） ──
_cached_embedding_function: Optional[EmbeddingFunction] = None


def reset_embedding_cache():
    """重置嵌入函数缓存（切换 API key 后调用）"""
    global _cached_embedding_function
    _cached_embedding_function = None
    logger.info("嵌入函数缓存已重置")


def _get_zhipu_embedding(texts: list[str]) -> list[list[float]]:
    """调用智谱 Embedding API (embedding-3) 获取 2048 维语义向量"""
    import requests
    from src.config import ZHIPU_API_KEY, ZHIPU_EMBEDDING_ENDPOINT, ZHIPU_EMBEDDING_MODEL

    if not ZHIPU_API_KEY:
        raise ValueError("ZHIPU_API_KEY not set")

    url = ZHIPU_EMBEDDING_ENDPOINT
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ZHIPU_EMBEDDING_MODEL,
        "input": texts,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            else:
                logger.warning(f"智谱 Embedding API returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"智谱 Embedding API attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))

    raise RuntimeError(f"智谱 Embedding API failed after {max_retries} attempts")


class ZhipuEmbeddingFunction(EmbeddingFunction):
    """基于智谱 GLM Embedding API 的语义嵌入函数 (2048-dim)。

    使用智谱 embedding-3 模型，对中文文本有优秀的语义理解能力，
    支持同义词、语义相似度等高级检索需求。
    """
    def __call__(self, texts: Documents) -> list[list[float]]:
        if not texts:
            return []
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(_get_zhipu_embedding(batch))
        return all_embeddings


class SimpleEmbeddingFunction(EmbeddingFunction):
    """轻量级嵌入函数，基于字符 n-gram 哈希，无需下载任何模型。

    作为 DeepSeek 嵌入不可用时的降级方案。维度: 384。
    """
    def __call__(self, texts: Documents) -> list[list[float]]:
        embeddings = []
        for text in texts:
            vec = self._text_to_vector(text)
            embeddings.append(vec)
        return embeddings

    def _text_to_vector(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM_FALLBACK
        text = text.lower()

        for i in range(len(text)):
            bigram = text[i:i+2]
            if len(bigram) == 2:
                h = hashlib.md5(bigram.encode()).digest()
                idx = struct.unpack("<I", h[:4])[0] % EMBEDDING_DIM_FALLBACK
                val = (struct.unpack("<H", h[4:6])[0] / 65535.0) * 2 - 1
                vec[idx] += val * 0.3

        for i in range(len(text)):
            trigram = text[i:i+3]
            if len(trigram) == 3:
                h = hashlib.md5(trigram.encode()).digest()
                idx = struct.unpack("<I", h[:4])[0] % EMBEDDING_DIM_FALLBACK
                val = (struct.unpack("<H", h[4:6])[0] / 65535.0) * 2 - 1
                vec[idx] += val * 0.2

        norm = max(sum(v * v for v in vec), 1e-10) ** 0.5
        return [v / norm for v in vec]


def _create_embedding_function() -> EmbeddingFunction:
    """创建最优可用的嵌入函数：优先使用智谱 Embedding API，不可用时降级为简单哈希

    关键：结果会被全局缓存。一旦选定，不再切换。
    这避免了维度不匹配的问题（Zhipu 2048-dim vs Simple 384-dim）。
    """
    global _cached_embedding_function
    if _cached_embedding_function is not None:
        return _cached_embedding_function

    from src.config import ZHIPU_API_KEY
    if ZHIPU_API_KEY:
        # v1.2: 跳过 API 连通性测试（会阻塞启动 1-3s），直接使用智谱嵌入
        # 若 API 实际不可用，后续 _get_zhipu_embedding 内部的 3 次重试机制会兜底
        logger.info("使用智谱 Embedding API (embedding-3, 2048-dim 语义向量)")
        _cached_embedding_function = ZhipuEmbeddingFunction()
        return _cached_embedding_function
    else:
        logger.info("ZHIPU_API_KEY 未设置，使用字符 n-gram 哈希嵌入 (384-dim)")

    _cached_embedding_function = SimpleEmbeddingFunction()
    return _cached_embedding_function


def get_chroma_client():
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def build_index():
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = _create_embedding_function()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # ── 两阶段切分：标题层级切分 → 递归大小控制 ──
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        add_start_index=True,
    )

    docs_loaded = 0
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fname in files:
            # 跳过隐藏文件
            if fname.startswith("."):
                continue
            # 检查扩展名白名单
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            try:
                text = load_document(fpath)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"加载文档失败 {fname}: {e}")
                continue

            if not text or not text.strip():
                continue

            # 阶段 1：按 Markdown 标题层级切分
            md_chunks = markdown_splitter.split_text(text)

            # 阶段 2：对超大块递归切分，控制块大小 + 保留重叠
            if md_chunks:
                sub_docs = final_splitter.split_documents(md_chunks)
            else:
                # 无标题结构时，直接对全文递归切分
                from langchain_core.documents import Document
                sub_docs = final_splitter.split_documents([Document(page_content=text)])

            chunks = [doc.page_content for doc in sub_docs]
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    ids=[f"{fname}_{i}"],
                    metadatas=[{"source": fname, "chunk": i}],
                )
                docs_loaded += 1

    # ── 重建 BM25 索引（与向量索引保持同步） ──
    try:
        from src.rag.retriever_engine import _rebuild_bm25_index
        _rebuild_bm25_index()
    except Exception as e:
        logger.warning(f"BM25 索引重建跳过: {e}")

    # v1.1: 清空检索缓存（索引已变更，旧缓存失效）
    try:
        from src.rag.retriever_engine import clear_retrieval_cache
        clear_retrieval_cache()
    except Exception:
        pass

    # ── v1.2: 索引构建完成后自动评估检索质量 ──
    try:
        from src.rag.evaluator import evaluate_current_knowledge
        eval_results = evaluate_current_knowledge()
        if "error" not in eval_results:
            from src.config import DATA_DIR
            import json as _json
            metrics_path = os.path.join(DATA_DIR, "metrics.json")
            existing = {}
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as _f:
                        existing = _json.load(_f)
                except Exception:
                    pass
            existing["rag_eval"] = {
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                **eval_results,
            }
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as _f:
                _json.dump(existing, _f, ensure_ascii=False, indent=2)
            logger.info(
                f"RAG 评估完成: recall@1={eval_results.get('recall@1', 'N/A')}, "
                f"recall@3={eval_results.get('recall@3', 'N/A')}, "
                f"mrr={eval_results.get('mrr', 'N/A')}"
            )
        else:
            logger.info(f"跳过 RAG 评估: {eval_results['error']}")
    except Exception as e:
        logger.warning(f"RAG 评估失败（非致命，索引已就绪）: {e}")

    return docs_loaded


def _split_document(text: str, source_name: str) -> list[dict]:
    """将文档文本切分为块，返回 [{documents, ids, metadatas}] 格式的批次列表"""
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        add_start_index=True,
    )

    md_chunks = markdown_splitter.split_text(text)
    if md_chunks:
        sub_docs = final_splitter.split_documents(md_chunks)
    else:
        from langchain_core.documents import Document
        sub_docs = final_splitter.split_documents([Document(page_content=text)])

    chunks = [doc.page_content for doc in sub_docs]
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "documents": [chunk],
            "ids": [f"{source_name}_{i}"],
            "metadatas": [{"source": source_name, "chunk": i}],
        })
    return result


def add_to_index(file_path: str) -> int:
    """增量添加单个文件到向量索引（不重建全库）

    Args:
        file_path: 文件路径

    Returns:
        添加的文本块数量
    """
    import os as _os
    fname = _os.path.basename(file_path)

    # 检查扩展名
    ext = _os.path.splitext(fname)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"不支持的文件格式，跳过: {fname}")
        return 0

    # 加载文档
    try:
        text = load_document(file_path)
    except Exception as e:
        logger.warning(f"加载文档失败 {fname}: {e}")
        return 0

    if not text or not text.strip():
        logger.warning(f"文档内容为空: {fname}")
        return 0

    # 获取或创建 collection
    collection = get_collection()
    if collection is None:
        logger.info("向量集合不存在，执行全量构建")
        return build_index()

    # 删除该文件的旧块（幂等更新）
    try:
        existing = collection.get(where={"source": fname})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
            logger.info(f"已删除 {fname} 的旧索引块 ({len(existing['ids'])} 块)")
    except Exception as e:
        logger.warning(f"删除旧索引块失败（可忽略）: {e}")

    # 切分并添加新块
    batches = _split_document(text, fname)
    added = 0
    for batch in batches:
        try:
            collection.add(
                documents=batch["documents"],
                ids=batch["ids"],
                metadatas=batch["metadatas"],
            )
            added += 1
        except Exception as e:
            logger.warning(f"添加索引块失败 {batch['ids']}: {e}")

    logger.info(f"增量索引完成: {fname} → {added} 块")

    # 同步 BM25
    try:
        from src.rag.retriever_engine import _rebuild_bm25_index
        _rebuild_bm25_index()
    except Exception as e:
        logger.warning(f"BM25 索引同步跳过: {e}")

    # v1.1: 清空检索缓存
    try:
        from src.rag.retriever_engine import clear_retrieval_cache
        clear_retrieval_cache()
    except Exception:
        pass

    return added


def remove_from_index(filename: str) -> int:
    """从向量索引中删除指定文件的所有块

    Args:
        filename: 文件名（与索引中的 source 元数据匹配）

    Returns:
        删除的块数量
    """
    collection = get_collection()
    if collection is None:
        return 0

    try:
        existing = collection.get(where={"source": filename})
        if existing and existing.get("ids"):
            count = len(existing["ids"])
            collection.delete(ids=existing["ids"])
            logger.info(f"已从索引中删除: {filename} ({count} 块)")

            # 同步 BM25
            try:
                from src.rag.retriever_engine import _rebuild_bm25_index
                _rebuild_bm25_index()
            except Exception as e:
                logger.warning(f"BM25 索引同步跳过: {e}")

            return count
    except Exception as e:
        logger.warning(f"从索引删除失败 {filename}: {e}")

    return 0


def get_collection():
    client = get_chroma_client()
    ef = _create_embedding_function()
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return None
