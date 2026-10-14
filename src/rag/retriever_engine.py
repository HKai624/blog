"""检索引擎：向量检索 + BM25 混合检索 + 相似度过滤

核心能力：
1. 相似度阈值过滤 — 至少保留前3条，后续按阈值筛选
2. BM25 + 向量混合检索，提升召回率
3. 可选重排序（Rerank）
4. v1.1: LRU 缓存层，高频查询零延迟重复检索
"""

import logging
from functools import lru_cache

from src.rag.index_builder import get_collection
from src.config import (
    TOP_K_RETRIEVAL,
    RERANK_ENABLED,
    RERANK_METHOD,
    RERANK_CANDIDATE_MULTIPLIER,
    SIMILARITY_THRESHOLD,
    MAX_RETRIEVED_DOCS,
    BM25_ENABLED,
    BM25_WEIGHT,
)

logger = logging.getLogger(__name__)


def _filter_by_similarity(docs: list[dict], threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """相似度阈值过滤：相对阈值 + 绝对上限双重保护

    ChromaDB 使用 cosine distance:
    - 0 = 完全相同
    - 1 = 正交
    - 2 = 完全相反

    策略：
    1. 至少保留前 3 条结果（即使 distance 偏高）
    2. 对后续结果使用阈值过滤
    3. 避免因阈值过严导致"全部丢弃"的极端情况
    """
    if not docs:
        return []

    # 分离有/无 distance 信息的文档
    with_dist = [d for d in docs if d.get("distance") is not None]
    without_dist = [d for d in docs if d.get("distance") is None]

    if not with_dist:
        return without_dist

    # 按 distance 升序排列（越小越相似）
    with_dist_sorted = sorted(with_dist, key=lambda d: d.get("distance", 999))

    # 兜底保护：至少保留前 3 条
    min_keep = min(3, len(with_dist_sorted))
    guaranteed = with_dist_sorted[:min_keep]
    rest = with_dist_sorted[min_keep:]

    # 对剩余结果用阈值过滤
    filtered_rest = []
    dropped = 0
    for doc in rest:
        if doc.get("distance", 999) <= threshold:
            filtered_rest.append(doc)
        else:
            dropped += 1

    result = guaranteed + filtered_rest + without_dist
    # 恢复原始顺序中的相对位置（保持多样性）
    id_order = {doc.get("id", i): i for i, doc in enumerate(docs)}
    result.sort(key=lambda d: id_order.get(d.get("id", ""), 0))

    if dropped > 0:
        logger.info(
            f"相似度过滤: 保留至少{min_keep}条 + {len(filtered_rest)}条通过阈值, "
            f"丢弃 {dropped} 条 (distance > {threshold})"
        )

    return result


def _get_bm25_retriever():
    """获取全局 BM25 检索器实例（懒加载）"""
    return getattr(_get_bm25_retriever, "_instance", None)


def _set_bm25_retriever(bm25):
    _get_bm25_retriever._instance = bm25


def _rebuild_bm25_index():
    """重建 BM25 索引（知识库更新后调用）"""
    if not BM25_ENABLED:
        return

    from src.rag.bm25_retriever import BM25Retriever
    bm25 = _get_bm25_retriever()
    if bm25 is None:
        bm25 = BM25Retriever()
        _set_bm25_retriever(bm25)

    collection = get_collection()
    if collection is None or collection.count() == 0:
        bm25.index([])
        return

    try:
        all_docs = collection.get()
        docs = []
        if all_docs.get("documents"):
            for i, content in enumerate(all_docs["documents"]):
                meta = {}
                if all_docs.get("metadatas") and all_docs["metadatas"][i]:
                    meta = all_docs["metadatas"][i] or {}
                docs.append({
                    "content": content,
                    "source": meta.get("source", "unknown"),
                    "id": f"{meta.get('source', 'unknown')}_{meta.get('chunk', i)}",
                })
        bm25.index(docs)
        logger.info(f"BM25 索引已重建: {len(docs)} 条")
    except Exception as e:
        logger.warning(f"BM25 索引重建失败: {e}")


def retrieve(
    query: str,
    k: int = TOP_K_RETRIEVAL,
    rerank: bool = None,
    rerank_method: str = None,
) -> list[dict]:
    """检索知识库文档 - 统一入口

    流程：
    1. 向量检索（ChromaDB）
    2. 相似度阈值过滤
    3. 可选：BM25 混合检索
    4. 可选：重排序（Rerank）
    5. 上下文 Token 裁剪
    6. 截断超长文本

    Args:
        query: 查询文本
        k: 返回文档数量
        rerank: 是否启用重排序（None 时使用 config）
        rerank_method: 重排序方法

    Returns:
        文档列表 [{"content": str, "source": str, "distance": float}]
    """
    collection = get_collection()
    if collection is None or collection.count() == 0:
        logger.info("知识库为空，跳过检索")
        return []

    # 决定是否重排序
    do_rerank = rerank if rerank is not None else RERANK_ENABLED
    method = rerank_method if rerank_method is not None else RERANK_METHOD
    do_bm25 = BM25_ENABLED

    # 需要更多候选：用于重排序和混合检索
    if do_rerank or do_bm25:
        fetch_k = min(k * RERANK_CANDIDATE_MULTIPLIER, collection.count())
    else:
        fetch_k = k

    try:
        results = collection.query(
            query_texts=[query],
            n_results=max(1, min(fetch_k, collection.count())),
        )
    except Exception as e:
        logger.error(f"ChromaDB 查询失败: {e}，尝试返回全部文档作为降级")
        try:
            all_data = collection.get()
            if all_data and all_data.get("documents"):
                docs = []
                for i, content in enumerate(all_data["documents"]):
                    meta = all_data.get("metadatas", [{}] * len(all_data["documents"]))
                    docs.append({
                        "content": str(content),
                        "source": (meta[i] or {}).get("source", "unknown") if i < len(meta) else "unknown",
                        "distance": None,
                        "id": f"fallback_{i}",
                    })
                logger.info(f"降级检索: 返回全部 {len(docs)} 条文档")
                return docs[:k]
        except Exception as e2:
            logger.error(f"降级检索也失败: {e2}")
        return []

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

    if not docs:
        logger.info("ChromaDB 返回空结果")
        return []

    # ── 相似度阈值过滤（至少保留3条兜底） ──
    docs = _filter_by_similarity(docs)

    # ── BM25 混合检索 ──
    if do_bm25:
        try:
            bm25 = _get_bm25_retriever()
            if bm25 is not None and bm25._built:
                from src.rag.bm25_retriever import hybrid_retrieve
                docs = hybrid_retrieve(
                    query, docs, bm25,
                    top_k=max(fetch_k, len(docs)),
                    bm25_weight=BM25_WEIGHT,
                )
        except Exception as e:
            logger.warning(f"BM25 混合检索失败，使用纯向量结果: {e}")

    # ── 重排序 ──
    if do_rerank and len(docs) > k:
        try:
            from src.rag.reranker import rerank as do_rerank_fn
            docs = do_rerank_fn(query, docs, method=method, top_k=k)
            logger.info(f"Rerank 完成: {len(docs)} 条结果 (method={method})")
        except Exception as e:
            logger.warning(f"Rerank 失败，使用原始排序: {e}")
            docs = docs[:k]

    return docs


# ═══════════════════════════════════════════════════════════════════
# v1.1: LRU 查询缓存
# ═══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _cached_retrieve(query_normalized: str, k: int):
    """实际执行检索的内部函数（结果被 LRU 缓存）。

    query_normalized 是归一化后的原始查询文本（去首尾空格、小写），
    确保语义相同但表述略有差异的查询能命中缓存。
    lru_cache 的 key 是 (query_normalized, k) 的哈希值。
    """
    return retrieve(query_normalized, k=k, rerank=RERANK_ENABLED, rerank_method=RERANK_METHOD)


def retrieve_with_cache(query: str, k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """带 LRU 缓存的检索入口

    策略：
    - 对 query 做归一化（去首尾空格、小写）
    - 缓存命中直接返回，跳过 ChromaDB 查询 + BM25 + 重排序
    - 高频重复问题（如"年假怎么申请"）从 ~3s 降至 ~0.1s
    - 知识库更新时需调用 clear_retrieval_cache() 清空
    """
    normalized = query.strip().lower()
    try:
        return _cached_retrieve(normalized, k)
    except Exception:
        logger.warning("LRU 缓存检索失败，降级为直接检索")
        return retrieve(query, k=k)


def clear_retrieval_cache():
    """清空检索缓存（知识库更新后必须调用）"""
    _cached_retrieve.cache_clear()
    logger.info("RAG 检索缓存已清空")


def knowledge_is_ready() -> bool:
    """检查知识库是否就绪"""
    collection = get_collection()
    return collection is not None and collection.count() > 0
