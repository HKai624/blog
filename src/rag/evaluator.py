"""RAG 检索评估模块：Recall@k / Precision@k / MRR / NDCG@k"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@k：相关文档中被检索到的比例"""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Precision@k：检索结果中相关文档的比例"""
    if k <= 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / min(k, len(retrieved_set))


def mean_reciprocal_rank(all_retrieved: list[list[str]], all_relevant: list[list[str]]) -> float:
    """Mean Reciprocal Rank (MRR)：第一个相关文档排名的倒数的平均值"""
    if not all_retrieved:
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        relevant_set = set(relevant)
        rank = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                rank = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k

    使用二元相关性（related=1, not=0），计算 DCG / IDCG。
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    k = min(k, len(retrieved_ids))

    # DCG
    dcg = 0.0
    for i in range(k):
        if retrieved_ids[i] in relevant_set:
            # 位置从 1 开始，log2(i+1) 的折扣
            dcg += 1.0 / (__import__("math").log2(i + 2))

    # IDCG（理想排序：所有相关文档在前 k 位）
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / (__import__("math").log2(i + 2))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    queries: list[str],
    ground_truth: list[dict],
    retrieve_fn,
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """对检索系统进行完整评估

    Args:
        queries: 查询文本列表
        ground_truth: 真实相关文档列表，格式 [{"query": "年假怎么申请", "relevant_docs": ["企业制度FAQ.txt_0"]}]
        retrieve_fn: 检索函数，签名为 fn(query: str, k: int) -> list[dict]，每个 dict 需含 "id" 或 "source"
        k_values: 需要评估的 k 值列表

    Returns:
        包含各指标得分的字典
    """
    if len(queries) != len(ground_truth):
        raise ValueError(f"查询数量 ({len(queries)}) 与标注数量 ({len(ground_truth)}) 不匹配")

    # 构建 query → relevant_ids 的映射
    gt_map = {}
    for item in ground_truth:
        gt_map[item["query"]] = set(item.get("relevant_docs", []))

    all_retrieved_ids = []
    all_relevant_ids = []

    for query in queries:
        relevant_ids = list(gt_map.get(query, set()))
        all_relevant_ids.append(relevant_ids)

        # 检索：取最大 k 值的结果
        max_k = max(k_values)
        results = retrieve_fn(query, k=max_k)
        retrieved_ids = [r.get("id", r.get("source", f"doc_{i}")) for i, r in enumerate(results)]
        all_retrieved_ids.append(retrieved_ids)

    # 计算各指标
    metrics = {}

    for k in k_values:
        recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]
        precisions = [precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]
        ndcgs = [ndcg_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]

        metrics[f"recall@{k}"] = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        metrics[f"precision@{k}"] = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
        metrics[f"ndcg@{k}"] = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0

    metrics["mrr"] = round(mean_reciprocal_rank(all_retrieved_ids, all_relevant_ids), 4)

    return metrics


def load_eval_dataset(file_path: str) -> list[dict]:
    """从 JSON 文件加载评估数据集"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载评估数据集: {file_path} → {len(data)} 条标注")
    return data


# ── 便捷函数：从 ChromaDB 直接评估 ──────────────────────────────

def evaluate_current_knowledge(
    k_values: list[int] = [1, 3, 5],
    eval_path: Optional[str] = None,
) -> dict:
    """评估当前知识库的检索效果

    Args:
        k_values: 评估的 k 值列表
        eval_path: 评估数据集路径，默认使用 config 中的 EVAL_DATASET_PATH

    Returns:
        评估指标字典
    """
    from src.config import EVAL_DATASET_PATH
    from src.rag.retriever_engine import retrieve

    path = eval_path or EVAL_DATASET_PATH

    try:
        dataset = load_eval_dataset(path)
    except FileNotFoundError:
        logger.warning(f"评估数据集不存在: {path}，跳过评估")
        return {"error": f"评估数据集不存在: {path}"}
    except Exception as e:
        logger.error(f"加载评估数据集失败: {e}")
        return {"error": str(e)}

    queries = [item["query"] for item in dataset]

    def _retrieve(query: str, k: int) -> list[dict]:
        docs = retrieve(query, k=k, rerank=False)
        return [{"id": d.get("id", d.get("source", f"doc_{i}")), "source": d.get("source", "")} for i, d in enumerate(docs)]

    metrics = evaluate_retrieval(queries, dataset, _retrieve, k_values=k_values)
    logger.info(f"RAG 评估完成: {metrics}")
    return metrics
