"""BM25 稀疏检索器：关键词匹配 + TF-IDF 变体评分

与向量检索（ChromaDB）互补：
- 向量检索擅长语义相似 → 同义词、近义表达
- BM25 擅长精确关键词匹配 → 专有名词、编号、ID

混合检索 (Hybrid Retrieval) = BM25 × weight + Vector × (1 - weight)
"""

import re
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 评分检索器

    基于 Okapi BM25 公式，适合中文关键词匹配。
    使用 jieba 分词（如已安装）或字符级 bigram 作为降级方案。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数 (默认 1.5)
            b: 文档长度归一化参数 (默认 0.75)
        """
        self.k1 = k1
        self.b = b
        self._docs: list[dict] = []        # [{"content": str, "source": str, ...}]
        self._tokenized: list[list[str]] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}      # document frequency
        self._idf: dict[str, float] = {}
        self._built = False

    # ── 分词 ──────────────────────────────────────────────────────
    @staticmethod
    def tokenize(text: str) -> list[str]:
        """中文分词：优先使用 jieba，不可用时回退为 bigram + 字符混合"""
        try:
            import jieba
            tokens = list(jieba.cut(text.lower()))
            return [t.strip() for t in tokens if t.strip() and not t.isspace()]
        except ImportError:
            pass

        # 降级方案：英文单词 + 中文 bigram + 中文字符
        text_lower = text.lower()
        tokens = []

        # 英文单词
        tokens.extend(re.findall(r"[a-zA-Z0-9]+", text_lower))

        # 中文 bigram
        chinese = re.findall(r"[一-鿿]+", text_lower)
        for segment in chinese:
            if len(segment) <= 2:
                tokens.append(segment)
            else:
                for i in range(len(segment) - 1):
                    tokens.append(segment[i:i + 2])
                # 也保留单字
                tokens.extend(list(segment))

        return tokens

    # ── 构建索引 ──────────────────────────────────────────────────
    def index(self, docs: list[dict]):
        """构建 BM25 索引

        Args:
            docs: 文档列表 [{"content": str, "source": str, ...}]
        """
        if not docs:
            self._built = True
            return

        self._docs = list(docs)
        self._tokenized = [self.tokenize(d["content"]) for d in docs]
        self._doc_len = [len(t) for t in self._tokenized]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0

        # 计算文档频率 (DF)
        self._df = {}
        for tokens in self._tokenized:
            seen = set(tokens)
            for token in seen:
                self._df[token] = self._df.get(token, 0) + 1

        # 计算 IDF
        n = len(docs)
        for token, df in self._df.items():
            self._idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        self._built = True
        logger.info(f"BM25 索引构建完成: {len(docs)} 篇文档, {len(self._df)} 个词项")

    # ── 搜索 ──────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回文档数

        Returns:
            排序后的文档列表 (含 _bm25_score 字段)
        """
        if not self._built or not self._docs:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return self._docs[:top_k]

        scores = []
        for idx, doc_tokens in enumerate(self._tokenized):
            score = 0.0
            doc_len = self._doc_len[idx]

            # 计算 TF（查询词在文档中的频率）
            tf = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1

            for token in query_tokens:
                if token not in self._idf:
                    continue
                idf = self._idf[token]
                tf_t = tf.get(token, 0)
                # BM25 公式
                numerator = tf_t * (self.k1 + 1)
                denominator = tf_t + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self._avgdl, 1.0)
                )
                score += idf * numerator / max(denominator, 1e-10)

            scores.append((score, idx))

        # 按分数降序
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:top_k]

        return [
            {**self._docs[idx], "_bm25_score": score}
            for score, idx in top_scores
            if score > 0 or len(top_scores) <= top_k
        ]


# ── 混合检索 ─────────────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    vector_docs: list[dict],
    bm25: BM25Retriever,
    top_k: int = 4,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """BM25 + 向量混合检索（加权融合排序）

    算法：
    1. 分别获取 BM25 和向量检索的结果
    2. 对两路结果做分数归一化 (min-max)
    3. 加权求和: final_score = bm25_weight * bm25_norm + (1 - bm25_weight) * vector_norm
    4. 按最终分数降序排列

    Args:
        query: 查询文本
        vector_docs: 向量检索结果（含 distance 字段）
        bm25: 已构建的 BM25 检索器实例
        top_k: 最终返回文档数
        bm25_weight: BM25 权重 (0~1)

    Returns:
        混合排序后的文档列表
    """
    # BM25 搜索
    bm25_docs = bm25.search(query, top_k=min(top_k * 2, len(vector_docs) + top_k))
    if not bm25_docs and not vector_docs:
        return []

    # 如果只有一路结果，直接返回
    if not bm25_docs:
        return vector_docs[:top_k]
    if not vector_docs:
        return bm25_docs[:top_k]

    # 构建文档 ID → 分数映射
    doc_scores: dict[str, dict] = {}

    # 向量检索分数归一化（distance 越小越相关，转换为 score）
    vec_dists = [d.get("distance", 1.0) for d in vector_docs if d.get("distance") is not None]
    if vec_dists:
        v_min, v_max = min(vec_dists), max(vec_dists)
        v_range = v_max - v_min if v_max != v_min else 1.0
    else:
        v_min, v_max, v_range = 0, 1, 1

    for rank, doc in enumerate(vector_docs):
        doc_id = doc.get("id", f"v_{rank}")
        dist = doc.get("distance", 1.0) or 1.0
        # distance 越小分越高: normalized = 1 - (d - min) / range
        vec_score = 1.0 - (dist - v_min) / v_range if v_range > 0 else 1.0
        # 原排名衰减
        vec_score += (1.0 / (rank + 1)) * 0.05
        doc_scores[doc_id] = {"doc": doc, "vec_score": vec_score, "bm25_score": 0.0}

    # BM25 分数归一化
    bm25_scores = [d.get("_bm25_score", 0.0) for d in bm25_docs]
    if bm25_scores and max(bm25_scores) > 0:
        b_min, b_max = 0.0, max(bm25_scores)
        b_range = b_max - b_min if b_max != b_min else 1.0
    else:
        b_min, b_max, b_range = 0, 1, 1

    existing_ids = set(doc_scores.keys())
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.get("id", f"b_{rank}")
        raw_score = doc.get("_bm25_score", 0.0)
        bm25_norm = (raw_score - b_min) / b_range if b_range > 0 else 0.0
        bm25_norm += (1.0 / (rank + 1)) * 0.03

        if doc_id in doc_scores:
            doc_scores[doc_id]["bm25_score"] = bm25_norm
        else:
            doc_scores[doc_id] = {
                "doc": doc,
                "vec_score": 0.0,
                "bm25_score": bm25_norm,
            }

    # 加权融合
    fused = []
    for doc_id, scores in doc_scores.items():
        final_score = (
            bm25_weight * scores["bm25_score"]
            + (1 - bm25_weight) * scores["vec_score"]
        )
        fused.append((final_score, scores["doc"]))

    fused.sort(key=lambda x: x[0], reverse=True)

    result = [doc for _, doc in fused[:top_k]]
    if fused and fused[0][0] > 0:
        logger.info(
            f"混合检索: {len(vector_docs)}向量 + {len(bm25_docs)}BM25 "
            f"→ {len(result)}最终 (top_score={fused[0][0]:.3f})"
        )

    return result
