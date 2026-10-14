"""RAG 重排序模块：关键词粗排 + LLM 精排 + 混合排序"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def rerank_keyword(query: str, docs: list[dict], top_k: int = 4) -> list[dict]:
    """基于关键词重叠度的轻量级重排序

    评分维度：
    1. 查询词在文档中的覆盖率（TF 得分）
    2. 标题/首段匹配加分
    3. 原始排名衰减（原排名越靠前，轻微加分）
    """
    if not docs:
        return []

    # 提取查询关键词（中文按字符切分，英文按空格切分）
    def tokenize(text: str) -> set:
        text_lower = text.lower()
        # 英文单词
        import re
        words = set(re.findall(r"[a-zA-Z]+", text_lower))
        # 中文单字
        chars = set(re.findall(r"[一-鿿]", text_lower))
        # 中文双字词
        bigrams = set()
        chinese = re.findall(r"[一-鿿]+", text_lower)
        for segment in chinese:
            for i in range(len(segment) - 1):
                bigrams.add(segment[i:i+2])
        return words | chars | bigrams

    query_tokens = tokenize(query)
    if not query_tokens:
        return docs[:top_k]

    scored = []
    for rank, doc in enumerate(docs):
        content = doc.get("content", "")
        doc_tokens = tokenize(content)

        # 关键词覆盖率
        overlap = query_tokens & doc_tokens
        if query_tokens:
            coverage = len(overlap) / len(query_tokens)
        else:
            coverage = 0.0

        # 标题/首段匹配加分
        title_bonus = 0.0
        first_line = content.split("\n")[0] if content else ""
        first_tokens = tokenize(first_line)
        if first_tokens and query_tokens:
            title_bonus = len(query_tokens & first_tokens) / len(query_tokens) * 0.3

        # 原始排名衰减
        rank_decay = 1.0 / (rank + 1) * 0.1

        score = coverage + title_bonus + rank_decay
        scored.append((score, doc))

    # 按得分降序排列
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def rerank_llm(query: str, docs: list[dict], top_k: int = 4) -> list[dict]:
    """使用 DeepSeek LLM 对检索结果重排序

    将候选文档列表发送给 LLM，要求其按与 query 的相关性排序。
    适合对检索质量要求高的场景。
    """
    if not docs or len(docs) <= 1:
        return docs[:top_k]

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

        if not DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY 未设置，降级为关键词重排序")
            return rerank_keyword(query, docs, top_k)

        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0,
        )

        # 构建重排序 prompt
        docs_text = []
        for i, doc in enumerate(docs):
            content = doc.get("content", "")[:500]  # 截断长文本
            source = doc.get("source", f"doc_{i}")
            docs_text.append(f"[{i}] 来源:{source}\n{content}")

        prompt = f"""请根据以下查询，对候选文档按相关性从高到低排序。

查询：{query}

候选文档：
{chr(10).join(docs_text)}

请只输出排序后的文档编号（如：3,0,1,2），不要输出其他内容。"""

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # 解析排序结果
        import re
        indices = [int(x) for x in re.findall(r"\d+", text)]
        indices = [i for i in indices if 0 <= i < len(docs)]

        # 补全未出现的文档
        seen = set(indices)
        for i in range(len(docs)):
            if i not in seen:
                indices.append(i)

        reranked = [docs[i] for i in indices[:top_k]]
        logger.info(f"LLM 重排序完成: {len(docs)} → {len(reranked)}")
        return reranked

    except Exception as e:
        logger.error(f"LLM 重排序失败: {e}，降级为关键词重排序")
        return rerank_keyword(query, docs, top_k)


def rerank(
    query: str,
    docs: list[dict],
    method: str = "hybrid",
    top_k: int = 4,
) -> list[dict]:
    """统一重排序入口

    Args:
        query: 用户查询文本
        docs: 候选文档列表（含 content, source 字段）
        method: 排序方法
            - "keyword": 仅关键词重排序（快速）
            - "llm": 仅 LLM 重排序（准确）
            - "hybrid": 关键词粗排 + LLM 精排（推荐）
        top_k: 最终返回的文档数量

    Returns:
        重排序后的文档列表
    """
    if not docs:
        return []

    if method == "keyword":
        return rerank_keyword(query, docs, top_k)

    if method == "llm":
        return rerank_llm(query, docs, top_k)

    if method == "hybrid":
        # 关键词粗排 → 取 (top_k * 2) 候选 → LLM 精排
        coarse = rerank_keyword(query, docs, top_k=min(top_k * 2, len(docs)))
        return rerank_llm(query, coarse, top_k=top_k)

    logger.warning(f"未知的重排序方法: {method}，使用默认 hybrid")
    return rerank(query, docs, method="hybrid", top_k=top_k)
