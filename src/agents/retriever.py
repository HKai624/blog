"""检索 Agent：调用检索引擎获取相关文档

v1.1: 使用 retrieve_with_cache() 替代 retrieve()，支持 LRU 缓存
"""

from src.rag.retriever_engine import retrieve_with_cache


def retrieve_docs(state: dict) -> dict:
    query = state.get("rewritten_query", "") or state.get("query", "")
    docs = retrieve_with_cache(query)

    if not docs:
        return {"retrieved_docs": [], "retrieved_context": ""}

    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[文档{i+1} 来源:{doc['source']}]\n{doc['content']}")

    return {
        "retrieved_docs": docs,
        "retrieved_context": "\n\n---\n\n".join(context_parts),
    }
