"""AI 新闻抓取模块：多平台聚合（Hacker News + ArXiv + 知乎 + Reddit），免费无需 API Key"""

import logging
import re
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

AI_KEYWORDS = [
    "AI", "LLM", "GPT", "Claude", "Gemini", "machine learning",
    "deep learning", "transformer", "OpenAI", "Anthropic", "DeepMind",
    "neural network", "language model", "diffusion", "RLHF", "RAG",
    "agent", "copilot", "chatgpt", "llama", "mistral", "groq",
]

ZHIHU_AI_KEYWORDS = [
    "AI", "人工智能", "大模型", "ChatGPT", "GPT", "Claude", "OpenAI",
    "机器学习", "深度学习", "LLM", "智能体", "Agent", "模型",
    "算力", "GPU", "NLP", "计算机视觉", "生成式", "AIGC",
    "自动驾驶", "机器人", "具身智能", "transformer", "推理",
]


def _is_ai_related(title: str) -> bool:
    lower = title.lower()
    return any(kw.lower() in lower for kw in AI_KEYWORDS)


def _jaccard_similarity(a: str, b: str) -> float:
    """计算两个字符串的 Jaccard 相似度（基于字符 2-gram）"""
    def ngrams(s: str, n: int = 2) -> set:
        s = s.lower()
        return {s[i:i+n] for i in range(len(s) - n + 1)}
    set_a, set_b = ngrams(a), ngrams(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate_by_title(items: list[dict], threshold: float = 0.65) -> list[dict]:
    """基于标题 Jaccard 相似度去重，保留先出现的条目"""
    clean_items = []
    for item in items:
        is_dup = False
        for kept in clean_items:
            if _jaccard_similarity(item["title"], kept["title"]) > threshold:
                is_dup = True
                break
        if not is_dup:
            clean_items.append(item)
    return clean_items


# ═══════════════════════════════════════════════════════════════
# 新闻源
# ═══════════════════════════════════════════════════════════════

def fetch_hacker_news_ai(max_items: int = 5) -> list[dict]:
    """从 Hacker News (Algolia API) 抓取 AI 相关热门文章"""
    try:
        url = "https://hn.algolia.com/api/v1/search_by_date"
        params = {
            "query": "AI OR LLM OR GPT OR machine learning OR OpenAI",
            "tags": "story",
            "hitsPerPage": max_items * 3,
        }
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        hits = data.get("hits", [])

        results = []
        for hit in hits:
            title = hit.get("title", "") or hit.get("story_title", "")
            link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            if title and _is_ai_related(title) and link:
                results.append({
                    "title": title.strip(),
                    "link": link,
                    "published": hit.get("created_at", ""),
                    "source": "Hacker News",
                })
            if len(results) >= max_items:
                break

        logger.info(f"Hacker News: 获取 {len(results)} 条 AI 相关文章")
        return results
    except Exception as e:
        logger.error(f"Hacker News 抓取失败: {e}")
        return []


def fetch_arxiv_ai(max_items: int = 5) -> list[dict]:
    """从 ArXiv cs.AI 分类抓取最新论文"""
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": "cat:cs.AI",
            "sortBy": "submittedDate",
            "start": 0,
            "max_results": max_items,
        }
        resp = requests.get(url, params=params, timeout=20)
        root = ET.fromstring(resp.text)

        ns = {"a": "http://www.w3.org/2005/Atom"}
        results = []

        for entry in root.findall("a:entry", ns):
            title_el = entry.find("a:title", ns)
            link_el = entry.find("a:id", ns)
            published_el = entry.find("a:published", ns)

            title_text = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
            link_text = link_el.text.strip() if link_el is not None and link_el.text else ""
            published_text = published_el.text.strip() if published_el is not None and published_el.text else ""

            if title_text and link_text:
                results.append({
                    "title": title_text,
                    "link": link_text,
                    "published": published_text,
                    "source": "ArXiv cs.AI",
                })

        logger.info(f"ArXiv: 获取 {len(results)} 篇论文")
        return results
    except Exception as e:
        logger.error(f"ArXiv 抓取失败: {e}")
        return []


def fetch_zhihu_hot(max_items: int = 5) -> list[dict]:
    """从知乎热榜抓取 AI 相关热门话题"""
    try:
        url = "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WorkMate/1.0)"}
        params = {"limit": 50, "desktop": "true"}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()

        results = []
        for item in data.get("data", []):
            target = item.get("target", {})
            title = target.get("title", "")
            # 提取纯文本（去除 HTML 标签）
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            link = target.get("url", "")
            if not link and target.get("id"):
                link = f"https://www.zhihu.com/question/{target['id']}"

            if title_clean and any(kw.lower() in title_clean.lower() for kw in ZHIHU_AI_KEYWORDS):
                results.append({
                    "title": title_clean,
                    "link": link,
                    "published": "",  # 知乎热榜无精确时间
                    "source": "知乎热榜",
                })

            if len(results) >= max_items:
                break

        logger.info(f"知乎热榜: 获取 {len(results)} 条 AI 相关话题")
        return results
    except Exception as e:
        logger.error(f"知乎热榜抓取失败: {e}")
        return []


def fetch_reddit_ml(max_items: int = 5) -> list[dict]:
    """从 Reddit r/MachineLearning + r/artificial 抓取热门文章"""
    results = []
    subreddits = ["MachineLearning", "artificial"]

    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; WorkMate/1.0)"}
            params = {"limit": 15}
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            data = resp.json()

            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                title = post.get("title", "")
                link = post.get("url", "")
                created_utc = post.get("created_utc", 0)
                # 跳过置顶帖
                if post.get("stickied", False):
                    continue

                if title and link and not title.startswith("["):
                    from datetime import datetime, timezone
                    published = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else ""

                    results.append({
                        "title": title.strip(),
                        "link": link,
                        "published": published,
                        "source": f"Reddit r/{sub}",
                    })

                if len(results) >= max_items:
                    break
        except Exception as e:
            logger.error(f"Reddit r/{sub} 抓取失败: {e}")
            continue

    logger.info(f"Reddit: 获取 {len(results)} 条文章")
    return results[:max_items]


# ═══════════════════════════════════════════════════════════════
# 聚合入口
# ═══════════════════════════════════════════════════════════════

def fetch_all_news(max_total: int = 5) -> list[dict]:
    """从全部新闻源抓取并合并，去重排序，取前 max_total 条"""
    all_news = []

    # 每个源取 2-3 条，保证多样性
    fetchers = [
        (fetch_hacker_news_ai, 3),
        (fetch_arxiv_ai, 2),
        (fetch_zhihu_hot, 2),
        (fetch_reddit_ml, 2),
    ]

    for fetcher, limit in fetchers:
        try:
            items = fetcher(max_items=limit)
            all_news.extend(items)
        except Exception as e:
            logger.warning(f"新闻源 {fetcher.__name__} 异常: {e}")
            continue

    # 去重
    all_news = _deduplicate_by_title(all_news)

    # 按发布时间排序（带时间的在前，无时间的在后）
    def sort_key(item):
        published = item.get("published", "")
        return published if published else "0"

    all_news.sort(key=sort_key, reverse=True)
    return all_news[:max_total]
