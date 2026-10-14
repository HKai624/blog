import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM 配置 ──────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
MODEL_NAME_REASONER = "deepseek-reasoner"

EMBEDDING_MODEL = "deepseek-chat"
EMBEDDING_DIM = 1024

# ── 智谱 Embedding 配置 ─────────────────────────────────────────
ZHIPU_EMBEDDING_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/embeddings"
ZHIPU_EMBEDDING_MODEL = "embedding-3"
ZHIPU_EMBEDDING_DIM = 2048

# ── RAG 配置 ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RETRIEVAL = 4

# ── RAG 重排序配置 ──────────────────────────────────────────────
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_METHOD = os.getenv("RERANK_METHOD", "hybrid")  # "llm" | "keyword" | "hybrid"
RERANK_CANDIDATE_MULTIPLIER = 3  # 粗排候选数是最终返回数的倍数

# ── RAG 优化配置 (Token 降本) ────────────────────────────────────
# 相似度阈值：distance > 此值才丢弃 (ChromaDB cosine distance: 0=完全相同, 1=正交, 2=完全相反)
# 1.0 表示至少非负相关（cosine similarity >= 0），是合理的默认值
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.0"))
# 最多返回的检索文档数（降低输入 Token）
MAX_RETRIEVED_DOCS = int(os.getenv("MAX_RETRIEVED_DOCS", "2"))
# 每个文档片段最大字符数（截断超长文本，保留核心信息）
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "800"))
# 传入大模型的上下文总 Token 上限（最高 3000）
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
# 大模型回答最大 Token 数
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# ── 混合检索配置 ──────────────────────────────────────────────────
# 是否启用 BM25 + 向量混合检索
BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"
# BM25 权重 (0-1)，剩余为向量检索权重
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# ── 向量库配置 ──────────────────────────────────────────────────
# 向量库后端: "chroma" | "milvus"
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma")
# Milvus 连接配置（使用 Milvus 时生效）
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "workmate_knowledge"

# ── RAG 评估配置 ────────────────────────────────────────────────
EVAL_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "eval", "qrels.json"
)

# ── 数据存储 ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "checkpoints.db")

# ── 飞书基础配置 ──────────────────────────────────────────────
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
FEISHU_ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY", "")

# ── 服务运行配置 ──────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# ── Agent 核心参数 ────────────────────────────────────────────
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "10"))
API_RETRY_TIMES = int(os.getenv("API_RETRY_TIMES", "2"))
SESSION_MAX_LENGTH = int(os.getenv("SESSION_MAX_LENGTH", "8"))
SESSION_EXPIRE_DAYS = int(os.getenv("SESSION_EXPIRE_DAYS", "7"))
SESSION_EXPIRE_SECONDS = SESSION_EXPIRE_DAYS * 24 * 3600

# ── 稳定性治理参数 ────────────────────────────────────────────
FALLBACK_MODELS = ["deepseek-chat", "deepseek-reasoner"]
RATE_LIMIT_PER_USER = int(os.getenv("RATE_LIMIT_PER_USER", "10"))
TOOL_CALL_TIMEOUT = int(os.getenv("TOOL_CALL_TIMEOUT", "30"))

STREAM_ENABLED = True
MAX_HISTORY_MESSAGES = 20

# ── MinerU 文档解析 API 配置 ───────────────────────────────────
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")
MINERU_BASE_URL = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4")

# ── 智谱视觉 + Embedding 配置 ───────────────────────────────────
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
# 视觉模型选择（按视觉效果从强到弱）：
#   glm-4v-plus  — 最强视觉理解（推荐：详细OCR、复杂图表、文档分析）
#   glm-4.6v     — 4.6 系列视觉模型
#   glm-4v       — 基础视觉模型
#   glm-4v-flash — 轻量快速视觉模型
VISION_MODEL = os.getenv("VISION_MODEL", "glm-4v-plus")
VISION_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# ── 图片上传配置 ────────────────────────────────────────────────
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", f"http://localhost:{SERVER_PORT}")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")

# ── 每日早报配置 ──────────────────────────────────────────────
ENABLE_LLM_SUMMARY = os.getenv("ENABLE_LLM_SUMMARY", "true").lower() == "true"
NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", "5"))
