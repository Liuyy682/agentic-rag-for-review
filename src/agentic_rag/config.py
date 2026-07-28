import os
from dotenv import load_dotenv

# --- Directory Configuration ---
_BASE_DIR = os.path.abspath(os.getcwd())
_RUNTIME_DIR = os.path.join(_BASE_DIR, "runtime")

load_dotenv(os.path.join(_BASE_DIR, ".env"))

# Force offline mode after dotenv loads to prevent the transformers background
# thread (auto_conversion) from trying to reach huggingface.co without network.
os.environ["HF_HUB_OFFLINE"] = "1"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value.strip())


HF_CACHE_DIR = os.path.join(_BASE_DIR, ".cache", "huggingface")
HF_HOME = os.environ.get("HF_HOME", HF_CACHE_DIR)
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", HF_HOME)
SENTENCE_TRANSFORMERS_HOME = os.environ.get("SENTENCE_TRANSFORMERS_HOME", HF_HOME)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", SENTENCE_TRANSFORMERS_HOME)

MARKDOWN_DIR = os.path.join(_RUNTIME_DIR, "markdown_docs")
MARKDOWN_CLEANED_DIR = os.path.join(_RUNTIME_DIR, "markdown_docs_cleaned")
MARKDOWN_CLEANING_LOG_DIR = os.path.join(_RUNTIME_DIR, "markdown_cleaning_logs")
MARKDOWN_CLEANING_DIFF_DIR = os.path.join(_RUNTIME_DIR, "markdown_cleaning_diffs")
DOCUMENT_IMAGE_DIR = os.path.join(_RUNTIME_DIR, "document_images")
INGESTION_LOG_DIR = os.path.join(_RUNTIME_DIR, "ingestion_logs")
INDEX_STATE_DIR = os.path.join(_RUNTIME_DIR, "index_state")

EVALUATION_REPORTS_DIR = os.path.join(_RUNTIME_DIR, "evaluation_reports")
COURSE_STRUCTURE_PATH = os.path.join(INDEX_STATE_DIR, "course_structure.json")
SESSION_MEMORY_PATH = os.path.join(_RUNTIME_DIR, "session_memory.sqlite3")

# --- Authentication ---
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
AUTH_JWT_SECRET = os.environ.get("AUTH_JWT_SECRET", "").strip()
AUTH_TENANT_ID = os.environ.get("AUTH_TENANT_ID", "local-tenant").strip()
AUTH_COOKIE_NAME = "agentic_rag_session"
AUTH_COOKIE_SECURE = _env_bool("AUTH_COOKIE_SECURE", APP_ENV == "production")
AUTH_TOKEN_TTL_SECONDS = _env_int("AUTH_TOKEN_TTL_SECONDS", 7 * 24 * 60 * 60)
INITIAL_ADMIN_EMAIL = os.environ.get("INITIAL_ADMIN_EMAIL", "").strip()
INITIAL_ADMIN_PASSWORD = os.environ.get("INITIAL_ADMIN_PASSWORD", "")

# --- Redis hot conversation memory ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0").strip()
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = _env_int("REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS", 2)
REDIS_SOCKET_TIMEOUT_SECONDS = _env_int("REDIS_SOCKET_TIMEOUT_SECONDS", 2)
MEMORY_TTL_SECONDS = _env_int("MEMORY_TTL_SECONDS", 7 * 24 * 60 * 60)
MEMORY_RECENT_TURNS = _env_int("MEMORY_RECENT_TURNS", 6)
MEMORY_CONTEXT_TOKEN_BUDGET = _env_int("MEMORY_CONTEXT_TOKEN_BUDGET", 6000)
MEMORY_LOCK_WAIT_SECONDS = _env_int("MEMORY_LOCK_WAIT_SECONDS", 3)
MEMORY_LOCK_LEASE_SECONDS = _env_int("MEMORY_LOCK_LEASE_SECONDS", 180)
MEMORY_LOCK_RENEW_INTERVAL_SECONDS = _env_int("MEMORY_LOCK_RENEW_INTERVAL_SECONDS", 30)

# --- Database Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag",
)

# --- Object Storage Configuration ---
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "")
MINIO_SECURE = _env_bool("MINIO_SECURE", True)

# --- Retrieval Fusion Configuration ---
RETRIEVAL_FUSION_MODE = "rrf"
# Options: "rrf", "dense", "sparse"
DENSE_TOP_K = 70
SPARSE_TOP_K = 30
RRF_TOP_K = 20
RRF_K = 60
RETRIEVAL_DEBUG = False
RETRIEVAL_CONTEXT_POLICY = os.environ.get("RETRIEVAL_CONTEXT_POLICY", "adaptive").strip().lower()
RETRIEVAL_NEIGHBOR_WINDOW = int(os.environ.get("RETRIEVAL_NEIGHBOR_WINDOW", "1"))
RETRIEVAL_PARENT_EXPAND_MIN_HITS = int(os.environ.get("RETRIEVAL_PARENT_EXPAND_MIN_HITS", "2"))

# --- Cross-Encoder Reranker Configuration ---
RERANKER_ENABLED = True
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_DEVICE = "auto"
RERANKER_BATCH_SIZE = 8
RERANKER_TOP_N = 40
RERANKER_FINAL_TOP_K = 3
RERANKER_MAX_LENGTH = 512
RERANKER_SCORE_THRESHOLD = 0.6
RERANKER_LOCAL_FILES_ONLY = _env_bool("RERANKER_LOCAL_FILES_ONLY", _env_bool("HF_HUB_OFFLINE", False))

# --- Model Configuration ---
DENSE_MODEL = "BAAI/bge-base-zh-v1.5"
DENSE_EMBEDDING_DIMENSION = 768
DENSE_EMBEDDING_DEVICE = os.environ.get("DENSE_EMBEDDING_DEVICE", "auto")
DENSE_EMBEDDING_BATCH_SIZE = int(os.environ.get("DENSE_EMBEDDING_BATCH_SIZE", "32"))
DENSE_QUERY_INSTRUCTION = os.environ.get("DENSE_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章：")
DENSE_NORMALIZE_EMBEDDINGS = _env_bool("DENSE_NORMALIZE_EMBEDDINGS", True)
DENSE_LOCAL_FILES_ONLY = _env_bool("DENSE_LOCAL_FILES_ONLY", _env_bool("HF_HUB_OFFLINE", False))
SPARSE_RETRIEVAL_BACKEND = "postgres_full_text_jieba"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
MAX_ANSWER_EVALUATION_RETRIES = 2
GRAPH_RECURSION_LIMIT = 50
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = _env_int("CHILD_CHUNK_SIZE", 300)
CHILD_CHUNK_OVERLAP = _env_int("CHILD_CHUNK_OVERLAP", 60)
MIN_PARENT_SIZE = _env_int("MIN_PARENT_SIZE", 2000)
MAX_PARENT_SIZE = _env_int("MAX_PARENT_SIZE", 4000)
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Markdown Cleaning Configuration ---
MARKDOWN_CLEANING_ENABLED = True
HEADER_FOOTER_SCAN_LINES = 3
MIN_REPEAT_PAGES = 3
MIN_REPEAT_RATIO = 0.3

# --- Document Conversion ---
DOCUMENT_CONVERTER = os.environ.get("DOCUMENT_CONVERTER", "markitdown")
SUPPORTED_DOCUMENT_EXTENSIONS = [
    item.strip().lower()
    for item in os.environ.get("SUPPORTED_DOCUMENT_EXTENSIONS", ".pdf,.md,.docx,.pptx").split(",")
    if item.strip()
]
PAGE_LEVEL_INCREMENTAL_INDEXING = _env_bool("PAGE_LEVEL_INCREMENTAL_INDEXING", False)
INGESTION_SKIP_UNCHANGED_FILES = _env_bool("INGESTION_SKIP_UNCHANGED_FILES", True)
INGESTION_STAGE_LOG_ENABLED = _env_bool("INGESTION_STAGE_LOG_ENABLED", True)

# --- Multimodal Document Ingestion ---
# PDF 图片提取：启用后 PDF 通过 pymupdf4llm 转换（同时完成文本+图片提取），
# 禁用则回退到 MarkItDown 纯文本转换（不提取图片）。
PDF_EXTRACT_IMAGES = _env_bool("PDF_EXTRACT_IMAGES", True)
# 提取图片的输出分辨率 (DPI) 和格式 (png / jpg)，仅 pymupdf4llm 路径生效。
PDF_IMAGE_DPI = int(os.environ.get("PDF_IMAGE_DPI", "150"))
PDF_IMAGE_FORMAT = os.environ.get("PDF_IMAGE_FORMAT", "png")

VLM_IMAGE_CAPTION_ENABLED = _env_bool("VLM_IMAGE_CAPTION_ENABLED", False)
LOCAL_VLM_BASE_URL = os.environ.get("LOCAL_VLM_BASE_URL", "http://localhost:8000/v1")
LOCAL_VLM_API_KEY = os.environ.get("LOCAL_VLM_API_KEY", "EMPTY")
LOCAL_VLM_MODEL = os.environ.get("LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
LOCAL_VLM_TIMEOUT_SECONDS = float(os.environ.get("LOCAL_VLM_TIMEOUT_SECONDS", "120"))
LOCAL_VLM_MAX_TOKENS = int(os.environ.get("LOCAL_VLM_MAX_TOKENS", "800"))
VLM_IMAGE_MIN_WIDTH = int(os.environ.get("VLM_IMAGE_MIN_WIDTH", "80"))
VLM_IMAGE_MIN_HEIGHT = int(os.environ.get("VLM_IMAGE_MIN_HEIGHT", "40"))
VLM_IMAGE_CONTEXT_CHARS = int(os.environ.get("VLM_IMAGE_CONTEXT_CHARS", "1200"))
VLM_IMAGE_MAX_PER_DOC = int(os.environ.get("VLM_IMAGE_MAX_PER_DOC", "80"))
VLM_IMAGE_ANALYSIS_WORKERS = int(os.environ.get("VLM_IMAGE_ANALYSIS_WORKERS", "1"))

# --- Image Analysis Engine ---
# 图片分析引擎: "paddleocr", "vlm", "none"
IMAGE_ANALYSIS_ENGINE = os.environ.get("IMAGE_ANALYSIS_ENGINE", "paddleocr")
# PaddleOCR 配置 (当 IMAGE_ANALYSIS_ENGINE=paddleocr 时生效)
PADDLEOCR_LANG = os.environ.get("PADDLEOCR_LANG", "ch")  # ch / en / ch_en
PADDLEOCR_USE_GPU = _env_bool("PADDLEOCR_USE_GPU", False)
# OCR 通用配置
OCR_IMAGE_MIN_WIDTH = int(os.environ.get("OCR_IMAGE_MIN_WIDTH", "80"))
OCR_IMAGE_MIN_HEIGHT = int(os.environ.get("OCR_IMAGE_MIN_HEIGHT", "40"))
OCR_IMAGE_MAX_PER_DOC = int(os.environ.get("OCR_IMAGE_MAX_PER_DOC", "80"))
OCR_IMAGE_ANALYSIS_WORKERS = int(os.environ.get("OCR_IMAGE_ANALYSIS_WORKERS", "2"))

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
