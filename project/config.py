import os
from dotenv import load_dotenv

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_PROJECT_DIR = os.path.dirname(__file__)
_RUNTIME_DIR = os.path.join(_BASE_DIR, "runtime")

load_dotenv(os.path.join(_BASE_DIR, ".env"))
load_dotenv(os.path.join(_PROJECT_DIR, ".env"), override=True)

HF_CACHE_DIR = os.path.join(_BASE_DIR, ".cache", "huggingface")
_DEFAULT_HF_HOME = os.path.expanduser("~/.cache/huggingface")
if not os.path.exists(_DEFAULT_HF_HOME) or not os.access(_DEFAULT_HF_HOME, os.W_OK):
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(HF_CACHE_DIR, "sentence-transformers"))

MARKDOWN_DIR = os.path.join(_RUNTIME_DIR, "markdown_docs")
MARKDOWN_CLEANED_DIR = os.path.join(_RUNTIME_DIR, "markdown_docs_cleaned")
MARKDOWN_CLEANING_LOG_DIR = os.path.join(_RUNTIME_DIR, "markdown_cleaning_logs")
MARKDOWN_CLEANING_DIFF_DIR = os.path.join(_RUNTIME_DIR, "markdown_cleaning_diffs")
DOCUMENT_IMAGE_DIR = os.path.join(_RUNTIME_DIR, "document_images")
INGESTION_LOG_DIR = os.path.join(_RUNTIME_DIR, "ingestion_logs")
PARENT_STORE_PATH = os.path.join(_RUNTIME_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_RUNTIME_DIR, "qdrant_db")
EVALUATION_REPORTS_DIR = os.path.join(_RUNTIME_DIR, "evaluation_reports")
COURSE_STRUCTURE_PATH = os.path.join(_RUNTIME_DIR, "course_structure.json")
SESSION_MEMORY_PATH = os.path.join(_RUNTIME_DIR, "session_memory.sqlite3")

# --- Database Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag",
)

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Retrieval Fusion Configuration ---
RETRIEVAL_FUSION_MODE = "rrf"
# Options: "qdrant_hybrid", "rrf", "dense", "sparse"
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
RRF_TOP_K = 10
RRF_K = 60
RETRIEVAL_DEBUG = False
RETRIEVAL_CONTEXT_POLICY = os.environ.get("RETRIEVAL_CONTEXT_POLICY", "adaptive").strip().lower()
RETRIEVAL_NEIGHBOR_WINDOW = int(os.environ.get("RETRIEVAL_NEIGHBOR_WINDOW", "1"))
RETRIEVAL_PARENT_EXPAND_MIN_HITS = int(os.environ.get("RETRIEVAL_PARENT_EXPAND_MIN_HITS", "2"))

# --- Cross-Encoder Reranker Configuration ---
RERANKER_ENABLED = True
RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
RERANKER_DEVICE = "auto"
RERANKER_BATCH_SIZE = 8
RERANKER_TOP_N = 40
RERANKER_FINAL_TOP_K = 5
RERANKER_MAX_LENGTH = 512
RERANKER_SCORE_THRESHOLD = None
RERANKER_LOCAL_FILES_ONLY = os.environ.get("RERANKER_LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes", "on"}

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
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
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
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

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

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
PDF_EXTRACT_IMAGES = _env_bool("PDF_EXTRACT_IMAGES", True)
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

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
