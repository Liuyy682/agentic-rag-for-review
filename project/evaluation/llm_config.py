import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not value:
        raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY before running evaluation.")
    return value


def base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL


def answer_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.environ.get("ANSWER_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL


def judge_model() -> str:
    return os.environ.get("RAGAS_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
