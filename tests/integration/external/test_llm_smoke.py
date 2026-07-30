import os

import pytest
from langchain_openai import ChatOpenAI

from agentic_rag import config


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_TESTS") != "1",
    reason="Set RUN_REAL_LLM_TESTS=1 and DEEPSEEK_API_KEY to run the LLM smoke test.",
)


def test_deepseek_returns_non_empty_text():
    if not config.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not configured.")

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        max_tokens=32,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        request_timeout=60,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    response = llm.invoke("请只回复 PONG。")

    assert str(response.content).strip()
