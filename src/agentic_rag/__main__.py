import logging
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path.cwd() / ".env")
os.environ["HF_HUB_OFFLINE"] = "1"


class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()


logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())


def main():
    import uvicorn

    print("\nStarting Agentic RAG server on http://0.0.0.0:7860")
    uvicorn.run("agentic_rag.server:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
