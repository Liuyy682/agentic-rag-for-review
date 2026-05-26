import sys
import os
import logging

# Suppress transformers safetensors auto-conversion (background thread tries to
# reach huggingface.co; times out without network, producing noisy tracebacks).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Suppress OTel "Failed to detach context" warning caused by generator/context interaction.
# Tracing is unaffected.
# Known bug: https://github.com/open-telemetry/opentelemetry-python/issues/2606
class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

import uvicorn

if __name__ == "__main__":
    print("\nStarting Agentic RAG server on http://0.0.0.0:7860")
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
