import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Force offline mode after dotenv loads, so the transformers background thread
# (auto_conversion → safetensors) does not try to reach huggingface.co.
# Must be a hard assignment, not setdefault — config.py loads dotenv with
# override=True and would undo a setdefault.
os.environ["HF_HUB_OFFLINE"] = "1"

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
