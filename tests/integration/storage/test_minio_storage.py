import os
import tempfile
import uuid
from pathlib import Path

import pytest

from agentic_rag.storage.object_storage import MinioObjectStorage


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MINIO_TESTS") != "1",
    reason="Set RUN_MINIO_TESTS=1 and MINIO_* variables to run the MinIO smoke test.",
)


def test_minio_upload_download_and_delete_round_trip():
    storage = MinioObjectStorage.from_config()
    object_key = f"documents/smoke-{uuid.uuid4()}/original/smoke.txt"

    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "source.txt"
        target = Path(temp_dir) / "target.txt"
        source.write_text("minio smoke", encoding="utf-8")

        try:
            storage.upload_file(object_key, source, "text/plain")
            assert storage.object_exists(object_key)
            storage.download_file(object_key, target)
            assert target.read_text(encoding="utf-8") == "minio smoke"
        finally:
            storage.delete_object(object_key)

        assert not storage.object_exists(object_key)
