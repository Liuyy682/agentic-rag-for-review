from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_rag import config


class MinioObjectStorage:
    """Small synchronous adapter around the MinIO Python SDK."""

    def __init__(self, client: Any, bucket: str):
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_config(cls) -> "MinioObjectStorage":
        values = {
            "MINIO_ENDPOINT": config.MINIO_ENDPOINT,
            "MINIO_ACCESS_KEY": config.MINIO_ACCESS_KEY,
            "MINIO_SECRET_KEY": config.MINIO_SECRET_KEY,
            "MINIO_BUCKET": config.MINIO_BUCKET,
        }
        missing = [name for name, value in values.items() if not str(value).strip()]
        if missing:
            raise RuntimeError(f"Missing required MinIO configuration: {', '.join(missing)}")

        try:
            from minio import Minio
        except ImportError as exc:
            raise RuntimeError("Install the 'minio' package before starting the RAG app.") from exc

        client = Minio(
            endpoint=config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE,
        )
        storage = cls(client, config.MINIO_BUCKET)
        storage.ensure_bucket()
        return storage

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            raise RuntimeError(
                f"MinIO bucket '{self.bucket}' does not exist; create it before starting the app."
            )

    def upload_file(self, object_key: str, path: str | Path, content_type: str | None = None) -> None:
        kwargs = {"content_type": content_type} if content_type else {}
        self.client.fput_object(self.bucket, object_key, str(path), **kwargs)

    def download_file(self, object_key: str, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, object_key, str(destination))

    def object_exists(self, object_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except Exception as exc:
            code = getattr(exc, "code", "")
            if code in {"NoSuchKey", "NoSuchObject", "NotFound"}:
                return False
            raise

    def delete_object(self, object_key: str) -> None:
        if object_key:
            self.client.remove_object(self.bucket, object_key)

    def delete_objects(self, object_keys: list[str]) -> None:
        keys = [key for key in dict.fromkeys(object_keys) if key]
        if not keys:
            return
        try:
            from minio.deleteobjects import DeleteObject
        except ImportError as exc:
            raise RuntimeError("Install the 'minio' package before deleting objects.") from exc
        errors = list(
            self.client.remove_objects(
                self.bucket,
                (DeleteObject(key) for key in keys),
            )
        )
        if errors:
            detail = "; ".join(f"{item.object_name}: {item.message}" for item in errors)
            raise RuntimeError(f"MinIO object deletion failed: {detail}")


def document_object_keys(document_id: str, original_file: str) -> dict[str, str]:
    safe_name = Path(original_file).name or "upload"
    prefix = f"documents/{document_id}"
    return {
        "raw": f"{prefix}/original/{safe_name}",
        "markdown": f"{prefix}/derived/document.md",
        "images_prefix": f"{prefix}/derived/images",
    }
