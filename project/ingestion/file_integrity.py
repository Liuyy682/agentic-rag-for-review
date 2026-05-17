from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hash of a local file without loading it all at once."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
