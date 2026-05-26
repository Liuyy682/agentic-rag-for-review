import asyncio
import threading
import time
import uuid


class TaskStore:
    """In-memory task tracker for upload progress."""

    def __init__(self):
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": "processing",
                "progress": 0.0,
                "description": "Starting...",
                "created_at": time.time(),
            }
        return task_id

    def update(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            return self._tasks.get(task_id)

    def cleanup_expired(self, max_age_seconds: int = 600):
        now = time.time()
        with self._lock:
            expired = [
                tid for tid, t in self._tasks.items()
                if now - t.get("created_at", 0) > max_age_seconds
            ]
            for tid in expired:
                del self._tasks[tid]

    async def _cleanup_loop(self, interval: int = 60, max_age_seconds: int = 600):
        while True:
            await asyncio.sleep(interval)
            self.cleanup_expired(max_age_seconds)


task_store = TaskStore()
