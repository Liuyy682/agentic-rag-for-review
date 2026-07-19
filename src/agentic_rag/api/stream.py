import asyncio
import json
import threading
from typing import AsyncGenerator


def _format_sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _put_sse_str(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, sse: str):
    asyncio.run_coroutine_threadsafe(queue.put(sse), loop)


async def stream_chat(
    chat_interface,
    message: str,
    history: list[dict],
    course_name: str | None,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Bridge the sync ChatInterface.chat() generator to async SSE StreamingResponse."""
    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    def _run_sync():
        try:
            generator = chat_interface.chat(
                message=message,
                history=history,
                course_name=course_name,
                session_id=session_id,
            )
            for item in generator:
                if cancel_event.is_set():
                    break
                if isinstance(item, str):
                    _put_sse_str(queue, loop, _format_sse("error", json.dumps({"message": item}, ensure_ascii=False)))
                    break
                # Send the full message list. list() snapshots the list but dict
                # values are captured by json.dumps atomically in this same thread.
                payload = json.dumps(list(item), ensure_ascii=False)
                _put_sse_str(queue, loop, _format_sse("messages", payload))
            _put_sse_str(queue, loop, _format_sse("done", "{}"))
        except Exception as e:
            _put_sse_str(queue, loop, _format_sse("error", json.dumps({"message": str(e)}, ensure_ascii=False)))
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop.run_in_executor(None, _run_sync)

    try:
        while True:
            sse_str = await queue.get()
            if sse_str is None:
                break
            yield sse_str
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        cancel_event.set()
        raise
