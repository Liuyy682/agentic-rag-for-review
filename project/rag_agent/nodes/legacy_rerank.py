import logging

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

import config
from retrieval.reranker import RerankerUnavailable, get_reranker

from ..graph_state import AgentState

logger = logging.getLogger(__name__)


def _parse_child_chunk_output(text: str) -> list[dict]:
    if not text:
        return []
    if text.startswith("NO_RELEVANT_CHUNKS") or text.startswith("RETRIEVAL_ERROR"):
        return []

    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("Parent ID: ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    parsed: list[dict] = []
    for block in blocks:
        parent_id = ""
        file_name = ""
        content_idx = None
        for idx, line in enumerate(block):
            if line.startswith("Parent ID: "):
                parent_id = line[len("Parent ID: "):].strip()
            elif line.startswith("File Name: "):
                file_name = line[len("File Name: "):].strip()
            elif line.startswith("Content:"):
                content_idx = idx
                break

        if content_idx is None:
            continue

        content_line = block[content_idx]
        content_first = content_line[len("Content:"):].lstrip()
        content_tail = block[content_idx + 1:]
        content = "\n".join([content_first] + content_tail).strip()

        extra_lines: list[str] = []
        for line in block:
            if line.startswith("Parent ID: "):
                continue
            if line.startswith("File Name: "):
                continue
            if line.startswith("Content:"):
                continue
            if line.startswith("Rerank Score:") or line.startswith("Rerank Rank:"):
                continue
            if not line.strip():
                continue
            extra_lines.append(line)

        parsed.append({
            "parent_id": parent_id,
            "file_name": file_name,
            "content": content,
            "extra_lines": extra_lines,
        })

    return parsed


def _format_child_chunk_output(docs: list[Document]) -> str:
    blocks: list[str] = []
    for doc in docs:
        metadata = doc.metadata or {}
        lines = [
            f"Parent ID: {metadata.get('parent_id', '')}",
            f"File Name: {metadata.get('source', '')}",
        ]
        extra_lines = metadata.get("_extra_lines") or []
        lines.extend(extra_lines)

        if "rerank_score" in metadata:
            try:
                lines.append(f"Rerank Score: {float(metadata.get('rerank_score')):.6f}")
            except (TypeError, ValueError):
                lines.append(f"Rerank Score: {metadata.get('rerank_score')}")
        if config.RETRIEVAL_DEBUG and "rerank_rank" in metadata:
            lines.append(f"Rerank Rank: {metadata.get('rerank_rank')}")

        content = doc.page_content or ""
        content_lines = content.splitlines()
        if content_lines:
            lines.append(f"Content: {content_lines[0]}")
            lines.extend(content_lines[1:])
        else:
            lines.append("Content: ")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def rerank_search_results(state: AgentState):
    if not config.RERANKER_ENABLED:
        return {}

    tool_calls = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls = msg.tool_calls or []
            break

    if not tool_calls:
        return {}

    queries: list[str] = []
    query_by_call_id: dict[str, str] = {}
    for call in tool_calls:
        if call.get("name") != "search_child_chunks":
            continue
        query = (call.get("args") or {}).get("query")
        if query:
            queries.append(query)
            if call.get("id"):
                query_by_call_id[call["id"]] = query

    if not queries:
        return {}

    updates: list = []
    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage) or getattr(msg, "name", "") != "search_child_chunks":
            continue

        tool_call_id = getattr(msg, "tool_call_id", None)
        query = None
        if tool_call_id and tool_call_id in query_by_call_id:
            query = query_by_call_id[tool_call_id]
        elif len(queries) == 1:
            query = queries[0]
        else:
            query = state.get("question", "")

        if not query:
            continue

        parsed = _parse_child_chunk_output(msg.content)
        if not parsed:
            continue

        docs: list[Document] = []
        for item in parsed[: config.RERANKER_TOP_N]:
            docs.append(
                Document(
                    page_content=item["content"],
                    metadata={
                        "parent_id": item["parent_id"],
                        "source": item["file_name"],
                        "_extra_lines": item["extra_lines"],
                    },
                )
            )

        top_k = min(len(docs), config.RERANKER_FINAL_TOP_K)
        if top_k <= 0:
            continue

        try:
            reranked = get_reranker().rerank(
                query=query,
                documents=docs,
                top_k=top_k,
                score_threshold=config.RERANKER_SCORE_THRESHOLD,
            )
        except RerankerUnavailable as exc:
            logger.warning(str(exc))
            reranked = docs[:top_k]
        except Exception:
            logger.exception("Rerank failed during scoring; using original retrieval order")
            reranked = docs[:top_k]

        new_content = _format_child_chunk_output(reranked)
        updates.append(ToolMessage(content=new_content, name=msg.name, tool_call_id=tool_call_id, id=msg.id))

    return {"messages": updates} if updates else {}
