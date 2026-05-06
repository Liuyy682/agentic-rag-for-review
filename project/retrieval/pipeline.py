import json
import logging
from typing import List, Optional

from langchain_core.documents import Document

import config
from db.parent_store_manager import ParentStoreManager
from rag_agent.reranker import get_reranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, collection, vector_db=None, collection_name=None, parent_store_manager=None):
        self.collection = collection
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.parent_store_manager = parent_store_manager or ParentStoreManager()

    def format_child_chunk_results(self, results) -> str:
        formatted_results = []
        for doc in results:
            parts = [
                f"Parent ID: {doc.metadata.get('parent_id', '')}",
                f"File Name: {doc.metadata.get('source', '')}",
            ]
            if config.RETRIEVAL_DEBUG:
                if "rrf_score" in doc.metadata:
                    parts.append(f"RRF Score: {doc.metadata.get('rrf_score'):.6f}")
                if "rrf_rank_details" in doc.metadata:
                    parts.append(f"RRF Rank Details: {doc.metadata.get('rrf_rank_details')}")
            if "rerank_score" in doc.metadata:
                try:
                    parts.append(f"Rerank Score: {float(doc.metadata.get('rerank_score')):.6f}")
                except (TypeError, ValueError):
                    parts.append(f"Rerank Score: {doc.metadata.get('rerank_score')}")
            if config.RETRIEVAL_DEBUG and "rerank_rank" in doc.metadata:
                parts.append(f"Rerank Rank: {doc.metadata.get('rerank_rank')}")
            parts.append(f"Content: {doc.page_content.strip()}")
            formatted_results.append("\n".join(parts))

        return "\n\n".join(formatted_results)

    def search_child_chunk_documents(self, query: str, limit: int) -> List[Document]:
        limit = limit or config.RRF_TOP_K
        retrieval_limit = limit
        if config.RERANKER_ENABLED:
            retrieval_limit = max(retrieval_limit, config.RERANKER_TOP_N)
        mode = config.RETRIEVAL_FUSION_MODE

        if mode == "rrf":
            if not self.vector_db or not self.collection_name:
                raise ValueError("RRF retrieval requires vector_db and collection_name")
            results = self.vector_db.rrf_search(
                collection_name=self.collection_name,
                query=query,
                dense_k=config.DENSE_TOP_K,
                sparse_k=config.SPARSE_TOP_K,
                fused_k=retrieval_limit,
                rrf_k=config.RRF_K,
            )
        elif mode == "dense":
            if not self.vector_db or not self.collection_name:
                raise ValueError("Dense retrieval requires vector_db and collection_name")
            results = self.vector_db.dense_search(self.collection_name, query, k=retrieval_limit)
        elif mode == "sparse":
            if not self.vector_db or not self.collection_name:
                raise ValueError("Sparse retrieval requires vector_db and collection_name")
            results = self.vector_db.sparse_search(self.collection_name, query, k=retrieval_limit)
        elif mode == "qdrant_hybrid":
            results = self.collection.similarity_search(query, k=retrieval_limit, score_threshold=0.7)
        else:
            raise ValueError(f"Unsupported retrieval fusion mode: {mode}")

        return list(results or [])[:retrieval_limit]

    def search_child_chunks(self, query: str, limit: int) -> str:
        try:
            results = self.search_child_chunk_documents(query, limit)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            return self.format_child_chunk_results(results)

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    def rerank_child_documents(self, query: str, docs: List[Document]) -> List[Document]:
        if not config.RERANKER_ENABLED or not docs:
            return docs[: config.RERANKER_FINAL_TOP_K]

        candidates = docs[: config.RERANKER_TOP_N]
        top_k = min(len(candidates), config.RERANKER_FINAL_TOP_K)
        if top_k <= 0:
            return []

        try:
            return get_reranker().rerank(
                query=query,
                documents=candidates,
                top_k=top_k,
                score_threshold=config.RERANKER_SCORE_THRESHOLD,
            )
        except Exception:
            logger.exception("Rag research rerank failed; using original retrieval order")
            return candidates[:top_k]

    def context_from_child_doc(self, doc: Document) -> dict:
        metadata = doc.metadata or {}
        score = metadata.get("rerank_score") or metadata.get("rrf_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        return {
            "parent_id": metadata.get("parent_id", ""),
            "source": metadata.get("source", ""),
            "content": doc.page_content or "",
            "score": score,
        }

    def parent_contexts(self, parent_ids: List[str], fallback_docs: List[Document]) -> List[dict]:
        contexts: List[dict] = []
        fallback_by_parent = {
            (doc.metadata or {}).get("parent_id", ""): doc
            for doc in fallback_docs
            if (doc.metadata or {}).get("parent_id")
        }

        try:
            raw_parents = self.parent_store_manager.load_content_many(parent_ids)
        except Exception:
            logger.exception("Parent retrieval failed during rag_research")
            raw_parents = []

        seen = set()
        for parent in raw_parents or []:
            parent_id = parent.get("parent_id", "")
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            contexts.append({
                "parent_id": parent_id,
                "source": parent.get("metadata", {}).get("source", "unknown"),
                "content": parent.get("content", "").strip(),
                "score": None,
            })

        for parent_id in parent_ids:
            if parent_id in seen:
                continue
            fallback = fallback_by_parent.get(parent_id)
            if fallback:
                seen.add(parent_id)
                contexts.append(self.context_from_child_doc(fallback))

        return contexts

    def rag_research(
        self,
        query: str,
        focus: Optional[str] = None,
        keep_parent_ids: Optional[List[str]] = None,
        exclude_parent_ids: Optional[List[str]] = None,
        retry_reason: Optional[str] = None,
    ) -> str:
        keep_parent_ids = keep_parent_ids or []
        exclude_parent_ids = set(exclude_parent_ids or [])
        effective_query = f"{query}\nFocus: {focus}" if focus else query
        diagnostics = {
            "retry_reason": retry_reason or "",
            "retrieval_mode": config.RETRIEVAL_FUSION_MODE,
            "reranker_enabled": config.RERANKER_ENABLED,
        }

        try:
            child_docs = self.search_child_chunk_documents(effective_query, config.RRF_TOP_K)
            diagnostics["child_candidates"] = len(child_docs)
            child_docs = [
                doc for doc in child_docs
                if (doc.metadata or {}).get("parent_id", "") not in exclude_parent_ids
            ]
            reranked_docs = self.rerank_child_documents(effective_query, child_docs)
            diagnostics["reranked_contexts"] = len(reranked_docs)

            parent_ids = []
            for parent_id in keep_parent_ids:
                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)
            for doc in reranked_docs:
                parent_id = (doc.metadata or {}).get("parent_id", "")
                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)

            contexts = self.parent_contexts(parent_ids, reranked_docs)
            sources = sorted({ctx["source"] for ctx in contexts if ctx.get("source")})
            gaps = [] if contexts else ["No relevant document context was retrieved."]
            diagnostics["parent_ids"] = len(parent_ids)
            diagnostics["contexts"] = len(contexts)

            return json.dumps({
                "query": query,
                "focus": focus or "",
                "contexts": contexts,
                "sources": sources,
                "parent_ids": parent_ids,
                "gaps": gaps,
                "diagnostics": diagnostics,
            }, ensure_ascii=False)
        except Exception as e:
            diagnostics["error"] = str(e)
            return json.dumps({
                "query": query,
                "focus": focus or "",
                "contexts": [],
                "sources": [],
                "parent_ids": keep_parent_ids,
                "gaps": [f"RAG_RESEARCH_ERROR: {str(e)}"],
                "diagnostics": diagnostics,
            }, ensure_ascii=False)

    def retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    def retrieve_parent_chunks(self, parent_id: str) -> str:
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
