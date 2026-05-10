from typing import List

from langchain_core.tools import tool

from retrieval.pipeline import RetrievalPipeline


class ToolFactory:
    def __init__(self, collection, vector_db=None, collection_name=None):
        self.pipeline = RetrievalPipeline(
            collection,
            vector_db=vector_db,
            collection_name=collection_name,
        )
        self.parent_store_manager = self.pipeline.parent_store_manager

    def _format_child_chunk_results(self, results) -> str:
        return self.pipeline.format_child_chunk_results(results)

    def _search_child_chunk_documents(self, query: str, limit: int):
        return self.pipeline.search_child_chunk_documents(query, limit)

    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks."""
        return self.pipeline.search_child_chunks(query, limit)

    def _rerank_child_documents(self, query: str, docs):
        return self.pipeline.rerank_child_documents(query, docs)

    def _context_from_child_doc(self, doc) -> dict:
        return self.pipeline.context_from_child_doc(doc)

    def _parent_contexts(self, parent_ids, fallback_docs):
        return self.pipeline.parent_contexts(parent_ids, fallback_docs)

    def _rag_research(
        self,
        query: str,
        focus=None,
        keep_parent_ids=None,
        exclude_parent_ids=None,
        retry_reason=None,
    ) -> str:
        """Run the deterministic RAG retrieval pipeline for one task."""
        return self.pipeline.rag_research(
            query=query,
            focus=focus,
            keep_parent_ids=keep_parent_ids,
            exclude_parent_ids=exclude_parent_ids,
            retry_reason=retry_reason,
        )

    def set_allowed_source_files(self, source_files=None):
        self.pipeline.set_allowed_source_files(source_files)

    def _retrieve_many_parent_chunks(self, parent_ids) -> str:
        return self.pipeline.retrieve_many_parent_chunks(parent_ids)

    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve a full parent chunk by ID."""
        return self.pipeline.retrieve_parent_chunks(parent_id)

    def create_tools(self) -> List:
        """Create and return the tools exposed to the task executor LLM."""
        rag_tool = tool("rag_research")(self._rag_research)
        return [rag_tool]

    def create_legacy_tools(self) -> List:
        """Create low-level retrieval tools for debugging or legacy experiments."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        return [search_tool, retrieve_tool]
