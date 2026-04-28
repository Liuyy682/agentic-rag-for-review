from typing import List
from langchain_core.tools import tool
import config
from db.parent_store_manager import ParentStoreManager

class ToolFactory:
    
    def __init__(self, collection, vector_db=None, collection_name=None):
        self.collection = collection
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.parent_store_manager = ParentStoreManager()

    def _format_child_chunk_results(self, results) -> str:
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
            parts.append(f"Content: {doc.page_content.strip()}")
            formatted_results.append("\n".join(parts))

        return "\n\n".join(formatted_results)
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            limit = limit or config.RRF_TOP_K
            mode = config.RETRIEVAL_FUSION_MODE

            if mode == "rrf":
                if not self.vector_db or not self.collection_name:
                    raise ValueError("RRF retrieval requires vector_db and collection_name")
                results = self.vector_db.rrf_search(
                    collection_name=self.collection_name,
                    query=query,
                    dense_k=config.DENSE_TOP_K,
                    sparse_k=config.SPARSE_TOP_K,
                    fused_k=limit,
                    rrf_k=config.RRF_K,
                )
            elif mode == "dense":
                if not self.vector_db or not self.collection_name:
                    raise ValueError("Dense retrieval requires vector_db and collection_name")
                results = self.vector_db.dense_search(self.collection_name, query, k=limit)
            elif mode == "sparse":
                if not self.vector_db or not self.collection_name:
                    raise ValueError("Sparse retrieval requires vector_db and collection_name")
                results = self.vector_db.sparse_search(self.collection_name, query, k=limit)
            elif mode == "qdrant_hybrid":
                results = self.collection.similarity_search(query, k=limit, score_threshold=0.7)
            else:
                raise ValueError(f"Unsupported retrieval fusion mode: {mode}")

            if not results:
                return "NO_RELEVANT_CHUNKS"

            return self._format_child_chunk_results(results)

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
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
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
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
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]
