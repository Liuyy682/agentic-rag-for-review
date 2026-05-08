import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from retrieval.fusion import reciprocal_rank_fusion

class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=len(self.__dense_embeddings.embed_query("test")), distance=qmodels.Distance.COSINE),
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def delete_by_parent_ids(self, collection_name: str, parent_ids: list[str]) -> None:
        """Remove child vectors for stale parent chunks during page-level updates."""
        if not parent_ids:
            return
        self.__client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.parent_id",
                            match=qmodels.MatchAny(any=parent_ids),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_by_source_file(self, collection_name: str, source_file: str) -> None:
        """Remove all vectors for a document before its first manifest-backed index."""
        self.__client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.source_file",
                            match=qmodels.MatchValue(value=source_file),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def get_collection(
        self,
        collection_name,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                    client=self.__client,
                    collection_name=collection_name,
                    embedding=self.__dense_embeddings,
                    sparse_embedding=self.__sparse_embeddings,
                    retrieval_mode=retrieval_mode,
                    sparse_vector_name=config.SPARSE_VECTOR_NAME
                )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")

    def dense_search(self, collection_name: str, query: str, k: int):
        """Return dense-only retrieval results."""
        collection = self.get_collection(collection_name, RetrievalMode.DENSE)
        return collection.similarity_search(query, k=k)

    def sparse_search(self, collection_name: str, query: str, k: int):
        """Return sparse/BM25-only retrieval results."""
        collection = self.get_collection(collection_name, RetrievalMode.SPARSE)
        return collection.similarity_search(query, k=k)

    def rrf_search(
        self,
        collection_name: str,
        query: str,
        dense_k: int,
        sparse_k: int,
        fused_k: int,
        rrf_k: int,
    ):
        """Run dense + sparse retrieval separately and fuse results using RRF."""
        dense_results = self.dense_search(collection_name, query, dense_k)
        sparse_results = self.sparse_search(collection_name, query, sparse_k)
        return reciprocal_rank_fusion(
            rankings=[dense_results, sparse_results],
            k=rrf_k,
            top_k=fused_k,
        )
