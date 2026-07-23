from agentic_rag import config
from agentic_rag.chat.chat_interface import ChatInterface
from agentic_rag.chat.hot_session_memory import HotSessionMemoryStore
from agentic_rag.chat.pg_session_memory import PgSessionMemoryStore
from agentic_rag.chat.redis_memory import RedisSessionMemoryCache
from agentic_rag.ingestion.cloud_document_manager import CloudDocumentManager
from agentic_rag.core.rag_system import RAGSystem
from agentic_rag.storage.metadata_repository import PgCourseStructureStore, PgDocumentRepository
from agentic_rag.storage.object_storage import MinioObjectStorage


class RagApplication:
    def __init__(self, rag_system, document_manager, chat_interface):
        self.rag_system = rag_system
        self.document_manager = document_manager
        self.chat_interface = chat_interface

    @classmethod
    def create(cls):
        rag_system = RAGSystem()
        rag_system.initialize()
        object_storage = MinioObjectStorage.from_config()
        repository = PgDocumentRepository()
        course_store = PgCourseStructureStore()
        return cls(
            rag_system=rag_system,
            document_manager=CloudDocumentManager(
                rag_system,
                object_storage=object_storage,
                repository=repository,
                course_store=course_store,
            ),
            chat_interface=ChatInterface(
                rag_system,
                course_store=course_store,
                session_memory=HotSessionMemoryStore(
                    PgSessionMemoryStore(),
                    RedisSessionMemoryCache.from_config(),
                    recent_turns=config.MEMORY_RECENT_TURNS,
                ),
            ),
        )

    def close(self) -> None:
        close = getattr(self.chat_interface.session_memory, "close", None)
        if close:
            close()
