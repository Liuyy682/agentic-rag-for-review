from chat.chat_interface import ChatInterface
from ingestion.document_manager import DocumentManager
from core.rag_system import RAGSystem


class RagApplication:
    def __init__(self, rag_system, document_manager, chat_interface):
        self.rag_system = rag_system
        self.document_manager = document_manager
        self.chat_interface = chat_interface

    @classmethod
    def create(cls):
        rag_system = RAGSystem()
        rag_system.initialize()
        return cls(
            rag_system=rag_system,
            document_manager=DocumentManager(rag_system),
            chat_interface=ChatInterface(rag_system),
        )
