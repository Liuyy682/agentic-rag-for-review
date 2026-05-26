import uuid
from langchain_openai import ChatOpenAI
import config
from storage.pg_vector_store import PgVectorManager
from storage.pg_parent_store import PgParentStoreManager
from ingestion.chunking import DocumentChunker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from observability.langfuse import Observability

class RAGSystem:

    def __init__(self):
        self.vector_db = PgVectorManager()
        self.parent_store = PgParentStoreManager()
        self.chunker = DocumentChunker()
        self.observability = Observability()
        self.agent_graph = None
        self.tool_factory = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("Set DEEPSEEK_API_KEY in project/.env before starting the RAG app.")

        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            request_timeout=120,
            extra_body={"thinking": {"type": "disabled"}},
        )
        self.tool_factory = ToolFactory(
            vector_db=self.vector_db,
            parent_store_manager=self.parent_store,
        )
        tools = self.tool_factory.create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def set_course_scope(self, source_files=None):
        if self.tool_factory:
            self.tool_factory.set_allowed_source_files(source_files)

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())
