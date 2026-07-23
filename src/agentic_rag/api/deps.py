from agentic_rag.application.rag_application import RagApplication

_rag_app: RagApplication | None = None


def get_rag_app() -> RagApplication:
    global _rag_app
    if _rag_app is None:
        _rag_app = RagApplication.create()
    return _rag_app


def close_rag_app() -> None:
    global _rag_app
    if _rag_app is not None:
        _rag_app.close()
        _rag_app = None
