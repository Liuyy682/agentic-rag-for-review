from application.rag_application import RagApplication

_rag_app: RagApplication | None = None


def get_rag_app() -> RagApplication:
    global _rag_app
    if _rag_app is None:
        _rag_app = RagApplication.create()
    return _rag_app
