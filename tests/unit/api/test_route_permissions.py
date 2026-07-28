from agentic_rag.api import routes
from agentic_rag.security.auth import get_admin, get_principal


def _route(path, method):
    return next(route for route in routes.router.routes if route.path == path and method in route.methods)


def _dependency_calls(route):
    return {dependency.call for dependency in route.dependant.dependencies}


def test_document_and_task_routes_require_administrator():
    protected = [
        ("/documents/upload", "POST"),
        ("/documents/tasks/{task_id}", "GET"),
        ("/documents/files", "GET"),
        ("/documents/courses", "GET"),
        ("/documents/clear", "POST"),
        ("/documents/courses/rename", "POST"),
        ("/documents/sections/rename", "POST"),
    ]
    for path, method in protected:
        assert get_admin in _dependency_calls(_route(path, method))


def test_chat_and_session_routes_require_authenticated_principal():
    protected = [
        ("/sessions", "GET"),
        ("/sessions", "POST"),
        ("/sessions/{session_id}", "DELETE"),
        ("/sessions/{session_id}/turns", "GET"),
        ("/chat", "POST"),
        ("/chat/clear", "POST"),
        ("/auth/me", "GET"),
        ("/auth/logout", "POST"),
    ]
    for path, method in protected:
        assert get_principal in _dependency_calls(_route(path, method))
