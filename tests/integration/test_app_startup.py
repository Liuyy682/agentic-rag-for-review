from fastapi.testclient import TestClient

from agentic_rag.server import app


def test_application_serves_static_homepage_and_api_routes():
    route_paths = set(app.openapi()["paths"])
    assert "/" in route_paths
    assert "/api/chat" in route_paths

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Agentic RAG" in response.text
