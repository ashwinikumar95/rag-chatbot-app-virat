# tests/test_api.py - API endpoint tests
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_returns_active_sessions(self):
        response = client.get("/health")
        data = response.json()
        assert "active_sessions" in data
        assert isinstance(data["active_sessions"], int)


class TestCrawlEndpoint:
    """Tests for the /crawl endpoint."""
    
    def test_crawl_missing_session_id(self):
        response = client.post("/crawl", json={"baseUrl": "https://example.com"})
        assert response.status_code == 422  # Pydantic validation error
    
    def test_crawl_empty_session_id(self):
        response = client.post("/crawl", json={
            "baseUrl": "https://example.com",
            "session_id": ""
        })
        assert response.status_code == 422
    
    def test_crawl_invalid_session_id_characters(self):
        response = client.post("/crawl", json={
            "baseUrl": "https://example.com",
            "session_id": "test@#$%"
        })
        assert response.status_code == 422
    
    def test_crawl_missing_url(self):
        response = client.post("/crawl", json={"session_id": "test_session"})
        assert response.status_code == 422
    
    def test_crawl_empty_url(self):
        response = client.post("/crawl", json={
            "baseUrl": "",
            "session_id": "test_session"
        })
        assert response.status_code == 422
    
    def test_crawl_invalid_url_no_scheme(self):
        response = client.post("/crawl", json={
            "baseUrl": "example.com",
            "session_id": "test_session"
        })
        assert response.status_code == 422
    
    def test_crawl_invalid_url_bad_scheme(self):
        response = client.post("/crawl", json={
            "baseUrl": "ftp://example.com",
            "session_id": "test_session"
        })
        assert response.status_code == 422
    
    def test_crawl_localhost_blocked(self):
        response = client.post("/crawl", json={
            "baseUrl": "http://localhost:8080",
            "session_id": "test_session"
        })
        assert response.status_code == 422
    
    def test_crawl_url_too_long(self):
        long_url = "https://example.com/" + "a" * 2100
        response = client.post("/crawl", json={
            "baseUrl": long_url,
            "session_id": "test_session"
        })
        assert response.status_code == 422


class TestAskEndpoint:
    """Tests for the /ask endpoint."""
    
    def test_ask_missing_session_id(self):
        response = client.post("/ask", json={"question": "What is this?"})
        assert response.status_code == 422
    
    def test_ask_missing_question(self):
        response = client.post("/ask", json={"session_id": "test_session"})
        assert response.status_code == 422
    
    def test_ask_empty_question(self):
        response = client.post("/ask", json={
            "question": "",
            "session_id": "test_session"
        })
        assert response.status_code == 422
    
    def test_ask_question_too_long(self):
        long_question = "What is " + "a" * 2500 + "?"
        response = client.post("/ask", json={
            "question": long_question,
            "session_id": "test_session"
        })
        assert response.status_code == 422
    
    def test_ask_no_data_ingested(self):
        response = client.post("/ask", json={
            "question": "What is this about?",
            "session_id": "nonexistent_session_12345"
        })
        assert response.status_code == 400
        assert "No data has been ingested" in response.json()["detail"]


class TestFileIngestEndpoint:
    """Tests for the /ingest/file endpoint."""
    
    def test_ingest_missing_file(self):
        response = client.post("/ingest/file", data={"session_id": "test_session"})
        assert response.status_code == 422
    
    def test_ingest_missing_session_id(self):
        response = client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world", "text/plain")}
        )
        assert response.status_code == 422
    
    def test_ingest_empty_session_id(self):
        response = client.post(
            "/ingest/file",
            data={"session_id": ""},
            files={"file": ("test.txt", b"Hello world", "text/plain")}
        )
        assert response.status_code == 400
    
    def test_ingest_unsupported_file_type(self):
        response = client.post(
            "/ingest/file",
            data={"session_id": "test_session"},
            files={"file": ("test.xyz", b"Hello world", "application/octet-stream")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_ingest_empty_file(self):
        response = client.post(
            "/ingest/file",
            data={"session_id": "test_session"},
            files={"file": ("test.txt", b"", "text/plain")}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


class TestURLValidation:
    """Tests for URL validation edge cases."""
    
    def test_url_with_port(self):
        # Valid URL with port should be accepted if not localhost
        response = client.post("/crawl", json={
            "baseUrl": "https://example.com:8080/path",
            "session_id": "test_session"
        })
        # Will fail at network level, but should pass validation
        assert response.status_code in [502, 500]  # Network error expected
    
    def test_url_with_path(self):
        response = client.post("/crawl", json={
            "baseUrl": "https://en.wikipedia.org/wiki/Python",
            "session_id": "test_session"
        })
        # Should pass validation
        assert response.status_code != 422
    
    def test_url_with_query_params(self):
        response = client.post("/crawl", json={
            "baseUrl": "https://example.com/search?q=test",
            "session_id": "test_session"
        })
        assert response.status_code != 422


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_root_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        # Should return HTML file
    
    def test_invalid_json_body(self):
        response = client.post(
            "/crawl",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_session_id_max_length(self):
        long_session = "a" * 100
        response = client.post("/crawl", json={
            "baseUrl": "https://example.com",
            "session_id": long_session
        })
        assert response.status_code == 422  # Should fail validation


# Run tests with: pytest tests/test_api.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
