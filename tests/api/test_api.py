"""
Tests for API endpoints.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestAPI:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "mcts-orchestrator"}

    def test_search_endpoint(self):
        """Test search invocation."""
        payload = {
            "problem": "Write a python function to add two numbers",
            "test_cases": ["assert add(1, 2) == 3"],
            "context": "Context for adding",
        }
        # Note: Dependencies usually mocked in real fixture, here relying on mock inside dependencies
        response = client.post("/search/", json=payload)

        # Check basic response structure
        if response.status_code != 200:
            print(f"\nSearch Error Body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert "solution" in data
        assert "agent_used" in data
        assert "execution_time_ms" in data

    def test_model_registration(self):
        """Test uploading a file."""
        # Create dummy file
        files = {"file": ("model.pt", b"dummy content", "application/octet-stream")}
        data = {"model_type": "hrm"}

        response = client.post("/models/register", files=files, data=data)

        if response.status_code != 200:
            print(f"\nRegister Error Body: {response.text}")

        assert response.status_code == 200
        result = response.json()
        assert result["version_id"].startswith("hrm_")
        assert result["filepath"].endswith(".pt")
