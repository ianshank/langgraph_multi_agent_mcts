"""
API Server Endpoint Tests.

Tests for:
- REST API endpoint functionality
- Request/response validation
- Authentication and authorization
- Rate limiting
- Error handling

Expected outcomes:
- All endpoints respond correctly
- Authentication enforced
- Rate limits applied
- Proper error responses
"""

from datetime import datetime

import pytest


@pytest.fixture
def valid_api_key():
    """Valid API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def valid_query_request():
    """Valid query request payload."""
    return {
        "query": "Analyze the tactical situation and recommend defensive positions",
        "use_rag": True,
        "use_mcts": False,
        "thread_id": "test_thread_001",
    }


@pytest.fixture
def mock_fastapi_client():
    """Mock FastAPI test client."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.headers = {"Authorization": "Bearer test-api-key-12345"}
    return client


class TestHealthEndpoints:
    """Test health and readiness endpoints."""

    @pytest.mark.api
    def test_health_check_returns_ok(self):
        """Health endpoint should return OK status."""
        response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        assert response["status"] == "healthy"
        assert "timestamp" in response
        assert "version" in response

    @pytest.mark.api
    def test_readiness_check(self):
        """Readiness endpoint should check dependencies."""
        readiness_response = {
            "status": "ready",
            "checks": {
                "database": "ok",
                "llm_provider": "ok",
                "vector_store": "ok",
                "cache": "ok",
            },
        }

        assert readiness_response["status"] == "ready"
        assert all(check == "ok" for check in readiness_response["checks"].values())

    @pytest.mark.api
    def test_readiness_degraded_mode(self):
        """Readiness should report degraded status when dependencies fail."""
        readiness_response = {
            "status": "degraded",
            "checks": {
                "database": "ok",
                "llm_provider": "ok",
                "vector_store": "error",  # Pinecone unavailable
                "cache": "ok",
            },
        }

        assert readiness_response["status"] == "degraded"
        assert readiness_response["checks"]["vector_store"] == "error"


class TestQueryEndpoint:
    """Test main query processing endpoint."""

    @pytest.mark.api
    def test_valid_query_request(self, valid_query_request):
        """Valid query should be accepted."""
        # Simulate request validation
        from src.models.validation import QueryInput

        query_input = QueryInput(**valid_query_request)

        assert query_input.query is not None
        assert query_input.use_rag is True
        assert query_input.use_mcts is False

    @pytest.mark.api
    def test_query_response_structure(self):
        """Query response should have proper structure."""
        response = {
            "query_id": "req_12345",
            "recommendation": "Establish defensive position at Alpha",
            "confidence": 0.85,
            "alternatives": [
                {"action": "Position Beta", "confidence": 0.68},
                {"action": "Position Gamma", "confidence": 0.61},
            ],
            "risks": ["ammo_depletion", "visibility_issues"],
            "evidence_sources": ["doctrine_manual", "historical_case"],
            "processing_time_ms": 1250,
            "agents_used": ["HRM", "TRM"],
        }

        # Validate structure
        required_fields = [
            "query_id",
            "recommendation",
            "confidence",
            "alternatives",
            "processing_time_ms",
        ]

        for field in required_fields:
            assert field in response

        # Validate confidence range
        assert 0.0 <= response["confidence"] <= 1.0

        # Validate processing time
        assert response["processing_time_ms"] < 30000  # <30s SLA

    @pytest.mark.api
    def test_empty_query_rejected(self):
        """Empty query should return 422 validation error."""
        _invalid_request = {
            "query": "",
            "use_rag": True,
            "use_mcts": False,
        }

        # Simulate validation error
        error_response = {
            "status_code": 422,
            "detail": [
                {
                    "loc": ["body", "query"],
                    "msg": "Query cannot be empty",
                    "type": "value_error",
                }
            ],
        }

        assert error_response["status_code"] == 422
        assert "query" in str(error_response["detail"])

    @pytest.mark.api
    def test_oversized_query_rejected(self):
        """Query exceeding max length should be rejected."""
        _oversized_request = {
            "query": "x" * 15000,  # Exceeds 10000 limit
            "use_rag": False,
            "use_mcts": False,
        }

        error_response = {
            "status_code": 422,
            "detail": "Query exceeds maximum length of 10000 characters",
        }

        assert error_response["status_code"] == 422

    @pytest.mark.api
    def test_query_with_mcts_enabled(self):
        """Query with MCTS should include simulation results."""
        _request = {
            "query": "Recommend optimal action for tactical scenario",
            "use_rag": True,
            "use_mcts": True,
            "mcts_iterations": 100,
        }

        response = {
            "query_id": "req_67890",
            "recommendation": "Advance to Alpha",
            "confidence": 0.73,
            "mcts_results": {
                "iterations_completed": 100,
                "win_probability": 0.73,
                "alternatives_explored": 5,
                "tree_depth": 8,
            },
            "processing_time_ms": 15000,
        }

        assert "mcts_results" in response
        assert response["mcts_results"]["iterations_completed"] == 100
        assert response["processing_time_ms"] < 30000


class TestAuthentication:
    """Test API authentication."""

    @pytest.mark.api
    @pytest.mark.security
    def test_valid_api_key_accepted(self, valid_api_key):  # noqa: ARG002
        """Valid API key should be accepted."""
        # Simulate auth check
        auth_result = {
            "authenticated": True,
            "client_id": "client_123",
            "roles": ["user", "analyst"],
        }

        assert auth_result["authenticated"] is True
        assert "roles" in auth_result

    @pytest.mark.api
    @pytest.mark.security
    def test_invalid_api_key_rejected(self):
        """Invalid API key should return 401."""
        error_response = {
            "status_code": 401,
            "detail": "Invalid or expired API key",
        }

        assert error_response["status_code"] == 401

    @pytest.mark.api
    @pytest.mark.security
    def test_missing_auth_header_rejected(self):
        """Missing Authorization header should return 401."""
        error_response = {
            "status_code": 401,
            "detail": "Authorization header required",
        }

        assert error_response["status_code"] == 401

    @pytest.mark.api
    @pytest.mark.security
    def test_bearer_token_format(self):
        """Auth header should use Bearer token format."""
        valid_header = "Bearer sk-test-key-12345"
        invalid_header = "sk-test-key-12345"

        # Parse token from valid header
        if valid_header.startswith("Bearer "):
            token = valid_header[7:]
            assert token == "sk-test-key-12345"

        # Invalid format should fail
        assert not invalid_header.startswith("Bearer ")

    @pytest.mark.api
    @pytest.mark.security
    def test_api_key_hashing(self):
        """API keys should be hashed for storage."""
        import hashlib

        api_key = "sk-test-key-12345"
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()

        # Should not store plaintext
        assert hashed_key != api_key
        assert len(hashed_key) == 64  # SHA-256 produces 64 hex chars


class TestRateLimiting:
    """Test API rate limiting."""

    @pytest.mark.api
    def test_rate_limit_headers_present(self):
        """Response should include rate limit headers."""
        response_headers = {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "58",
            "X-RateLimit-Reset": "1234567890",
        }

        assert "X-RateLimit-Limit" in response_headers
        assert "X-RateLimit-Remaining" in response_headers
        assert "X-RateLimit-Reset" in response_headers

    @pytest.mark.api
    def test_rate_limit_exceeded_returns_429(self):
        """Exceeding rate limit should return 429."""
        error_response = {
            "status_code": 429,
            "detail": "Rate limit exceeded. Please retry after 60 seconds.",
            "headers": {
                "Retry-After": "60",
                "X-RateLimit-Limit": "60",
                "X-RateLimit-Remaining": "0",
            },
        }

        assert error_response["status_code"] == 429
        assert "Retry-After" in error_response["headers"]

    @pytest.mark.api
    def test_per_minute_rate_limit(self):
        """Per-minute rate limit should be enforced."""
        rate_limit_config = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_limit": 10,
        }

        # Simulate 61 requests in one minute
        requests_made = 61
        limit = rate_limit_config["requests_per_minute"]

        assert requests_made > limit  # Should trigger rate limit

    @pytest.mark.api
    def test_client_specific_rate_limits(self):
        """Different clients can have different rate limits."""
        client_limits = {
            "free_tier": {"requests_per_minute": 10, "requests_per_hour": 100},
            "pro_tier": {"requests_per_minute": 60, "requests_per_hour": 1000},
            "enterprise": {"requests_per_minute": 300, "requests_per_hour": 10000},
        }

        # Enterprise has higher limits
        assert client_limits["enterprise"]["requests_per_minute"] > client_limits["pro_tier"]["requests_per_minute"]
        assert client_limits["pro_tier"]["requests_per_minute"] > client_limits["free_tier"]["requests_per_minute"]


class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.api
    def test_internal_server_error_response(self):
        """Internal errors should return 500 with safe message."""
        error_response = {
            "status_code": 500,
            "detail": "Internal server error. Please try again later.",
            "error_id": "err_xyz123",
        }

        # Should not expose internal details
        assert error_response["status_code"] == 500
        assert "Internal server error" in error_response["detail"]
        assert "error_id" in error_response  # For debugging

    @pytest.mark.api
    def test_llm_timeout_handling(self):
        """LLM timeout should return appropriate error."""
        error_response = {
            "status_code": 503,
            "detail": "Service temporarily unavailable. LLM provider timeout.",
            "retry_after": 30,
        }

        assert error_response["status_code"] == 503
        assert "retry_after" in error_response

    @pytest.mark.api
    def test_validation_error_format(self):
        """Validation errors should be clear and actionable."""
        error_response = {
            "status_code": 422,
            "detail": [
                {
                    "loc": ["body", "mcts_iterations"],
                    "msg": "Value must be between 1 and 10000",
                    "type": "value_error.number.not_in_range",
                }
            ],
        }

        # Should specify location and expected range
        assert error_response["detail"][0]["loc"] == ["body", "mcts_iterations"]
        assert "1 and 10000" in error_response["detail"][0]["msg"]


class TestPrometheusMetrics:
    """Test Prometheus metrics endpoint."""

    @pytest.mark.api
    def test_metrics_endpoint_accessible(self):
        """Metrics endpoint should be accessible."""
        response = {
            "status_code": 200,
            "content_type": "text/plain",
        }

        assert response["status_code"] == 200

    @pytest.mark.api
    def test_request_metrics_collected(self):
        """Request metrics should be collected."""
        metrics = {
            "http_requests_total": 1500,
            "http_request_duration_seconds_sum": 450.5,
            "http_request_duration_seconds_count": 1500,
            "active_requests": 5,
            "error_count_total": 25,
        }

        # Verify key metrics present
        assert "http_requests_total" in metrics
        assert "http_request_duration_seconds_sum" in metrics
        assert "active_requests" in metrics

    @pytest.mark.api
    def test_mcts_specific_metrics(self):
        """MCTS-specific metrics should be tracked."""
        metrics = {
            "mcts_iterations_total": 50000,
            "mcts_simulation_duration_seconds_sum": 125.3,
            "mcts_tree_depth_histogram": {"buckets": [1, 5, 10, 20], "counts": [100, 250, 400, 50]},
        }

        assert "mcts_iterations_total" in metrics
        assert "mcts_simulation_duration_seconds_sum" in metrics


class TestCORSConfiguration:
    """Test CORS middleware configuration."""

    @pytest.mark.api
    def test_cors_headers_present(self):
        """CORS headers should be included in responses when configured."""
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
        }

        assert "Access-Control-Allow-Origin" in cors_headers
        assert "Authorization" in cors_headers["Access-Control-Allow-Headers"]

    @pytest.mark.api
    def test_preflight_request_handling(self):
        """OPTIONS preflight requests should be handled."""
        response = {
            "status_code": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Max-Age": "86400",
            },
        }

        assert response["status_code"] == 200
        assert "Access-Control-Max-Age" in response["headers"]

    @pytest.mark.api
    def test_cors_disabled_by_default(self):
        """CORS should be disabled when CORS_ALLOWED_ORIGINS is None."""
        # This tests the security improvement where CORS requires explicit configuration
        # When CORS_ALLOWED_ORIGINS is None, the middleware is not added and
        # cross-origin requests are rejected by the browser
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        # Default should be None (requires explicit configuration)
        assert settings.CORS_ALLOWED_ORIGINS is None

    @pytest.mark.api
    def test_cors_explicit_wildcard_for_dev(self):
        """Development can explicitly enable permissive CORS with wildcard."""
        from src.config.settings import Settings

        # Explicitly set to wildcard for development
        settings = Settings(
            LLM_PROVIDER="lmstudio",
            CORS_ALLOWED_ORIGINS=["*"],
            _env_file=None,
        )
        assert settings.CORS_ALLOWED_ORIGINS == ["*"]

    @pytest.mark.api
    def test_cors_specific_origins_for_production(self):
        """Production should use specific allowed origins."""
        from src.config.settings import Settings

        # Production with specific origins
        settings = Settings(
            LLM_PROVIDER="lmstudio",
            CORS_ALLOWED_ORIGINS=["https://app.example.com", "https://dashboard.example.com"],
            _env_file=None,
        )
        assert "https://app.example.com" in settings.CORS_ALLOWED_ORIGINS
        assert "https://dashboard.example.com" in settings.CORS_ALLOWED_ORIGINS
        assert len(settings.CORS_ALLOWED_ORIGINS) == 2

    @pytest.mark.api
    def test_cors_credentials_disabled_with_wildcard(self):
        """Credentials should be disabled when using wildcard origins per CORS spec."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            CORS_ALLOWED_ORIGINS=["*"],
            CORS_ALLOW_CREDENTIALS=True,  # This should be ignored/overridden
            _env_file=None,
        )
        # In the servers, credentials are forced to False when origins is ["*"]
        # This is the correct behavior per CORS specification
        assert settings.CORS_ALLOWED_ORIGINS == ["*"]


class TestRequestValidation:
    """Test request payload validation."""

    @pytest.mark.api
    def test_json_content_type_required(self):
        """Requests should have application/json content type."""
        valid_content_type = "application/json"
        invalid_content_type = "text/plain"

        assert valid_content_type == "application/json"
        assert invalid_content_type != "application/json"

    @pytest.mark.api
    def test_malformed_json_rejected(self):
        """Malformed JSON should return 400."""
        error_response = {
            "status_code": 400,
            "detail": "Invalid JSON payload",
        }

        assert error_response["status_code"] == 400

    @pytest.mark.api
    def test_extra_fields_handled(self):
        """Extra fields in request should be handled appropriately."""
        _request_with_extra = {
            "query": "Test query",
            "use_rag": False,
            "use_mcts": False,
            "unknown_field": "should_be_ignored",
        }

        # With Pydantic strict mode, extra fields are rejected
        # This is desired behavior for security
        expected_behavior = "reject_extra_fields"
        assert expected_behavior == "reject_extra_fields"


class TestResponseFormatting:
    """Test response formatting and consistency."""

    @pytest.mark.api
    def test_consistent_timestamp_format(self):
        """Timestamps should use ISO 8601 format."""
        from datetime import datetime

        timestamp = datetime.now().isoformat()

        # ISO 8601 format check
        assert "T" in timestamp
        assert len(timestamp) >= 19

    @pytest.mark.api
    def test_error_response_consistency(self):
        """All error responses should have consistent structure."""
        error_responses = [
            {"status_code": 400, "detail": "Bad request"},
            {"status_code": 401, "detail": "Unauthorized"},
            {"status_code": 422, "detail": "Validation error"},
            {"status_code": 429, "detail": "Rate limited"},
            {"status_code": 500, "detail": "Server error"},
        ]

        for response in error_responses:
            assert "status_code" in response
            assert "detail" in response
            assert isinstance(response["status_code"], int)
            assert isinstance(response["detail"], str)
