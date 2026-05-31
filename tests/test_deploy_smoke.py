"""
Unit tests for deploy_smoke.py — health check SHA matching and cache backend detection.

Tests the core logic of run_health_check() in isolation by mocking
requests.get() and environment variables, covering:

  - Git commit SHA matching (expected vs actual, unknown, empty, missing env)
  - Document cache backend detection (s3, local, unavailable, unknown)
  - Query cache mode detection (disabled, local_fallback, redis)
  - Health status (healthy, degraded, unhealthy)
  - HTTP errors and network failures
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from scripts import deploy_smoke

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module-level globals before each test to prevent cross-test pollution."""
    deploy_smoke.API_URL = "http://test-deploy.example.com"
    yield


@pytest.fixture
def mock_response():
    """Build a mock requests.Response with configurable JSON body."""

    def _factory(status_code=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = json.dumps(json_data) if json_data else "{}"
        return resp

    return _factory


# =========================================================================
# Git Commit SHA Matching Tests
# =========================================================================


class TestGitCommitSHAMatching:
    """Tests for the git_commit_sha verification logic in run_health_check()."""

    def test_sha_matches_expected(self, mock_response):
        """SHA matches EXPECTED_GIT_SHA → returns True, prints success."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc123def456",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": "abc123def456"}, clear=False
        ):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_sha_mismatch_fails(self, mock_response):
        """SHA does not match EXPECTED_GIT_SHA → returns False."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "old-sha-value",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": "new-sha-value"}, clear=False
        ):
            result = deploy_smoke.run_health_check()
            assert result is False

    def test_sha_unknown_does_not_fail(self, mock_response):
        """SHA is 'unknown' → prints warning but does NOT fail (returns True if other checks pass)."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "unknown",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": "abc123"}, clear=False
        ):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_sha_empty_does_not_fail(self, mock_response):
        """SHA is empty string → prints warning but does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": "abc123"}, clear=False
        ):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_sha_none_does_not_fail(self, mock_response):
        """SHA is None → prints warning but does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": None,
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": "abc123"}, clear=False
        ):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_no_expected_sha_env_skips_check(self, mock_response):
        """EXPECTED_GIT_SHA is not set → skips SHA comparison, returns True."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "some-sha",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        # Only remove EXPECTED_GIT_SHA — don't wipe all env vars
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": ""}, clear=False
        ):
            # Clear the value so os.getenv returns None
            if "EXPECTED_GIT_SHA" in os.environ:
                del os.environ["EXPECTED_GIT_SHA"]
            deploy_smoke.API_URL = "http://test-deploy.example.com"
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_sha_missing_from_response(self, mock_response):
        """git_commit_sha key is missing from response → defaults to 'unknown', does not fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            # git_commit_sha key deliberately absent
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp), patch.dict(
            os.environ, {"EXPECTED_GIT_SHA": "abc123"}, clear=False
        ):
            result = deploy_smoke.run_health_check()
            assert result is True


# =========================================================================
# Document Cache Backend Detection Tests
# =========================================================================


class TestDocumentCacheBackend:
    """Tests for the document cache backend detection logic in run_health_check()."""

    def test_cache_backend_s3_success(self, mock_response):
        """doc_cache_backend == 's3' → prints success, returns True."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_cache_backend_local_fallback(self, mock_response):
        """doc_cache_backend == 'local' → prints warning, does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "local",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_cache_backend_unavailable(self, mock_response):
        """doc_cache_backend contains 'unavailable' → prints warning, does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "unavailable (configured: s3)",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_cache_backend_unknown_value_and_missing(self, mock_response):
        """doc_cache_backend is unrecognized or missing → falls to else/default, does NOT fail."""
        for backend_value in ["s3_disabled", "unknown", "gcs"]:
            health_data = {
                "status": "healthy",
                "version": "0.1.0",
                "git_commit_sha": "abc",
                "services": {
                    "document_cache_backend": backend_value,
                    "query_cache_mode": "redis",
                },
            }
            resp = mock_response(200, health_data)
            with patch.object(deploy_smoke.requests, "get", return_value=resp):
                result = deploy_smoke.run_health_check()
                assert result is True, f"Failed for backend value: {backend_value}"

    def test_cache_backend_missing_defaults_to_unknown(self, mock_response):
        """doc_cache_backend key is missing from services → defaults to 'unknown', does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                # document_cache_backend deliberately omitted
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_cache_backend_with_error(self, mock_response):
        """doc_cache_error is present → prints warning, does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "document_cache_error": "Bucket 'idop-cache' not found",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True


# =========================================================================
# Query Cache Mode Tests
# =========================================================================


class TestQueryCacheMode:
    """Tests for the query cache mode detection logic in run_health_check()."""

    def test_query_cache_disabled_fails(self, mock_response):
        """query_cache_mode == 'disabled' → fails the check, returns False."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "disabled",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is False

    def test_query_cache_local_fallback_warns(self, mock_response):
        """query_cache_mode == 'local_fallback' → prints warning, does NOT fail."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "local_fallback",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_query_cache_redis_connected(self, mock_response):
        """query_cache_mode == 'redis' → prints success, returns True."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_query_cache_unknown_prints_connected(self, mock_response):
        """query_cache_mode is an unrecognized string → prints as 'connected', returns True."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "memcached",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True


# =========================================================================
# Health Status Tests
# =========================================================================


class TestHealthStatus:
    """Tests for the overall health status logic in run_health_check()."""

    def test_healthy_status_returns_true(self, mock_response):
        """status == 'healthy' → returns True."""
        health_data = {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_degraded_status_passes(self, mock_response):
        """status == 'degraded' → returns True (non-fatal, API still serving traffic)."""
        health_data = {
            "status": "degraded",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is True

    def test_unhealthy_status_fails(self, mock_response):
        """status == 'unhealthy' → returns False."""
        health_data = {
            "status": "unhealthy",
            "version": "0.1.0",
            "git_commit_sha": "abc",
            "services": {
                "document_cache_backend": "s3",
                "query_cache_mode": "redis",
            },
        }
        resp = mock_response(200, health_data)
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is False


# =========================================================================
# HTTP Error & Network Failure Tests
# =========================================================================


class TestHTTPErrors:
    """Tests for HTTP and network error handling in run_health_check()."""

    def test_non_200_status_code_fails(self, mock_response):
        """HTTP status is 500 → returns False."""
        resp = mock_response(500, {"detail": "Internal server error"})
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is False

    def test_non_200_status_404(self, mock_response):
        """HTTP status is 404 → returns False."""
        resp = mock_response(404, {"detail": "Not found"})
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is False

    def test_network_timeout_returns_false(self):
        """requests.get raises a timeout exception → returns False."""
        with patch.object(
            deploy_smoke.requests, "get", side_effect=Exception("Connection timeout")
        ):
            result = deploy_smoke.run_health_check()
            assert result is False

    def test_connection_refused_returns_false(self):
        """requests.get raises ConnectionError → returns False."""
        with patch.object(
            deploy_smoke.requests,
            "get",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = deploy_smoke.run_health_check()
            assert result is False

    def test_json_decode_error_returns_false(self, mock_response):
        """Response contains invalid JSON → returns False."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        resp.text = "not json"
        with patch.object(deploy_smoke.requests, "get", return_value=resp):
            result = deploy_smoke.run_health_check()
            assert result is False
