"""
Unit tests for the IDOP authentication module (app.api.auth).

Covers:
- password hashing + verification roundtrip
- JWT creation + decode roundtrip
- get_current_user with a valid token
- get_current_user with a missing token (401)
- get_current_user with an invalid / expired token (401)
- dev-mode bypass when jwt_secret_key == "CHANGE-ME-IN-PRODUCTION"
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from jose import jwt

from app.api.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_SECRET = "super-secret-test-key-not-for-prod"
_TEST_ALGORITHM = "HS256"


def _make_settings(**overrides):
    """Return a lightweight settings-like object for patching."""
    defaults = {
        "jwt_secret_key": _TEST_SECRET,
        "jwt_algorithm": _TEST_ALGORITHM,
        "jwt_expire_minutes": 60,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Password hashing tests
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    """Verify bcrypt hash / verify roundtrip."""

    def test_hash_and_verify_succeeds(self):
        plain = "my-s3cur3-passw0rd!"
        hashed = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed) is True

    def test_verify_wrong_password_fails(self):
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_different_hashes_for_same_password(self):
        hashed_a = hash_password("same")
        hashed_b = hash_password("same")
        # bcrypt salts differ each time
        assert hashed_a != hashed_b


# ---------------------------------------------------------------------------
# JWT creation / decode tests
# ---------------------------------------------------------------------------


class TestJWT:
    """Verify JWT encode / decode roundtrip."""

    @patch("app.api.auth.get_settings")
    def test_create_and_decode_token(self, mock_settings):
        mock_settings.return_value = _make_settings()
        token = create_access_token({"sub": "alice@example.com", "role": "admin"})
        payload = jwt.decode(token, _TEST_SECRET, algorithms=[_TEST_ALGORITHM])
        assert payload["sub"] == "alice@example.com"
        assert payload["role"] == "admin"
        assert "exp" in payload

    @patch("app.api.auth.get_settings")
    def test_custom_expiry(self, mock_settings):
        mock_settings.return_value = _make_settings()
        token = create_access_token(
            {"sub": "bob@example.com", "role": "user"},
            expires_delta=timedelta(minutes=5),
        )
        payload = jwt.decode(token, _TEST_SECRET, algorithms=[_TEST_ALGORITHM])
        assert payload["sub"] == "bob@example.com"


# ---------------------------------------------------------------------------
# get_current_user tests
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    """Verify the FastAPI dependency under various conditions."""

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_valid_token_returns_user(self, mock_settings):
        mock_settings.return_value = _make_settings()
        token = create_access_token({"sub": "alice@example.com", "role": "admin"})
        user = await get_current_user(token=token)
        assert user["sub"] == "alice@example.com"
        assert user["role"] == "admin"

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_missing_token_raises_401(self, mock_settings):
        mock_settings.return_value = _make_settings()
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_invalid_token_raises_401(self, mock_settings):
        mock_settings.return_value = _make_settings()
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token="not-a-valid-jwt")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_expired_token_raises_401(self, mock_settings):
        mock_settings.return_value = _make_settings()
        token = create_access_token(
            {"sub": "expired@example.com", "role": "user"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_token_missing_sub_raises_401(self, mock_settings):
        mock_settings.return_value = _make_settings()
        # Craft a token without the "sub" claim
        token = create_access_token({"role": "user"})
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Dev-mode bypass
# ---------------------------------------------------------------------------


class TestDevMode:
    """When jwt_secret_key is the default sentinel, auth is bypassed."""

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_dev_mode_returns_dev_user(self, mock_settings):
        mock_settings.return_value = _make_settings(
            jwt_secret_key="CHANGE-ME-IN-PRODUCTION",
        )
        user = await get_current_user(token=None)
        assert user["sub"] == "dev@localhost"
        assert user["role"] == "admin"

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_dev_mode_ignores_token(self, mock_settings):
        mock_settings.return_value = _make_settings(
            jwt_secret_key="CHANGE-ME-IN-PRODUCTION",
        )
        # Even a garbage token is ignored in dev mode
        user = await get_current_user(token="garbage-token")
        assert user["sub"] == "dev@localhost"
        assert user["role"] == "admin"
