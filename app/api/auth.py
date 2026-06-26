"""
Core authentication module for IDOP.

Provides JWT token creation / validation, password hashing helpers,
user CRUD against the ``idop_users`` table in PostgreSQL, and the
``get_current_user`` FastAPI dependency.

When ``jwt_secret_key`` equals the default sentinel
``"CHANGE-ME-IN-PRODUCTION"`` authentication is disabled (dev mode)
and a synthetic user dict is returned for every request.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import psycopg2
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import get_settings

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return ``True`` when *plain* matches the bcrypt *hashed* value."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
_DEV_SENTINEL = "CHANGE-ME-IN-PRODUCTION"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create a signed JWT containing *data*.

    If *expires_delta* is ``None`` the token expires after the configured
    ``jwt_expire_minutes`` from settings.
    """
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

_DEV_USER: dict[str, str] = {"sub": "dev@localhost", "role": "admin"}


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
) -> dict[str, str]:
    """Decode the JWT from the ``Authorization`` header and return a user dict.

    Returns ``{"sub": <email>, "role": <role>}``.

    In dev mode (``jwt_secret_key == "CHANGE-ME-IN-PRODUCTION"``) the
    dependency returns a hard-coded admin user without requiring a token.
    """
    settings = get_settings()

    # --- dev mode bypass ---------------------------------------------------
    if secrets.compare_digest(settings.jwt_secret_key, _DEV_SENTINEL):
        return _DEV_USER

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception

    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        email: str | None = payload.get("sub")
        role: str = payload.get("role", "user")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    return {"sub": email, "role": role}


# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_role(*allowed_roles: str):
    """Factory for role-based dependency."""

    async def _check_role(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(allowed_roles)}",
            )
        return user

    return _check_role


# ---------------------------------------------------------------------------
# Database helpers (idop_users table via psycopg2)
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS idop_users (
    id             SERIAL PRIMARY KEY,
    email          VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role           VARCHAR(50) DEFAULT 'user',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_connection() -> psycopg2.extensions.connection | None:
    """Return a psycopg2 connection or ``None`` on failure."""
    settings = get_settings()
    try:
        return psycopg2.connect(settings.supabase_db_url)
    except Exception:
        return None


def create_users_table() -> None:
    """Create the ``idop_users`` table if it does not exist.

    Silently returns ``None`` when the database is unreachable.
    """
    conn = _get_connection()
    if conn is None:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
    finally:
        conn.close()
    return None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Fetch a user row by *email*. Returns ``None`` when not found."""
    conn = _get_connection()
    if conn is None:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, hashed_password, role, created_at "
                    "FROM idop_users WHERE email = %s",
                    (email,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": row[0],
                    "email": row[1],
                    "hashed_password": row[2],
                    "role": row[3],
                    "created_at": row[4],
                }
    finally:
        conn.close()


def create_user(email: str, password: str, role: str = "user") -> dict[str, Any]:
    """Insert a new user and return the created row as a dict."""
    hashed = hash_password(password)
    conn = _get_connection()
    if conn is None:
        raise RuntimeError("Unable to connect to the database")
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO idop_users (email, hashed_password, role) "
                    "VALUES (%s, %s, %s) "
                    "RETURNING id, email, role, created_at",
                    (email, hashed, role),
                )
                row = cur.fetchone()
                return {
                    "id": row[0],
                    "email": row[1],
                    "role": row[2],
                    "created_at": row[3],
                }
    finally:
        conn.close()
