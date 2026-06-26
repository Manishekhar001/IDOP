"""
Authentication routes for IDOP.

Endpoints
---------
POST /auth/register  -- create a new user account
POST /auth/login     -- obtain a JWT access token
GET  /auth/me        -- return the current authenticated user
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.api.auth import (
    create_access_token,
    create_user,
    get_current_user,
    get_user_by_email,
    require_admin,
    verify_password,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "user"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    email: str
    role: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    """Create a new user account."""
    existing = get_user_by_email(body.email)
    if existing is not None:
        logger.warning("Registration attempt with existing email: %s", body.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )
    try:
        user = create_user(body.email, body.password, body.role)
    except RuntimeError as exc:
        # Database connection error - return 503 Service Unavailable
        logger.error("Database unavailable for user creation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Failed to create user %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {exc}",
        ) from exc

    logger.info("User created: %s", user["email"])
    return {"message": "User created", "email": user["email"]}


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and return a JWT access token.

    Accepts standard OAuth2 ``username`` / ``password`` form fields.
    The ``username`` field is treated as the user's email address.
    """
    user = get_user_by_email(form_data.username)
    if user is None or not verify_password(form_data.password, user["hashed_password"]):
        logger.warning("Failed login attempt for: %s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user["email"], "role": user["role"]})
    logger.info("User logged in: %s", user["email"])
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return UserResponse(email=current_user["sub"], role=current_user["role"])


@router.get("/admin-only", response_model=UserResponse)
async def admin_only(current_user: dict = Depends(require_admin)):
    """Admin-only endpoint example."""
    return UserResponse(email=current_user["sub"], role=current_user["role"])
