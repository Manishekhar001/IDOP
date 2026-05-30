"""
Utility classes for request validation, standardized error responses, and file size formatting.

Provides:
- FileValidator: Validates uploaded files (type, size, extension)
- QueryValidator: Validates question/query inputs
- ValidationError: Custom exception for validation failures
- ErrorResponse: Factory methods for standardized error responses
- format_file_size: Human-readable file size formatting
"""

from fastapi import UploadFile


class ValidationError(Exception):
    """Custom exception raised when request validation fails."""

    pass


class FileValidator:
    """Validates uploaded document files against allowed types and size limits.

    Must stay in sync with DocumentProcessor.SUPPORTED_EXTENSIONS.
    """

    ALLOWED_EXTENSIONS: dict[str, str] = {
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }

    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """Validate that the uploaded file has an allowed extension and is within size limits.

        Args:
            file: The uploaded file to validate.

        Raises:
            ValidationError: If the file is missing, has an unsupported extension, or exceeds the size limit.
        """
        if not file or not file.filename:
            raise ValidationError("No file provided or filename is empty")

        file_ext = (
            "." + file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else ""
        )

        if file_ext not in FileValidator.ALLOWED_EXTENSIONS:
            allowed = ", ".join(FileValidator.ALLOWED_EXTENSIONS.keys())
            raise ValidationError(
                f"Invalid file type '{file_ext}'. Allowed types: {allowed}"
            )

        if hasattr(file, "size") and file.size:
            if file.size > FileValidator.MAX_FILE_SIZE:
                max_mb = FileValidator.MAX_FILE_SIZE / (1024 * 1024)
                raise ValidationError(
                    f"File size exceeds maximum allowed size of {max_mb:.0f} MB"
                )

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Extract the file extension (with dot) from a filename.

        Args:
            filename: The filename to extract the extension from.

        Returns:
            The file extension including the dot (e.g. '.pdf'), or empty string if no extension.
        """
        return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    @staticmethod
    def validate_extension(extension: str) -> str:
        """Validate and normalize a file extension.

        Args:
            extension: The file extension (with or without leading dot).

        Returns:
            The normalized extension without the leading dot.

        Raises:
            ValidationError: If the extension is not in the allowed list.
        """
        ext = extension.lstrip(".").lower() if extension else ""
        full_ext = f".{ext}" if ext else ""
        if full_ext and full_ext not in FileValidator.ALLOWED_EXTENSIONS:
            allowed = ", ".join(FileValidator.ALLOWED_EXTENSIONS.keys())
            raise ValidationError(
                f"Unsupported file extension '.{ext}'. Allowed: {allowed}"
            )
        return ext


class QueryValidator:
    """Validates user query/question inputs for length and content constraints."""

    MIN_QUESTION_LENGTH: int = 3
    MAX_QUESTION_LENGTH: int = 2000

    @staticmethod
    def validate_question(question: str, allow_empty: bool = False) -> str:
        """Validate and sanitize a user question string.

        Args:
            question: The raw question string to validate.
            allow_empty: If True, allows empty strings to pass through.

        Returns:
            The trimmed and validated question string.

        Raises:
            ValidationError: If the question is too short, too long, or empty (when not allowed).
        """
        if not question or not question.strip():
            if allow_empty:
                return ""
            raise ValidationError("Question cannot be empty.")

        question = question.strip()
        if len(question) < QueryValidator.MIN_QUESTION_LENGTH:
            raise ValidationError(
                f"Question too short (minimum {QueryValidator.MIN_QUESTION_LENGTH} characters)"
            )
        if len(question) > QueryValidator.MAX_QUESTION_LENGTH:
            raise ValidationError(
                f"Question too long (maximum {QueryValidator.MAX_QUESTION_LENGTH} characters)"
            )
        return question

    @staticmethod
    def validate_top_k(top_k: int) -> int:
        """Validate and clamp the top_k retrieval parameter.

        Args:
            top_k: The number of documents to retrieve.

        Returns:
            The clamped top_k value (1-20).

        Raises:
            ValidationError: If top_k is outside the allowed range.
        """
        if top_k < 1:
            raise ValidationError("top_k must be at least 1")
        if top_k > 20:
            raise ValidationError("top_k must be at most 20")
        return top_k


class ErrorResponse:
    """Factory class for creating standardized error response dictionaries.

    All methods return a dictionary matching the ErrorResponse Pydantic model schema
    with 'error', 'message', and optional 'detail' fields.
    """

    @staticmethod
    def validation_error(message: str, field: str | None = None) -> dict:
        """Create a standardized validation error response.

        Args:
            message: Human-readable description of the validation failure.
            field: Optional field name that failed validation.

        Returns:
            A dictionary with error type, message, and optional detail.
        """
        return {
            "error": "ValidationError",
            "message": message,
            "detail": f"Field: {field}" if field else None,
        }

    @staticmethod
    def service_unavailable(service: str, resolution: str | None = None) -> dict:
        """Create a standardized service unavailable error response.

        Args:
            service: Name of the unavailable service.
            resolution: Optional hint on how to resolve the issue.

        Returns:
            A dictionary with error type, message, and optional detail.
        """
        return {
            "error": "ServiceUnavailable",
            "message": f"{service} is not available",
            "detail": resolution,
        }

    @staticmethod
    def internal_error(operation: str, exception: Exception | None = None) -> dict:
        """Create a standardized internal server error response.

        Args:
            operation: Description of the operation that failed.
            exception: Optional exception instance for detail extraction.

        Returns:
            A dictionary with error type, message, and optional detail.
        """
        return {
            "error": "InternalError",
            "message": f"Failed to {operation}",
            "detail": f"{type(exception).__name__}: {exception}" if exception else None,
        }

    @staticmethod
    def not_found(resource: str, identifier: str) -> dict:
        """Create a standardized resource not found error response.

        Args:
            resource: Type of resource that was not found (e.g. 'Document', 'Thread').
            identifier: The identifier that was searched for.

        Returns:
            A dictionary with error type, message, and detail.
        """
        return {
            "error": "NotFoundError",
            "message": f"{resource} not found",
            "detail": f"No {resource.lower()} found for identifier: {identifier}",
        }


def format_file_size(size_bytes: int) -> str:
    """Convert a file size in bytes to a human-readable string.

    Args:
        size_bytes: The file size in bytes.

    Returns:
        A human-readable size string (e.g. '2.5 MB', '340 B', '1.2 KB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
