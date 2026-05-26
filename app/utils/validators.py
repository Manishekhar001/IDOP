import re
from typing import Optional
from fastapi import UploadFile


class ValidationError(Exception):
    pass


class FileValidator:
    ALLOWED_EXTENSIONS = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain",
        ".html": "text/html",
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024

    @staticmethod
    def validate_file(file: UploadFile) -> None:
        if not file or not file.filename:
            raise ValidationError("No file provided or filename is Empty")

        file_ext = (
            "." + file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else ""
        )

        if file_ext not in FileValidator.ALLOWED_EXTENSIONS:
            allowed = ", ".join(FileValidator.ALLOWED_EXTENSIONS.keys())
            raise ValidationError(
                f"Invalid file type '{file_ext}. Allowed Types: {allowed}'"
            )

        if hasattr(file, "size") and file.size:
            if file.size > FileValidator.MAX_FILE_SIZE:
                max_mb = FileValidator.MAX_FILE_SIZE / (1024 * 1024)
                raise ValidationError(
                    f"File size exceeds maximum allowed size of {max_mb:.0f} MB"
                )

    @staticmethod
    def get_file_extension(filename: str) -> str:
        return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


class QueryValidator:
    MIN_QUESTION_LENGTH = 3
    MAX_QUESTION_LENGTH = 1000

    @staticmethod
    def validate_question(question: str, allow_empty: bool = False) -> str:
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
