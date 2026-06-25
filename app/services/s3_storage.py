import io
import json
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.services.storage_backend import StorageBackend
from app.utils.logger import get_logger

logger = get_logger(__name__)


class S3StorageBackend(StorageBackend):
    """
    S3-based storage for production Lambda deployment.
    """

    def __init__(self, bucket_name: str | None = None):
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            logger.warning(
                "boto3 not installed. S3StorageBackend will not work properly."
            )
            self.enabled = False
            return

        self.enabled = True
        settings = get_settings()
        self.bucket_name = bucket_name or settings.s3_cache_bucket
        self.region = settings.aws_region

        boto_config = Config(
            region_name=self.region, retries={"max_attempts": 3, "mode": "adaptive"}
        )

        # Build boto3 client
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self.s3_client = boto3.client(
                "s3",
                config=boto_config,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        else:
            self.s3_client = boto3.client("s3", config=boto_config)

        self.validation_error: str | None = None
        try:
            self._validate_bucket()
            logger.info(
                f"S3Storage initialized with bucket: {self.bucket_name} (region: {self.region})"
            )
        except Exception as e:
            self.validation_error = str(e)
            logger.warning(f"S3 bucket validation failed, disabling S3 backend: {e}")
            self.enabled = False

    def _validate_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' is accessible.")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise ValueError(f"S3 bucket '{self.bucket_name}' does not exist.")
            elif error_code == "403":
                raise PermissionError(
                    f"Access denied to S3 bucket '{self.bucket_name}'"
                )
            raise

    def _get_s3_key(self, document_id: str, file_extension: str, filename: str) -> str:
        return f"{file_extension}/{document_id}/{filename}"

    def _object_exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def exists(self, document_id: str, file_extension: str) -> bool:
        if not self.enabled:
            return False
        # NOTE: document.{ext} is intentionally excluded — the upload flow only saves
        # chunks.json, embeddings.npy, and metadata.json via save_chunks_and_embeddings().
        # LocalStorageBackend matches this check for consistency.
        required_files = ["chunks.json", "embeddings.npy", "metadata.json"]
        for filename in required_files:
            key = self._get_s3_key(document_id, file_extension, filename)
            if not self._object_exists(key):
                logger.debug(f"S3 cache miss for {document_id} (missing: {filename})")
                return False
        logger.debug(f"S3 cache hit for {document_id}")
        return True

    def save_document(
        self, document_id: str, file_path: Path, file_extension: str
    ) -> None:
        if not self.enabled:
            return
        key = self._get_s3_key(
            document_id=document_id,
            file_extension=file_extension,
            filename=f"document.{file_extension}",
        )
        try:
            with open(file_path, "rb") as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=f.read(),
                    ServerSideEncryption="AES256",
                )
            logger.info(f"Uploaded original document to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to upload document to S3: {e}")
            raise

    def save_chunks(
        self, document_id: str, file_extension: str, chunks: list[dict]
    ) -> None:
        if not self.enabled:
            return
        key = self._get_s3_key(
            document_id=document_id,
            file_extension=file_extension,
            filename="chunks.json",
        )
        try:
            body = json.dumps(chunks, indent=2).encode("utf-8")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
            logger.debug(f"Saved {len(chunks)} chunks to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to save chunks to S3: {e}")
            raise

    def save_embeddings(
        self, document_id: str, file_extension: str, embeddings: np.ndarray
    ) -> None:
        if not self.enabled:
            return
        key = self._get_s3_key(document_id, file_extension, "embeddings.npy")
        try:
            buffer = io.BytesIO()
            np.save(buffer, embeddings)
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
                ServerSideEncryption="AES256",
            )
            logger.debug(f"Saved embeddings {embeddings.shape} to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to S3: {e}")
            raise

    def save_metadata(
        self, document_id: str, file_extension: str, metadata: dict
    ) -> None:
        if not self.enabled:
            return
        key = self._get_s3_key(document_id, file_extension, "metadata.json")
        try:
            body = json.dumps(metadata, indent=2).encode("utf-8")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
            logger.debug(f"Saved metadata to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to save metadata to S3: {e}")
            raise

    def load_chunks(self, document_id: str, file_extension: str) -> list[dict]:
        key = self._get_s3_key(document_id, file_extension, "chunks.json")
        from botocore.exceptions import ClientError

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            chunks = json.loads(response["Body"].read().decode("utf-8"))
            logger.debug(f"Loaded {len(chunks)} chunks from S3: {key}")
            return chunks
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Chunks file not found in S3: {key}")
            raise

    def load_embeddings(self, document_id: str, file_extension: str) -> np.ndarray:
        key = self._get_s3_key(document_id, file_extension, "embeddings.npy")
        from botocore.exceptions import ClientError

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            buffer = io.BytesIO(response["Body"].read())
            embeddings = np.load(buffer)
            logger.debug(f"Loaded embeddings {embeddings.shape} from S3: {key}")
            return embeddings
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Embeddings file not found in S3: {key}")
            raise

    def load_metadata(self, document_id: str, file_extension: str) -> dict:
        key = self._get_s3_key(document_id, file_extension, "metadata.json")
        from botocore.exceptions import ClientError

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            metadata = json.loads(response["Body"].read().decode("utf-8"))
            logger.debug(f"Loaded metadata from S3: {key}")
            return metadata
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Metadata file not found in S3: {key}")
            raise

    def delete(self, document_id: str, file_extension: str) -> None:
        if not self.enabled:
            return
        keys_to_delete = [
            {
                "Key": self._get_s3_key(
                    document_id, file_extension, f"document.{file_extension}"
                )
            },
            {"Key": self._get_s3_key(document_id, file_extension, "chunks.json")},
            {"Key": self._get_s3_key(document_id, file_extension, "embeddings.npy")},
            {"Key": self._get_s3_key(document_id, file_extension, "metadata.json")},
        ]
        try:
            self.s3_client.delete_objects(
                Bucket=self.bucket_name, Delete={"Objects": keys_to_delete}
            )
            logger.info(f"Deleted S3 cache for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            raise

    def delete_all(self) -> int:
        if not self.enabled:
            return 0
        total_deleted = 0
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" not in page:
                    continue
                keys_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
                if keys_to_delete:
                    response = self.s3_client.delete_objects(
                        Bucket=self.bucket_name, Delete={"Objects": keys_to_delete}
                    )
                    deleted_count = len(response.get("Deleted", []))
                    total_deleted += deleted_count
            logger.info(f"Cleared entire S3 cache: {total_deleted} objects deleted")
            return total_deleted
        except Exception as e:
            logger.error(f"Failed to delete all from S3: {e}")
            raise

    def list_documents(self) -> list[str]:
        if not self.enabled:
            return []
        document_ids = set()
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    key_parts = obj["Key"].split("/")
                    if len(key_parts) >= 2:
                        document_ids.add(key_parts[1])
            return list(document_ids)
        except Exception as e:
            logger.error(f"Failed to list documents from S3: {e}")
            return []

    def get_stats(self) -> dict:
        if not self.enabled:
            return {"backend": "s3", "status": "disabled"}
        total_size = 0
        total_objects = 0
        doc_type_counts = {}
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    total_size += obj["Size"]
                    total_objects += 1
                    doc_type = (
                        obj["Key"].split("/")[0] if "/" in obj["Key"] else "unknown"
                    )
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            total_size_bytes_value = total_size
            if total_size_bytes_value < 1024:
                total_size_human_value = f"{total_size_bytes_value} B"
            elif total_size_bytes_value < 1024 * 1024:
                total_size_human_value = f"{total_size_bytes_value / 1024:.1f} KB"
            else:
                total_size_human_value = (
                    f"{total_size_bytes_value / (1024 * 1024):.2f} MB"
                )

            stats = {
                "backend": "s3",
                "bucket": self.bucket_name,
                "region": self.region,
                "total_documents": len(self.list_documents()),
                "total_objects": total_objects,
                "total_size_bytes": total_size_bytes_value,
                "total_size_human": total_size_human_value,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "documents_by_type": doc_type_counts,
            }
            logger.info(f"S3 storage stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get S3 stats: {e}")
            return {"backend": "s3", "bucket": self.bucket_name, "error": str(e)}
