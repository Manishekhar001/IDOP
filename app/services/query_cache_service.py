import json
import hashlib
import logging
import fnmatch
from typing import Optional, Dict, Any
from app.config import get_settings

logger = logging.getLogger(__name__)


class QueryCacheService:
    """Redis-based cache service for query results, embeddings, and SQL."""

    # Shared class-level local cache so all instances share the same data
    _local_cache_shared: Dict[str, str] = {}

    def __init__(
        self, redis_url: Optional[str] = None, redis_token: Optional[str] = None
    ):
        settings = get_settings()
        self.enabled = False
        self.client = None

        self.stats = {
            "embedding": {"hits": 0, "misses": 0},
            "rag": {"hits": 0, "misses": 0},
            "sql_gen": {"hits": 0, "misses": 0},
            "sql_result": {"hits": 0, "misses": 0},
        }

        redis_url = redis_url or settings.upstash_redis_url
        redis_token = redis_token or settings.upstash_redis_token
        self.use_local = False
        self._local_cache = self._local_cache_shared

        if redis_url and redis_token:
            try:
                from upstash_redis import Redis

                self.client = Redis(url=redis_url, token=redis_token)
                self.client.ping()
                self.enabled = True
                logger.info("Upstash Redis cache connected successfully")

            except ImportError:
                logger.warning(
                    "upstash-redis package not installed. Falling back to local in-memory cache."
                )
                self.use_local = True
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Upstash Redis: {e}. Falling back to local in-memory cache."
                )
                self.use_local = True
        else:
            logger.info(
                "Upstash Redis credentials not configured. Falling back to local in-memory cache."
            )
            self.use_local = True

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()

    def _serialize(self, value: Any) -> str:
        return json.dumps(value, default=str)

    def _deserialize(self, value: str) -> Any:
        return json.loads(value)

    def get(self, key: str, cache_type: str = "rag") -> Optional[Dict]:
        if not self.enabled and not self.use_local:
            self._record_miss(cache_type)
            return None

        if self.use_local:
            if key in self._local_cache:
                self._record_hit(cache_type)
                logger.debug(f"Local Cache HIT: {key}")
                return self._deserialize(self._local_cache[key])
            self._record_miss(cache_type)
            logger.debug(f"Local Cache MISS: {key}")
            return None

        try:
            result = self.client.get(key)
            if result is None:
                self._record_miss(cache_type)
                logger.debug(f"Cache MISS: {key}")
                return None

            self._record_hit(cache_type)
            logger.debug(f"Cache HIT: {key}")
            return self._deserialize(result)

        except Exception as e:
            logger.warning(f"Cache GET error for key {key}: {e}")
            self._record_miss(cache_type)
            return None

    def set(self, key: str, value: Dict, ttl: int, cache_type: str = "rag") -> bool:
        if not self.enabled and not self.use_local:
            return False

        if self.use_local:
            try:
                serialized = self._serialize(value)
                self._local_cache[key] = serialized
                logger.debug(f"Local Cache SET: {key}")
                return True
            except Exception as e:
                logger.warning(f"Local Cache SET error for key {key}: {e}")
                return False

        try:
            serialized = self._serialize(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Cache SET error for key {key}: {e}")
            return False

    def delete(self, pattern: str) -> int:
        if not self.enabled and not self.use_local:
            return 0

        if self.use_local:
            keys_to_delete = [k for k in self._local_cache if fnmatch.fnmatch(k, pattern)]
            for k in keys_to_delete:
                del self._local_cache[k]
            logger.info(f"Local cache invalidation: Deleted {len(keys_to_delete)} keys matching '{pattern}'")
            return len(keys_to_delete)

        try:
            keys = self.client.keys(pattern)
            if not keys:
                return 0

            deleted = 0
            for key in keys:
                self.client.delete(key)
                deleted += 1

            logger.info(f"Cache invalidation: Deleted {deleted} keys matching '{pattern}'")
            return deleted

        except Exception as e:
            logger.warning(f"Cache DELETE error for pattern {pattern}: {e}")
            return 0

    def flush_all(self) -> bool:
        if self.use_local:
            self._local_cache.clear()
            logger.info("Local cache flushed: All keys deleted")
            return True

        if not self.enabled:
            return False

        try:
            self.client.flushdb()
            logger.info("Cache flushed: All keys deleted")
            return True
        except Exception as e:
            logger.warning(f"Cache FLUSH error: {e}")
            return False

    def get_embedding_key(self, text: str) -> str:
        text_hash = self._compute_hash(text)
        return f"embedding:{text_hash}"

    def get_rag_key(self, question: str, top_k: int) -> str:
        question_hash = self._compute_hash(question.lower())
        return f"rag:{question_hash}:{top_k}"

    def get_sql_gen_key(self, question: str) -> str:
        question_hash = self._compute_hash(question.lower())
        return f"sql_gen:{question_hash}"

    def get_sql_result_key(self, sql_query: str) -> str:
        normalized_sql = " ".join(sql_query.strip().lower().split())
        sql_hash = self._compute_hash(normalized_sql)
        return f"sql_result:{sql_hash}"

    def _record_hit(self, cache_type: str):
        if cache_type in self.stats:
            self.stats[cache_type]["hits"] += 1

    def _record_miss(self, cache_type: str):
        if cache_type in self.stats:
            self.stats[cache_type]["misses"] += 1

    def get_stats(self) -> Dict:
        stats_with_rates = {}
        for cache_type, counts in self.stats.items():
            total = counts["hits"] + counts["misses"]
            hit_rate = (counts["hits"] / total * 100) if total > 0 else 0
            stats_with_rates[cache_type] = {
                "hits": counts["hits"],
                "misses": counts["misses"],
                "total_queries": total,
                "hit_rate": f"{hit_rate:.1f}%",
            }

        return {
            "enabled": self.enabled,
            "cache_types": stats_with_rates,
        }

    def reset_stats(self):
        for cache_type in self.stats:
            self.stats[cache_type] = {"hits": 0, "misses": 0}
        logger.info("Cache statistics reset")

    def health_check(self) -> Dict:
        if self.use_local:
            return {"status": "healthy", "message": "Local in-memory cache active", "mode": "local"}
        if not self.enabled:
            return {"status": "disabled", "message": "Redis cache not configured"}

        try:
            self.client.ping()
            return {"status": "healthy", "message": "Redis connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Redis error: {str(e)}"}
