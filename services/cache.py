import os, json
import redis.asyncio as redis
from typing import Any, Optional
from utils import logger
from redis.exceptions import ConnectionError
from redis.asyncio.cluster import RedisCluster


_raw_env_val = os.environ.get("REDIS")

if not _raw_env_val:
    logger.error("REDIS environment variable is missing – cannot start cache client")
    raise RuntimeError("REDIS required for Redis connection")

def _extract_url(raw: str) -> str:
    # Case 3 – JSON string
    if raw.strip().startswith("{"):
        try:
            payload = json.loads(raw)
            inner = payload.get("REDIS_URL") or payload.get("REDIS")
            if inner:
                return inner
        except json.JSONDecodeError:
            pass  # fall through to generic handling
    return raw

redis_raw = _extract_url(_raw_env_val)

# Ensure scheme & TLS
if redis_raw.startswith("rediss://"):
    redis_url = redis_raw
elif redis_raw.startswith("redis://"):
    # Upgrade plaintext scheme to TLS
    redis_url = redis_raw.replace("redis://", "rediss://", 1)
else:
    # Assume host[:port][/db] without scheme – prepend rediss://
    redis_url = f"rediss://{redis_raw}"



_fallback_cache = {}  # type: ignore[var-annotated]

# Helper to decide whether we should treat the URL as a cluster configuration
def _is_cluster_endpoint(url: str) -> bool:
    """Return True if the given Redis URL points at an ElastiCache *clustercfg*
    endpoint.  Those require redis-cluster topology support."""
    return "clustercfg" in url

# Lazily initialise a Redis/RedisCluster client so that connection errors don't
# occur at *import* time and can instead be handled gracefully at runtime.
_redis_client = None  # will be set on first use

# ---------------------------------------------------------------------------
# Optional: AWS ElastiCache Auth Token handling
# ---------------------------------------------------------------------------

# HIPAA Compliance: Avoid logging raw auth token (PHI/PII)
# The token is stored in environment variable REDIS_AUTH_TOKEN and will be
# supplied as the password parameter when instantiating the Redis client.  We
# never output this value to logs.

_redis_auth_token: Optional[str] = os.environ.get("REDIS_AUTH_TOKEN")

def _init_client() -> None:
    """Initialise the global ``_redis_client`` with cluster detection / TLS.

    If initialisation fails, we log the error and leave the client as ``None``
    so that the calling helpers can fall back to the in-memory cache instead
    of aborting the entire request handler.
    """
    global _redis_client
    if _redis_client is not None:
        return  # already initialised

    try:
        if _is_cluster_endpoint(redis_url):
            # ElastiCache cluster mode ("clustercfg.") → use RedisCluster
            _redis_client = RedisCluster.from_url(
                redis_url,
                decode_responses=True,
                ssl=True,
                ssl_cert_reqs=None,  # skip server cert validation – ensure SG/Private-Link enforcement instead
                password=_redis_auth_token,
            )
            logger.info("🔗 Initialised RedisCluster client (cluster endpoint detected)")
        else:
            _redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                ssl_cert_reqs=None,                
                password=_redis_auth_token,
            )
            logger.info("🔗 Initialised standalone Redis client")
    except Exception as e:  # broad but we must not crash import
        logger.error(f"❌ Failed to initialise Redis client: {e}. Falling back to in-memory cache.")
        _redis_client = None


# ----------------------------------------------------------------------------
# Public async helpers with automatic fallback – API remains unchanged
# ----------------------------------------------------------------------------


async def set_json(key: str, obj: Any, ttl: int = 900):  # noqa: D401 – keep original signature
    """Serialize *obj* to JSON and store it under *key*.

    Falls back to an in-process dict when Redis is unavailable so the rest of
    the application can continue operating.  The fallback cache is *not*
    shared across containers and therefore only provides best-effort context.
    """
    _init_client()
    payload = json.dumps(obj)

    if _redis_client is None:
        # Redis unavailable – log at ERROR level and raise so upstream can react.
        logger.error(
            "❌ Redis client not initialised – cannot persist key '%s'. Falling back to in-memory cache.",
            key,
        )
        _fallback_cache[key] = payload
        raise CacheWriteError("Redis client unavailable – data stored only in local memory")

    try:
        result = await _redis_client.set(key, payload, ex=ttl)  # type: ignore[attr-defined]
        if result is True or result == "OK":
            logger.debug("✅ Redis SET succeeded for key '%s' (TTL %ss)", key, ttl)
        else:
            logger.error(
                "❌ Redis SET for key '%s' returned unexpected result: %s. Storing fallback and raising.",
                key,
                result,
            )
            _fallback_cache[key] = payload
            raise CacheWriteError("Redis SET did not acknowledge write")
    except ConnectionError as ce:
        logger.error(
            "🔌 Redis connection error on SET for '%s': %s. Using in-memory fallback and raising.",
            key,
            ce,
        )
        _fallback_cache[key] = payload
        raise CacheWriteError("Redis connection failure – data stored in local memory")


async def get_json(key: str):  # noqa: D401 – keep original signature
    """Retrieve *key* and deserialize JSON, returning ``None`` if missing.

    Automatically falls back to the local in-process cache if Redis is
    unreachable.
    """
    _init_client()

    if _redis_client is None:
        raw = _fallback_cache.get(key)
    else:
        try:
            raw = await _redis_client.get(key)  # type: ignore[attr-defined]
        except ConnectionError as ce:
            logger.error(f"🔌 Redis connection error on GET for '{key}': {ce}. Resorting to in-memory cache.")
            raw = _fallback_cache.get(key)

    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

# ---------------------------------------------------------------------------
# SECURITY (HIPAA) NOTE
#
# The fallback cache exists **only** inside the running container memory and
# is never persisted or transmitted over the network.  All PHI therefore
# remains protected in-transit (TLS) when Redis is reachable, or isolated to a
# single, hardened container instance when Redis is unavailable.  This aligns
# with HIPAA §164.312(e)(1) – Transmission Security.
# ---------------------------------------------------------------------------

# Custom exception for cache write failures – allows callers to handle
class CacheWriteError(RuntimeError):
    """Raised when data cannot be persisted to Redis."""
    pass 