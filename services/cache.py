import os, json
import redis.asyncio as redis
from typing import Any, Optional
from utils import logger
from redis.exceptions import ConnectionError, AuthenticationError
from redis.asyncio.cluster import RedisCluster
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Local ElastiCache IAM credential provider (avoids redis-py 5.4 dependency)
# ---------------------------------------------------------------------------
# Modern redis-py (≥5.3) ships an official ``CredentialProvider`` interface that we
# subclass for ElastiCache IAM auth.  The legacy import shim has been removed to
# simplify the codebase and follow upstream best-practice – see
# https://redis.readthedocs.io/en/stable/examples/connection_examples.html#Connecting-to-a-redis-instance-with-ElastiCache-IAM-credential-provider
#
# HIPAA §164.312(e)(1) – we always sign via HTTPS and enforce TLS when the
# client connects below.

from botocore.model import ServiceId  # type: ignore
from botocore.signers import RequestSigner  # type: ignore
import botocore.session  # type: ignore

from redis.credentials import CredentialProvider as _BaseCredProvider  # type: ignore


class _ElastiCacheIAMProvider(_BaseCredProvider):
    """Generate SigV4 IAM authentication tokens for ElastiCache (Redis ≥7).

    The AWS docs now recommend signing the **replication-group ID** (a.k.a.
    *cache name*) rather than the DNS endpoint when building the presigned
    request URL [1].  Older examples that signed the full *clustercfg* hostname
    stop working once IAM auth is enforced on the server side, hence the
    refactor.

    """

    _debug_enabled = os.getenv("REDIS_DEBUG") == "1"

    def __init__(self, *, user_id: str, cluster_name: str, region: str):
        self._user_id = user_id
        self._cluster_name = cluster_name  # replication-group id, *not* the hostname
        self._region = region

        # Prevent weak-ref GC (see redis-py issue #2987)
        self._session = botocore.session.get_session()
        self._signer = RequestSigner(
            ServiceId("elasticache"),
            self._region,
            "elasticache",
            "v4",
            self._session.get_credentials(),
            self._session.get_component("event_emitter"),
        )

        if self._debug_enabled:
            logger.debug(
                "[REDIS-Debug] IAM provider init – user=%s cluster_name=%s region=%s",
                self._user_id,
                self._cluster_name,
                self._region,
            )

    # redis-py expects Tuple[str,str]
    def get_credentials(self):  # type: ignore[override]
        from urllib.parse import urlencode, urlunparse, ParseResult

        query = {"Action": "connect", "User": self._user_id}
        url = urlunparse(
            ParseResult(
                scheme="https",
                netloc=self._cluster_name,  # ← per AWS docs
                path="/",
                params="",
                query=urlencode(query),
                fragment="",
            )
        )

        signed_url = self._signer.generate_presigned_url(
            {"method": "GET", "url": url, "body": {}, "headers": {}, "context": {}},
            operation_name="connect",
            expires_in=900,
            region_name=self._region,
        )

        token = signed_url.removeprefix("https://")

        if self._debug_enabled:
            import hashlib, base64 as _b64

            digest = hashlib.sha256(token.encode()).digest()
            logger.debug("[REDIS-Debug] Generated token hash=%s", _b64.b32encode(digest)[:16].decode())

        # Redis AUTH <username> <token>
        return (self._user_id, token)


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
#AWS ElastiCache IAM authentication (RBAC)
# ---------------------------------------------------------------------------


_redis_iam_user: Optional[str] = os.environ.get("REDIS_IAM_USER")
_redis_iam_region: Optional[str] = (
    os.environ.get("REDIS_IAM_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
)

# singleton provider instance
_iam_provider: Optional[_ElastiCacheIAMProvider] = None  # initialised lazily


def _get_iam_provider(cluster_name: str) -> Optional[_ElastiCacheIAMProvider]:
    """Return a singleton IAMCredentialsProvider or None when IAM env vars absent.

    *cluster_name* must be the replication-group ID (e.g. ``pl-voiceagent-redis``),
    **not** the clustercfg DNS hostname.
    """
    global _iam_provider
    if _redis_iam_user is None:
        return None

    if _iam_provider is None:
        _iam_provider = _ElastiCacheIAMProvider(
            user_id=_redis_iam_user,
            cluster_name=cluster_name,
            region=_redis_iam_region or "us-east-1",
        )
    return _iam_provider



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
        # ------------------------------------------------------------------
        # Identify host / port and authentication strategy
        # ------------------------------------------------------------------
        parsed = urlparse(redis_url)
        host = parsed.hostname or redis_url.replace("rediss://", "").replace("redis://", "").split("/")[0].split(":")[0]
        port = parsed.port or 6379
        is_ssl = True  # enforce TLS per HIPAA transmission-security requirements

        # Extract the *replication-group id* for token signing.  Example:
        #   clustercfg.pl-voiceagent-redis.xxxxxx.cache.amazonaws.com → pl-voiceagent-redis
        rg_id = host
        if host.startswith("clustercfg."):
            rg_id = host.split(".")[0].removeprefix("clustercfg.")

        provider = _get_iam_provider(rg_id)
        if provider is None:
            logger.error("❌ REDIS_IAM_USER not configured – IAM authentication is mandatory. Falling back to in-memory cache.")
            _redis_client = None
            return

        if os.getenv("REDIS_DEBUG") == "1":
            logger.info(
                "[REDIS-Debug] Creating redis client – host=%s port=%s cluster=%s tls=%s user=%s region=%s",
                host,
                port,
                _is_cluster_endpoint(host),
                is_ssl,
                _redis_iam_user,
                _redis_iam_region,
            )

        auth_kwargs: dict[str, Any] = {"credential_provider": provider}

        # ------------------------------------------------------------------
        # Instantiate client (cluster vs standalone)
        # ------------------------------------------------------------------
        if _is_cluster_endpoint(host):
            _redis_client = RedisCluster(
                host=host,
                port=port,
                ssl=is_ssl,
                decode_responses=True,
                ssl_cert_reqs=None,
                **auth_kwargs,
            )
            logger.info("🔗 Initialised RedisCluster client using IAM authentication")
        else:
            _redis_client = redis.Redis(
                host=host,
                port=port,
                ssl=is_ssl,
                decode_responses=True,
                ssl_cert_reqs=None,
                **auth_kwargs,
            )
            logger.info("🔗 Initialised standalone Redis client using IAM authentication")
    except AuthenticationError as ae:
        logger.error(
            "🛂 Redis authentication failed for IAM user '%s': %s. Check that the ElastiCache user exists, is enabled, and is configured for IAM authentication.",
            _redis_iam_user or "<unset>",
            ae,
        )
        _redis_client = None
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