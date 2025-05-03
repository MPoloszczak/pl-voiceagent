import os, json
import redis.asyncio as redis
from typing import Any
from utils import logger


# ---------------------------------------------------------------------------
# Resolve Redis URL – handles three scenarios:
#   1) Secret injected as plain host:port (e.g. "cache.example.com:6379")
#   2) Secret injected as full URI      (e.g. "rediss://cache.example.com:6379/0")
#   3) Secret injected as JSON payload  (because the entire Secrets Manager JSON
#      was mapped to the env var).  We inspect keys "REDIS_URL" or "REDIS".
# ---------------------------------------------------------------------------

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

# When using ElastiCache in-transit encryption, the scheme should be 'rediss://'
redis_client = redis.from_url(
    redis_url,
    decode_responses=True,
    ssl_cert_reqs=None  # pass a CA bundle path here if you require server cert validation
)


async def set_json(key: str, obj: Any, ttl: int = 900):
    """Serialize obj to JSON and store with expiry ttl seconds."""
    await redis_client.set(key, json.dumps(obj), ex=ttl)


async def get_json(key: str):
    """Fetch a key and deserialize JSON, returning None if missing."""
    val = await redis_client.get(key)
    if val is None:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return None 