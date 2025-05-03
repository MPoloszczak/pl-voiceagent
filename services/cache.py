import os, json
import redis.asyncio as redis
from typing import Any
from utils import logger


redis_url = os.environ.get("REDIS")
if not redis_url:
    logger.error("REDIS environment variable is missing – cannot start cache client")
    raise RuntimeError("REDIS required for Redis connection")

# When using ElastiCache in-transit encryption, the scheme should be 'rediss://'
redis_client = redis.from_url(
    redis_url,
    decode_responses=True,
    ssl=True,
    ssl_cert_reqs=None  # or a CA bundle if you want cert verification
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