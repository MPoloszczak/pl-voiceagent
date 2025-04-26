import os
import json
import aioredis
from typing import Any


# Create Redis client using URL provided in env var
redis = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)


async def set_json(key: str, obj: Any, ttl: int = 900):
    """Serialize obj to JSON and store with expiry ttl seconds."""
    await redis.set(key, json.dumps(obj), ex=ttl)


async def get_json(key: str):
    """Fetch a key and deserialize JSON, returning None if missing."""
    val = await redis.get(key)
    if val is None:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return None 