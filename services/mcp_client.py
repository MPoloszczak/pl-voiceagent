import os
import httpx
from typing import List
from agents import Tool

_timeout = httpx.Timeout(5.0)
_client = httpx.AsyncClient(timeout=_timeout, http2=True)

# Centralised Redis cache helpers (with TLS + automatic in-memory fallback)
from services.cache import get_json, set_json


async def tools_for(tenant: str) -> List[Tool]:
    """Return the tool list for *tenant*.

    If the MCP service is unreachable we fall back to:
    1. The most recently cached copy for this tenant (if any)
    2. An **empty** list, allowing the upstream agents to continue operating.

    This mirrors the fault-tolerant approach used in ``services.cache`` and
    ensures we do not interrupt the user experience – a core HIPAA safeguard
    (§164.308(a)(1)(ii)(A)) requiring availability of ePHI in spite of system
    failures.  Toolbox metadata contains no ePHI, so a plaintext fallback is
    acceptable; when the MCP endpoint is provisioned we attempt TLS (https)
    by default to maintain encryption in-transit (§164.312(e)(1)).
    """

    from utils import logger  # local import to avoid cycles

    cache_key = f"mcp:tenant:{tenant}:tools"

    # -------------------------------------------------------------------
    # 1) Attempt to read toolbox from Redis (encrypted in-transit & at rest)
    # -------------------------------------------------------------------
    cached = await get_json(cache_key)
    if cached:
        try:
            return [Tool(**item) for item in cached]
        except Exception as parse_exc:  # malformed cache – log and continue
            logger.warning(
                f"Cached toolbox for tenant '{tenant}' is invalid: {parse_exc}. Will attempt remote fetch."
            )

    # -------------------------------------------------------------------
    # 2) Fetch from MCP service over TLS, then persist to Redis (TTL 15 min)
    # -------------------------------------------------------------------
    base = os.environ.get("MCP_BASE", "https://mcp.local")  # default to TLS
    url = f"{base.rstrip('/')}/{tenant}/toolbox.json"

    try:
        resp = await _client.get(url)
        resp.raise_for_status()
        data = resp.json()
        tools = [Tool(**item) for item in data]

        # Cache raw JSON so it can be reconstructed easily and is JSON-serialisable.
        await set_json(cache_key, data, ttl=900)  # 15-minute TTL
        return tools

    except (httpx.HTTPError, OSError) as exc:
        logger.error(
            f"Failed to fetch tools for tenant '{tenant}' from MCP: {exc}. Using cached/empty toolbox."
        )

        if cached:
            # We already tried to parse earlier but may return anyway (better than empty)
            try:
                return [Tool(**item) for item in cached]
            except Exception:
                pass  # fall through to empty

        return [] 