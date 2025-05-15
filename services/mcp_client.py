import os
import httpx
import json
from typing import List
from agents import Tool

# Centralised Redis cache helpers (with TLS + automatic in-memory fallback)
from services.cache import get_json, set_json

# A slightly longer read timeout (15s) accommodates SSE handshakes while
# keeping connect / write timeouts strict.  HIPAA §164.312(e)(1) requires
# encryption in-transit but does not prescribe RTTs – we tune for UX.

_timeout = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=None)
_client = httpx.AsyncClient(timeout=_timeout, http2=True, follow_redirects=True)

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
    base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com")

    async def _remote_fetch() -> List[Tool]:
        """Fetch toolbox over HTTPS/SSE per MCP best-practices.

        The endpoint *may* emit `text/event-stream` – in that case we read the
        first `data:` frame (JSON-RPC payload) and immediately close the
        connection.  This matches the reference FastMCP implementation and
        avoids holding an open stream longer than needed (least-privilege &
        resource stewardship – HIPAA §164.308(a)(3)(i)).
        """

        url = f"{base.rstrip('/')}/{tenant}/toolbox.json"

        headers = {"Accept": "application/json, text/event-stream"}

        async with _client.stream("GET", url, headers=headers, follow_redirects=True) as resp:
            resp.raise_for_status()

            ctype = resp.headers.get("content-type", "")

            if "text/event-stream" in ctype:
                # Iterate until first `data:` frame.
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        # comment / heartbeat per SSE spec
                        continue
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        data = json.loads(payload)
                        return [Tool(**item) for item in data]

                raise ValueError("SSE stream closed before toolbox payload received")

            # Non-SSE JSON response.
            data = await resp.json()
            return [Tool(**item) for item in data]

    try:
        tools = await _remote_fetch()

        # Cache raw JSON so it can be reconstructed easily and is JSON-serialisable.
        def _dump(t: Tool):
            if hasattr(t, "model_dump"):
                return t.model_dump()
            if hasattr(t, "dict"):
                return t.dict()
            return t.__dict__

        await set_json(cache_key, [_dump(t) for t in tools], ttl=900)
        return tools

    except (httpx.HTTPError, OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error(
            f"Failed to fetch tools for tenant '{tenant}' from MCP: {exc}. Using cached/empty toolbox."
        )

        if cached:
            try:
                return [Tool(**item) for item in cached]
            except Exception:
                pass  # fall through

        return [] 