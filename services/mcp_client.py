import os
import json
import asyncio
from typing import List

from agents import Tool

# New MCP Streamable HTTP client (OpenAI Agents SDK >= 0.0.15)
# Fallbacks to dict typing when the SDK is unavailable (CI mocks).
try:
    from agents.mcp.server import (
        MCPServerStreamableHttp,
        MCPServerStreamableHttpParams,
    )
except ImportError:  # pragma: no cover – SDK not available in CI mocks
    MCPServerStreamableHttp = None  # type: ignore
    MCPServerStreamableHttpParams = dict  # type: ignore

# MCP spec version header (kept in-sync with oai.MCP_PROTOCOL_VERSION)
MCP_PROTOCOL_VERSION = "2025-03-26"

import httpx  # retained for backwards-compat TTL caches – not strictly required

# Centralised Redis cache helpers (with TLS + automatic in-memory fallback)
from services.cache import get_json, set_json

# A slightly longer read timeout (15s) accommodates SSE handshakes while
# keeping connect / write timeouts strict.  HIPAA §164.312(e)(1) requires
# encryption in-transit but does not prescribe RTTs – we tune for UX.

_timeout = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=None)
_client = httpx.AsyncClient(timeout=_timeout, http2=True, follow_redirects=True)  # kept for legacy callers

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
    # 2) Fetch via MCP JSON-RPC list_tools call (preferred as of 2025-03-26)
    # -------------------------------------------------------------------

    async def _remote_fetch() -> List[Tool]:
        """Request `list_tools` over the per-tenant Streamable-HTTP endpoint.

        The OpenAI Agents SDK >= 0.0.15 transparently handles JSON-RPC framing
        and optional SSE upgrades, ensuring full compliance with the 2025-03-26
        MCP specification.  We prefer this over custom HTTP clients now that
        the legacy `/toolbox.json` endpoint has been removed.
        """

        if MCPServerStreamableHttp is None:
            raise RuntimeError("MCPStreamableHttp class unavailable – cannot fetch tools list")

        base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com").rstrip("/")

        params: "MCPServerStreamableHttpParams" = {
            "url": f"{base}/{tenant}/mcp",
            "headers": {
                "Mcp-Protocol-Version": MCP_PROTOCOL_VERSION,
                "Mcp-Session-Id": None,
            },
        }

        server = MCPServerStreamableHttp(
            params=params,
            cache_tools_list=False,  # We handle our own Redis cache
            name=f"{tenant}-tools-http",
        )

        try:
            # Perform the JSON-RPC call with a short timeout (6s prevents UX lag)
            tools: List[Tool] = await asyncio.wait_for(server.list_tools(), timeout=6.0)
            return tools
        finally:
            # Explicitly close underlying HTTP session if SDK exposes it
            aclose = getattr(server, "aclose", None)
            if asyncio.iscoroutinefunction(aclose):
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 – non-fatal cleanup
                    pass

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