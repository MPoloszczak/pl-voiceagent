from __future__ import annotations

"""Cross-worker CallSid ➜ tenant mapping helpers.

Rationale
---------
The webhook that receives the initial voice call (``/voice``) and the
WebSocket handler (``/twilio-stream``) may be served by **different**
Uvicorn workers or even separate AWS Fargate tasks.  Relying on an
in-memory ``call_to_tenant`` dict therefore breaks under horizontal
scaling – the later handler cannot look up the tenant and falls back to
the WebSocket host ("stream"), causing MCP look-ups like
``/stream/toolbox.json`` to 404 and the OpenAI Agent to run without the
expected tools.

To solve this we persist the mapping in Redis (encrypted in-transit &
at-rest, see ``services.cache``) for a short TTL so that any worker can
retrieve it.  We store a **single JSON string** to keep the schema
simple and avoid ePHI exposure – the CallSid itself is not considered
PHI per §164.514(b)(2) and is required for operations.
"""
from typing import Optional

from services.cache import get_json, set_json
from utils import logger

_KEY_FMT = "tenant_map:{call_sid}"
_TTL = 3600  # 1 hour is more than enough for a single phone call


async def set_tenant(call_sid: str, tenant: str, ttl: int = _TTL) -> None:
    """Persist **tenant** for **call_sid** with an expiring TTL."""
    try:
        await set_json(_KEY_FMT.format(call_sid=call_sid), {"t": tenant}, ttl=ttl)
        logger.debug("✅ Stored tenant mapping %s ➜ %s (TTL %ss)", call_sid, tenant, ttl)
    except Exception as exc:
        # Non-fatal: we still have in-memory mapping; log and continue.
        logger.error("❌ Failed to persist tenant mapping for %s: %s", call_sid, exc)


async def get_tenant(call_sid: str) -> Optional[str]:
    """Return tenant for **call_sid** or *None* if unset / expired."""
    try:
        data = await get_json(_KEY_FMT.format(call_sid=call_sid))
        if isinstance(data, dict):
            return data.get("t")
        if isinstance(data, str):  # legacy future-proofing
            return data
    except Exception as exc:
        logger.error("❌ Failed to read tenant mapping for %s: %s", call_sid, exc)
    return None 