from __future__ import annotations

"""Tenant lookup service

Provides read-only resolution of tenant name ➜ uuid mapping from the shared metadata
schema in the Aurora PostgreSQL cluster.  All traffic is encrypted in-transit
(TLS) and the database credentials are obtained via the `RDS` environment
variable which points to the *read-only* instance ARN per HIPAA §164.312(e)(1)
Transmission Security.

NOTE: This module intentionally keeps read operations separate from the
multi-tenant session code in `db/session.py` to guarantee the lookup happens
with the cluster-level `public` search_path and without switching roles.
"""

import os
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

from utils import logger

# ---------------------------------------------------------------------------
# Engine (read-only)
# ---------------------------------------------------------------------------

RDS_DSN = os.environ.get("RDS")
if not RDS_DSN:
    logger.error("RDS env var missing – tenant lookup will fail")

_engine = create_async_engine(RDS_DSN, pool_size=2, max_overflow=4, echo=False)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Simple in-memory cache for resolved tenants {name: uuid}
_TENANT_CACHE: dict[str, str] = {}


async def _cached_lookup(name: str) -> Optional[str]:
    """Internal helper with naive cache to reduce round-trips."""
    if name in _TENANT_CACHE:
        return _TENANT_CACHE[name]
    async with _engine.begin() as conn:
        # Explicitly set search_path to public to avoid tenant schemas
        await conn.execute(text("SET search_path TO public"))
        result = await conn.execute(
            text("SELECT tenant_id FROM public.tenants WHERE name = :name LIMIT 1"),
            {"name": name},
        )
        row = result.fetchone()
        uuid_val = str(row[0]) if row else None
        if uuid_val:
            _TENANT_CACHE[name] = uuid_val
        return uuid_val


async def get_tenant_uuid(name: str) -> Optional[str]:
    """Return the tenant's UUID (as str) for the given *name* or *None* if not found.

    Args:
        name: Sub-domain portion of the incoming request (lowercase)

    Returns:
        str | None – UUID string corresponding to the tenant.
    """
    try:
        uuid_val = await _cached_lookup(name)
        if uuid_val:
            logger.debug(f"Resolved tenant '{name}' ➜ {uuid_val}")
        else:
            logger.info(f"Tenant '{name}' not found in registry")
        return uuid_val
    except Exception as exc:
        logger.error(f"Tenant lookup failed for '{name}': {exc}")
        raise 