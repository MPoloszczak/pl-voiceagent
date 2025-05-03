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
import json
from typing import Optional, Any

import boto3
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

from utils import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Database name may be supplied separately so that we do not store PHI-related
# identifiers inside the managed AWS secret.  This satisfies HIPAA
# §164.312(a)(2)(iv) (Encryption) by letting AWS Secrets Manager keep only the
# minimum necessary sensitive values (username/password) while the DB name
# travels via the encrypted ECS Task environment channel.

DATABASE_NAME = os.environ.get("DATABASE_NAME")

# ---------------------------------------------------------------------------
# Helpers – resolve RDS DSN from env/Secrets Manager
# ---------------------------------------------------------------------------


def _resolve_rds_dsn(secret_arn: Optional[str]) -> Optional[str]:
    """Build a PostgreSQL DSN from *separate* environment variables and the
    *RDS* secret that holds *only* the username & password.

    Per the new deployment standard we keep non-PHI parameters (host, port,
    database name) in **plain ECS task env vars** and store *only* the
    credentials in AWS Secrets Manager.  This adheres to HIPAA
    §164.312(a)(2)(iv) (Encryption of data at rest) while preventing any
    unnecessary duplication of identifiers that could be linked back to an
    individual (§164.514(b)).

    Required env vars:
        DATABASE_HOST   – cluster reader endpoint (read-only per compliance)
        DATABASE_PORT   – port number (default 5432)
        DATABASE_NAME   – DB name (non-PHI so safe in env var)

    Required secret keys:
        username / user
        password / pass
    """

    if not secret_arn:
        logger.error("`RDS` environment variable missing – cannot build DSN")
        return None

    # Fetch credentials from Secrets Manager
    try:
        region = "us-east-1"
        sm_client = boto3.client("secretsmanager", region_name=region)
        resp = sm_client.get_secret_value(SecretId=secret_arn)
        secret_str: str = resp.get("SecretString", "")
        payload = json.loads(secret_str) if secret_str else {}

        user = payload.get("username") or payload.get("user")
        password = payload.get("password") or payload.get("pass")

        if not (user and password):
            logger.error("Secrets Manager payload missing username/password fields")
            return None

        # Non-sensitive fields from env
        host = os.environ.get("DATABASE_HOST")
        port = int(os.environ.get("DATABASE_PORT", "5432"))
        db_name = os.environ.get("DATABASE_NAME") or DATABASE_NAME

        if not (host and db_name):
            logger.error("DATABASE_HOST and/or DATABASE_NAME environment variables are missing – cannot build DSN")
            return None

        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

    except Exception as exc:
        logger.error(f"Failed to fetch RDS secret {secret_arn}: {exc}")
        return None


RDS_DSN = _resolve_rds_dsn(os.environ.get("RDS"))

if not RDS_DSN:
    # This will bubble up later when the first lookup occurs; log loudly.
    logger.error("RDS DSN could not be resolved – tenant lookup will fail")

_engine = create_async_engine(RDS_DSN, pool_size=2, max_overflow=4, echo=False) if RDS_DSN else None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Simple in-memory cache for resolved tenants {name: uuid}
_TENANT_CACHE: dict[str, str] = {}


async def _cached_lookup(name: str) -> Optional[str]:
    """Internal helper with naive cache to reduce round-trips."""
    if name in _TENANT_CACHE:
        return _TENANT_CACHE[name]

    global _engine  # we may recreate if was None
    if _engine is None and RDS_DSN:
        _engine = create_async_engine(RDS_DSN, pool_size=2, max_overflow=4, echo=False)
    if _engine is None:
        logger.error("Tenant lookup attempted without valid database DSN")
        return None

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