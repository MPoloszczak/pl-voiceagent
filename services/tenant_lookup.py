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


def _resolve_rds_dsn(raw_val: Optional[str]) -> Optional[str]:
    """Return a PostgreSQL DSN usable by SQLAlchemy.

    The *raw_val* may be:
      1. A direct DSN string (e.g. postgresql+asyncpg://user:pass@host/db)
      2. An AWS Secrets Manager ARN whose payload is either that DSN directly
         or a JSON document with a key named "RDS" or "POSTGRES_DSN".

    For HIPAA §164.312(e)(1) Transmission Security, we avoid storing the DSN in
    plain-text environment variables and instead reference a Secrets Manager
    ARN.  This helper resolves the secret at runtime while the container
    assumes the task IAM role that grants *GetSecretValue*.
    """

    if not raw_val:
        return None

    # If the value looks like a Secrets Manager ARN, fetch it
    if raw_val.startswith("arn:aws:secretsmanager"):
        try:
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
            sm_client = boto3.client("secretsmanager", region_name=region)
            resp = sm_client.get_secret_value(SecretId=raw_val)
            secret_str: str = resp.get("SecretString", "")

            if secret_str:
                # If secret payload is JSON, pluck DSN field; otherwise assume raw DSN
                try:
                    payload = json.loads(secret_str)

                    # If secret is already a DSN string under a key
                    embedded_dsn = payload.get("RDS") or payload.get("POSTGRES_DSN")
                    if embedded_dsn:
                        return embedded_dsn

                    # Otherwise attempt to construct DSN from individual parts
                    user = payload.get("username") or payload.get("user")
                    password = payload.get("password") or payload.get("pass")
                    host = payload.get("host") or payload.get("endpoint") or payload.get("hostname")
                    port = payload.get("port", 5432)

                    if user and password and host and DATABASE_NAME:
                        # Use asyncpg driver for async SQLAlchemy
                        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{DATABASE_NAME}"

                    # As a last resort, fall back to treating the whole JSON as raw DSN
                    return secret_str
                except json.JSONDecodeError:
                    return secret_str
        except Exception as exc:
            logger.error(f"Failed to fetch RDS secret {raw_val}: {exc}")
            return None
    # Otherwise assume it is already a DSN
    if raw_val and DATABASE_NAME and '/' not in raw_val.split('@')[-1]:
        # DSN missing db path; append it
        return raw_val.rstrip('/') + f"/{DATABASE_NAME}"

    return raw_val


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