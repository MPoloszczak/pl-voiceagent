import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.sql import text

# ---------------------------------------------------------------------------
# Resolve DSN from env/Secrets Manager (shared helper)
# ---------------------------------------------------------------------------

from services.tenant_lookup import _resolve_rds_dsn  # pylint: disable=protected-access

# The primary (read-write) endpoint is provided via the HIPAA-compliant
# environment variable `RDS`.  Its value may be either:
#   1. A plain PostgreSQL DSN string; or
#   2. An AWS Secrets Manager ARN whose payload contains the DSN or the
#      components needed to build one.
#
# To maintain compliance with HIPAA §164.312(a)(2)(iv) (Encryption) we prefer
# option ② because it keeps the credentials encrypted at rest and in transit.

_RAW_RDS_VAL = os.environ.get("RDS")

# Attempt to resolve the raw value into a DSN using the shared helper
_DSN = _resolve_rds_dsn(_RAW_RDS_VAL)

if not _DSN:
    raise RuntimeError("RDS DSN could not be resolved from the `RDS` environment variable or its referenced secret")

engine = create_async_engine(_DSN, pool_size=10, max_overflow=20, echo=False)


@asynccontextmanager
async def session_for(tenant: str):
    """Yield an AsyncSession scoped to the given tenant (schema-per-tenant, RLS).
    CALLER must `async with session_for(tenant) as session:`
    """
    async with engine.connect() as conn:
        # switch tenant role & search_path then expose tenant via GUC
        await conn.execute(text(f"SET ROLE {tenant}_role"))
        await conn.execute(text(f"SET search_path TO {tenant},public"))
        await conn.execute(text("SET app.tenant=:t"), {"t": tenant})
        async with AsyncSession(bind=conn, expire_on_commit=False) as session:
            yield session 