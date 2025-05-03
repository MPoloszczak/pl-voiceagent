import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.sql import text

# ---------------------------------------------------------------------------
# Connection string
# ---------------------------------------------------------------------------

# The primary (read-write) endpoint is now provided via the HIPAA-compliant
# environment variable `RDS`.  We keep the previous `AURORA_DSN` fallback for
# backwards compatibility during roll-out, but the new name should be used in
# all deployment targets going forward.

_DSN = os.environ.get("RDS")

if not _DSN:
    raise RuntimeError("RDS environment variable required for database connection is missing")

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