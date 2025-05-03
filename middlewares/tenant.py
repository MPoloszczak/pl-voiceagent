from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.tenant_lookup import get_tenant_uuid  # NEW


class TenantCtx(BaseHTTPMiddleware):
    """Resolve tenant context from request host (subdomain) and stash on request.state.

    For HIPAA compliance (§164.308(a)(3)(i) Workforce Security, §164.312(a)(1) Access Control),
    every request is validated against the authoritative tenant registry stored in the encrypted
    Aurora PostgreSQL cluster.  This ensures that data access is restricted to duly provisioned
    tenants and prevents unauthorized cross-tenant leakage.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow unauthenticated access for health checks so the ALB can probe (§164.312(b)).
        if path in ("/", "/health", "/hc"):
            request.state.tenant_id = "default"
            return await call_next(request)

        # Extract tenant from the hostname (e.g. clinicdev.pololabsai.com -> clinicdev)
        host_header = request.headers.get("host", "")
        hostname = host_header.split(":")[0]  # strip port if present
        subdomain_parts = hostname.split(".")
        if len(subdomain_parts) < 3:  # Expect at least subdomain + root domain
            return JSONResponse({"detail": "invalid host header, tenant subdomain missing"}, status_code=400)

        tenant_name = subdomain_parts[0].lower()

        # Validate tenant against RDS mapping table (read-only query)
        try:
            tenant_uuid = await get_tenant_uuid(tenant_name)
        except Exception as exc:
            # Log but avoid leaking internal details to client
            return JSONResponse({"detail": "internal tenant resolution error"}, status_code=500)

        if not tenant_uuid:
            return JSONResponse({"detail": "unknown tenant"}, status_code=400)

        # Stash identifiers for downstream handlers
        request.state.tenant_id = tenant_name
        request.state.tenant_uuid = tenant_uuid

        response = await call_next(request)
        return response 