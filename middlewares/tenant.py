from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class TenantCtx(BaseHTTPMiddleware):
    """Extract X-Tenant-Id header and stash it on request.state. Reject if missing."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path not in ("/", "/health", "/hc"):
            tenant = request.headers.get("x-tenant-id")
            if not tenant:
                return JSONResponse({"detail": "x-tenant-id header missing"}, status_code=400)
            request.state.tenant_id = tenant.lower()
        else:
            # default tenant context for health checks
            request.state.tenant_id = "default"
        response = await call_next(request)
        return response 