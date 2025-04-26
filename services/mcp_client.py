import os
import httpx
from typing import List
from agents import Tool

_timeout = httpx.Timeout(5.0)
_client = httpx.AsyncClient(timeout=_timeout)


async def tools_for(tenant: str) -> List[Tool]:
    """Fetch a tenant's toolbox.json from the MCP service and return Tool objects."""
    base = os.environ.get("MCP_BASE", "http://mcp.local")
    url = f"{base.rstrip('/')}/{tenant}/toolbox.json"
    resp = await _client.get(url)
    resp.raise_for_status()
    data = resp.json()
    return [Tool(**item) for item in data] 