import asyncio
import httpx
from openai import AsyncOpenAI

from agents import Agent, Runner, set_default_openai_client
from openai.types.responses import ResponseTextDeltaEvent

from utils import logger
# Dynamic MCP integration per tenant
import os


try:
    from agents.mcp.server import (
        MCPServerStreamableHttp,
        MCPServerStreamableHttpParams,
    )
except ImportError:  # pragma: no cover – SDK not available in CI mocks
    MCPServerStreamableHttp = None  # type: ignore
    MCPServerStreamableHttpParams = dict  # type: ignore

# Protocol version header used for all outbound MCP requests.
MCP_PROTOCOL_VERSION = "2025-03-26"

from services.mcp_client import tools_for

# ------------------------------------------------------------------
# Single Medspa assistant agent (dynamic toolbox set per-tenant)
# ------------------------------------------------------------------

# Main conversational agent
medspa_agent = Agent(
    name="Medspa Agent",
    instructions="""You are a friendly medspa assistant. Your role is to provide information about the medspa, answer questions, 
and build rapport with potential clients. 

Information about our medspa:
- We offer a range of services including facials, botox, fillers, laser treatments, and skin consultations
- We have licensed medical professionals on staff
- We use only premium products and advanced technology
- First consultations are complimentary

When responding to the user:
1. Be warm, friendly, and professional
2. Match the user's communication style and tone to build rapport
3. Provide helpful, accurate information about our services
4. Subtly encourage the user to book a consultation but avoid being pushy
5. If the user expresses interest in booking, suggest they schedule a consultation

Remember to be conversational, not corporate. Make the user feel valued and understood.
""",
    model="gpt-4o"
)


_httpx_client: httpx.AsyncClient = httpx.AsyncClient()


_oai_client: AsyncOpenAI = AsyncOpenAI(http_client=_httpx_client)

# Register the client once for the entire process (thread-safe).
set_default_openai_client(_oai_client)

class StreamingHandle:
    """Wrap an async generator to allow cancellation of the LLM stream."""
    def __init__(self, agen):
        self._agen = agen
    def cancel(self):
        """Cancel the async generator to stop the LLM streaming."""
        try:
            aclose = getattr(self._agen, 'aclose', None)
            if aclose:
                # schedule generator close without awaiting
                try:
                    asyncio.create_task(self._agen.aclose())
                except RuntimeError:
                    pass
        except Exception:
            pass


async def _init_mcp_server(agent: Agent, tenant_id: str) -> None:
    """Initialise and attach a *Streamable HTTP* ``MCPServerStreamableHttp`` instance.

    As recommended by the 2025-03-26 MCP specification (§3.1.4), we prefer the
    consolidated HTTP+JSON-RPC transport which optionally upgrades to SSE for
    streaming.  This approach eliminates the task-group race condition present
    in the legacy SSE-only client implementation (see Anthropic Agents SDK
    issue #1390) and therefore provides a more robust connection strategy for
    latency-sensitive ECS workloads.  No alternative transports are attempted:
    if the HTTP initialisation fails we propagate the error so that the caller
    can respond appropriately (in accordance with the "no fallbacks" policy).
    """

    # Remove any previous server registration to avoid stale handles across
    # multiple tenant invocations within the same process.
    try:
        agent.mcp_servers.clear()  # type: ignore[attr-defined]
    except AttributeError:
        pass


    if MCPServerStreamableHttp is None:
        logger.error(
            "MCPServerStreamableHttp class not available – cannot initialise MCP HTTP transport."
        )
        raise RuntimeError("MCP HTTP transport unavailable in current runtime")

    base = os.getenv("MCP_BASE", "https://mcp.pololabsai.com").rstrip("/")

    params: "MCPServerStreamableHttpParams" = {
        "url": f"{base}/{tenant_id}/mcp",
        "headers": {
            "Mcp-Protocol-Version": MCP_PROTOCOL_VERSION,
        },
    }

    server = MCPServerStreamableHttp(
        params=params,
        cache_tools_list=True,
        name=f"{tenant_id}-mcp-http",
    )

    # Some SDK versions expose an async connect(); use it when present to
    # perform eager endpoint discovery and auth validation.
    try:
        if hasattr(server, "connect") and asyncio.iscoroutinefunction(server.connect):
            await asyncio.wait_for(server.connect(), timeout=6.0)
        agent.mcp_servers = [server]
        logger.info(
            f"✅ Connected to MCP HTTP transport for tenant {tenant_id} at {base}/{tenant_id}/mcp"
        )
    except Exception as exc:  # noqa: BLE001 – surface failures to caller
        logger.error(
            f"❌ Failed to initialise MCP HTTP transport for tenant {tenant_id}: {exc}"
        )
        # Propagate – the caller may choose to handle this, but we do *not*
        # fall back to alternative transports to respect the no-fallback
        # policy.
        raise

# ---------------------------------------------------------------------------
# Public helpers – consolidated HTTP transport
# ---------------------------------------------------------------------------


async def get_agent_response(
    transcript: str,
    call_conversation_history: list,
    tenant_id: str,
) -> tuple[list, str]:
    """Generate a full assistant response using the HTTP+JSON-RPC MCP transport.

    Parameters
    ----------
    transcript : str
        The latest user utterance.
    call_conversation_history : list
        The existing dialog history for this call (MCP-compliant OpenAI format).
    tenant_id : str
        Logical tenant identifier used to isolate tool scopes and auth.

    Returns
    -------
    tuple(updated_history, response_text)
        *updated_history* – The dialog history including the assistant reply
        *response_text* – The assistant's natural-language response
    """

    logger.info("Running medspa agent with input: %s", transcript)

    # Ensure up-to-date MCP server configuration for this tenant.
    await _init_mcp_server(medspa_agent, tenant_id)

    agent_input = call_conversation_history + [
        {"role": "user", "content": transcript},
    ]

    result = await Runner.run(medspa_agent, agent_input)
    return result.to_input_list(), result.final_output


async def stream_agent_deltas(
    transcript: str,
    call_conversation_history: list,
    tenant_id: str,
):
    """Stream assistant deltas for real-time TTS while preserving history.

    The function returns a tuple consisting of:

    1. *updated_history_future* – resolves to the updated chat history once
       the stream concludes.
    2. *delta_generator* – async generator that yields str deltas as produced
       by the LLM.
    3. *handle* – `StreamingHandle` instance that allows the caller to cancel
       the in-flight stream (used by VAD).
    """

    await _init_mcp_server(medspa_agent, tenant_id)

    agent_input = call_conversation_history + [
        {"role": "user", "content": transcript},
    ]

    loop = asyncio.get_running_loop()
    updated_hist_future: asyncio.Future = loop.create_future()

    async def _delta_generator():
        streaming_result = Runner.run_streamed(medspa_agent, agent_input)

        async for event in streaming_result.stream_events():
            if (
                event.type == "raw_response_event"
                and isinstance(event.data, ResponseTextDeltaEvent)
            ):
                yield event.data.delta

        # Resolve history promise once stream completes.
        if not updated_hist_future.done():
            updated_hist_future.set_result(streaming_result.to_input_list())

    agen = _delta_generator()
    return updated_hist_future, agen, StreamingHandle(agen)