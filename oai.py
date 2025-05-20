import asyncio
import httpx
from openai import AsyncOpenAI
from typing import Optional
from agents import Agent, Runner, set_default_openai_client, FunctionTool
from openai.types.responses import ResponseTextDeltaEvent
from utils import logger
import os
from fastmcp import Client, FastMCP
import json







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




# ---------------------------------------------------------------------------
# Public helpers – consolidated HTTP transport
# ---------------------------------------------------------------------------


# Model Context Protocol version header for MCP requests, used by HTTP transport
MCP_PROTOCOL_VERSION = os.environ.get("MCP_PROTOCOL_VERSION", "2025-03-26")

async def get_agent_response(
    transcript: str,
    call_conversation_history: list,
    tenant_id: str,
    session_id: Optional[str] = None,
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
    session_id : Optional[str], optional
        Session ID for the MCP request, by default None

    Returns
    -------
    tuple(updated_history, response_text)
        *updated_history* – The dialog history including the assistant reply
        *response_text* – The assistant's natural-language response
    """

    # Compose the MCP transport URL for this tenant
    base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com").rstrip("/")
    transport_url = f"{base}/{tenant_id}/mcp"
    # Prepare messages for the agent
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
    # Connect to MCP server and dynamically load tools for this tenant
    async with Client(transport=transport_url) as mcp_client:
        # Ensure connection established
        logger.debug("Pinging MCP server for tenant '%s' at %s", tenant_id, transport_url)
        await mcp_client.ping()
        logger.debug("MCP server reachable for tenant '%s'", tenant_id)
        # Retrieve available tools
        mcp_tools = await mcp_client.list_tools()
        # Convert MCP tools into FunctionTools
        function_tools: list[FunctionTool] = []
        for tool in mcp_tools:
            name = tool.name
            # Some MCP servers may provide description or docs
            description = getattr(tool, 'description', '') or name
            # Use provided JSON schema if available
            params_schema = getattr(tool, 'params_json_schema', None) or getattr(tool, 'schema', None)
            # Create invocation function that calls MCP tool
            async def _invoke(ctx, args_json: str, _name=name, _mcp=mcp_client):
                params = json.loads(args_json)
                results = await _mcp.call_tool(_name, params)
                # Concatenate text outputs
                texts = []
                for content in results:
                    if hasattr(content, 'text'):
                        texts.append(content.text)
                return ''.join(texts)
            # Build the FunctionTool
            function_tools.append(
                FunctionTool(
                    name=name,
                    description=description,
                    params_json_schema=params_schema,
                    on_invoke_tool=_invoke,
                )
            )
        # Attach tools to the agent for this run
        medspa_agent.tools = function_tools
        # Run the agent loop with tools enabled
        result = await Runner.run(medspa_agent, agent_input)
        return result.to_input_list(), result.final_output


async def stream_agent_deltas(
    transcript: str,
    call_conversation_history: list,
    tenant_id: str,
    session_id: Optional[str] = None,
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

    # Prepare messages for the agent
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]

    loop = asyncio.get_running_loop()
    updated_hist_future: asyncio.Future = loop.create_future()

    async def _delta_generator():
        # Compose the MCP transport URL for this tenant
        base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com").rstrip("/")
        transport_url = f"{base}/{tenant_id}/mcp"
        # Connect to MCP server and dynamically load tools
        async with Client(transport=transport_url) as mcp_client:
            logger.debug("Pinging MCP server for tenant '%s' at %s", tenant_id, transport_url)
            await mcp_client.ping()
            logger.debug("MCP server reachable for tenant '%s'", tenant_id)
            mcp_tools = await mcp_client.list_tools()
            function_tools: list[FunctionTool] = []
            for tool in mcp_tools:
                name = tool.name
                description = getattr(tool, 'description', '') or name
                params_schema = getattr(tool, 'params_json_schema', None) or getattr(tool, 'schema', None)
                async def _invoke(ctx, args_json: str, _name=name, _mcp=mcp_client):
                    params = json.loads(args_json)
                    results = await _mcp.call_tool(_name, params)
                    texts = []
                    for content in results:
                        if hasattr(content, 'text'):
                            texts.append(content.text)
                    return ''.join(texts)
                function_tools.append(
                    FunctionTool(
                        name=name,
                        description=description,
                        params_json_schema=params_schema,
                        on_invoke_tool=_invoke,
                    )
                )
            # Attach tools to the agent for streaming run
            medspa_agent.tools = function_tools
            # Execute streaming agent with dynamic tools
            streaming_result = Runner.run_streamed(medspa_agent, agent_input)
            async for event in streaming_result.stream_events():
                if (
                    event.type == "raw_response_event"
                    and isinstance(event.data, ResponseTextDeltaEvent)
                ):
                    yield event.data.delta
            if not updated_hist_future.done():
                updated_hist_future.set_result(streaming_result.to_input_list())

    agen = _delta_generator()
    return updated_hist_future, agen, StreamingHandle(agen)