import asyncio
import httpx
from openai import AsyncOpenAI
from typing import Optional
from agents import Agent, Runner, set_default_openai_client, FunctionTool
from openai.types.responses import ResponseTextDeltaEvent
from utils import logger
import os
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from urllib.parse import urlparse
import json

##MCP Configuration------------------------------------------------
# Define a handler for server-emitted logs per FastMCP advanced features
async def log_handler(message: LogMessage):
    level = message.level.upper()
    logger_name = message.logger or 'default'
    data = message.data
    logger.debug(f"[MCP Server Log - {level}] {logger_name}: {data}")

# Remove the global client initialization - we'll create tenant-specific clients instead
# This follows the pattern from the server where each tenant has its own MCP instance

# ------------------------------------------------------------------
# Single Medspa assistant agent (dynamic toolbox set per-tenant)
# ------------------------------------------------------------------

# Main conversational agent - initialize without MCP servers since they're tenant-specific
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
    model="gpt-4o",
    # Remove mcp_servers parameter - we'll attach tools dynamically per request
)


_httpx_client = httpx.AsyncClient()
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

# Model Context Protocol version header for MCP requests
# Using the latest protocol version as specified in documentation
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

    # Initialize MCP client for this tenant
    logger.info("[MCP] Initializing for tenant '%s'", tenant_id)
    
    # Construct tenant-specific URL following the server's routing pattern
    # The server mounts tenants at /{tenant_id}/mcp
    raw_base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com")
    parsed = urlparse(raw_base)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    
    # Construct the tenant-specific MCP endpoint URL
    transport_url = f"{scheme}://{netloc}/{tenant_id}/mcp/"
    logger.debug("[MCP] Using Streamable HTTP transport URL: %s", transport_url)
    
    # Prepare messages for the agent
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
    
    # Connect to MCP server and dynamically load tools for this tenant
    # Using context manager ensures proper cleanup
    async with Client(transport=transport_url, log_handler=log_handler) as mcp_client:
        logger.info("[MCP] Opening connection to server at %s", transport_url)
        
        try:
            # Verify connection is working
            logger.debug("[MCP] Sending ping for tenant '%s'", tenant_id)
            await mcp_client.ping()
            logger.info("[MCP] Ping successful for tenant '%s'", tenant_id)
        except Exception as e:
            logger.error("[MCP] Failed to connect to MCP server for tenant '%s': %s", tenant_id, e)
            logger.error("[MCP] Check that the MCP server is running and accessible at %s", transport_url)
            # Return a fallback response without MCP tools
            result = await Runner.run(medspa_agent, agent_input)
            return result.to_input_list(), result.final_output
        
        # Retrieve available tools from the tenant's MCP server
        try:
            logger.info("[MCP] Listing tools for tenant '%s'", tenant_id)
            mcp_tools = await mcp_client.list_tools()
            logger.info("[MCP] Retrieved %d tools for tenant '%s'", len(mcp_tools), tenant_id)
            
            # Log tool names for debugging
            tool_names = [tool.name for tool in mcp_tools]
            logger.debug("[MCP] Available tools: %s", tool_names)
        except Exception as e:
            logger.error("[MCP] Failed to list tools for tenant '%s': %s", tenant_id, e)
            mcp_tools = []
        
        # Convert MCP tools into FunctionTools for the agent
        function_tools: list[FunctionTool] = []
        
        for tool in mcp_tools:
            name = tool.name
            description = tool.description or f"Tool: {name}"
            
            # Get the JSON schema for parameters
            # FastMCP tools have inputSchema attribute
            params_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            
            logger.debug("[MCP] Converting tool '%s' with schema: %s", name, params_schema)
            
            # Create invocation function that calls MCP tool
            async def _invoke(ctx, args_json: str, _name=name, _mcp=mcp_client):
                """Invoke MCP tool with proper error handling."""
                try:
                    params = json.loads(args_json) if args_json else {}
                    logger.debug("[MCP] Calling tool '%s' with params: %s", _name, params)
                    
                    # Call the tool through the MCP client
                    results = await _mcp.call_tool(_name, params)
                    
                    # Handle different result types
                    if hasattr(results, 'content'):
                        # Single result with content attribute
                        content_items = results.content if isinstance(results.content, list) else [results.content]
                    elif isinstance(results, list):
                        # List of results
                        content_items = results
                    else:
                        # Direct result
                        content_items = [results]
                    
                    # Extract text from results
                    texts = []
                    for item in content_items:
                        if hasattr(item, 'text'):
                            texts.append(item.text)
                        elif isinstance(item, str):
                            texts.append(item)
                        else:
                            texts.append(str(item))
                    
                    result_text = '\n'.join(texts)
                    logger.debug("[MCP] Tool '%s' returned: %s", _name, result_text[:200])
                    return result_text
                    
                except Exception as e:
                    logger.error("[MCP] Error calling tool '%s': %s", _name, e)
                    return f"Error calling tool {_name}: {str(e)}"
            
            # Build and collect the FunctionTool
            function_tool = FunctionTool(
                name=name,
                description=description,
                params_json_schema=params_schema,
                on_invoke_tool=_invoke,
            )
            function_tools.append(function_tool)
        
        logger.info("[MCP] Created %d FunctionTools for agent", len(function_tools))
        
        # Create a new agent instance with the tenant-specific tools
        # This ensures proper tool isolation per tenant
        tenant_agent = Agent(
            name=medspa_agent.name,
            instructions=medspa_agent.instructions,
            model=medspa_agent.model,
            tools=function_tools,  # Attach the tenant-specific tools
        )
        
        # Run the agent with the tenant-specific configuration
        logger.info("[MCP] Invoking agent run for tenant '%s' with %d tools", tenant_id, len(function_tools))
        result = await Runner.run(tenant_agent, agent_input)
        logger.info("[MCP] Agent run complete for tenant '%s'", tenant_id)
        
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
        # Construct tenant-specific URL
        raw_base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com")
        parsed = urlparse(raw_base)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        transport_url = f"{scheme}://{netloc}/{tenant_id}/mcp/"
        logger.debug("[MCP-STREAM] Using Streamable HTTP transport URL: %s", transport_url)
        
        # Connect to MCP server and dynamically load tools
        async with Client(transport=transport_url, log_handler=log_handler) as mcp_client:
            logger.info("[MCP-STREAM] Opening connection for tenant '%s'", tenant_id)
            
            try:
                logger.debug("[MCP-STREAM] Sending ping for tenant '%s'", tenant_id)
                await mcp_client.ping()
                logger.info("[MCP-STREAM] Ping successful for tenant '%s'", tenant_id)
            except Exception as e:
                logger.error("[MCP-STREAM] Failed to connect to MCP server for tenant '%s': %s", tenant_id, e)
                # Fall back to agent without tools
                streaming_result = Runner.run_streamed(medspa_agent, agent_input)
                async for event in streaming_result.stream_events():
                    if (
                        event.type == "raw_response_event"
                        and isinstance(event.data, ResponseTextDeltaEvent)
                    ):
                        yield event.data.delta
                if not updated_hist_future.done():
                    updated_hist_future.set_result(streaming_result.to_input_list())
                return
            
            # Get tools for streaming
            try:
                mcp_tools = await mcp_client.list_tools()
                logger.info("[MCP-STREAM] Retrieved %d tools for tenant '%s'", len(mcp_tools), tenant_id)
            except Exception as e:
                logger.error("[MCP-STREAM] Failed to list tools: %s", e)
                mcp_tools = []
            
            # Convert tools
            function_tools: list[FunctionTool] = []
            for tool in mcp_tools:
                name = tool.name
                description = tool.description or f"Tool: {name}"
                params_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                
                async def _invoke(ctx, args_json: str, _name=name, _mcp=mcp_client):
                    try:
                        params = json.loads(args_json) if args_json else {}
                        results = await _mcp.call_tool(_name, params)
                        
                        # Handle results
                        if hasattr(results, 'content'):
                            content_items = results.content if isinstance(results.content, list) else [results.content]
                        elif isinstance(results, list):
                            content_items = results
                        else:
                            content_items = [results]
                        
                        texts = []
                        for item in content_items:
                            if hasattr(item, 'text'):
                                texts.append(item.text)
                            elif isinstance(item, str):
                                texts.append(item)
                            else:
                                texts.append(str(item))
                        
                        return '\n'.join(texts)
                    except Exception as e:
                        logger.error("[MCP-STREAM] Error calling tool '%s': %s", _name, e)
                        return f"Error calling tool {_name}: {str(e)}"
                
                function_tools.append(
                    FunctionTool(
                        name=name,
                        description=description,
                        params_json_schema=params_schema,
                        on_invoke_tool=_invoke,
                    )
                )
            
            logger.info("[MCP-STREAM] Created %d FunctionTools", len(function_tools))
            
            # Create tenant-specific agent
            tenant_agent = Agent(
                name=medspa_agent.name,
                instructions=medspa_agent.instructions,
                model=medspa_agent.model,
                tools=function_tools,
            )
            
            # Execute streaming with tenant-specific agent
            logger.info("[MCP-STREAM] Starting streaming agent run for tenant '%s'", tenant_id)
            streaming_result = Runner.run_streamed(tenant_agent, agent_input)
            
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