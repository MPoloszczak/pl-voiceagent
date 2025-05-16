# 'logging' no longer required after tool removal
import asyncio
import httpx
from openai import AsyncOpenAI

from agents import Agent, Runner, set_default_openai_client
from openai.types.responses import ResponseTextDeltaEvent

from utils import logger
# Dynamic MCP integration per tenant
import os
# Preferred SDK-native MCP server integration
try:
    from agents.mcp.server import MCPServerSse  # type: ignore
except ImportError:  # fallback if extension not installed
    MCPServerSse = None  # type: ignore

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

# Pre-instantiated OpenAI client shared across all runner calls. Using
# `set_default_openai_client` lets the Agents SDK pick it up implicitly
# instead of passing an unsupported `client=` kwarg to Runner.* APIs.
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

async def get_agent_response(transcript, call_conversation_history, tenant_id: str):
    """
    Process transcripts with the agent and generate a response.
    
    Args:
        transcript: The transcript text from the user
        call_conversation_history: The conversation history for this call
        tenant_id: The ID of the tenant

    Returns:
        tuple: (updated_history, response_text)
    """
    logger.info(f"Running medspa agent with input: {transcript}")

  

    if MCPServerSse is not None:
        try:
            base = os.getenv("MCP_BASE", "https://mcp.pololabsai.com").rstrip("/")
            server = MCPServerSse(
                params={"url": f"{base}/sse/"},
                cache_tools_list=True,
                name=f"{tenant_id}-mcp",
            )
            try:
                # Ensure the SSE connection is opened before the server is used by the Agent.
                # The Agents SDK requires an explicit `connect()` call; otherwise, a
                # `Server not initialized` error is raised when the tool invocation occurs.
                await server.connect()
            except Exception as conn_exc:
                logger.error(
                    f"Failed to connect to MCP server for tenant {tenant_id}: {conn_exc}"
                )
                # Fallback to static tool loading so the application can still function.
                try:
                    medspa_agent.tools = await tools_for(tenant_id)
                except Exception as e2:
                    logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e2}")
            medspa_agent.mcp_servers = [server]
        except Exception as e:
            logger.error(f"Failed to configure MCP server for tenant {tenant_id}: {e}")
            try:
                medspa_agent.tools = await tools_for(tenant_id)
            except Exception as e2:
                logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e2}")
    else:
        try:
            medspa_agent.tools = await tools_for(tenant_id)
        except Exception as e:
            logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e}")

    # Create input with conversation history
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
    
    # Run the agent – the default OpenAI client has already been registered
    result = await Runner.run(medspa_agent, agent_input)
    
    # Get the agent response
    agent_response = result.final_output
    
    # Return updated history and response
    return result.to_input_list(), agent_response 

async def stream_agent_deltas(transcript: str, call_conversation_history: list, tenant_id: str):
    """
    Stream the agent response and expose a cancellation handle.

    Returns
    -------
    tuple(updated_history_future, async_generator, handle)
        updated_history_future : asyncio.Future  Resolves to the updated conversation history once the stream completes
        async_generator        : AsyncGenerator  Yields raw text deltas for TTS
        handle                 : StreamingHandle object with .cancel() used by VAD
    """
    # Load dynamic tools for tenant
    if MCPServerSse is not None:
        try:
            base = os.getenv("MCP_BASE", "https://mcp.pololabsai.com").rstrip("/")
            server = MCPServerSse(
                params={"url": f"{base}/sse/"},
                cache_tools_list=True,
                name=f"{tenant_id}-mcp",
            )
            try:
                await server.connect()
            except Exception as conn_exc:
                logger.error(
                    f"Failed to connect to MCP server for tenant {tenant_id}: {conn_exc}"
                )
                try:
                    medspa_agent.tools = await tools_for(tenant_id)
                except Exception as e2:
                    logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e2}")
            medspa_agent.mcp_servers = [server]
        except Exception as e:
            logger.error(f"Failed to configure MCP server for tenant {tenant_id}: {e}")
            try:
                medspa_agent.tools = await tools_for(tenant_id)
            except Exception as e2:
                logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e2}")

    # If a server was configured but failed to establish a connection, clear it
    try:
        if MCPServerSse is not None and server is not None and not getattr(server, "_connected", True):
            medspa_agent.mcp_servers.clear()
    except NameError:
        # 'server' may not be defined if MCP was not enabled – ignore gracefully.
        pass

    # Compose the input including history and the latest user message
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]

    loop = asyncio.get_running_loop()
    updated_hist_future: asyncio.Future = loop.create_future()

    async def _delta_generator():
        # 1) Create the streamed run **inside** the consumer coroutine.
        streaming_result = Runner.run_streamed(
            medspa_agent,
            agent_input,
        )

        # 2) Relay deltas to the caller in real-time.
        async for event in streaming_result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta

        # 3) When streaming completes, resolve the history Future so the
        #    caller can persist.  This happens *after* the final delta to
        #    guarantee ordering (required for consistent audit trail).
        try:
            if not updated_hist_future.done():
                updated_hist_future.set_result(streaming_result.to_input_list())
        except Exception as hist_exc:
            # Never propagate – instead set the original input so that the
            # caller at least has a consistent baseline.
            logger.error(f"Failed to build updated history: {hist_exc}")
            if not updated_hist_future.done():
                updated_hist_future.set_result(agent_input)

    agen = _delta_generator()
    handle = StreamingHandle(agen)

    return updated_hist_future, agen, handle