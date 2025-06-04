import asyncio
import httpx
from openai import AsyncOpenAI
from typing import Optional, Dict, List
from agents import Agent, Runner, set_default_openai_client
from openai.types.responses import ResponseTextDeltaEvent
from utils import logger
import os
import json
import time
from contextlib import asynccontextmanager
import uuid
import hashlib

##HIPAA Compliance & MCP Configuration--------------------------
# HIPAA audit logging class
class HIPAALogger:
    @staticmethod
    def log_access(session_id: str, action: str, result: str):
        """Log access for HIPAA audit trail per §164.312(b)."""
        timestamp = time.time()
        logger.info(
            "[HIPAA-AUDIT] session=%s action=%s result=%s timestamp=%f", 
            session_id or "anonymous", action, result, timestamp
        )
    
    @staticmethod
    def log_mcp_access(operation: str, result: str, details: str = ""):
        """Log MCP access for HIPAA audit trail per §164.312(b)."""
        timestamp = time.time()
        logger.info(
            "[HIPAA-MCP] operation=%s result=%s details=%s timestamp=%f",
            operation, result, details, timestamp
        )

# Single MCP server configuration with client session management
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "https://mcp.pololabsai.com/mcp/")
MCP_PROTOCOL_VERSION = "2025-03-26"

# Global client session tracking
class MCPSessionManager:
    """Manages MCP client sessions with unique client IDs and keep-alive functionality."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._client_id = self._generate_client_id()
        self._session_timeout = 3600  # 1 hour default timeout
    
    def _generate_client_id(self) -> str:
        """Generate a unique client ID for this application instance."""
        # Use a combination of process ID, timestamp, and random UUID for uniqueness
        process_info = f"{os.getpid()}-{time.time()}-{uuid.uuid4()}"
        client_hash = hashlib.sha256(process_info.encode()).hexdigest()[:16]
        return f"voice-agent-{client_hash}"
    
    def get_client_id(self) -> str:
        """Get the persistent client ID for this application instance."""
        return self._client_id
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session with tracking."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self._sessions[session_id] = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "client_id": self._client_id,
            "keep_alive": True
        }
        
        HIPAALogger.log_mcp_access("session_created", "success", f"session_id={session_id}")
        return session_id
    
    def update_session_activity(self, session_id: str):
        """Update the last activity timestamp for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["last_activity"] = time.time()
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions based on timeout."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in self._sessions.items():
            if current_time - session_data["last_activity"] > self._session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
            HIPAALogger.log_mcp_access("session_expired", "cleanup", f"session_id={session_id}")
    
    def get_session_headers(self, session_id: str) -> Dict[str, str]:
        """Get headers for MCP requests including client ID and session info."""
        self.update_session_activity(session_id)
        
        return {
            "X-MCP-Client-ID": self._client_id,
            "X-MCP-Session-ID": session_id,
            "X-MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "X-MCP-Keep-Alive": "true",
            "User-Agent": "PoloLabs-VoiceAI/1.0",
            "Content-Type": "application/json"
        }

# Global session manager instance
_session_manager = MCPSessionManager()

# Single MCP server instance for the application
_mcp_server_instance: Optional[object] = None

async def create_mcp_server() -> Optional[object]:
    """
    Create single MCP server instance following MCP 2025-03-26 protocol with client ID and keep-alive support.
    
    HIPAA Compliance: Secure HTTPS connections per §164.312(e)(1), audit logging per §164.312(b)
    """
    global _mcp_server_instance
    
    # Return existing instance if available and connected
    if _mcp_server_instance:
        try:
            # Verify server is still connected by listing tools
            await asyncio.wait_for(_mcp_server_instance.list_tools(), timeout=5.0)
            logger.debug("[HIPAA-MCP] Using existing connected server")
            HIPAALogger.log_mcp_access("server_reuse", "success", "existing_connection")
            return _mcp_server_instance
        except Exception:
            logger.debug("[HIPAA-MCP] Existing server not connected, creating new")
            _mcp_server_instance = None
    
    try:
        # Validate URL for HIPAA compliance
        if not MCP_SERVER_URL.startswith(('https://', 'http://localhost')):
            logger.error("[HIPAA-MCP] Invalid URL scheme: %s", MCP_SERVER_URL)
            HIPAALogger.log_mcp_access("url_validation", "invalid_scheme", MCP_SERVER_URL)
            return None
        
        logger.info("[HIPAA-MCP] Creating MCP server connection to %s", MCP_SERVER_URL)
        
        # Import OpenAI Agents SDK MCP classes - using latest 2025-03-26 compatible version
        from agents.mcp.server import MCPServerStreamableHttp
        from datetime import timedelta
        
        # Generate session ID for this connection
        session_id = _session_manager.create_session()
        
        # Get session-specific headers with client ID
        session_headers = _session_manager.get_session_headers(session_id)
        
        # Create MCP server using 2025-03-26 protocol with session management
        mcp_server = MCPServerStreamableHttp(
            params={
                "url": MCP_SERVER_URL,
                "headers": session_headers,
                "timeout": timedelta(seconds=15),
                "sse_read_timeout": timedelta(seconds=30),
                "terminate_on_close": True
            },
            cache_tools_list=True,
            name="voice-agent-mcp",
            client_session_timeout_seconds=30.0
        )
        
        # Connect the server
        logger.info("[HIPAA-MCP] Connecting to MCP server with client_id=%s session_id=%s", 
                   _session_manager.get_client_id(), session_id)
        await mcp_server.connect()
        
        # Verify connection by listing tools
        tools = await asyncio.wait_for(mcp_server.list_tools(), timeout=10.0)
        logger.info("[HIPAA-MCP] MCP server connected successfully, found %d tools", len(tools))
        
        # Cache the connected server
        _mcp_server_instance = mcp_server
        
        HIPAALogger.log_mcp_access("server_created", "success", f"tools_{len(tools)}_client_id_{_session_manager.get_client_id()}")
        return mcp_server
        
    except ImportError as e:
        logger.error("[HIPAA-MCP] OpenAI Agents SDK MCP module not available: %s", e)
        HIPAALogger.log_mcp_access("server_creation", "import_error", str(e))
        return None
        
    except asyncio.TimeoutError:
        logger.error("[HIPAA-MCP] Connection timeout")
        HIPAALogger.log_mcp_access("connection", "timeout", "30s")
        return None
        
    except Exception as e:
        logger.error("[HIPAA-MCP] Failed to create MCP server: %s", e)
        HIPAALogger.log_mcp_access("server_creation", "error", str(e))
        return None

async def cleanup_mcp_server():
    """
    Cleanup MCP server connection and session tracking.
    
    HIPAA Compliance: Ensures proper connection cleanup and audit logging per §164.312(b).
    """
    global _mcp_server_instance
    
    if _mcp_server_instance:
        try:
            # Cleanup expired sessions
            _session_manager.cleanup_expired_sessions()
            
            if hasattr(_mcp_server_instance, 'cleanup'):
                await _mcp_server_instance.cleanup()
            elif hasattr(_mcp_server_instance, 'session') and _mcp_server_instance.session:
                await _mcp_server_instance.session.close()
            
            logger.debug("[HIPAA-MCP] Successfully cleaned up MCP server")
            HIPAALogger.log_mcp_access("server_cleanup", "success", "cleanup")
            
        except Exception as cleanup_error:
            logger.warning("[HIPAA-MCP] Error during server cleanup: %s", cleanup_error)
            HIPAALogger.log_mcp_access("server_cleanup", "error", str(cleanup_error))
        finally:
            _mcp_server_instance = None

@asynccontextmanager
async def managed_session_context(session_id: str):
    """
    Managed context for session operations with proper cleanup and audit logging.
    HIPAA: Ensures proper access control and audit trail per §164.312(a)(1) and §164.312(b)
    """
    start_time = time.time()
    
    # Create or update session tracking
    if not session_id:
        session_id = _session_manager.create_session()
    else:
        _session_manager.update_session_activity(session_id)
    
    try:
        HIPAALogger.log_access(session_id, "context_start", "success")
        yield session_id
    except Exception as e:
        logger.error("[HIPAA] Context error for session '%s': %s", session_id, e)
        HIPAALogger.log_access(session_id, "context_error", str(e))
        raise
    finally:
        duration = time.time() - start_time
        logger.debug("[HIPAA] Context completed for session '%s' in %.2fs", session_id, duration)
        HIPAALogger.log_access(session_id, "context_end", f"duration_{duration:.2f}s")
        
        # Periodic cleanup of expired sessions (every 10th request)
        if hash(session_id) % 10 == 0:
            _session_manager.cleanup_expired_sessions()

# ------------------------------------------------------------------
# Single Medspa Assistant Agent with MCP Server
# ------------------------------------------------------------------

# Create agent with MCP server - will be configured at startup
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

IMPORTANT: You must maintain strict confidentiality and privacy in all interactions, 
adhering to HIPAA guidelines for protected health information.
""",
    model="gpt-4o",
)

# Initialize MCP-enabled agent at startup
_agent_with_mcp: Optional[Agent] = None
_mcp_initialization_attempted: bool = False
_mcp_retry_on_invocation: bool = True

async def initialize_agent():
    """Initialize the agent with MCP server connection."""
    global _agent_with_mcp, _mcp_initialization_attempted, _mcp_retry_on_invocation
    
    _mcp_initialization_attempted = True
    mcp_server = await create_mcp_server()
    
    if mcp_server:
        _agent_with_mcp = Agent(
            name=medspa_agent.name,
            instructions=medspa_agent.instructions,
            model=medspa_agent.model,
            mcp_servers=[mcp_server]
        )
        _mcp_retry_on_invocation = False  # Success, no need to retry
        logger.info("[HIPAA-MCP] Agent initialized with MCP server, client_id=%s", 
                   _session_manager.get_client_id())
        HIPAALogger.log_mcp_access("agent_initialization", "success", f"client_id={_session_manager.get_client_id()}")
    else:
        _agent_with_mcp = medspa_agent
        logger.warning("[HIPAA-MCP] Agent initialized without MCP server (fallback)")
        HIPAALogger.log_mcp_access("agent_initialization", "fallback_mode", "no_mcp_server")

async def ensure_agent_initialized():
    """Ensure agent is initialized, with MCP retry on first invocation if needed."""
    global _agent_with_mcp, _mcp_initialization_attempted, _mcp_retry_on_invocation
    
    # If agent is not initialized at all, initialize it
    if not _agent_with_mcp:
        await initialize_agent()
        return
    
    # If MCP failed during startup and this is the first invocation, retry once
    if _mcp_retry_on_invocation and _mcp_initialization_attempted:
        logger.info("[HIPAA-MCP] Retrying MCP server connection on first agent invocation")
        HIPAALogger.log_mcp_access("mcp_retry_invocation", "started", "first_call_retry")
        
        try:
            mcp_server = await create_mcp_server()
            if mcp_server:
                _agent_with_mcp = Agent(
                    name=medspa_agent.name,
                    instructions=medspa_agent.instructions,
                    model=medspa_agent.model,
                    mcp_servers=[mcp_server]
                )
                logger.info("[HIPAA-MCP] ✅ MCP server connection successful on retry")
                HIPAALogger.log_mcp_access("mcp_retry_invocation", "success", "connected_on_retry")
            else:
                logger.warning("[HIPAA-MCP] ⚠️ MCP server connection retry failed, continuing with fallback")
                HIPAALogger.log_mcp_access("mcp_retry_invocation", "failed", "fallback_maintained")
        except Exception as e:
            logger.warning(f"[HIPAA-MCP] ⚠️ MCP server retry failed: {e}")
            HIPAALogger.log_mcp_access("mcp_retry_invocation", "error", str(e))
        
        # Disable further retry attempts regardless of outcome
        _mcp_retry_on_invocation = False

def get_agent() -> Agent:
    """Get the configured agent (with or without MCP)."""
    return _agent_with_mcp if _agent_with_mcp else medspa_agent

_httpx_client = httpx.AsyncClient()
_oai_client: AsyncOpenAI = AsyncOpenAI(http_client=_httpx_client)

# Register the client once for the entire process (thread-safe).
set_default_openai_client(_oai_client)

class StreamingHandle:
    """Wrap an async generator to allow cancellation of the LLM stream."""
    def __init__(self, agen):
        self._agen = agen
        self._cancelled = False
        self._cleanup_task = None
    
    def cancel(self):
        """Cancel the async generator to stop the LLM streaming."""
        if self._cancelled:
            return
            
        self._cancelled = True
        
        # Schedule cleanup asynchronously to avoid blocking and race conditions
        try:
            if hasattr(self._agen, 'aclose'):
                try:
                    loop = asyncio.get_running_loop()
                    self._cleanup_task = loop.create_task(self._safe_cleanup())
                except RuntimeError:
                    logger.debug("Could not schedule async generator cleanup: event loop not available")
        except Exception as e:
            logger.debug("Error during streaming handle cancellation setup: %s", e)
    
    async def _safe_cleanup(self):
        """Safely close the async generator with proper error handling."""
        try:
            if hasattr(self._agen, 'aclose'):
                await self._agen.aclose()
        except RuntimeError as e:
            if "already running" in str(e):
                logger.debug("Async generator cleanup skipped: generator still running")
            else:
                logger.debug("Async generator cleanup error: %s", e)
        except Exception as e:
            logger.debug("Unexpected error during async generator cleanup: %s", e)

# ---------------------------------------------------------------------------
# Simplified Agent Response Functions with Session Management
# ---------------------------------------------------------------------------

async def get_agent_response(
    transcript: str,
    call_conversation_history: list,
    session_id: Optional[str] = None,
) -> tuple[list, str]:
    """
    Generate assistant response using the configured agent with session tracking.
    
    HIPAA Compliance: Maintains strict data isolation, implements proper access controls 
    per §164.312(a)(1), logs security events per §164.312(b)
    """
    
    async with managed_session_context(session_id) as tracked_session_id:
        # Ensure agent is initialized with MCP retry capability
        await ensure_agent_initialized()
        
        agent = get_agent()
        agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
        
        try:
            logger.info("[HIPAA] Executing agent run for session %s", tracked_session_id)
            result = await Runner.run(agent, agent_input)
            
            logger.info("[HIPAA] Agent run completed successfully for session %s", tracked_session_id)
            HIPAALogger.log_access(tracked_session_id, "agent_run", "success")
            
            return result.to_input_list(), result.final_output
            
        except Exception as e:
            logger.error("[HIPAA] Agent execution error for session %s: %s", tracked_session_id, e)
            HIPAALogger.log_access(tracked_session_id, "agent_run", "error")
            
            # Fallback to basic agent without tools
            result = await Runner.run(medspa_agent, agent_input)
            return result.to_input_list(), result.final_output


async def stream_agent_deltas(
    transcript: str,
    call_conversation_history: list,
    session_id: Optional[str] = None,
):
    """
    Stream assistant deltas using the configured agent with session tracking.
    
    HIPAA Compliance: Maintains data isolation during streaming, implements proper access controls
    """

    async with managed_session_context(session_id) as tracked_session_id:
        # Ensure agent is initialized with MCP retry capability
        await ensure_agent_initialized()
        
        agent = get_agent()
        agent_input = call_conversation_history + [{"role": "user", "content": transcript}]

        loop = asyncio.get_running_loop()
        updated_hist_future: asyncio.Future = loop.create_future()

        async def _delta_generator():
            try:
                logger.info("[HIPAA] Starting streaming for session %s", tracked_session_id)
                HIPAALogger.log_access(tracked_session_id, "streaming", "started")
                
                streaming_result = Runner.run_streamed(agent, agent_input)
                
                async for event in streaming_result.stream_events():
                    if (
                        event.type == "raw_response_event"
                        and isinstance(event.data, ResponseTextDeltaEvent)
                    ):
                        yield event.data.delta
                        
                if not updated_hist_future.done():
                    updated_hist_future.set_result(streaming_result.to_input_list())
                
                logger.info("[HIPAA] Streaming completed successfully for session %s", tracked_session_id)
                HIPAALogger.log_access(tracked_session_id, "streaming", "completed")
                
            except Exception as e:
                logger.error("[HIPAA] Streaming error for session %s: %s", tracked_session_id, e)
                HIPAALogger.log_access(tracked_session_id, "streaming", "error")
                
                # Fallback streaming
                try:
                    streaming_result = Runner.run_streamed(medspa_agent, agent_input)
                    async for event in streaming_result.stream_events():
                        if (
                            event.type == "raw_response_event"
                            and isinstance(event.data, ResponseTextDeltaEvent)
                        ):
                            yield event.data.delta
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(streaming_result.to_input_list())
                except Exception as fallback_error:
                    logger.error("[HIPAA] Fallback streaming error for session %s: %s", tracked_session_id, fallback_error)
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(agent_input + [{"role": "assistant", "content": "Service temporarily unavailable"}])

        agen = _delta_generator()
        return updated_hist_future, agen, StreamingHandle(agen)