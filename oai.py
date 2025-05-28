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
from urllib.parse import urlparse

##HIPAA Compliance & MCP Configuration--------------------------
# HIPAA audit logging class
class HIPAALogger:
    @staticmethod
    def log_tenant_access(tenant_id: str, session_id: str, action: str, result: str):
        """Log tenant access for HIPAA audit trail per §164.312(b)."""
        timestamp = time.time()
        logger.info(
            "[HIPAA-AUDIT] tenant=%s session=%s action=%s result=%s timestamp=%f", 
            tenant_id, session_id or "anonymous", action, result, timestamp
        )
    
    @staticmethod
    def log_mcp_access(tenant_id: str, operation: str, result: str, details: str = ""):
        """Log MCP access for HIPAA audit trail per §164.312(b)."""
        timestamp = time.time()
        logger.info(
            "[HIPAA-MCP] tenant=%s operation=%s result=%s details=%s timestamp=%f",
            tenant_id, operation, result, details, timestamp
        )

# Enhanced tenant validation with rate limiting
_tenant_access_times: Dict[str, List[float]] = {}
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 100

# Optimized MCP server caching for performance
_mcp_server_cache: Dict[str, object] = {}
_cache_creation_times: Dict[str, float] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL
MCP_PROTOCOL_VERSION = os.environ.get("MCP_PROTOCOL_VERSION", "2025-03-26")

# Periodic cache cleanup for memory management
async def periodic_cache_cleanup():
    """Periodically clean up expired MCP cache entries."""
    while True:
        try:
            await asyncio.sleep(1800)  # Clean up every 30 minutes
            cleaned = await cleanup_expired_cache_entries()
            if cleaned > 0:
                logger.info(f"[HIPAA-MCP] Periodic cleanup removed {cleaned} expired cache entries")
        except Exception as e:
            logger.error(f"[HIPAA-MCP] Error in periodic cache cleanup: {e}")

# Cleanup task reference
_cleanup_task = None

async def cleanup_expired_cache_entries():
    """Clean up expired cache entries and disconnected servers to prevent memory leaks."""
    current_time = time.time()
    expired_tenants = []
    disconnected_tenants = []
    
    for tenant_id, creation_time in _cache_creation_times.items():
        age = current_time - creation_time
        if age >= CACHE_TTL_SECONDS:
            expired_tenants.append(tenant_id)
        elif tenant_id in _mcp_server_cache:
            # Check if server is still connected
            server = _mcp_server_cache[tenant_id]
            if hasattr(server, 'session') and not server.session:
                disconnected_tenants.append(tenant_id)
    
    # Clean up expired entries
    for tenant_id in expired_tenants:
        if tenant_id in _mcp_server_cache:
            server = _mcp_server_cache.pop(tenant_id, None)
            _cache_creation_times.pop(tenant_id, None)
            # Cleanup server connection if it exists
            if server and hasattr(server, 'cleanup'):
                try:
                    await server.cleanup()
                except Exception as e:
                    logger.debug("[HIPAA-MCP] Error cleaning up server for tenant '%s': %s", tenant_id, e)
            logger.debug("[HIPAA-MCP] Cleaned up expired cache entry for tenant '%s'", tenant_id)
    
    # Clean up disconnected entries
    for tenant_id in disconnected_tenants:
        if tenant_id in _mcp_server_cache:
            _mcp_server_cache.pop(tenant_id, None)
            _cache_creation_times.pop(tenant_id, None)
            logger.debug("[HIPAA-MCP] Cleaned up disconnected cache entry for tenant '%s'", tenant_id)
    
    total_cleaned = len(expired_tenants) + len(disconnected_tenants)
    if total_cleaned > 0:
        logger.info("[HIPAA-MCP] Cleaned up %d cache entries (%d expired, %d disconnected)", 
                   total_cleaned, len(expired_tenants), len(disconnected_tenants))
    
    return total_cleaned

async def create_optimized_mcp_server(tenant_id: str) -> Optional[object]:
    """
    Create or retrieve cached MCP server for tenant with optimizations.
    
    Performance optimizations:
    - Server instance caching with TTL
    - Tool list caching enabled  
    - Connection timeout optimized for real-time voice
    - Streamable HTTP transport for low latency
    - Proper connection lifecycle management
    - Direct URL construction without redirect handling
    
    HIPAA Compliance: Tenant isolation per §164.312(a)(1), audit logging per §164.312(b),
    direct HTTPS connections per §164.312(e)(1)
    """
    
    # Check cache first (with TTL for production stability)
    current_time = time.time()
    if tenant_id in _mcp_server_cache:
        cache_age = current_time - _cache_creation_times.get(tenant_id, 0)
        if cache_age < CACHE_TTL_SECONDS:
            cached_server = _mcp_server_cache[tenant_id]
            # Verify server is still connected
            if hasattr(cached_server, 'session') and cached_server.session:
                logger.debug("[HIPAA-MCP] Using cached connected server for tenant '%s' (age: %.1fs)", 
                            tenant_id, cache_age)
                HIPAALogger.log_mcp_access(tenant_id, "cache_hit", "success", f"age_{cache_age:.1f}s")
                return cached_server
            else:
                logger.debug("[HIPAA-MCP] Cached server for tenant '%s' not connected, recreating", tenant_id)
                _mcp_server_cache.pop(tenant_id, None)
                _cache_creation_times.pop(tenant_id, None)
        else:
            # Cache expired, remove old entry
            logger.debug("[HIPAA-MCP] Cache expired for tenant '%s', creating new server", tenant_id)
            _mcp_server_cache.pop(tenant_id, None)
            _cache_creation_times.pop(tenant_id, None)
    
    try:
        # Construct tenant-specific MCP URL directly (server now mounts at /{tenant_id}/mcp)
        raw_base = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com")
        parsed = urlparse(raw_base)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        
        # Construct direct URL to the mounted MCP endpoint
        base_url = f"{scheme}://{netloc}".rstrip('/')
        transport_url = f"{base_url}/{tenant_id}/mcp"
        
        # Validate URL format for HIPAA compliance
        if not transport_url.startswith(('https://', 'http://localhost')):
            logger.error("[HIPAA-MCP] Invalid URL scheme for tenant '%s': %s", tenant_id, transport_url)
            HIPAALogger.log_mcp_access(tenant_id, "url_validation", "invalid_scheme", transport_url)
            return None
        
        logger.info("[HIPAA-MCP] Creating optimized MCP server for tenant '%s' at %s", 
                   tenant_id, transport_url)
        
        # Import OpenAI Agents SDK MCP classes
        from agents.mcp.server import MCPServerStreamableHttp
        from datetime import timedelta
        
        # Create optimized MCP server for real-time voice AI
        mcp_server = MCPServerStreamableHttp(
            params={
                "url": transport_url,
                "headers": {
                    "X-MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
                    "X-Tenant-ID": tenant_id,
                    "X-Application": "voice-ai",
                    "User-Agent": "PoloLabs-VoiceAI/1.0"
                },
                "timeout": timedelta(seconds=10),  # Optimized for real-time
                "sse_read_timeout": timedelta(seconds=30),  # Shorter timeout for voice
                "follow_redirects": False,  # No redirects expected with direct mounting
                "max_redirects": 0  # Disable redirect following
            },
            cache_tools_list=True,  # Enable automatic tool caching
            name=f"tenant-{tenant_id}-voice",
            client_session_timeout_seconds=15.0  # Real-time voice optimization
        )
        
        # Connect the server before caching (critical for streaming)
        logger.info("[HIPAA-MCP] Connecting MCP server for tenant '%s'", tenant_id)
        try:
            await asyncio.wait_for(mcp_server.connect(), timeout=30.0)
            
            # Verify connection by attempting to list tools
            tools = await asyncio.wait_for(mcp_server.list_tools(), timeout=10.0)
            logger.info("[HIPAA-MCP] MCP server connected successfully for tenant '%s', found %d tools", 
                       tenant_id, len(tools))
            
        except asyncio.TimeoutError:
            logger.error("[HIPAA-MCP] Connection timeout for tenant '%s' after 30s", tenant_id)
            HIPAALogger.log_mcp_access(tenant_id, "connection", "timeout", "30s")
            await safe_cleanup_mcp_server(mcp_server, tenant_id)
            return None
        except Exception as connect_error:
            logger.error("[HIPAA-MCP] Connection failed for tenant '%s': %s", tenant_id, connect_error)
            HIPAALogger.log_mcp_access(tenant_id, "connection", "failed", str(connect_error))
            await safe_cleanup_mcp_server(mcp_server, tenant_id)
            return None
        
        # Cache the connected server and record creation time
        _mcp_server_cache[tenant_id] = mcp_server
        _cache_creation_times[tenant_id] = current_time
        
        logger.info("[HIPAA-MCP] Created, connected and cached MCP server for tenant '%s'", tenant_id)
        HIPAALogger.log_mcp_access(tenant_id, "server_created_connected", "success", f"url_{transport_url}")
        
        return mcp_server
        
    except ImportError as e:
        logger.error("[HIPAA-MCP] OpenAI Agents SDK MCP module not available: %s", e)
        HIPAALogger.log_mcp_access(tenant_id, "server_creation", "import_error", str(e))
        return None
        
    except Exception as e:
        logger.error("[HIPAA-MCP] Failed to create MCP server for tenant '%s': %s", tenant_id, e)
        HIPAALogger.log_mcp_access(tenant_id, "server_creation", "error", str(e))
        return None

async def safe_cleanup_mcp_server(server: object, tenant_id: str):
    """
    Safely cleanup MCP server connections with proper async generator handling.
    
    HIPAA Compliance: Ensures proper connection cleanup and audit logging per §164.312(b).
    """
    if server:
        try:
            # Check if server has cleanup method
            if hasattr(server, 'cleanup'):
                await server.cleanup()
            # Also check for session cleanup
            elif hasattr(server, 'session') and server.session:
                try:
                    await server.session.close()
                except Exception:
                    pass  # Session may already be closed
            
            logger.debug("[HIPAA-MCP] Successfully cleaned up MCP server for tenant '%s'", tenant_id)
            HIPAALogger.log_mcp_access(tenant_id, "server_cleanup", "success", "safe_cleanup")
            
        except Exception as cleanup_error:
            logger.warning("[HIPAA-MCP] Error during server cleanup for tenant '%s': %s", 
                         tenant_id, cleanup_error)
            HIPAALogger.log_mcp_access(tenant_id, "server_cleanup", "error", str(cleanup_error))

# ------------------------------------------------------------------
# MCP Connection Health Check for HIPAA Monitoring
# ------------------------------------------------------------------

async def check_mcp_server_health(tenant_id: str) -> bool:
    """
    Check MCP server health for HIPAA monitoring and compliance.
    Returns True if server is healthy, False otherwise.
    """
    try:
        if tenant_id not in _mcp_server_cache:
            return False
            
        server = _mcp_server_cache[tenant_id]
        if not hasattr(server, 'session') or not server.session:
            return False
            
        # Quick health check by listing tools
        tools = await asyncio.wait_for(server.list_tools(), timeout=5.0)
        HIPAALogger.log_mcp_access(tenant_id, "health_check", "success", f"tools_{len(tools)}")
        return True
        
    except Exception as e:
        logger.warning("[HIPAA-MCP] Health check failed for tenant '%s': %s", tenant_id, e)
        HIPAALogger.log_mcp_access(tenant_id, "health_check", "failed", str(e))
        return False

# ------------------------------------------------------------------
# HIPAA-Compliant Tenant Validation with Rate Limiting
# ------------------------------------------------------------------

async def validate_tenant_with_rate_limit(tenant_id: str) -> bool:
    """
    Enhanced tenant validation with rate limiting for HIPAA compliance.
    HIPAA: Implements access controls per §164.312(a)(1) and audit logging per §164.312(b)
    """
    current_time = time.time()
    
    # Rate limiting per tenant
    if tenant_id not in _tenant_access_times:
        _tenant_access_times[tenant_id] = []
    
    # Clean old entries
    window_start = current_time - RATE_LIMIT_WINDOW
    _tenant_access_times[tenant_id] = [
        t for t in _tenant_access_times[tenant_id] if t > window_start
    ]
    
    # Check rate limit
    if len(_tenant_access_times[tenant_id]) >= MAX_REQUESTS_PER_WINDOW:
        logger.warning("[HIPAA] Rate limit exceeded for tenant '%s'", tenant_id)
        HIPAALogger.log_tenant_access(tenant_id, "", "rate_limit_exceeded", "denied")
        return False
    
    # Record this access
    _tenant_access_times[tenant_id].append(current_time)
    
    # Basic tenant ID validation (alphanumeric, underscore, hyphen only)
    if not tenant_id or not isinstance(tenant_id, str):
        logger.warning("[HIPAA] Invalid tenant_id format: %s", tenant_id)
        HIPAALogger.log_tenant_access(tenant_id, "", "invalid_format", "denied")
        return False
        
    if not tenant_id.replace('_', '').replace('-', '').isalnum():
        logger.warning("[HIPAA] Tenant ID contains invalid characters: %s", tenant_id)
        HIPAALogger.log_tenant_access(tenant_id, "", "invalid_characters", "denied")
        return False
    
    try:
        base_url = os.environ.get("MCP_BASE", "https://mcp.pololabsai.com").rstrip('/')
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/tenants")
            
            if response.status_code == 200:
                tenant_data = response.json()
                available_tenants = set(tenant_data.get("tenants", {}).keys())
                
                is_valid = tenant_id in available_tenants
                
                if is_valid:
                    logger.info("[HIPAA] Tenant '%s' validated successfully", tenant_id)
                    HIPAALogger.log_tenant_access(tenant_id, "", "validation", "success")
                else:
                    logger.warning("[HIPAA] Tenant '%s' not found in available tenants: %s", 
                                 tenant_id, list(available_tenants))
                    HIPAALogger.log_tenant_access(tenant_id, "", "validation", "not_found")
                
                return is_valid
            else:
                logger.error("[HIPAA] Failed to validate tenant '%s': HTTP %d", 
                           tenant_id, response.status_code)
                HIPAALogger.log_tenant_access(tenant_id, "", "validation", "http_error")
                return False
                
    except Exception as e:
        logger.error("[HIPAA] Tenant validation error for '%s': %s", tenant_id, e)
        HIPAALogger.log_tenant_access(tenant_id, "", "validation", "exception")
        return False

# MCP server creation function is now integrated above

@asynccontextmanager
async def managed_tenant_context(tenant_id: str, session_id: str):
    """
    Managed context for tenant operations with proper cleanup and audit logging.
    HIPAA: Ensures proper access control and audit trail per §164.312(a)(1) and §164.312(b)
    """
    start_time = time.time()
    try:
        HIPAALogger.log_tenant_access(tenant_id, session_id, "context_start", "success")
        yield
    except Exception as e:
        logger.error("[HIPAA] Context error for tenant '%s', session '%s': %s", tenant_id, session_id, e)
        HIPAALogger.log_tenant_access(tenant_id, session_id, "context_error", str(e))
        raise
    finally:
        duration = time.time() - start_time
        logger.debug("[HIPAA] Context completed for tenant '%s' in %.2fs", tenant_id, duration)
        HIPAALogger.log_tenant_access(tenant_id, session_id, "context_end", f"duration_{duration:.2f}s")

# ------------------------------------------------------------------
# Single Optimized Medspa Assistant Agent
# ------------------------------------------------------------------

# Single agent instance with dynamic MCP configuration for optimal performance
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
    # MCP servers will be configured dynamically per request via RunnerContext
)

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
                # Create a fire-and-forget cleanup task
                try:
                    loop = asyncio.get_running_loop()
                    self._cleanup_task = loop.create_task(self._safe_cleanup())
                except RuntimeError:
                    # Event loop may be closed or in a different context
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
                # Generator is still actively running, which is expected during cancellation
                logger.debug("Async generator cleanup skipped: generator still running")
            else:
                logger.debug("Async generator cleanup error: %s", e)
        except Exception as e:
            logger.debug("Unexpected error during async generator cleanup: %s", e)

# ---------------------------------------------------------------------------
# Optimized Agent Response Functions with Native MCP Support
# ---------------------------------------------------------------------------

def ensure_cleanup_task_running():
    """Ensure the periodic cache cleanup task is running."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(periodic_cache_cleanup())
        logger.info("[HIPAA-MCP] Started periodic cache cleanup task")

async def get_agent_response(
    transcript: str,
    call_conversation_history: list,
    tenant_id: str,
    session_id: Optional[str] = None,
) -> tuple[list, str]:
    """
    Generate optimized assistant response using OpenAI Agents SDK native MCP support.
    
    Optimizations implemented:
    - Uses single agent instance with dynamic MCP configuration
    - Implements connection pooling and configuration caching
    - Enhanced HIPAA audit logging with rate limiting
    - Streamlined error handling with proper fallbacks
    - Automatic 307 redirect handling for production deployments
    
    HIPAA Compliance: Validates tenant authorization, maintains strict data isolation,
    implements proper access controls per §164.312(a)(1), logs security events per §164.312(b)
    """
    
    # Ensure cleanup task is running for optimal performance
    ensure_cleanup_task_running()
    
    async with managed_tenant_context(tenant_id, session_id or "anonymous"):
        # Step 1: Enhanced tenant validation with rate limiting
        if not await validate_tenant_with_rate_limit(tenant_id):
            logger.error("[HIPAA] Tenant validation failed for '%s' - using secure fallback", tenant_id)
            # Secure fallback without tools for HIPAA compliance
            agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
            result = await Runner.run(medspa_agent, agent_input)
            return result.to_input_list(), result.final_output
        
        # Step 2: Create tenant-specific MCP server (cached for performance)
        mcp_server = await create_optimized_mcp_server(tenant_id)
        
        if not mcp_server:
            logger.error("[HIPAA] Failed to create MCP server for tenant '%s' - using fallback", tenant_id)
            # Fallback to agent without tools for security
            agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
            result = await Runner.run(medspa_agent, agent_input)
            return result.to_input_list(), result.final_output
        
        # Step 3: Configure agent with tenant-specific MCP server
        # Verify server health before proceeding
        is_healthy = await check_mcp_server_health(tenant_id)
        if not is_healthy:
            logger.warning("[HIPAA] MCP server health check failed for tenant '%s', using fallback", tenant_id)
            agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
            result = await Runner.run(medspa_agent, agent_input)
            return result.to_input_list(), result.final_output
        
        # Using the single agent instance with dynamic MCP server configuration
        tenant_agent = Agent(
            name=medspa_agent.name,
            instructions=medspa_agent.instructions,
            model=medspa_agent.model,
            mcp_servers=[mcp_server]  # Use the tenant-specific MCP server
        )
        
        # Step 4: Execute optimized agent run with native MCP support
        agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
        
        try:
            logger.info("[HIPAA] Executing optimized agent run for tenant '%s' with MCP server %s", 
                       tenant_id, mcp_server.name)
            
            # Execute with proper error handling for tracing
            try:
                result = await Runner.run(tenant_agent, agent_input)
                
                logger.info("[HIPAA] Agent run completed successfully for tenant '%s'", tenant_id)
                HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "agent_run", "success")
                
                # Ensure result is properly formatted for tracing
                return result.to_input_list(), result.final_output
            
            except Exception as runner_error:
                logger.error("[HIPAA] Agent runner error for tenant '%s': %s", tenant_id, runner_error)
                HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "agent_run", "runner_error")
                # Fallback to basic agent without tools
                result = await Runner.run(medspa_agent, agent_input)
                return result.to_input_list(), result.final_output
            
        except Exception as e:
            logger.error("[HIPAA] Agent execution error for tenant '%s': %s", tenant_id, e)
            HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "agent_run", "error")
            
            # Secure fallback on any execution error
            result = await Runner.run(medspa_agent, agent_input)
            return result.to_input_list(), result.final_output


async def stream_agent_deltas(
    transcript: str,
    call_conversation_history: list,
    tenant_id: str,
    session_id: Optional[str] = None,
):
    """
    Stream assistant deltas with optimized MCP configuration reuse.
    
    Optimizations implemented:
    - Reuses MCP configuration from get_agent_response
    - Single agent instance with dynamic configuration
    - Shared validation and context management
    - Optimized streaming with proper error handling
    - Improved async generator cleanup
    
    HIPAA Compliance: Maintains tenant isolation during streaming, implements proper access controls
    """

    async with managed_tenant_context(tenant_id, session_id or "anonymous"):
        # Prepare messages for the agent
        agent_input = call_conversation_history + [{"role": "user", "content": transcript}]

        loop = asyncio.get_running_loop()
        updated_hist_future: asyncio.Future = loop.create_future()

        async def _delta_generator():
            # Step 1: Enhanced tenant validation (shared with get_agent_response)
            if not await validate_tenant_with_rate_limit(tenant_id):
                logger.error("[HIPAA] Streaming tenant validation failed for '%s'", tenant_id)
                HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "streaming", "validation_failed")
                
                # Secure fallback without tools
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
                except Exception as e:
                    logger.error("[HIPAA] Fallback streaming error for tenant '%s': %s", tenant_id, e)
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(agent_input + [{"role": "assistant", "content": "Service temporarily unavailable"}])
                return
            
            # Step 2: Reuse MCP server (cached for performance)
            mcp_server = await create_optimized_mcp_server(tenant_id)
            
            if not mcp_server:
                logger.error("[HIPAA] Failed to create streaming MCP server for tenant '%s'", tenant_id)
                # Fall back to agent without tools
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
                except Exception as e:
                    logger.error("[HIPAA] Fallback streaming error for tenant '%s': %s", tenant_id, e)
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(agent_input + [{"role": "assistant", "content": "Service temporarily unavailable"}])
                return
            
            # Step 3: Configure tenant-specific agent for streaming
            # Verify server health before proceeding
            is_healthy = await check_mcp_server_health(tenant_id)
            if not is_healthy:
                logger.warning("[HIPAA] MCP server health check failed for streaming tenant '%s', using fallback", tenant_id)
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
                except Exception as e:
                    logger.error("[HIPAA] Fallback streaming error for tenant '%s': %s", tenant_id, e)
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(agent_input + [{"role": "assistant", "content": "Service temporarily unavailable"}])
                return
            
            tenant_agent = Agent(
                name=medspa_agent.name,
                instructions=medspa_agent.instructions,
                model=medspa_agent.model,
                mcp_servers=[mcp_server]  # Use the tenant-specific MCP server
            )
            
            # Step 4: Execute optimized streaming with native MCP support
            try:
                logger.info("[HIPAA] Starting optimized streaming for tenant '%s'", tenant_id)
                HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "streaming", "started")
                
                streaming_result = Runner.run_streamed(tenant_agent, agent_input)
                
                # Stream events with proper error handling
                try:
                    async for event in streaming_result.stream_events():
                        if (
                            event.type == "raw_response_event"
                            and isinstance(event.data, ResponseTextDeltaEvent)
                        ):
                            yield event.data.delta
                            
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(streaming_result.to_input_list())
                    
                    logger.info("[HIPAA] Streaming completed successfully for tenant '%s'", tenant_id)
                    HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "streaming", "completed")
                
                except Exception as stream_error:
                    logger.error("[HIPAA] Streaming event error for tenant '%s': %s", tenant_id, stream_error)
                    HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "streaming", "stream_error")
                    # Try to complete with basic result
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(agent_input + [{"role": "assistant", "content": "Error occurred during streaming"}])
                
            except Exception as e:
                logger.error("[HIPAA] Streaming setup error for tenant '%s': %s", tenant_id, e)
                HIPAALogger.log_tenant_access(tenant_id, session_id or "anonymous", "streaming", "setup_error")
                
                # Secure fallback on streaming error
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
                    logger.error("[HIPAA] Fallback streaming error for tenant '%s': %s", tenant_id, fallback_error)
                    if not updated_hist_future.done():
                        updated_hist_future.set_result(agent_input + [{"role": "assistant", "content": "Service temporarily unavailable"}])

        agen = _delta_generator()
        return updated_hist_future, agen, StreamingHandle(agen)