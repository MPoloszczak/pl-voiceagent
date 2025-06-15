import os
import uvicorn
from fastapi import FastAPI, Request, WebSocket, Response, Depends, HTTPException, Header, status
from aws_xray_sdk.core import patch_all
import time
from contextlib import asynccontextmanager
import base64
from twilio.request_validator import RequestValidator

from utils import logger, setup_logging
from twilio import twilio_service
from dpg import get_deepgram_service
from ell import tts_service
from oai import initialize_agent, cleanup_mcp_server

patch_all()  # instrument std libs

# Create the singleton instance after all imports are resolved
deepgram_service = get_deepgram_service()

# -------------------------
# Authentication utilities
# -------------------------

# 1. AWS IAM – verify X-Amzn-Principal-Id injected by API Gateway

def verify_iam_principal(principal_id: str | None = Header(None, alias="X-Amzn-Principal-Id")):
    """Dependency that ensures a valid IAM principal header is present."""
    if not principal_id:
        logger.warning("[AUTH] Missing X-Amzn-Principal-Id header")
        raise HTTPException(status_code=403, detail="Unauthorized")
    return principal_id

# 2. Twilio request signature verification for webhook calls

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not TWILIO_AUTH_TOKEN:
    logger.warning("TWILIO_AUTH_TOKEN is not set – Twilio signature validation will fail")

twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None


async def verify_twilio_signature(request: Request):
    """Dependency that validates the X-Twilio-Signature header."""
    if not twilio_validator:
        # Mis-configuration – treat as server error to avoid false sense of security
        raise HTTPException(status_code=500, detail="Twilio signature validator not configured")

    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    body = await request.body()

    if not twilio_validator.validate(url, body, signature):
        logger.warning("[AUTH] Invalid Twilio webhook signature for %s", url)
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

# 3. Twilio WebSocket Basic-Auth validation (AccountSid:AuthToken)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")


async def authorize_twilio_websocket(websocket: WebSocket) -> bool:
    """Rejects WebSocket connections that do not present valid Twilio Basic Auth."""
    auth_header = websocket.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    try:
        decoded = base64.b64decode(auth_header.split(" ")[1]).decode()
        account_sid, token = decoded.split(":", 1)
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    if account_sid != TWILIO_ACCOUNT_SID or token != TWILIO_AUTH_TOKEN:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    # Passed validation
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("Starting Programmable Voice Agent service")
    
    # Log environment variable status
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    
    logger.info(f"DEEPGRAM_API_KEY found: {bool(deepgram_api_key)}")
    logger.info(f"ELEVENLABS_API_KEY found: {bool(elevenlabs_api_key)}")
    logger.info(f"OPENAI_API_KEY found: {bool(openai_api_key)}")
    logger.info(f"MCP_SERVER_URL configured: {bool(mcp_server_url)} ({'default' if not mcp_server_url else 'custom'})")
    
    if not deepgram_api_key:
        logger.error("DEEPGRAM_API_KEY is missing. Deepgram transcription will not work.")
    if not elevenlabs_api_key:
        logger.error("ELEVENLABS_API_KEY is missing. TTS responses will not work.")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY is missing. Agent responses will not work.")
    
    # Initialize MCP agent with server connection - non-blocking fallback
    try:
        await initialize_agent()
        logger.info("✅ MCP agent initialization completed successfully")
    except Exception as e:
        logger.warning(f"⚠️ MCP agent initialization failed during startup: {e}")
        logger.info("🔄 Application will continue and retry MCP connection on first agent invocation")
        # HIPAA Compliance: Log startup failure for audit trail per §164.312(b)
        logger.info("[HIPAA-AUDIT] mcp_startup_failure logged for compliance tracking")


    
    yield
    
    # Shutdown
    logger.info("Shutting down Programmable Voice Agent service")
    
    # Clean up MCP server connection
    try:
        await cleanup_mcp_server()
        logger.info("✅ MCP server cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️ Error during MCP server cleanup: {e}")
    
    # Close all active Deepgram connections
    try:
        for call_sid in list(deepgram_service.active_connections.keys()):
            await deepgram_service.close_connection(call_sid)
    except Exception as e:
        logger.warning(f"⚠️ Error during Deepgram cleanup: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Programmable Voice Agent", lifespan=lifespan)

# ----- API Endpoints -----

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Programmable Voice Agent"}

@app.get("/websocket-metrics")
async def websocket_metrics():
    """Return metrics about WebSocket connections."""
    return {
        "active_connections": len(twilio_service.active_connections),
        "connection_attempts": twilio_service.websocket_connection_attempts
    }

@app.get("/websocket-status", dependencies=[Depends(verify_iam_principal)])
async def websocket_status():
    """Return detailed status about active WebSocket connections and connection attempts."""
    return {
        "active_connections_count": len(twilio_service.active_connections),
        "active_connections": [
            {
                "session_id": session_id,
                "client_state": str(ws.client_state) if hasattr(ws, "client_state") else "unknown",
                "application_state": str(ws.application_state) if hasattr(ws, "application_state") else "unknown"
            }
            for session_id, ws in twilio_service.active_connections.items()
        ],
        "connection_attempts": twilio_service.websocket_connection_attempts
    }

@app.post("/voice", dependencies=[Depends(verify_twilio_signature)])
async def handle_incoming_call(request: Request):
    """
    Handle incoming Twilio voice calls and initiate a WebSocket stream.
    
    This endpoint receives the initial webhook from Twilio when a call comes in,
    and responds with TwiML that instructs Twilio to open a WebSocket connection
    to our streaming endpoint.
    """
    return await twilio_service.handle_voice_webhook(request)

@app.websocket("/twilio-stream")
async def twilio_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio audio streaming.

    The handshake is validated via Basic Auth (AccountSid:AuthToken) before the
    connection is accepted. Unauthenticated attempts are closed with code 1008
    per RFC 6455. Only `wss://` URLs are issued in production (see /voice).
    """
    if not await authorize_twilio_websocket(websocket):
        return  # Connection already closed due to auth failure

    # Delegate to TwilioService – it will call `accept()` internally after auth
    await twilio_service.handle_twilio_stream(websocket)

# For direct execution
if __name__ == "__main__":
    # Use environment variable for port to allow configuration via Docker
    port = int(os.environ.get("PORT", 8080))
    
    # Add debug logging to see what port value we're getting
    logger.info(f"PORT environment variable: {os.environ.get('PORT')}")
    logger.info(f"Using port: {port}")
    
    # Disable reload in production, enable it only in development
    reload_mode = os.environ.get("ENV", "production").lower() == "development"
    
    # Start the server
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=reload_mode
    ) 