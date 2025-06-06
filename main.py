import os
import uvicorn
from fastapi import FastAPI, Request, WebSocket, Response
from aws_xray_sdk.core import patch_all
import time
from contextlib import asynccontextmanager

from utils import logger, setup_logging
from twilio import twilio_service
from dpg import get_deepgram_service
from ell import tts_service
from oai import initialize_agent, cleanup_mcp_server

patch_all()  # instrument std libs

# Create the singleton instance after all imports are resolved
deepgram_service = get_deepgram_service()

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

    # Warm up ElevenLabs realtime TTS to avoid cold start on first call
    try:
        await tts_service.warmup()
        logger.info("✅ ElevenLabs TTS warm-up completed")
    except Exception as e:
        logger.debug(f"ElevenLabs warm-up failed: {e}")
    
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

@app.get("/websocket-status")
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

@app.post("/voice")
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
    
    This endpoint:
    1. Accepts the WebSocket connection from Twilio
    2. Waits for initial events to extract CallSid
    3. Processes incoming audio data
    4. Sends periodic pings to maintain the connection
    5. Handles disconnection gracefully
    """
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