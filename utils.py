import asyncio
import json
import logging
import os
import time
from datetime import datetime
from starlette.websockets import WebSocketState
import base64
from typing import Dict
import sys
import boto3

# Configure logging
def setup_logging(app_name="pl-voiceagent"):
    """
    Set up application logging with file and console handlers.
    
    Args:
        app_name: Name of the application for the logger
        
    Returns:
        The configured logger
    """
    
    # Initialize named logger
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    # Prevent duplicate log propagation to the root logger
    logger.propagate = False
    
    return logger

# Initialize logger
logger = setup_logging()
logger.info("Logger initialized")

# -----------------------------------------------------------------------------
# Centralised secret loading – fetch JSON blob from a single Secrets Manager ARN
# and project each key/value into os.environ. This lets the application continue
# to use simple os.getenv look-ups while keeping all sensitive config in one
# place.
# -----------------------------------------------------------------------------

_SECRETS_LOADED = False  # module-level guard

def _load_secrets_from_aws() -> None:
    """Populate os.environ from a JSON secret if ENV_VARS_ARN is set.

    The secret is expected to contain a JSON object where each top-level key
    corresponds to one environment variable needed by the application.
    Existing os.environ keys are left untouched to allow overrides (e.g. for
    local development).  Any failure is logged but does *not* raise so the app
    can still start with whatever configuration is available.
    """
    global _SECRETS_LOADED
    if _SECRETS_LOADED:
        return
    secret_arn = os.getenv("ENV_VARS_ARN")
    if not secret_arn:
        return  # nothing to do

    try:
        # Use default credentials/region resolution chain – override region for
        # local dev when AWS_REGION is not set.
        sm = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1")
        response = sm.get_secret_value(SecretId=secret_arn)
        secret_string = response.get("SecretString")
        if not secret_string:
            logger.error("Secrets Manager returned no SecretString for %s", secret_arn)
            return

        try:
            secrets = json.loads(secret_string)
        except json.JSONDecodeError:
            logger.error("Secret %s is not valid JSON – skipping automatic injection", secret_arn)
            return

        injected = 0
        if os.getenv("DEBUG_SECRETS") == "1" or os.getenv("REDIS_DEBUG") == "1":
            logger.info("[Secrets-Debug] Inspecting retrieved secrets keys: %s", list(secrets.keys()))

        for key, value in secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
                injected += 1
            elif os.getenv("DEBUG_SECRETS") == "1" or os.getenv("REDIS_DEBUG") == "1":
                logger.info("[Secrets-Debug] Skipped existing env key %s", key)
        logger.info("🔐 Loaded %d secrets from Secrets Manager", injected)
        _SECRETS_LOADED = True
    except Exception as e:  # broad catch to avoid startup failure
        logger.error("❌ Failed to load application secrets from %s: %s", secret_arn, e)


# Invoke at import time so that any later module import sees populated env vars
_load_secrets_from_aws()

async def send_periodic_pings(websocket, session_id, interval=25):
    """
    Send periodic WebSocket ping frames to keep the connection alive.
    
    Args:
        websocket: The WebSocket connection
        session_id: Unique identifier for the session
        interval: Time in seconds between pings (default: 25)
    """
    try:
        while True:
            await asyncio.sleep(interval)
            # Check if the connection is still open before sending ping
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug(f"WebSocket disconnected, stopping ping for session: {session_id}")
                break
            
            logger.debug(f"Sending WebSocket ping frame to session: {session_id}")
            # Use proper WebSocket ping frame instead of text message
            # This is more standards-compliant and will be recognized by Twilio
            try:
                # Try using the native ping method if available
                await websocket.ping()
            except AttributeError:
                # Fallback to empty bytes if ping method not available
                await websocket.send_bytes(b'')
                
    except asyncio.CancelledError:
        logger.debug(f"Ping task cancelled for session: {session_id}")
        raise
    except Exception as e:
        logger.error(f"Error sending ping to session {session_id}: {str(e)}")
        logger.error(f"Error details: {repr(e)}")

# Twiml for initiating a WebSocket stream
TWIML_STREAM_TEMPLATE = """
<Response>
    <Connect>
        <!-- receive inbound (caller ➜ Twilio) audio only -->
        <Stream url="{websocket_url}" track="inbound_track" />
    </Connect>
</Response>
"""


# Constants for keep-alive mechanism
KEEPALIVE_INTERVAL = 5.0  # seconds between silence packets (must be < 10s)
KEEPALIVE_CHUNK_MS = 20   # milliseconds of silence per packet

async def send_silence_keepalive(websocket, stream_sid, session_id, tts_service):
    """
    Send periodic silence packets to keep the Twilio stream alive during periods of inactivity.
    
    Args:
        websocket: The WebSocket connection to Twilio
        stream_sid: The Twilio Stream SID
        session_id: Unique identifier for the session
        tts_service: TTS service instance for generating silence
    """
    logger.info(f"Starting silence keep-alive task for session: {session_id}")
    
    # Generate silence once and reuse
    silent_bytes = tts_service.generate_silence(KEEPALIVE_CHUNK_MS)
    silent_payload = base64.b64encode(silent_bytes).decode('utf-8')
    
    # Prepare the message packet
    packet = json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": { 
            "payload": silent_payload 
        }
    })
    
    try:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            
            # Check if the connection is still open
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug(f"WebSocket disconnected, stopping silence keep-alive for session: {session_id}")
                break
                
            # Send the silence packet
            logger.debug(f"Sending silence keep-alive packet for session: {session_id}")
            await websocket.send_text(packet)
            
    except asyncio.CancelledError:
        logger.debug(f"Silence keep-alive task cancelled for session: {session_id}")
        # Let the cancellation propagate
        raise
    except Exception as e:
        logger.error(f"Error in silence keep-alive for session {session_id}: {str(e)}")
        logger.error(f"Error details: {repr(e)}") 

async def cancel_keepalive_if_needed(session_id: str, keepalive_tasks: Dict[str, asyncio.Task]):
    """
    Cancel the silence keep-alive task for a session if it exists.

    Args:
        session_id: The session identifier
        keepalive_tasks: Mapping of session IDs to their keep-alive task
    """
    task = keepalive_tasks.get(session_id)
    if task:
        logger.info(f"Cancelling silence keep-alive task for session: {session_id}")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        del keepalive_tasks[session_id] 

#Change to 15 minutes, instead of 1 minute
async def enforce_call_length_limit(call_sid: str, websocket, deepgram_service, duration_sec: int = 15 * 60):
    """Enforce a hard upper bound on the duration of a call.

    Once *duration_sec* seconds have elapsed, the call will be terminated
    gracefully – i.e. the Deepgram connection will be closed and the
    WebSocket to Twilio will be closed if still open.

    Args:
        call_sid: The Twilio Call SID that uniquely identifies the call.
        websocket: The WebSocket instance tied to the streaming call.
        deepgram_service: An instance of the DeepgramService used by the
            application – required so we can close the upstream ASR
            connection in the same way other call-ending paths do.
        duration_sec: Maximum allowed call length in seconds (default 15
            minutes = 900 s).
    """

    # HIPAA Compliance: §164.312(b) – Maintain audit controls for
    # electronic PHI. We log the timer start/stop events for traceability.
    logger.info(f"[HIPAA-AUDIT] call_length_timer_started call_sid={call_sid} limit={duration_sec}s")

    try:
        await asyncio.sleep(duration_sec)

        # If the coroutine is still running after *duration_sec* we have hit
        # the upper bound. Proceed with graceful shutdown mirroring the
        # silence-watchdog logic.
        logger.debug(f"Call length timeout hit for {call_sid}")
        logger.info(f"[HIPAA-AUDIT] call_length_timeout_reached call_sid={call_sid}")

        # Mark the Deepgram connection as closed to avoid automatic
        # reconnection attempts.
        try:
            deepgram_service.call_closed[call_sid] = True
        except Exception:
            # Deepgram service dict may not be initialised yet – fail soft.
            pass

        # Close Deepgram connection gracefully.
        try:
            await deepgram_service.close_connection(call_sid)
        except Exception:
            pass

        # Close the WebSocket to Twilio if it is still open. Use the same
        # status code employed elsewhere in the codebase for normal
        # shutdown.
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1000, reason="Maximum call duration reached")
            except Exception:
                pass
    except asyncio.CancelledError:
        # Normal cancellation (e.g. caller hung up before timeout).
        logger.info(f"[HIPAA-AUDIT] call_length_timer_cancelled call_sid={call_sid}")
        raise 

# ---------------------------------------------------------------------------
# Logging helpers – masking potential PHI / secrets
# ---------------------------------------------------------------------------

# Lower-cased header names that must be masked in any logs.  Includes common
# authentication and identity fields that could contain PHI or credentials.
_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "x-twilio-signature",
    "cookie",
    "set-cookie",
    "x-api-key",
    "apikey",
    "api_key",
}


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of *headers* with sensitive values redacted.

    A best-effort helper to avoid leaking authentication secrets or PHI in the
    application logs.  Any header whose *name* matches one of the keys in
    ``_SENSITIVE_HEADER_KEYS`` (case-insensitive) has its value replaced with
    the literal string ``"REDACTED"``.

    HIPAA §164.312(b) – Audit Controls: we keep structural information (the
    presence of the header) for debugging/audit purposes while masking
    contents that could identify an individual or disclose credentials.
    """

    sanitized: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _SENSITIVE_HEADER_KEYS:
            sanitized[k] = "REDACTED"
        else:
            sanitized[k] = v
    return sanitized 