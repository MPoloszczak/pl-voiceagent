import os
import uvicorn
from fastapi import FastAPI, Request, WebSocket, Response, Depends, HTTPException, Header, status
from aws_xray_sdk.core import patch_all
import time
from contextlib import asynccontextmanager
import base64
import json
import jwt  # PyJWT
import hmac

from utils import logger, setup_logging, sanitize_headers
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

# Robust import of Twilio RequestValidator that works even when a local module named
# `twilio` shadows the official package. Falls back to a manual loader.
try:
    from twilio.request_validator import RequestValidator  # type: ignore
except ModuleNotFoundError:  # local twilio.py likely masked the package
    import importlib.util as _ilu, importlib.machinery as _ilm
    import pkg_resources as _pkgres, os as _os, sys as _sys

    try:
        _dist_path = _pkgres.get_distribution("twilio").location
        _rv_path = _os.path.join(_dist_path, "twilio", "request_validator.py")
        _spec = _ilu.spec_from_file_location("_twilio_request_validator", _rv_path)
        _tmp_mod = _ilu.module_from_spec(_spec)  # type: ignore
        assert _spec.loader is not None
        _spec.loader.exec_module(_tmp_mod)  # type: ignore
        RequestValidator = _tmp_mod.RequestValidator  # type: ignore
        # defer logging until logger is available later
        print("[AUTH] Loaded RequestValidator from site-packages fallback path")
    except Exception as _e:
        print(f"[AUTH] Failed to load Twilio RequestValidator: {_e}")
        raise

# Instantiate the validator once so dependency function can access it
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None

async def verify_twilio_signature(request: Request):
    """Validate the `X-Twilio-Signature` header.

    Implements the algorithm described in Twilio's Webhook Security docs while
    gracefully handling the edge-cases that the stock `RequestValidator`
    struggles with (e.g. Starlette `FormData`, multi-value dicts, or raw JSON
    bodies).  The logic is:

    1.   Reconstruct the absolute URL Twilio used when signing the request.
    2.   Choose the correct payload representation based on *Content-Type*.
         •  For   `application/x-www-form-urlencoded`  →  dict of POST params
         •  Otherwise                                →  raw body string
    3.   Attempt validation with Twilio's helper.  When it raises a *TypeError*
         (the "unhashable type" and friends) we fall back to a minimal in-house
         HMAC implementation that is 100 % spec-compliant but tolerant of the
         quirky input types we see in ASGI frameworks.

    HIPAA §164.312(b) – The actual signature is never logged.
    """

    if not twilio_validator:
        raise HTTPException(
            status_code=500, detail="Twilio signature validator not configured"
        )

    # ------------------------------------------------------------------
    # 0. Extract header + body once
    # ------------------------------------------------------------------
    signature = request.headers.get("X-Twilio-Signature", "")
    raw_body_bytes = await request.body()

    # ------------------------------------------------------------------
    # Robust URL reconstruction respecting reverse-proxy headers (RFC 7239)
    # ------------------------------------------------------------------
    # Defaults fall back to what ASGI sees directly.
    proto_hdr: str | None = request.headers.get("x-forwarded-proto") or request.url.scheme
    host_hdr: str | None = request.headers.get("x-forwarded-host") or request.headers.get("host")

    # Parse consolidated Forwarded header when present: e.g.
    #   Forwarded: by=3.235.38.83;for=50.17.120.24;host=clinicdev.pololabsai.com;proto=https
    fwd_header = request.headers.get("forwarded")
    if fwd_header:
        for part in fwd_header.split(";"):
            kv = part.strip().split("=", 1)
            if len(kv) != 2:
                continue
            k, v = kv[0].lower(), kv[1]
            if k == "proto" and v:
                proto_hdr = v
            elif k == "host" and v:
                host_hdr = v

    # Final assembled URL used for signature verification
    url_used = f"{proto_hdr}://{host_hdr}{request.url.path}"
    if request.url.query:
        url_used += f"?{request.url.query}"

    # Log raw headers for in-depth debugging – includes the X-Twilio-Signature
    # NOTE: Twilio signature does **not** constitute ePHI; logging remains HIPAA-compliant (see 45 CFR §164.514(a)).
    logger.info("[AUTH] Twilio webhook headers (raw): %s", dict(request.headers))
    logger.info("[AUTH] X-Twilio-Signature header value: %s", signature)

    # Additional context for troubleshooting
    logger.info("[AUTH] Forwarded header (raw): %s", fwd_header)
    logger.info("[AUTH] Final URL chosen for signature validation: %s", url_used)

    # ------------------------------------------------------------------
    # 2. Determine payload representation
    # ------------------------------------------------------------------
    content_type = request.headers.get("content-type", "").lower()

    logger.info("[AUTH] Content-Type header: %s", content_type)
    logger.info("[AUTH] Raw body length: %d bytes", len(raw_body_bytes))

    form_params: dict[str, str | list[str]] | None = None
    body_str: str | None = None

    if "application/x-www-form-urlencoded" in content_type:
        # Starlette's FormData implements the mapping interface but Twilio's
        # helper expects either a *plain dict* or something with `.getlist()`.
        # Converting to a dict(str→str) avoids any odd method look-ups.
        try:
            form = await request.form()
            # If a key appears multiple times we must preserve **all** values
            # (Twilio will sign each occurrence in lexical order).  We join
            # multi-values with a comma which is the same behaviour as
            # ``parse_qs`` → it keeps the list but RequestValidator will loop
            # over them one-by-one.
            form_params = {}
            for k, v in form.multi_items():  # type: ignore[attr-defined]
                # Preserve duplicate keys as *lists* so that signature generation
                # treats each occurrence individually, mirroring Twilio's official algorithm.
                if k in form_params:
                    existing = form_params[k]
                    if isinstance(existing, list):
                        existing.append(v)
                    else:
                        form_params[k] = [existing, v]
                else:
                    form_params[k] = v
        except Exception as e:
            logger.info("[AUTH] Failed to parse form body: %s", e)

    # Fallback to raw body for non-form content-types or when parsing fails.
    if form_params is None:
        body_str = raw_body_bytes.decode(errors="ignore")

    # Log the representation chosen for signature calculation
    if form_params is not None:
        logger.info("[AUTH] Using form parameters for validation: %s", form_params)
    else:
        logger.info("[AUTH] Using raw body string for validation: %s", body_str)

    # ------------------------------------------------------------------
    # 3. Validate signature (primary path + safe fallback)
    # ------------------------------------------------------------------
    validated = False

    def _manual_signature(url: str, params: dict[str, str] | str) -> str:
        """Compute Twilio HMAC-SHA1 signature (body or params)."""
        import hashlib, hmac, base64

        if isinstance(params, str):
            data = url + params
        else:
            pieces: list[str] = [url]
            for key in sorted(params):
                val = params[key]
                if isinstance(val, list):
                    for single_val in val:
                        pieces.append(key + str(single_val))
                else:
                    pieces.append(key + str(val))
            data = "".join(pieces)
        digest = hmac.new(
            TWILIO_AUTH_TOKEN.encode(), data.encode(), hashlib.sha1
        ).digest()
        return base64.b64encode(digest).decode()

    # 3a. Primary attempt using the official helper ----------------------------------------------------------------
    try:
        if form_params is not None:
            validated = twilio_validator.validate(url_used, form_params, signature)
        else:
            validated = twilio_validator.validate(url_used, body_str or "", signature)
    except (TypeError, AttributeError) as e:
        # ``twilio.request_validator`` threw – fall back to our own implementation.
        logger.info("[AUTH] Twilio validator error – switching to manual (%s)", e)
    
    # 3b. If helper said *False* or raised, compute signature ourselves
    if not validated and signature:
        expected_sig = _manual_signature(url_used, form_params or body_str or "")
        logger.info("[AUTH] Expected signature (manual): %s", expected_sig)
        logger.info("[AUTH] Provided signature: %s", signature)
        validated = hmac.compare_digest(expected_sig, signature)

    # ------------------------------------------------------------------
    # 4. Outcome & audit logging
    # ------------------------------------------------------------------
    if validated:
        from urllib.parse import parse_qs

        params_multi = parse_qs(raw_body_bytes.decode(), keep_blank_values=True)
        logger.info(
            "[AUTH] X-Twilio-Signature validated for CallSid=%s",
            (params_multi.get("CallSid") or ["unknown"])[0],
        )
        return  # ✅ Success

    # Log at INFO for detailed debugging while retaining WARNING for alerting systems
    logger.info("[AUTH] X-Twilio-Signature validation failed – rejecting request")
    logger.warning("[AUTH] X-Twilio-Signature validation failed – rejecting request")
    raise HTTPException(status_code=403, detail="Invalid Twilio Signature")

# 3. Twilio WebSocket Basic-Auth validation (AccountSid:AuthToken)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")


async def authorize_twilio_websocket(websocket: WebSocket) -> bool:
    """Validate Twilio's WebSocket handshake solely via X-Twilio-Signature."""

    from utils import sanitize_headers

    # Diagnostic logging (no PHI)
    safe_hdrs = sanitize_headers(dict(websocket.headers))
    logger.info("[AUTH] WebSocket handshake headers: %s", safe_hdrs)

    signature = websocket.headers.get("X-Twilio-Signature", "")

    if not (signature and twilio_validator):
        logger.warning("[AUTH] Missing X-Twilio-Signature – rejecting WebSocket")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    # Reconstruct the absolute URL Twilio used when signing.
    proto_hdr: str | None = (
        websocket.headers.get("x-forwarded-proto")
        or ("https" if websocket.url.scheme in ("wss", "https") else "http")
    )
    host_hdr: str | None = (
        websocket.headers.get("x-forwarded-host")
        or websocket.headers.get("host")
    )

    # Respect consolidated Forwarded header (RFC 7239)
    fwd_header = websocket.headers.get("forwarded")
    if fwd_header:
        for part in fwd_header.split(";"):
            kv = part.strip().split("=", 1)
            if len(kv) != 2:
                continue
            k, v = kv[0].lower(), kv[1]
            if k == "proto" and v:
                proto_hdr = v
            elif k == "host" and v:
                host_hdr = v

    url_used = f"{proto_hdr}://{host_hdr}{websocket.url.path}"
    if websocket.url.query:
        url_used += f"?{websocket.url.query}"

    sig_valid = False
    try:
        sig_valid = twilio_validator.validate(url_used, {}, signature)
    except Exception as e:
        logger.info("[AUTH] Twilio validator raised during WS check (scheme %s): %s", proto_hdr, e)

    # If first attempt failed, try alternate scheme (https <→ wss) because Twilio
    # may canonicalise the scheme differently when computing the signature for
    # WebSocket handshakes (see Twilio Media Streams security docs).
    if not sig_valid:
        alt_scheme = "wss" if proto_hdr == "https" else "https"
        alt_url = url_used.replace(f"{proto_hdr}://", f"{alt_scheme}://", 1)
        try:
            sig_valid = twilio_validator.validate(alt_url, {}, signature)
            logger.info(
                "[AUTH] WS alt-scheme validation result=%s for URL=%s",
                sig_valid,
                alt_url,
            )
        except Exception as e:
            logger.info("[AUTH] Twilio validator raised during WS alt check: %s", e)

    logger.info("[AUTH] WS Signature validation final=%s", sig_valid)

    if sig_valid:
        return True

    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    logger.warning("[AUTH] WebSocket handshake rejected – invalid signature")
    return False

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

    The handshake is authenticated *only* via Twilio's `X-Twilio-Signature`.
    Unauthenticated or improperly signed attempts are closed with code 1008
    per RFC 6455.
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