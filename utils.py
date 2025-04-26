import asyncio
import json
import logging
import os
import time
from datetime import datetime
from starlette.websockets import WebSocketState
import requests
import base64
from typing import Dict

# Configure logging
def setup_logging(app_name="pl-voiceagent"):
    """
    Set up application logging with file and console handlers.
    
    Args:
        app_name: Name of the application for the logger
        
    Returns:
        The configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Set up logger
    logger = logging.getLogger(app_name)
    
    # Set file handler for persistent logging
    file_handler = logging.FileHandler(f"logs/{app_name}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

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

def get_ngrok_url():
    """
    Get the ngrok public URL from the ngrok API or environment.
    
    Returns:
        str: The ngrok public URL or empty string if not found
    """
    ngrok_url = os.environ.get("NGROK_URL", "")
    
    if not ngrok_url:
        try:
            # Try to get the URL from ngrok's API
            # Wait briefly to ensure ngrok is up
            time.sleep(2)
            
            # Try to get the URL from ngrok's API
            response = requests.get("http://ngrok:4040/api/tunnels", timeout=5)
            tunnels = response.json().get("tunnels", [])
            
            if tunnels:
                # Find the HTTPS tunnel
                https_tunnels = [t for t in tunnels if t["public_url"].startswith("https://")]
                if https_tunnels:
                    ngrok_url = https_tunnels[0]["public_url"]
                    os.environ["NGROK_URL"] = ngrok_url
                    
                    # Log the important URLs for Twilio configuration
                    logger.info("======================= TWILIO CONFIGURATION URLS =======================")
                    logger.info(f"NGROK HTTPS URL: {ngrok_url}")
                    logger.info(f"Twilio Voice Webhook URL: {ngrok_url}/voice")
                    logger.info(f"Twilio WebSocket Stream URL: {ngrok_url}/twilio-stream")
                    logger.info("======================================================================")
                else:
                    logger.warning("No HTTPS ngrok tunnels found. Make sure --https=true is set in ngrok command.")
                    ngrok_url = tunnels[0]["public_url"]
                    logger.info(f"Using HTTP ngrok URL instead: {ngrok_url}")
            else:
                logger.warning("No ngrok tunnels found. Check ngrok container status.")
        except Exception as e:
            logger.error(f"Failed to get ngrok URL: {str(e)}")
            logger.info("If using ngrok, please check the ngrok container logs for the public URL.")
    
    return ngrok_url 

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