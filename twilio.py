import json
import asyncio
import uuid
from typing import Dict, Optional, List, Any
from datetime import datetime
import traceback
from fastapi import WebSocket, Response, Request
from fastapi.responses import Response
from starlette.websockets import WebSocketState
import base64
from collections import deque

from utils import logger, TWIML_STREAM_TEMPLATE, get_ngrok_url, send_periodic_pings, send_silence_keepalive, cancel_keepalive_if_needed
from dpg import get_deepgram_service
from ell import tts_service
from oai import stream_agent_deltas
from vad_events import interruption_manager
from services.cache import get_json, set_json  # NEW

class TwilioService:
    """Service for handling Twilio calls and WebSocket interactions"""
    
    def __init__(self):
        """Initialize the Twilio service"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.websocket_connection_attempts: Dict[str, List[Dict[str, Any]]] = {}
        self.call_to_stream_sid: Dict[str, str] = {}
        # Map call_sid -> tenant id to build cache keys
        self.call_to_tenant: Dict[str, str] = {}
        # Track keepalive tasks for each session
        self.keepalive_tasks: Dict[str, asyncio.Task] = {}
        # Track received media frame counts for logging
        self.media_frame_counts: Dict[str, int] = {}
        # Buffer for media frames before Deepgram connection is ready
        self.media_buffer: Dict[str, deque] = {}
        
        # Get the deepgram service instance
        self.deepgram_service = get_deepgram_service()
        
        # Set transcript callback on deepgram service
        self.deepgram_service.set_transcript_callback(self.process_transcript)
        
        logger.info("TwilioService initialized")
        
    async def _flush_media_buffer(self, call_sid: str):
        """Flush any buffered media frames to Deepgram."""
        buf = self.media_buffer.get(call_sid)
        if not buf:
            return
        logger.info(f"🔍 Flushing {len(buf)} buffered audio frames for call: {call_sid}")
        while buf:
            frame = buf.popleft()
            await self.deepgram_service.send_audio(call_sid, frame)
        del self.media_buffer[call_sid]
        
    async def handle_voice_webhook(self, request: Request) -> Response:
        """
        Handle incoming Twilio voice calls and initiate a WebSocket stream.
        
        Args:
            request: The HTTP request from Twilio
            
        Returns:
            Response with TwiML instructions
        """
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        
        if not call_sid:
            logger.error("❌ Critical error: No CallSid provided in Twilio webhook")
            return Response(
                content="<Response><Say>We're sorry, but there was an error processing your call. Please try again later.</Say></Response>",
                media_type="application/xml"
            )
        
        logger.info(f"Incoming call received: {call_sid}")
        
        # Log additional Twilio parameters for debugging
        try:
            twilio_params = dict(form_data)
            logger.info(f"Twilio parameters for call {call_sid}: {json.dumps(twilio_params)}")
        except Exception as e:
            logger.warning(f"Could not log Twilio parameters: {str(e)}")
        
        # Generate WebSocket URL - prioritize ngrok HTTPS URL if available
        ngrok_url = get_ngrok_url()
        
        if ngrok_url and ngrok_url.startswith("https://"):
            # Use the ngrok HTTPS URL which has valid certificates
            websocket_url = f"{ngrok_url.replace('https://', 'wss://')}/twilio-stream"
            logger.info(f"Using ngrok HTTPS URL for WebSocket: {websocket_url}")
        else:
            # Fallback to request-based URL determination
            host = request.headers.get("host", "localhost:8080")
            scheme = request.headers.get("x-forwarded-proto", "http")
            
            logger.info(f"🔍 Request headers: {dict(request.headers)}")
            logger.info(f"🔍 Detected host: {host}, scheme: {scheme}")
            
            websocket_url = f"wss://{host}/twilio-stream"
            if scheme == "http" and "localhost" in host:
                # Only use ws:// for local development
                websocket_url = f"ws://{host}/twilio-stream"
            
            logger.info(f"Using request-based WebSocket URL: {websocket_url}")
        
        logger.info(f"🔍 Generated WebSocket URL: {websocket_url}")
        
        # Store CallSid in a pending connections dictionary to associate it later
        self.websocket_connection_attempts[call_sid] = []
        
        # Generate TwiML response
        twiml = TWIML_STREAM_TEMPLATE.format(websocket_url=websocket_url)
        
        logger.info(f"Instructing Twilio to connect to WebSocket: {websocket_url}")
        logger.debug(f"🔍 Generated TwiML response: {twiml}")
        
        return Response(content=twiml, media_type="application/xml")
        
    async def handle_twilio_stream(self, websocket: WebSocket):
        """
        WebSocket endpoint for Twilio audio streaming.
        
        Args:
            websocket: The WebSocket connection from Twilio
        """
        # Enhanced logging for incoming connection request
        connection_request_details = {
            "headers": dict(websocket.headers),
            "path": websocket.url.path,
            "query_params": dict(websocket.query_params),
            "raw_path": str(websocket.url),
        }
        logger.info(f"🔍 WebSocket connection request details: {json.dumps(connection_request_details)}")
        
        # We'll identify the proper CallSid from the initial events
        call_sid = None
        session_id = None
        
        # Track last activity time for auto-termination
        last_activity_time = datetime.now().timestamp()
        # Timeout in seconds - 60 seconds of inactivity will close the connection
        inactivity_timeout = 60
        # Timeout for receiving CallSid - fail if we don't get it within this time
        callsid_timeout = 10  # seconds
        callsid_start_time = datetime.now().timestamp()
        
        # Enhanced connection logging
        headers = dict(websocket.headers)
        logger.info(f"🔍 WebSocket connection request received, awaiting CallSid")
        logger.info(f"🔍 WebSocket request headers: {json.dumps(headers)}")
        logger.info(f"🔍 WebSocket connection path: {websocket.url.path}")
        logger.info(f"🔍 WebSocket connection query params: {dict(websocket.query_params)}")
        
        # Log connection attempt with placeholder for CallSid
        connection_attempt = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_host": websocket.client.host if hasattr(websocket, "client") else "unknown",
            "client_port": websocket.client.port if hasattr(websocket, "client") else "unknown",
            "headers": headers,
            "status": "waiting_for_callsid"
        }
        
        # Store connection attempt temporarily until we get CallSid
        pending_connection_attempt = connection_attempt
        
        logger.info(f"WebSocket connection attempt from {connection_attempt['client_host']}:{connection_attempt['client_port']}, awaiting CallSid")
        
        try:
            # Log pre-acceptance state
            logger.info(f"🔍 Attempting to accept WebSocket connection, awaiting CallSid")
            
            # Accept the WebSocket connection
            await websocket.accept()
            logger.info(f"✅ WebSocket connection successfully established, awaiting CallSid")
            
            # Register DeepgramService main event loop for dispatching transcripts
            try:
                loop = asyncio.get_running_loop()
                self.deepgram_service.set_main_loop(loop)
            except Exception as e:
                logger.error(f"❌ Could not set DeepgramService main_loop: {e}")
            
            # Start the ping task with a placeholder session ID until we get CallSid
            placeholder_id = f"pending_{uuid.uuid4()}"
            self.active_connections[placeholder_id] = websocket
            logger.info(f"🔍 Starting ping task for WebSocket pending session")
            ping_task = asyncio.create_task(send_periodic_pings(websocket, placeholder_id))
            
            try:
                # Wait for initial messages to extract CallSid
                logger.info(f"🔍 Waiting for initial messages to extract CallSid")
                
                # Initial loop to wait for CallSid
                while not call_sid:
                    # Check for CallSid timeout
                    if datetime.now().timestamp() - callsid_start_time > callsid_timeout:
                        error_msg = f"Timeout waiting for CallSid after {callsid_timeout} seconds"
                        logger.error(f"❌ {error_msg}")
                        await websocket.close(code=1008, reason=error_msg)
                        return
                    
                    # Check for inactivity timeout
                    if datetime.now().timestamp() - last_activity_time > inactivity_timeout:
                        error_msg = f"Inactivity timeout reached while waiting for CallSid"
                        logger.error(f"❌ {error_msg}")
                        await websocket.close(code=1008, reason=error_msg)
                        return
                    
                    # Use wait_for to add a timeout to receive operation
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                        # Update last activity time whenever a message is received
                        last_activity_time = datetime.now().timestamp()
                    except asyncio.TimeoutError:
                        # No message received, continue the loop to check timeouts
                        continue
                    except RuntimeError as re:
                        # Handle "Cannot call 'receive' once a disconnect message has been received"
                        if "disconnect message has been received" in str(re):
                            logger.info(f"WebSocket disconnected while waiting for CallSid")
                            return
                        else:
                            # Re-raise if it's a different RuntimeError
                            raise re
                    
                    logger.debug(f"🔍 Received WebSocket message while waiting for CallSid")
                    
                    # Try to extract CallSid from the message
                    if "text" in message:
                        text_data = message["text"]
                        try:
                            data = json.loads(text_data)
                            event = data.get("event")
                            
                            if event == "start":
                                logger.info(f"Stream started event received")
                                stream_sid = data.get("streamSid")
                                if stream_sid:
                                    logger.info(f"✅ Extracted Stream SID: {stream_sid}")
                                    start_call_sid = data.get("start", {}).get("callSid")
                                    if start_call_sid:
                                        call_sid = start_call_sid
                                        self.call_to_stream_sid[call_sid] = stream_sid
                                        session_id = f"{call_sid}_{uuid.uuid4()}"
                                        self.active_connections[session_id] = websocket
                                        logger.info(f"🔍 Proactively starting Deepgram connection for call: {call_sid}")
                                        # Start Deepgram connection in background to reduce TTS latency
                                        asyncio.create_task(self.deepgram_service.setup_connection(call_sid))
                                        # Initialize media buffer for this call
                                        self.media_buffer[call_sid] = deque(maxlen=200)
                                        try:
                                            del self.active_connections[placeholder_id]
                                        except KeyError:
                                            pass
                                        
                                        logger.info(f"Generating welcome message immediately for call {call_sid} with Stream SID {stream_sid}")
                                        await tts_service.generate_welcome_message(call_sid, websocket, stream_sid)
                                        
                                        # Start the silence keep-alive task after welcome message completes
                                        logger.info(f"Starting silence keep-alive after welcome message for call {call_sid}")
                                        self.keepalive_tasks[session_id] = asyncio.create_task(
                                            send_silence_keepalive(websocket, stream_sid, session_id, tts_service)
                                        )
                                        
                                        # Map call to tenant id for later DB/Redis lookups
                                        self.call_to_tenant[call_sid] = websocket.headers.get("x-tenant-id", "default")
                                        
                                        # Ensure Redis history key exists
                                        history_key = f"{self.call_to_tenant[call_sid]}:hist:{call_sid}"
                                        existing_hist = await get_json(history_key)
                                        if existing_hist is None:
                                            await set_json(history_key, [])
                                        
                                        # Exit the loop now that we have CallSid
                                        break
                                    
                                    # Also look for stream SID in case it contains the CallSid
                                    elif stream_sid.startswith("CA"):
                                        logger.info(f"✅ Extracted CallSid {stream_sid} from streamSid field")
                                        call_sid = stream_sid
                                        self.call_to_stream_sid[call_sid] = stream_sid
                                        
                                        # Create a unique session ID using call_sid
                                        session_id = f"{call_sid}_{uuid.uuid4()}"
                                        
                                        # Update the active connections - replace placeholder with real session
                                        self.active_connections[session_id] = websocket
                                        logger.info(f"🔍 Proactively starting Deepgram connection for call: {call_sid}")
                                        asyncio.create_task(self.deepgram_service.setup_connection(call_sid))
                                        self.media_buffer[call_sid] = deque(maxlen=200)
                                        try:
                                            del self.active_connections[placeholder_id]
                                        except KeyError:
                                            pass
                                        
                                        # Generate and send welcome message immediately now that we have both Call SID and Stream SID
                                        logger.info(f"Generating welcome message immediately for call {call_sid} with Stream SID {stream_sid}")
                                        await tts_service.generate_welcome_message(call_sid, websocket, stream_sid)
                                        
                                        # Start the silence keep-alive task after welcome message completes
                                        logger.info(f"Starting silence keep-alive after welcome message for call {call_sid}")
                                        self.keepalive_tasks[session_id] = asyncio.create_task(
                                            send_silence_keepalive(websocket, stream_sid, session_id, tts_service)
                                        )
                                        
                                        # Map call to tenant id for later DB/Redis lookups
                                        self.call_to_tenant[call_sid] = websocket.headers.get("x-tenant-id", "default")
                                        
                                        # Exit the loop now that we have CallSid
                                        break
                                else:
                                    logger.warning(f"⚠️ No Stream SID found in start event")
                                    
                            elif event == "connected":
                                logger.info(f"✅ Twilio WebSocket connection established for call: {call_sid}")
                                
                                # Note: Removing the welcome message queuing logic here
                                # We'll send the welcome message directly when we get the "start" event
                                logger.info(f"Waiting for Stream SID and Call SID before sending welcome message")
                                
                            elif event == "media":
                                # Parse Twilio media event payload
                                payload_b64 = data.get("media", {}).get("payload")
                                if payload_b64:
                                    pcm_ulaw = base64.b64decode(payload_b64)
                                    # Feed frame into interruption detector before anything else
                                    interruption_manager.process_frame(call_sid, pcm_ulaw)
                                    count = self.media_frame_counts.get(session_id, 0) + 1
                                    self.media_frame_counts[session_id] = count
                                    await cancel_keepalive_if_needed(session_id, self.keepalive_tasks)
                                    if pcm_ulaw:
                                        if self.deepgram_service.connection_ready.get(call_sid):
                                            await self._flush_media_buffer(call_sid)
                                            success = await self.deepgram_service.send_audio(call_sid, pcm_ulaw)
                                            if not success:
                                                logger.warning(f"Failed to send audio to Deepgram for call {call_sid}")
                                        else:
                                            buf = self.media_buffer.get(call_sid)
                                            if buf is not None:
                                                buf.append(pcm_ulaw)
                                                logger.debug(f"🔍 Deepgram not ready, buffering media frame {count} for call: {call_sid}")
                                            else:
                                                logger.debug(f"🔍 Media buffer not found (already flushed), dropping frame {count} for call: {call_sid}")
                                else:
                                    logger.debug(f"📦 No media payload for call {call_sid}")
                                
                            elif event == "stop":
                                logger.info(f"Stream stop event received for call: {call_sid}")
                                
                                # Mark Deepgram call as closed to prevent reconnection
                                self.deepgram_service.call_closed[call_sid] = True
                                # Close Deepgram connection gracefully when the stream stops
                                logger.info(f"Gracefully closing Deepgram connection due to stream stop for call: {call_sid}")
                                await self.deepgram_service.close_connection(call_sid)
                                logger.info(f"✅ Deepgram connection closed due to stream stop for call: {call_sid}")
                                
                                # Keep the WebSocket connection open for user interaction
                                logger.info(f"Keeping WebSocket connection open for user interaction after stream stop for call: {call_sid}")
                                
                            elif event == "mark":
                                # Handle Twilio mark events for speech detection
                                mark_data = data.get("mark", {})
                                mark_name = mark_data.get("name", "unknown")
                                
                                logger.info(f"📢 Mark event received for call {call_sid}: {mark_name}")
                                logger.info(f"📢 Full mark event data: {json.dumps(data)}")
                                
                                # Different handling based on the mark name
                                if "speech-start" in mark_name or "start-speech" in mark_name:
                                    logger.info(f"🎤 Speech started for call {call_sid}")
                                    # Ensure Deepgram connection is active when speech starts
                                    await self.deepgram_service.setup_connection(call_sid)
                                elif "speech-end" in mark_name or "end-speech" in mark_name:
                                    logger.info(f"🎤 Speech ended for call {call_sid}")
                                elif "no-speech" in mark_name:
                                    logger.info(f"🔇 No speech detected for call {call_sid}")
                                else:
                                    # Log other mark events for debugging
                                    logger.info(f"📢 Other mark event '{mark_name}' for call {call_sid}")
                                    
                                # Always update activity time for any mark event
                                last_activity_time = datetime.now().timestamp()
                                
                            else:
                                logger.info(f"🔍 Received unknown event type: {event} for call: {call_sid}")
                                logger.info(f"🔍 Full event data: {json.dumps(data)}")
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON text message: {text_data}")
                    
                # Main message handling loop
                logger.info(f"Successfully extracted CallSid {call_sid}, entering main WebSocket loop")
                
                # Ensure tenant history exists in redis
                history_key = f"{self.call_to_tenant.get(call_sid, 'default')}:hist:{call_sid}"
                existing_hist = await get_json(history_key)
                if existing_hist is None:
                    await set_json(history_key, [])
                
                # Continue processing messages now that we have the CallSid
                while True:
                    # Check for inactivity timeout
                    if datetime.now().timestamp() - last_activity_time > inactivity_timeout:
                        logger.warning(f"Inactivity timeout reached for call {call_sid}, closing connection")
                        await websocket.close(1001, "Inactivity timeout")
                        break
                        
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
                        last_activity_time = datetime.now().timestamp()
                        
                        # Handle text messages (events from Twilio)
                        if "text" in message:
                            text_data = message["text"]
                            try:
                                data = json.loads(text_data)
                                event = data.get("event")
                                
                                if event == "stop":
                                    logger.info(f"Stream stop event received for call: {call_sid}")
                                    
                                    # Mark Deepgram call as closed to prevent reconnection
                                    self.deepgram_service.call_closed[call_sid] = True
                                    # Close Deepgram connection gracefully
                                    await self.deepgram_service.close_connection(call_sid)
                                    logger.info(f"✅ Deepgram connection closed due to stream stop for call: {call_sid}")
                                    
                                    # Keep WebSocket open but break the loop to clean up
                                    break
                                    
                                elif event == "mark":
                                    mark_data = data.get("mark", {})
                                    mark_name = mark_data.get("name", "unknown")
                                    
                                    logger.info(f"📢 Mark event received for call {call_sid}: {mark_name}")
                                    
                                    # Different handling based on the mark name
                                    if "speech-start" in mark_name or "start-speech" in mark_name:
                                        logger.info(f"🎤 Speech started for call {call_sid}")
                                        # Ensure Deepgram connection is active when speech starts
                                        await self.deepgram_service.setup_connection(call_sid)
                                        
                                elif event == "media":
                                    # Parse Twilio media event payload
                                    payload_b64 = data.get("media", {}).get("payload")
                                    if payload_b64:
                                        pcm_ulaw = base64.b64decode(payload_b64)
                                        # Feed frame into interruption detector before anything else
                                        interruption_manager.process_frame(call_sid, pcm_ulaw)
                                        count = self.media_frame_counts.get(session_id, 0) + 1
                                        self.media_frame_counts[session_id] = count
                                        await cancel_keepalive_if_needed(session_id, self.keepalive_tasks)
                                        if pcm_ulaw:
                                            if self.deepgram_service.connection_ready.get(call_sid):
                                                await self._flush_media_buffer(call_sid)
                                                success = await self.deepgram_service.send_audio(call_sid, pcm_ulaw)
                                                if not success:
                                                    logger.warning(f"Failed to send audio to Deepgram for call {call_sid}")
                                            else:
                                                buf = self.media_buffer.get(call_sid)
                                                if buf is not None:
                                                    buf.append(pcm_ulaw)
                                                    logger.debug(f"🔍 Deepgram not ready, buffering media frame {count} for call: {call_sid}")
                                                else:
                                                    logger.debug(f"🔍 Media buffer not found (already flushed), dropping frame {count} for call: {call_sid}")
                                    else:
                                        logger.debug(f"📦 No media payload for call {call_sid}")
                                
                                # Other events are handled the same as in the initial loop
                                else:
                                    logger.debug(f"Event {event} in main loop for call {call_sid}")
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Received non-JSON text message: {text_data}")
                                
                    except asyncio.TimeoutError:
                        # No message received, continue the loop to check timeouts
                        continue
                    except RuntimeError as re:
                        # Handle "Cannot call 'receive' once a disconnect message has been received"
                        if "disconnect message has been received" in str(re):
                            logger.info(f"WebSocket disconnected for call {call_sid}")
                            break
                        else:
                            # Re-raise if it's a different RuntimeError
                            raise re
                            
            except Exception as e:
                logger.error(f"❌ Error in main WebSocket loop for call {call_sid}: {str(e)}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            # This will catch exceptions during connection acceptance
            logger.error(f"❌ Failed to establish WebSocket connection: {str(e)}")
            logger.error(traceback.format_exc())
            # Add connection failure to tracking
            pending_connection_attempt["status"] = "failed"
            pending_connection_attempt["error"] = str(e)
            pending_connection_attempt["traceback"] = traceback.format_exc()
            
        finally:
            # Clean up
            if 'ping_task' in locals():
                logger.info(f"🔍 Cancelling ping task for call: {call_sid if call_sid else 'unknown'}")
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up keep-alive task if exists
            if session_id and session_id in self.keepalive_tasks:
                logger.info(f"🔍 Cancelling silence keep-alive task for session: {session_id}")
                self.keepalive_tasks[session_id].cancel()
                try:
                    await self.keepalive_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self.keepalive_tasks[session_id]

            # Mark the call as closed to suppress reconnection logic for Deepgram
            if call_sid:
                logger.debug(f"🔍 Marking Deepgram call_closed for call: {call_sid} in finally cleanup")
                self.deepgram_service.call_closed[call_sid] = True
                # Cleanup interruption detector resources
                interruption_manager.cleanup(call_sid)

            # Close Deepgram connection
            if call_sid:
                logger.info(f"🔍 Closing Deepgram connection for call: {call_sid}")
                await self.deepgram_service.close_connection(call_sid)
            
            # Remove from active connections
            if session_id and session_id in self.active_connections:
                logger.info(f"🔍 Removing session from active connections: {session_id}")
                del self.active_connections[session_id]
            elif placeholder_id and placeholder_id in self.active_connections:
                logger.info(f"🔍 Removing placeholder session from active connections: {placeholder_id}")
                del self.active_connections[placeholder_id]
            
            # Clean up conversation history
            if call_sid:
                # cleanup redis expiration left as-is; no explicit delete
                if call_sid in self.call_to_tenant:
                    del self.call_to_tenant[call_sid]
            
            logger.info(f"WebSocket connection closed and cleaned up for call: {call_sid if call_sid else 'unknown'}")
            
    async def process_transcript(self, transcript: str, call_sid: str, is_final: bool):
        """
        Process transcripts with OpenAI agent and generate responses
        
        Args:
            transcript: The text transcript from Deepgram
            call_sid: The Twilio call SID
            is_final: Whether this is a final transcript or interim
        """
        logger.debug(f"🔍 process_transcript invoked for call {call_sid}, is_final={is_final}: '{transcript}'")
        tenant = self.call_to_tenant.get(call_sid, "default")
        
        # Drop non-final transcripts and skip response if agent cannot speak yet
        if not is_final:
            return
        if not interruption_manager.can_agent_speak(call_sid):
            logger.info(f"Skipping agent response for call {call_sid} due to pending barge-in or speaking flag")
            return
        
        try:
            # Process with OpenAI agent
            history_key = f"{tenant}:hist:{call_sid}"
            conversation_history = await get_json(history_key) or []
            
            # Process with agent via streaming and capture cancellation handle
            updated_history, delta_generator, llm_handle = await stream_agent_deltas(transcript, conversation_history, tenant)
            # Register LLM streaming handle so it can be cancelled on barge-in
            interruption_manager.register_llm_handle(call_sid, llm_handle)
            
            # Update conversation history
            await set_json(history_key, updated_history)
            
            logger.info(f"⏩ PIPELINE: Streaming agent response for call {call_sid}")
            
            # Find active WebSocket for this call
            matching_sessions = [sid for sid in self.active_connections.keys() if sid.startswith(f"{call_sid}_")]
            
            if not matching_sessions:
                logger.error(f"❌ No active WebSocket connection found for call {call_sid}")
                return
            
            # Use the first matching session
            session_id = matching_sessions[0]
            websocket = self.active_connections[session_id]
            
            # Cancel silence keep-alive when sending an agent response
            if session_id in self.keepalive_tasks:
                logger.info(f"Cancelling silence keep-alive as agent is responding for session: {session_id}")
                self.keepalive_tasks[session_id].cancel()
                try:
                    await self.keepalive_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self.keepalive_tasks[session_id]
            
            # Get stream SID for this call
            stream_sid = self.call_to_stream_sid.get(call_sid)
            if not stream_sid:
                logger.error(f"❌ No Stream SID found for call {call_sid}. Cannot send response.")
                return
            
            # Stream the response with low-latency TTS
            await tts_service.stream_response_to_user(call_sid, delta_generator, websocket, stream_sid)
            
            # Clear handle registration once streaming completes
            interruption_manager.register_llm_handle(call_sid, None)
            
        except Exception as e:
            logger.error(f"❌ Error processing transcript for call {call_sid}: {str(e)}")
            logger.error(traceback.format_exc())

# Create a singleton instance
twilio_service = TwilioService() 