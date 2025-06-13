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
import os
import time

from utils import (
    logger,
    TWIML_STREAM_TEMPLATE,
    send_periodic_pings,
    send_silence_keepalive,
    cancel_keepalive_if_needed,
    enforce_call_length_limit,
    sanitize_headers,
)
from dpg import get_deepgram_service
from ell import tts_service
from oai import (
    stream_agent_deltas,
    get_agent,
    ensure_agent_initialized,
    schedule_openai_connection_warmup,
    schedule_agent_prewarm,
)
from vad_events import interruption_manager, HUMAN_GAP_SEC
from services.cache import get_json, set_json, CacheWriteError


class TwilioService:
    """Service for handling Twilio calls and WebSocket interactions"""

    def __init__(self):
        """Initialize the Twilio service"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.websocket_connection_attempts: Dict[str, List[Dict[str, Any]]] = {}
        self.call_to_stream_sid: Dict[str, str] = {}
        # Track keepalive tasks for each session
        self.keepalive_tasks: Dict[str, asyncio.Task] = {}
        # Track received media frame counts for logging
        self.media_frame_counts: Dict[str, int] = {}
        # Buffer for media frames before Deepgram connection is ready
        self.media_buffer: Dict[str, deque] = {}

        # Buffer transcripts spoken during a barge-in event
        self.barge_in_buffers: Dict[str, List[str]] = {}
        # Track callbacks registered with the interruption manager
        self.barge_in_registered: set[str] = set()
        # Temporary queue for transcripts skipped while agent is speaking
        self.skip_queues: Dict[str, List[str]] = {}
        self.skip_registered: set[str] = set()

        # Locks to serialize transcript processing per call
        self.transcript_locks: Dict[str, asyncio.Lock] = {}

        # Get the deepgram service instance
        self.deepgram_service = get_deepgram_service()

        # Set transcript callback on deepgram service
        self.deepgram_service.set_transcript_callback(self.process_transcript)

        # --- Silence watchdog state ---
        # Track last time we heard the user speak (epoch seconds)
        self.last_user_input: Dict[str, float] = {}
        # Whether we have already prompted the caller for silence this round
        self.silence_prompted: Dict[str, bool] = {}
        # Background tasks per call for silence monitoring
        self.silence_tasks: Dict[str, asyncio.Task] = {}

        # Background tasks per call for maximum call length enforcement
        self.call_length_tasks: Dict[str, asyncio.Task] = {}

        logger.info("TwilioService initialized")

    async def _flush_media_buffer(self, call_sid: str):
        """Flush any buffered media frames to Deepgram."""
        buf = self.media_buffer.get(call_sid)
        if not buf:
            return
        logger.info(
            f"🔍 Flushing {len(buf)} buffered audio frames for call: {call_sid}"
        )
        while buf:
            frame = buf.popleft()
            await self.deepgram_service.send_audio(call_sid, frame)
        del self.media_buffer[call_sid]

    async def _process_barge_in_buffer(self, call_sid: str) -> None:
        """Send any buffered user speech collected during a barge-in."""
        transcripts = self.barge_in_buffers.pop(call_sid, [])
        if not transcripts:
            return
        # ensure registration flag cleared
        self.barge_in_registered.discard(call_sid)
        await asyncio.sleep(HUMAN_GAP_SEC)
        combined = " ".join(transcripts)
        await self.process_transcript(combined, call_sid, True)

    async def _process_skip_queue(self, call_sid: str) -> None:
        """Replay transcripts queued while the agent was speaking."""
        try:
            queue = self.skip_queues.get(call_sid, [])
            resume_event = interruption_manager.get_resume_event(call_sid)
            while queue:
                matching_sessions = [
                    sid
                    for sid in self.active_connections
                    if sid.startswith(f"{call_sid}_")
                ]
                if call_sid not in [
                    sid.split("_")[0] for sid in self.active_connections
                ]:
                    break
                websocket = (
                    self.active_connections.get(matching_sessions[0])
                    if matching_sessions
                    else None
                )
                if (
                    websocket is None
                    or websocket.client_state == WebSocketState.DISCONNECTED
                ):
                    break
                if not interruption_manager.can_agent_speak(call_sid):
                    try:
                        await asyncio.wait_for(resume_event.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass
                    resume_event.clear()
                    continue
                transcript = queue.pop(0)
                await self.process_transcript(transcript, call_sid, True)
            self.skip_queues.pop(call_sid, None)
        finally:
            self.skip_registered.discard(call_sid)
            interruption_manager.resume_events.pop(call_sid, None)

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
                media_type="application/xml",
            )

        logger.info(f"Incoming call received: {call_sid}")

        # Fire-and-forget warm-up of the OpenAI connection so that the first
        # agent response will not suffer a cold TLS/TCP handshake delay.
        # This costs zero tokens and completes while the welcome prompt plays.
        try:
            schedule_openai_connection_warmup()
            schedule_agent_prewarm(call_sid)
        except Exception as warm_err:
            logger.debug(f"Warm-up scheduling failed: {warm_err}")

        # Log additional Twilio parameters for debugging
        try:
            twilio_params = dict(form_data)
            # HIPAA Compliance: redact fields that may contain PHI such as caller phone numbers.
            sanitized_params = {k: v for k, v in twilio_params.items() if k.lower() not in ("caller", "from", "to", "callername")}
            logger.debug(
                f"Twilio parameters (sanitized) for call {call_sid}: {json.dumps(sanitized_params)}"
            )
        except Exception as e:
            logger.debug(f"Could not log sanitized Twilio parameters: {str(e)}")

        # Generate WebSocket URL for Twilio streaming
        host = request.headers.get("host", "localhost:8080")
        scheme = request.headers.get("x-forwarded-proto", "http")

        safe_headers = sanitize_headers(dict(request.headers))
        logger.info(f"🔍 Request headers: {safe_headers}")
        logger.info(f"🔍 Detected host: {host}, scheme: {scheme}")

        websocket_url = f"wss://{host}/twilio-stream"
        if scheme == "http" and "localhost" in host:
            # Only use ws:// for local development
            websocket_url = f"ws://{host}/twilio-stream"

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
            "headers": sanitize_headers(dict(websocket.headers)),
            "path": websocket.url.path,
            "query_params": dict(websocket.query_params),
            "raw_path": str(websocket.url),
        }
        logger.info(
            f"🔍 WebSocket connection request details: {json.dumps(connection_request_details)}"
        )

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
        headers = sanitize_headers(dict(websocket.headers))
        logger.info(f"🔍 WebSocket connection request received, awaiting CallSid")
        logger.info(f"🔍 WebSocket request headers: {json.dumps(headers)}")
        logger.info(f"🔍 WebSocket connection path: {websocket.url.path}")
        logger.info(
            f"🔍 WebSocket connection query params: {dict(websocket.query_params)}"
        )

        # Log connection attempt with placeholder for CallSid
        connection_attempt = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_host": (
                websocket.client.host if hasattr(websocket, "client") else "unknown"
            ),
            "client_port": (
                websocket.client.port if hasattr(websocket, "client") else "unknown"
            ),
            "headers": headers,
            "status": "waiting_for_callsid",
        }

        # Store connection attempt temporarily until we get CallSid
        pending_connection_attempt = connection_attempt

        logger.info(
            f"WebSocket connection attempt from {connection_attempt['client_host']}:{connection_attempt['client_port']}, awaiting CallSid"
        )

        try:
            # Log pre-acceptance state
            logger.info(
                f"🔍 Attempting to accept WebSocket connection, awaiting CallSid"
            )

            # Accept the WebSocket connection
            await websocket.accept()
            logger.info(
                f"✅ WebSocket connection successfully established, awaiting CallSid"
            )

            # Register DeepgramService main event loop for dispatching transcripts
            try:
                loop = asyncio.get_running_loop()
                self.deepgram_service.set_main_loop(loop)
            except Exception as e:
                logger.error(f"❌ Could not set DeepgramService main_loop: {e}")

            # Generate placeholder session ID for early tracking
            placeholder_id = str(uuid.uuid4())[:8]
            self.active_connections[placeholder_id] = websocket

            logger.info(
                f"📬 Awaiting initial message to extract CallSid (timeout: {callsid_timeout}s)"
            )

            # Wait for the first message to extract CallSid
            while not call_sid:
                timeout_elapsed = datetime.now().timestamp() - callsid_start_time
                if timeout_elapsed >= callsid_timeout:
                    logger.error(
                        f"❌ CallSid timeout after {callsid_timeout}s, closing WebSocket"
                    )
                    await websocket.close(1008, "CallSid timeout")
                    return

                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1.0)

                    if "text" in message:
                        try:
                            data = json.loads(message["text"])
                            event = data.get("event")

                            if event == "start":
                                # Extract CallSid from the start event
                                call_sid = data.get("start", {}).get("callSid")
                                stream_sid = data.get("start", {}).get("streamSid")

                                if call_sid:
                                    logger.info(
                                        f"🎯 CallSid identified: {call_sid}, StreamSid: {stream_sid}"
                                    )

                                    # Store Stream SID for later use
                                    if stream_sid:
                                        self.call_to_stream_sid[call_sid] = stream_sid

                                    # Generate proper session ID now that we have CallSid
                                    session_id = f"{call_sid}_{uuid.uuid4().hex[:8]}"

                                    # Move from placeholder to proper session ID
                                    del self.active_connections[placeholder_id]
                                    self.active_connections[session_id] = websocket

                                    # Initialize media buffer for this call
                                    self.media_buffer[call_sid] = deque()

                                    # Record connection attempt with CallSid
                                    pending_connection_attempt["call_sid"] = call_sid
                                    pending_connection_attempt["session_id"] = (
                                        session_id
                                    )
                                    pending_connection_attempt["status"] = "success"
                                    self.websocket_connection_attempts.setdefault(
                                        call_sid, []
                                    ).append(pending_connection_attempt)

                                    logger.info(
                                        f"✅ WebSocket connection established successfully. Call: {call_sid}, Session: {session_id}"
                                    )

                                    # Prefer Stream SID from the start event
                                    if not stream_sid:
                                        logger.warning(
                                            f"⚠️ No StreamSid found in start event for call {call_sid}"
                                        )

                                    break
                                else:
                                    logger.warning("⚠️ No CallSid found in start event")
                            else:
                                logger.debug(
                                    f"📩 Event {event} received while waiting for start event"
                                )

                        except json.JSONDecodeError:
                            logger.warning(
                                f"⚠️ Received non-JSON text message while waiting for CallSid: {message['text']}"
                            )

                except asyncio.TimeoutError:
                    # Continue waiting for CallSid
                    continue

            # At this point we have CallSid, continue with normal processing
            if not call_sid:
                logger.error("❌ CallSid could not be extracted from start event")
                await websocket.close(1008, "Invalid start event")
                return

            # Now process calls with a known CallSid

            # Start Deepgram connection for this call
            await self.deepgram_service.setup_connection(call_sid)

            # Generate and send welcome message after Deepgram connection is ready
            # HIPAA Compliance: §164.312(b) - Administrative safeguards for PHI communications
            # Log welcome message generation for audit trail per §164.312(b)
            logger.info(
                f"[HIPAA-AUDIT] welcome_message_generation initiated for call_sid={call_sid}"
            )
            try:
                stream_sid = self.call_to_stream_sid.get(call_sid)
                if not stream_sid:
                    logger.warning(
                        f"⚠️ Stream SID not immediately available for call {call_sid}, waiting briefly"
                    )
                    for _ in range(5):
                        await asyncio.sleep(0.01)
                        stream_sid = self.call_to_stream_sid.get(call_sid)
                        if stream_sid:
                            logger.info(
                                f"✅ Stream SID obtained after retry: {stream_sid}"
                            )
                            break

                if stream_sid:
                    logger.info(
                        f"🔊 Generating welcome message for call {call_sid} with stream {stream_sid}"
                    )

                    # -------------------------------------------------
                    # Start maximum call length timer (15 minutes)
                    # -------------------------------------------------
                    length_task = asyncio.create_task(
                        enforce_call_length_limit(
                            call_sid,
                            websocket,
                            self.deepgram_service,
                        )
                    )
                    self.call_length_tasks[call_sid] = length_task

                    await tts_service.generate_welcome_message(
                        call_sid, websocket, stream_sid
                    )
                    # Pre-initialize the agent session while caller hears welcome
                    asyncio.create_task(ensure_agent_initialized())
                    # HIPAA Compliance: Log successful welcome message delivery per §164.312(b)
                    logger.info(
                        f"[HIPAA-AUDIT] welcome_message_delivered successfully for call_sid={call_sid}"
                    )

                    # Initialise silence tracking after welcome is delivered
                    self.last_user_input[call_sid] = time.time()
                    silence_task = asyncio.create_task(
                        self._monitor_user_silence(call_sid, websocket, stream_sid)
                    )
                    self.silence_tasks[call_sid] = silence_task
                else:
                    logger.error(
                        f"❌ Cannot generate welcome message: No Stream SID found for call {call_sid} after retry"
                    )
                    # HIPAA Compliance: Log failed welcome message for audit trail per §164.312(b)
                    logger.error(
                        f"[HIPAA-AUDIT] welcome_message_failed no_stream_sid for call_sid={call_sid}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ Error generating welcome message for call {call_sid}: {str(e)}"
                )
                # HIPAA Compliance: Log welcome message errors for audit trail per §164.312(b)
                logger.error(
                    f"[HIPAA-AUDIT] welcome_message_error for call_sid={call_sid}: {str(e)}"
                )

            # Start periodic ping task to keep connection alive
            ping_task = asyncio.create_task(send_periodic_pings(websocket, call_sid))
            logger.info(f"🏓 Ping task started for call: {call_sid}")

            # Handle conversation history initialization
            if call_sid:
                history_key = f"hist:{call_sid}"
                existing_hist = await get_json(history_key)
                if existing_hist is None:
                    await set_json(history_key, [])

                # Continue processing messages now that we have the CallSid
                while True:
                    # Check for inactivity timeout
                    if (
                        datetime.now().timestamp() - last_activity_time
                        > inactivity_timeout
                    ):
                        logger.warning(
                            f"Inactivity timeout reached for call {call_sid}, closing connection"
                        )
                        await websocket.close(1001, "Inactivity timeout")
                        break

                    try:
                        message = await asyncio.wait_for(
                            websocket.receive(), timeout=5.0
                        )
                        last_activity_time = datetime.now().timestamp()

                        # Handle text messages (events from Twilio)
                        if "text" in message:
                            text_data = message["text"]
                            try:
                                data = json.loads(text_data)
                                event = data.get("event")
                                # Capture Stream SID from any event if not already stored
                                sid_from_event = data.get("streamSid")
                                if (
                                    sid_from_event
                                    and call_sid
                                    and call_sid not in self.call_to_stream_sid
                                ):
                                    self.call_to_stream_sid[call_sid] = sid_from_event

                                if event == "stop":
                                    logger.info(
                                        f"Stream stop event received for call: {call_sid}"
                                    )

                                    # Mark Deepgram call as closed to prevent reconnection
                                    self.deepgram_service.call_closed[call_sid] = True
                                    # Close Deepgram connection gracefully
                                    try:
                                        await self.deepgram_service.close_connection(call_sid)
                                    except Exception:
                                        pass
                                    logger.info(
                                        f"✅ Deepgram connection closed due to stream stop for call: {call_sid}"
                                    )

                                    # Keep WebSocket open but break the loop to clean up
                                    break

                                elif event == "mark":
                                    mark_data = data.get("mark", {})
                                    mark_name = mark_data.get("name", "unknown")

                                    logger.info(
                                        f"📢 Mark event received for call {call_sid}: {mark_name}"
                                    )

                                    # Different handling based on the mark name
                                    if (
                                        "speech-start" in mark_name
                                        or "start-speech" in mark_name
                                    ):
                                        logger.info(
                                            f"🎤 Speech started for call {call_sid}"
                                        )
                                        # Ensure Deepgram connection is active when speech starts
                                        await self.deepgram_service.setup_connection(
                                            call_sid
                                        )

                                elif event == "media":
                                    # Parse Twilio media event payload
                                    payload_b64 = data.get("media", {}).get("payload")
                                    if payload_b64:
                                        pcm_ulaw = base64.b64decode(payload_b64)
                                        # Feed frame into interruption detector before anything else
                                        interruption_manager.process_frame(
                                            call_sid, pcm_ulaw
                                        )
                                        count = (
                                            self.media_frame_counts.get(session_id, 0)
                                            + 1
                                        )
                                        self.media_frame_counts[session_id] = count
                                        await cancel_keepalive_if_needed(
                                            session_id, self.keepalive_tasks
                                        )
                                        if pcm_ulaw:
                                            if self.deepgram_service.connection_ready.get(
                                                call_sid
                                            ):
                                                await self._flush_media_buffer(call_sid)
                                                success = await self.deepgram_service.send_audio(
                                                    call_sid, pcm_ulaw
                                                )
                                                if not success:
                                                    logger.warning(
                                                        f"Failed to send audio to Deepgram for call {call_sid}"
                                                    )
                                            else:
                                                buf = self.media_buffer.get(call_sid)
                                                if buf is not None:
                                                    buf.append(pcm_ulaw)
                                                    logger.debug(
                                                        f"🔍 Deepgram not ready, buffering media frame {count} for call: {call_sid}"
                                                    )
                                                else:
                                                    logger.debug(
                                                        f"🔍 Media buffer not found (already flushed), dropping frame {count} for call: {call_sid}"
                                                    )
                                    else:
                                        logger.debug(
                                            f"📦 No media payload for call {call_sid}"
                                        )

                                # Other events are handled the same as in the initial loop
                                else:
                                    logger.debug(
                                        f"Event {event} in main loop for call {call_sid}"
                                    )

                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Received non-JSON text message: {text_data}"
                                )

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
            # This will catch exceptions during connection acceptance and main loop
            logger.error(
                f"❌ Failed to establish WebSocket connection or error in main loop: {str(e)}"
            )
            logger.error(traceback.format_exc())
            # Add connection failure to tracking
            pending_connection_attempt["status"] = "failed"
            pending_connection_attempt["error"] = str(e)
            pending_connection_attempt["traceback"] = traceback.format_exc()

        finally:
            # Clean up resources - no MCP session termination needed with single-server architecture
            # The MCP server handles session cleanup internally via the keep-alive and connection management

            # Clean up
            if "ping_task" in locals():
                logger.info(
                    f"🔍 Cancelling ping task for call: {call_sid if call_sid else 'unknown'}"
                )
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

            # Clean up keep-alive task if exists
            if session_id and session_id in self.keepalive_tasks:
                logger.info(
                    f"🔍 Cancelling silence keep-alive task for session: {session_id}"
                )
                self.keepalive_tasks[session_id].cancel()
                try:
                    await self.keepalive_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self.keepalive_tasks[session_id]

            # Mark the call as closed to suppress reconnection logic for Deepgram
            if call_sid:
                logger.debug(
                    f"🔍 Marking Deepgram call_closed for call: {call_sid} in finally cleanup"
                )
                self.deepgram_service.call_closed[call_sid] = True
                # Cleanup interruption detector resources
                interruption_manager.cleanup(call_sid)
                self.barge_in_buffers.pop(call_sid, None)
                self.barge_in_registered.discard(call_sid)
                self.skip_queues.pop(call_sid, None)
                self.skip_registered.discard(call_sid)
                self.transcript_locks.pop(call_sid, None)

            # Close Deepgram connection
            if call_sid:
                logger.info(f"🔍 Closing Deepgram connection for call: {call_sid}")
                try:
                    # Mark as gracefully closed to prevent reconnection races
                    self.deepgram_service.call_closed[call_sid] = True
                    await self.deepgram_service.close_connection(call_sid)
                except Exception:
                    pass

            # Remove from active connections
            if session_id and session_id in self.active_connections:
                logger.info(
                    f"🔍 Removing session from active connections: {session_id}"
                )
                del self.active_connections[session_id]
            elif placeholder_id and placeholder_id in self.active_connections:
                logger.info(
                    f"🔍 Removing placeholder session from active connections: {placeholder_id}"
                )
                del self.active_connections[placeholder_id]

            # ---- Silence watchdog cleanup ----
            if call_sid in self.silence_tasks:
                logger.info(f"🔍 Cancelling silence watchdog for call: {call_sid}")
                self.silence_tasks[call_sid].cancel()
                try:
                    await self.silence_tasks[call_sid]
                except asyncio.CancelledError:
                    pass
                del self.silence_tasks[call_sid]
                self.last_user_input.pop(call_sid, None)
                self.silence_prompted.pop(call_sid, None)

            # ---- Call length timer cleanup ----
            if call_sid in self.call_length_tasks:
                logger.info(f"🔍 Cancelling call length timer for call: {call_sid}")
                self.call_length_tasks[call_sid].cancel()
                try:
                    await self.call_length_tasks[call_sid]
                except asyncio.CancelledError:
                    pass
                del self.call_length_tasks[call_sid]

            logger.info(
                f"WebSocket connection closed and cleaned up for call: {call_sid if call_sid else 'unknown'}"
            )

    async def process_transcript(self, transcript: str, call_sid: str, is_final: bool):
        """
        Process transcripts with OpenAI agent and generate responses

        Args:
            transcript: The text transcript from Deepgram
            call_sid: The Twilio call SID
            is_final: Whether this is a final transcript or interim
        """
        logger.debug(
            f"🔍 process_transcript invoked for call {call_sid}, is_final={is_final}, transcript_len={len(transcript)}"
        )

        lock = self.transcript_locks.setdefault(call_sid, asyncio.Lock())
        async with lock:
            try:
                # Ignore non-final transcripts
                if not is_final:
                    return

                # Record time of final user utterance to feed silence watchdog
                self.last_user_input[call_sid] = time.time()
                self.silence_prompted[call_sid] = False

                # If user is barging in, buffer transcript for later processing
                if interruption_manager.awaiting_user_end.get(call_sid):
                    self.barge_in_buffers.setdefault(call_sid, []).append(transcript)
                    if call_sid not in self.barge_in_registered:

                        async def cb():
                            await self._process_barge_in_buffer(call_sid)

                        interruption_manager.register_resume_callback(call_sid, cb)
                        self.barge_in_registered.add(call_sid)
                    # Clear barge-in immediately now that we have the user's utterance
                    await interruption_manager.clear_barge_in_now(call_sid)
                    return
                if not interruption_manager.can_agent_speak(call_sid):
                    logger.info(
                        f"Queuing transcript for call {call_sid} due to pending barge-in or speaking flag"
                    )
                    self.skip_queues.setdefault(call_sid, []).append(transcript)
                    if call_sid not in self.skip_registered:
                        self.skip_registered.add(call_sid)
                        asyncio.create_task(self._process_skip_queue(call_sid))
                    return
                # Process with OpenAI agent
                history_key = f"hist:{call_sid}"
                try:
                    conversation_history = await get_json(history_key) or []
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to load conversation history for call {call_sid}: {e}"
                    )
                    conversation_history = []  # Fallback to empty history

                # Identify session_id for MCP session header
                matching_sessions = [
                    sid
                    for sid in self.active_connections.keys()
                    if sid.startswith(f"{call_sid}_")
                ]
                if not matching_sessions:
                    logger.error(
                        f"❌ No active WebSocket connection found for call {call_sid}"
                    )
                    return
                session_id = matching_sessions[0]

                # Process with agent via streaming (single LLM call) and capture cancellation handle
                history_future, delta_generator, llm_handle = await stream_agent_deltas(
                    transcript, conversation_history, session_id
                )
                # Register LLM streaming handle so it can be cancelled on barge-in
                interruption_manager.register_llm_handle(call_sid, llm_handle)

                logger.info(
                    f"⏩ PIPELINE: Streaming agent response for call {call_sid}"
                )

                # Retrieve websocket for this session
                websocket = self.active_connections[session_id]

                # Cancel silence keep-alive when sending an agent response
                if session_id in self.keepalive_tasks:
                    logger.info(
                        f"Cancelling silence keep-alive as agent is responding for session: {session_id}"
                    )
                    self.keepalive_tasks[session_id].cancel()
                    try:
                        await self.keepalive_tasks[session_id]
                    except asyncio.CancelledError:
                        pass
                    del self.keepalive_tasks[session_id]

                # Get stream SID for this call
                stream_sid = self.call_to_stream_sid.get(call_sid)
                if not stream_sid:
                    logger.error(
                        f"❌ No Stream SID found for call {call_sid}. Cannot send response."
                    )
                    return

                # Stream the response with low-latency TTS – when this
                # coroutine returns the delta generator has completed, so the
                # history_future is now resolved.
                await tts_service.stream_response_to_user(
                    call_sid, delta_generator, websocket, stream_sid
                )

                # Persist updated history once available (should be resolved now)
                try:
                    conversation_history = await history_future
                    await set_json(history_key, conversation_history)
                except CacheWriteError as e:
                    logger.warning(
                        f"⚠️ Failed to persist conversation history for call {call_sid}: {e}"
                    )
                    # Continue without caching - the conversation can still work
                except Exception as e:
                    logger.error(
                        f"❌ Unexpected error persisting history for call {call_sid}: {e}"
                    )

                # Start silence detection for potential keep-alive messages
                try:
                    self.keepalive_tasks[session_id] = asyncio.create_task(
                        send_silence_keepalive(
                            websocket, stream_sid, session_id, tts_service
                        )
                    )
                    logger.info(f"Started silence keep-alive for session: {session_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to start silence keep-alive for session {session_id}: {e}"
                    )

            except Exception as e:
                logger.error(
                    f"❌ Error processing transcript for call {call_sid}: {str(e)}"
                )
                logger.error(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Silence monitoring helpers
    # ---------------------------------------------------------------------

    async def _monitor_user_silence(self, call_sid: str, websocket: WebSocket, stream_sid: str):
        """Watch for prolonged caller silence and act per business rules.

        1. If 30 s elapse without caller input, prompt with TTS("Hello? …").
        2. If another 15 s elapse (45 s total) without input, terminate call.
        """

        try:
            while True:
                await asyncio.sleep(1)
                last_ts = self.last_user_input.get(call_sid)
                if last_ts is None:
                    # Call likely cleaned up
                    return

                elapsed = time.time() - last_ts

                # First threshold – send prompt
                if not self.silence_prompted.get(call_sid, False) and elapsed >= 30:
                    logger.info(
                        f"🔔 30 s silence detected on call {call_sid}; prompting caller"
                    )
                    # HIPAA Compliance: §164.312(b) – Log prompt for audit trail
                    logger.info(
                        f"[HIPAA-AUDIT] silence_prompt_issued call_sid={call_sid}"
                    )
                    try:
                        await tts_service.prompt_user_still_there(
                            call_sid, websocket, stream_sid
                        )
                    except Exception as e:
                        logger.error(
                            f"❌ Error sending silence prompt for call {call_sid}: {e}"
                        )
                    # Reset timer and set flag
                    self.last_user_input[call_sid] = time.time()
                    self.silence_prompted[call_sid] = True

                # Second threshold – hang up (15 s after prompt)
                elif self.silence_prompted.get(call_sid, False) and elapsed >= 15:
                    logger.warning(
                        f"📞 Ending call {call_sid} due to 45 s of caller silence"
                    )
                    logger.debug(
                        f"call ended due to user silence timeout (conversation watchdog)"
                    )

                    # Close Deepgram and WebSocket gracefully
                    try:
                        # Mark as gracefully closed to prevent reconnection races
                        self.deepgram_service.call_closed[call_sid] = True
                        await self.deepgram_service.close_connection(call_sid)
                    except Exception:
                        pass

                    if websocket.client_state != WebSocketState.DISCONNECTED:
                        await websocket.close(code=1000, reason="User silence timeout")
                    return
        except asyncio.CancelledError:
            # Task cancelled on normal cleanup
            return


# Create singleton instance
twilio_service = TwilioService()
