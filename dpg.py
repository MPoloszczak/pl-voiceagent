import os
import asyncio
import json
import traceback
import time
from typing import Dict, Any, Optional, Callable

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents
)

from utils import logger

class DeepgramService:
    """Service for handling Deepgram speech-to-text transcription"""
    
    def __init__(self):
        """Initialize the Deepgram service with API key from environment"""
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            logger.error("❌ DEEPGRAM_API_KEY is missing. Deepgram transcription will not work.")
            return
            
        # Initialize client with appropriate options
        self.client = DeepgramClient(
            self.api_key,
            DeepgramClientOptions(
                options={
                    "keepalive": "true",  # Enable keep-alive
                    "keepalive_timeout": "30",  # 30 seconds keep-alive timeout
                    "termination_exception_connect": "true",  # Raise exceptions on connection failures
                }
            )
        )
        
        # Store active Deepgram connections
        self.active_connections = {}
        # Flags to track readiness and closed calls
        self.connection_ready: Dict[str, bool] = {}
        self.call_closed: Dict[str, bool] = {}
        # Dictionary to track keep-alive tasks for each call
        self.keepalive_tasks = {}
        # Store the transcript callback function
        self.transcript_callback = None
        
        # Capture the main asyncio loop for thread-safe scheduling
        try:
            self.main_loop = asyncio.get_event_loop()
        except Exception:
            self.main_loop = None
            logger.warning("⚠️ Could not capture main event loop for DeepgramService")
        
        logger.info("Deepgram service initialized")
        
    def set_transcript_callback(self, callback: Callable):
        """
        Set the callback function to handle transcripts
        
        Args:
            callback: Function that accepts (transcript, call_sid)
        """
        self.transcript_callback = callback

    def set_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Explicitly set the asyncio event loop for scheduling transcript callbacks.
        """
        self.main_loop = loop
        logger.info(f"🔄 DeepgramService main_loop set to {loop}")
        
    def _refresh_main_loop(self):
        """
        Refresh the stored main_loop to the currently running asyncio loop if available.
        """
        try:
            running = asyncio.get_running_loop()
            if running is not self.main_loop:
                self.main_loop = running
                logger.info("🔄 DeepgramService main_loop refreshed to active loop")
        except RuntimeError:
            # No running loop yet, ignore
            pass
        
    async def setup_connection(self, call_sid: str) -> bool:
        """
        Set up a persistent connection to Deepgram for a specific call.
        
        Args:
            call_sid: The Twilio call SID to associate with this connection
        
        Returns:
            bool: Success status of the connection setup
        """
        # Ensure we have the correct running loop before scheduling callbacks
        self._refresh_main_loop()
        if not self.api_key:
            logger.error("Cannot set up Deepgram connection: DEEPGRAM_API_KEY is missing")
            return False
            
        logger.info(f"🔍 Setting up Deepgram connection for call: {call_sid}")
        
        # Cancel any existing keep-alive task for this call
        if call_sid in self.keepalive_tasks:
            try:
                logger.info(f"Cancelling existing keep-alive task for call: {call_sid}")
                keep_alive_task = self.keepalive_tasks[call_sid]
                keep_alive_task.cancel()
                try:
                    await keep_alive_task
                except asyncio.CancelledError:
                    pass
                del self.keepalive_tasks[call_sid]
            except Exception as e:
                logger.error(f"Error cancelling keep-alive task for call {call_sid}: {str(e)}")
        
        # Check if there's already an active connection for this call
        if call_sid in self.active_connections:
            try:
                # Test if the connection is still alive
                connection = self.active_connections[call_sid]
                if connection.is_alive:
                    # Additional verification to check if WebSocket is truly connected
                    if connection.is_connected():
                        logger.info(f"✅ Using existing Deepgram connection for call: {call_sid}")
                        
                        # Start a new keep-alive task if it doesn't exist
                        if call_sid not in self.keepalive_tasks:
                            keep_alive_task = asyncio.create_task(self.send_periodic_keepalives(call_sid))
                            self.keepalive_tasks[call_sid] = keep_alive_task
                            logger.info(f"Started new keep-alive task for existing connection for call: {call_sid}")
                        
                        return True
                    else:
                        logger.warning(f"⚠️ Deepgram connection reports not connected despite open event for call: {call_sid}")
                        # Fall through to recreate the connection
                else:
                    logger.info(f"🔄 Existing Deepgram connection is dead, creating new one for call: {call_sid}")
                    # Clean up the dead connection
                    await self.close_connection(call_sid)
            except Exception as e:
                logger.error(f"❌ Error checking existing Deepgram connection for call {call_sid}: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue to create a new connection
        
        # Fixed max retry count and backoff multiplier
        max_retries = 3
        retry_count = 0
        backoff_seconds = 1.0
        
        while retry_count <= max_retries:
            try:
                # Configure Deepgram transcription options with optimal settings for voice calls
                options = LiveOptions(
                    model="nova-3",
                    language="en-US",
                    smart_format=True,
                    interim_results=True,
                    punctuate=True,
                    endpointing=200,  # 200ms of silence to detect end of speech
                    utterance_end_ms=1000,  # 1 second of silence to mark utterance end
                    encoding="mulaw",
                    channels=1,
                    sample_rate=8000,  # Twilio uses 8kHz audio
                    vad_events=True,  # Enable Voice Activity Detection events
                )
                
                logger.debug(f"Creating Deepgram live connection for call: {call_sid} (attempt {retry_count + 1}/{max_retries + 1})")
                connection = self.client.listen.live.v("1")
                
                # Store connection early so Open events map correctly
                self.active_connections[call_sid] = connection
                
                # Dump the options for debugging
                logger.info(f"🔍 Deepgram options for call {call_sid}: {options.to_dict()}")
                
                # Create event handler bindings with CallSid context
                self._bind_event_handlers(connection, call_sid)
                
                logger.debug(f"Starting Deepgram connection for call: {call_sid}")
                
                # The start method returns a boolean directly, not a coroutine
                start_success = connection.start(options)
                
                # Remove stale connection on start failure
                if not start_success:
                    if call_sid in self.active_connections:
                        del self.active_connections[call_sid]
                    self.connection_ready.pop(call_sid, None)
                
                if not start_success:
                    logger.error(f"❌ Failed to start Deepgram connection for call: {call_sid} on attempt {retry_count + 1}")
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = backoff_seconds * (2 ** retry_count)  # Exponential backoff
                        logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("Failed to start Deepgram connection after max retries")
                
                # Verify connection state after successful start
                if not connection.is_connected():
                    logger.error(f"❌ Deepgram WebSocket reports not connected despite successful start for call: {call_sid}")
                    # Remove stale connection on WebSocket connect failure
                    if call_sid in self.active_connections:
                        del self.active_connections[call_sid]
                        self.connection_ready.pop(call_sid, None)
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = backoff_seconds * (2 ** retry_count)
                        logger.info(f"WebSocket not connected, retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("Deepgram WebSocket never connected properly after successful start")
                
                # Log successful connection
                logger.info(f"✅ Deepgram WebSocket successfully connected for call: {call_sid}")
                
                # Send silent audio packet to prime the connection
                silent_audio = b'\x00' * 320  # 20ms of silence at 8kHz
                success = connection.send(silent_audio)
                logger.info(f"✅ Sent silent audio packet to prime Deepgram connection for call: {call_sid}: {success}")
                
                # Wait longer to ensure connection is stable before continuing
                await asyncio.sleep(1.0)
                
                # Verify the connection is fully ready with a test message
                logger.info(f"Performing connection readiness verification for call: {call_sid}")
                connection_ready = await self._validate_connection(connection, call_sid)
                if not connection_ready:
                    logger.error(f"❌ Connection readiness verification failed for call {call_sid}")
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = backoff_seconds * (2 ** retry_count)
                        logger.info(f"Connection not fully ready, retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("Deepgram connection failed readiness check after max retries")
                
                logger.info(f"✅ Connection readiness verification successful for call: {call_sid}")
                
                # Start a keep-alive task for this connection
                keep_alive_task = asyncio.create_task(self.send_periodic_keepalives(call_sid))
                self.keepalive_tasks[call_sid] = keep_alive_task
                
                logger.info(f"✅ Deepgram connection successfully initialized for call: {call_sid}")
                logger.info(f"✅ Deepgram keep-alive task started for call: {call_sid}")
                
                # Store in active connections
                self.active_connections[call_sid] = connection
                return True
                
            except Exception as e:
                logger.error(f"❌ Deepgram connection error for call {call_sid} on attempt {retry_count + 1}: {str(e)}")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                logger.error(traceback.format_exc())
                
                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = backoff_seconds * (2 ** retry_count)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to establish Deepgram connection after {max_retries + 1} attempts")
                    return False
        
        return False
        
    async def close_connection(self, call_sid: str) -> None:
        """
        Close the Deepgram connection for a specific call.
        
        Args:
            call_sid: The Twilio call SID associated with the connection to close
        """
        # Cancel keep-alive task if it exists
        if call_sid in self.keepalive_tasks:
            try:
                logger.info(f"Cancelling Deepgram keep-alive task for call: {call_sid}")
                keep_alive_task = self.keepalive_tasks[call_sid]
                keep_alive_task.cancel()
                try:
                    await keep_alive_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
                del self.keepalive_tasks[call_sid]
                logger.info(f"✅ Deepgram keep-alive task cancelled for call: {call_sid}")
            except Exception as e:
                logger.error(f"❌ Error cancelling Deepgram keep-alive task for call {call_sid}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Close Deepgram connection if it exists
        if call_sid in self.active_connections:
            connection = self.active_connections[call_sid]
            try:
                logger.debug(f"Closing Deepgram connection for call: {call_sid}")
                
                # The finish method returns a boolean directly, not a coroutine
                finish_success = connection.finish()
                if finish_success:
                    logger.info(f"✅ Deepgram connection successfully closed for call: {call_sid}")
                else:
                    logger.warning(f"⚠️ Deepgram connection may not have closed properly for call: {call_sid}")
                
                # Remove from active connections regardless of finish success
                del self.active_connections[call_sid]
                # Clean up readiness and closed flags
                self.connection_ready.pop(call_sid, None)
                self.call_closed.pop(call_sid, None)
            except Exception as e:
                logger.error(f"❌ Error closing Deepgram connection for call {call_sid}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Still remove from active connections on error to avoid stale references
                try:
                    del self.active_connections[call_sid]
                except KeyError:
                    pass
        else:
            logger.debug(f"No Deepgram connection to close for call: {call_sid}")

    async def send_audio(self, call_sid: str, audio_data: bytes) -> bool:
        """
        Send audio data to Deepgram for transcription
        
        Args:
            call_sid: The Twilio call SID
            audio_data: Binary audio data
            
        Returns:
            bool: Success status of the operation
        """
        # Do not send audio until connection is marked ready
        if not self.connection_ready.get(call_sid):
            logger.warning(f"⚠️ Deepgram connection not ready for call: {call_sid}")
            return False
        if call_sid not in self.active_connections:
            logger.warning(f"No Deepgram connection available for call: {call_sid}")
            logger.info(f"Attempting to establish Deepgram connection for call {call_sid}")
            success = await self.setup_connection(call_sid)
            if not success:
                logger.error(f"Failed to establish Deepgram connection for call {call_sid}")
                return False
        
        connection = self.active_connections[call_sid]
        
        # Audio sending retry parameters
        max_send_retries = 3
        retry_count = 0
        backoff_seconds = 0.5
        
        while retry_count <= max_send_retries:
            try:
                # Add enhanced logging to track audio processing
                if retry_count == 0:
                    # initial send logging removed to reduce log noise
                    pass
                else:
                    logger.info(f"⏩ PIPELINE: Retry {retry_count}/{max_send_retries} - Sending {len(audio_data)} bytes to Deepgram for call {call_sid}")
                
                # Check connection status before sending
                if not connection.is_connected():
                    logger.error(f"❌ Deepgram connection not connected before sending audio for call {call_sid}")
                    retry_count += 1
                    if retry_count <= max_send_retries:
                        wait_time = backoff_seconds * (2 ** retry_count)
                        logger.info(f"Waiting {wait_time:.2f}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        break
                
                # Implement empty packet detection
                if not audio_data or len(audio_data) == 0:
                    logger.warning(f"⚠️ Empty audio packet detected, not sending to Deepgram for call {call_sid}")
                    return False
                
                # Send the audio and get result
                success = connection.send(audio_data)
                if not success:
                    logger.error(f"❌ Error sending audio to Deepgram for call {call_sid} - send() returned False")
                    retry_count += 1
                    if retry_count <= max_send_retries:
                        wait_time = backoff_seconds * (2 ** retry_count)
                        logger.info(f"Waiting {wait_time:.2f}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        break
                else:
                    logger.debug(f"Forwarded {len(audio_data)} bytes of audio data to Deepgram for call: {call_sid}")
                    return True
            except Exception as e:
                logger.error(f"❌ Error sending audio to Deepgram: {str(e)}")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                logger.error(traceback.format_exc())
                
                retry_count += 1
                if retry_count <= max_send_retries:
                    wait_time = backoff_seconds * (2 ** retry_count)
                    logger.info(f"Waiting {wait_time:.2f}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    # We've exhausted our retries
                    logger.error(f"❌ Failed to send audio to Deepgram after {max_send_retries + 1} attempts")
                    
                    # Try to reestablish the Deepgram connection for future audio
                    logger.info(f"Attempting to recreate Deepgram connection for call {call_sid} after send failures")
                    await self.close_connection(call_sid)
                    try:
                        success = await self.setup_connection(call_sid)
                        if success:
                            logger.info(f"✅ Successfully recreated Deepgram connection for call {call_sid} after failures")
                        else:
                            logger.error(f"❌ Failed to recreate Deepgram connection for call {call_sid}")
                    except Exception as setup_error:
                        logger.error(f"❌ Failed to recreate Deepgram connection: {str(setup_error)}")
                    return False
        
        return False

    async def send_periodic_keepalives(self, call_sid: str, interval: int = 2):
        """
        Send periodic keep-alive messages to Deepgram to maintain the connection,
        especially during periods of silence.
        
        Args:
            call_sid: The Twilio call SID
            interval: Time in seconds between keep-alive messages (default: 2)
        """
        try:
            logger.info(f"Starting Deepgram keep-alive task for call: {call_sid}")
            
            # Track consecutive failures
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            # We'll keep sending keep-alives as long as the connection and task exist
            while call_sid in self.active_connections and call_sid in self.keepalive_tasks:
                # Sleep first to avoid immediate keep-alive after connection is established
                await asyncio.sleep(interval)
                
                # Check if the connection still exists after sleeping
                if call_sid not in self.active_connections:
                    logger.info(f"Deepgram connection gone, stopping keep-alive for call: {call_sid}")
                    break
                    
                connection = self.active_connections[call_sid]
                
                # More reliable connection state check
                # Check both SDK reported state and socket connection
                connection_alive = hasattr(connection, 'is_alive') and connection.is_alive
                socket_connected = connection.is_connected()
                
                logger.debug(f"🔍 Deepgram connection state check - is_alive: {connection_alive}, is_connected: {socket_connected}")
                
                if not connection_alive:
                    # Demote stale is_alive=False when socket still connected
                    if socket_connected:
                        logger.debug(f"⚠️ Deepgram connection reports not alive but socket still connected for call: {call_sid}")
                    else:
                        logger.warning(f"⚠️ Deepgram connection reports not alive, stopping keep-alive for call: {call_sid}")
                        break
                
                # Additional check for WebSocket state if is_alive is True
                if connection_alive and not socket_connected:
                    logger.error(f"❌ Deepgram WebSocket not connected despite is_alive=True for call: {call_sid}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"❌ Too many consecutive WebSocket state failures, recreating connection for call: {call_sid}")
                        # Trigger connection recreation
                        asyncio.create_task(self._recreate_connection(call_sid))
                        break
                    
                    # Skip this iteration but continue the loop
                    continue
                
                # Reset failures if both checks pass
                if connection_alive and socket_connected:
                    consecutive_failures = 0
                    
                try:
                    # Send keep-alive to Deepgram - try sending a keep-alive message
                    # First try an empty binary message
                    logger.debug(f"Sending keep-alive for call: {call_sid}")
                    json_keepalive = json.dumps({"type": "KeepAlive"})
                    success = connection.send(json_keepalive.encode('utf-8'))
                    
                    if success:
                        logger.debug(f"✅ Sent Deepgram keep-alive message for call: {call_sid}")
                        # Reset consecutive failures on success
                        consecutive_failures = 0
                    else:
                        logger.warning(f"⚠️ Deepgram keep-alive send failed for call: {call_sid}")
                        consecutive_failures += 1
                        
                        # If enough failures, recreate the connection
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"❌ Too many consecutive keep-alive failures, recreating connection for call: {call_sid}")
                            # Trigger connection recreation
                            asyncio.create_task(self._recreate_connection(call_sid))
                            break
                except Exception as e:
                    logger.error(f"❌ Failed to send Deepgram keep-alive for call {call_sid}: {str(e)}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"❌ Too many consecutive keep-alive exceptions, recreating connection for call: {call_sid}")
                        # Trigger connection recreation
                        asyncio.create_task(self._recreate_connection(call_sid))
                        break
            
            logger.info(f"Deepgram keep-alive task ended for call: {call_sid}")
        except asyncio.CancelledError:
            logger.info(f"Deepgram keep-alive task cancelled for call: {call_sid}")
        except Exception as e:
            logger.error(f"❌ Error in Deepgram keep-alive task for call {call_sid}: {str(e)}")
            logger.error(traceback.format_exc())
            
    async def _recreate_connection(self, call_sid: str):
        """
        Recreate a Deepgram connection when keep-alive detects issues.
        
        Args:
            call_sid: The Twilio call SID
        """
        try:
            logger.info(f"🔄 Recreating Deepgram connection for call: {call_sid}")
            
            # Close the existing connection first
            await self.close_connection(call_sid)
            
            # Short delay to ensure clean closure
            await asyncio.sleep(1)
            
            # Create a new connection
            await self.setup_connection(call_sid)
            
            logger.info(f"✅ Successfully recreated Deepgram connection for call: {call_sid}")
        except Exception as e:
            logger.error(f"❌ Failed to recreate Deepgram connection for call {call_sid}: {str(e)}")
            logger.error(traceback.format_exc())
            
    async def _validate_connection(self, connection, call_sid):
        """
        Validate that a Deepgram connection is fully ready by sending a test message.
        
        Args:
            connection: The Deepgram connection to validate
            call_sid: The Twilio call SID
            
        Returns:
            bool: True if connection is fully ready, False otherwise
        """
        try:
            # Send test message and verify connection is fully established
            test_message = json.dumps({"type": "ConnectionTest"}).encode('utf-8')
            success = connection.send(test_message)
            
            if not success:
                logger.error(f"❌ Connection validation failed for call {call_sid}")
                return False
                
            # Wait briefly to ensure message is processed
            await asyncio.sleep(0.5)
            return connection.is_connected()
        except Exception as e:
            logger.error(f"❌ Error during connection validation test: {str(e)}")
            return False
            
    def _bind_event_handlers(self, connection, call_sid):
        """
        Bind event handlers to the Deepgram connection with CallSid context
        
        Args:
            connection: The Deepgram connection
            call_sid: The Twilio call SID to associate with events
        """
        def bind_handler(handler):
            def wrapper(*args, **kw):
                # Extract actual event payload from kwargs first (preferred) or positional args
                if kw:
                    data = next(iter(kw.values()))
                elif len(args) > 1:
                    data = args[1]
                else:
                    data = args[0]
                
                # Add debug logging for Deepgram websocket events
                event_name = handler.__name__.replace('_on_', '')
                logger.debug(f"🔌 Deepgram {event_name} event for call {call_sid}")
                
                # Enhanced debugging: Log raw data for error investigation
                if event_name == "error":
                    logger.error(f"🔍 Deepgram error event raw data: {data}")
                elif event_name in ["open", "close"]:
                    logger.info(f"🔍 Deepgram {event_name} event raw data: {data}")
                else:
                    logger.debug(f"🔍 Deepgram {event_name} event received for call {call_sid}")
                
                # Use only the CallSid keyword argument and pass data as positional
                # This avoids potential parameter naming conflicts in the handler functions
                return handler(data, call_sid=call_sid)
            return wrapper
        
        # Register all event handlers using the wrapper
        connection.on(LiveTranscriptionEvents.Open, bind_handler(self._on_open))
        connection.on(LiveTranscriptionEvents.Close, bind_handler(self._on_close))
        connection.on(LiveTranscriptionEvents.Error, bind_handler(self._on_error))
        connection.on(LiveTranscriptionEvents.SpeechStarted, bind_handler(self._on_speech_started))
        connection.on(LiveTranscriptionEvents.UtteranceEnd, bind_handler(self._on_utterance_end))
        connection.on(LiveTranscriptionEvents.Metadata, bind_handler(self._on_metadata))
        connection.on(LiveTranscriptionEvents.Unhandled, bind_handler(self._on_unhandled))
        
        # Register transcript handler
        connection.on(LiveTranscriptionEvents.Transcript, bind_handler(self._on_transcript))
    
    # Event handler methods
    def _on_open(self, open_event, call_sid):
        """Handler for Deepgram WebSocket open event."""
        logger.info(f"✅ Deepgram WebSocket connection opened for call: {call_sid}")
        
        # Extract and log connection details for debugging
        connection_id = getattr(open_event, 'connection_id', 'No connection ID available')
        created = getattr(open_event, 'created', 'No creation timestamp available')
        
        logger.info(f"🔍 Deepgram connection ID: {connection_id}")
        logger.info(f"🔍 Deepgram connection created at: {created}")
        logger.info(f"🔍 Deepgram connection details: {open_event}")
        
        # Verify the connection is in active_connections
        if call_sid not in self.active_connections:
            logger.warning(f"⚠️ Deepgram connection opened but not found in active_connections for call: {call_sid}")
        else:
            connection = self.active_connections[call_sid]
            # Verify the connection is still valid
            try:
                if connection.is_connected():
                    logger.info(f"✅ Deepgram connection is valid and connected for call: {call_sid}")
                    # Mark the connection ready now that the socket is open
                    self.connection_ready[call_sid] = True
                    # Flush any buffered media frames now that Deepgram is ready
                    # Delay import to avoid circular dependency
                    import twilio
                    asyncio.create_task(twilio.twilio_service._flush_media_buffer(call_sid))
                else:
                    logger.warning(f"⚠️ Deepgram connection reports not connected despite open event for call: {call_sid}")
            except Exception as e:
                logger.error(f"❌ Error checking Deepgram connection status: {str(e)}")

    def _on_close(self, close_event, call_sid):
        """Handler for Deepgram WebSocket close event."""
        # Extract any available close information
        close_code = getattr(close_event, 'code', 'No close code available')
        close_reason = getattr(close_event, 'reason', 'No close reason available')
        
        logger.info(f"🔌 Deepgram WebSocket closed for call: {call_sid}")
        logger.info(f"🔌 Close code: {close_code}")
        logger.info(f"🔌 Close reason: {close_reason}")
        logger.info(f"🔌 Close event full details: {close_event}")
        
        # If this was an unexpected close and call not deliberately closed, try to reconnect
        if close_code not in (1000, 1001) and not self.call_closed.get(call_sid):
            logger.info(f"Unexpected close detected (code {close_code}), scheduling reconnection for call: {call_sid}")
            asyncio.create_task(self._recreate_connection(call_sid))
        else:
            logger.debug(f"Close event for call {call_sid} detected; no reconnection scheduled.")

    def _on_error(self, error_event, call_sid):
        """Handler for Deepgram error event."""
        # Extract as much diagnostic information as possible
        error_type = getattr(error_event, 'type', 'Unknown error type')
        error_message = getattr(error_event, 'message', 'No error message available')
        error_code = getattr(error_event, 'code', 'No error code available')
        
        logger.error(f"❌ Deepgram error for call: {call_sid}")
        logger.error(f"❌ Error type: {error_type}")
        logger.error(f"❌ Error message: {error_message}")
        logger.error(f"❌ Error code: {error_code}")
        logger.error(f"❌ Deepgram error event full details: {error_event}")
        
        # Downgrade 'no running event loop' to a warning and skip reconnect
        if "no running event loop" in str(error_message).lower():
            logger.warning(f"⚠️ Ignoring Deepgram error (no running event loop) for call: {call_sid}")
            return
        # Check if we need to reconnect
        if error_type in ("ConnectionClosed", "WebSocketException") or "connect" in str(error_message).lower():
            logger.info(f"Connection-related error detected, scheduling reconnection for call: {call_sid}")
            asyncio.create_task(self._recreate_connection(call_sid))

    def _on_speech_started(self, speech_event, call_sid):
        """Handler for Deepgram speech started event."""
        logger.info(f"🎤 Deepgram detected speech started for call: {call_sid}")
        logger.debug(f"🔍 Speech started details: {speech_event}")

    def _on_utterance_end(self, utterance_event, call_sid):
        """Handler for Deepgram utterance end event."""
        logger.info(f"🎤 Deepgram detected utterance end for call: {call_sid}")
        logger.debug(f"🔍 Utterance end details: {utterance_event}")

    def _on_metadata(self, metadata_event, call_sid):
        """Handler for Deepgram metadata event."""
        logger.info(f"ℹ️ Deepgram metadata received for call: {call_sid}")
        logger.debug(f"🔍 Metadata details: {metadata_event}")

    def _on_unhandled(self, unhandled_event, call_sid):
        """Handler for unhandled Deepgram events."""
        logger.warning(f"⚠️ Deepgram unhandled event for call: {call_sid}")
        logger.warning(f"🔍 Unhandled event details: {unhandled_event}")

    def _on_transcript(self, transcript_event, call_sid):
        """Handler for Deepgram transcript event."""
        # Process the transcript data
        transcript_found = False
        transcript = ""
        
        # First check if this is a valid transcript response
        if not hasattr(transcript_event, 'channel'):
            logger.warning(f"🎯 No channel found in transcript event for call {call_sid}")
            logger.warning(f"🎯 Transcript event type: {type(transcript_event).__name__}")
            logger.warning(f"🎯 Transcript event attributes: {dir(transcript_event)}")
            logger.warning(f"🎯 Raw transcript event: {transcript_event}")
            return
            
        if not hasattr(transcript_event.channel, 'alternatives'):
            logger.warning(f"🎯 No alternatives found in transcript channel for call {call_sid}")
            logger.warning(f"🎯 Channel attributes: {dir(transcript_event.channel)}")
            return
            
        if not transcript_event.channel.alternatives:
            logger.warning(f"🎯 Empty alternatives list in transcript for call {call_sid}")
            return
        
        # Extract the transcript from the result
        transcript = transcript_event.channel.alternatives[0].transcript
        transcript_found = True
        
        # Extract is_final status
        is_final = getattr(transcript_event, 'is_final', False)
        
        # Get confidence if available
        confidence = None
        if hasattr(transcript_event.channel.alternatives[0], 'confidence'):
            confidence = transcript_event.channel.alternatives[0].confidence
        
        if transcript.strip():
            if is_final:
                logger.info(f"🎯 FINAL TRANSCRIPT: {transcript}")
                if confidence is not None:
                    logger.info(f"🎯 Confidence: {confidence:.2f}")
            else:
                logger.info(f"🎯 PARTIAL TRANSCRIPT: {transcript}")
                if confidence is not None:
                    logger.debug(f"🎯 Confidence: {confidence:.2f}")
        else:
            logger.debug(f"🎯 Empty transcript received for call {call_sid} (is_final={is_final})")
        
        # Debug full transcript event at debug level
        logger.debug(f"🎯 TRANSCRIPT FULL EVENT: {transcript_event}")
        
        # Use create_task to handle async processing without waiting
        if transcript_found and is_final and transcript.strip() and self.transcript_callback:
            logger.info(f"🎯 Scheduling final transcript in loop {self.main_loop} for call: {call_sid}")
            if self.main_loop and self.main_loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.transcript_callback(transcript, call_sid, is_final),
                        self.main_loop
                    )
                except Exception as e:
                    logger.error(f"❌ Error scheduling transcript callback for call {call_sid}: {e}")
            else:
                logger.error("❌ DeepgramService main_loop not set or not running; cannot dispatch transcript")
        else:
            if not transcript_found:
                logger.warning(f"🎯 No transcript found to process for call {call_sid}")
            elif not is_final:
                logger.debug(f"🎯 Skipping non-final transcript for call {call_sid}")
            elif not transcript.strip():
                logger.debug(f"🎯 Skipping empty transcript for call {call_sid}")
            elif not self.transcript_callback:
                logger.warning(f"🎯 No transcript callback registered")

# Singleton instance - initially None
_deepgram_service_instance = None

def get_deepgram_service():
    """
    Factory function to get or create the DeepgramService singleton instance.
    
    This pattern prevents circular imports by deferring instantiation until explicitly requested.
    
    Returns:
        DeepgramService: The singleton instance of the DeepgramService
    """
    global _deepgram_service_instance
    if _deepgram_service_instance is None:
        _deepgram_service_instance = DeepgramService()
    return _deepgram_service_instance 