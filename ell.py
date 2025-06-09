import os
import json
import base64
import asyncio
import logging
from typing import Iterator, Optional
import threading, queue, contextlib
import time

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from starlette.websockets import WebSocketState, WebSocketDisconnect

from utils import logger
from vad_events import interruption_manager

class TTSService:
    """Service for text-to-speech conversion using ElevenLabs"""
    
    def __init__(self):
        """Initialize the TTS service with API key from environment"""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("❌ ELEVENLABS_API_KEY not found in environment variables")
            self.client: Optional[ElevenLabs] = None
        else:
            self.client = ElevenLabs(api_key=self.api_key)

    def generate_silence(self, duration_ms: int = 20) -> bytes:
        """
        Generate μ-law 0x7F silence for the given duration at 8 kHz.
        
        Args:
            duration_ms: Duration of silence in milliseconds
            
        Returns:
            bytes: μ-law encoded silence bytes
        """
        # μ-law silence using Twilio canonical digital 0xFF for μ-law
        return b'\xFF' * int(8000 * duration_ms / 1000)

    async def generate_welcome_message(self, call_sid, websocket, stream_sid):
        """
        Generate a welcome message using ElevenLabs API and send it to the user.
        
        Args:
            call_sid: The Twilio call SID
            websocket: The WebSocket connection
            stream_sid: The Twilio Stream SID
        """
        try:
            # mark agent as speaking so VAD is active during welcome
            interruption_manager.set_speaking(call_sid, True)
            logger.info(f"🔊 Starting welcome message generation for call {call_sid}")
            
            # Add more detailed WebSocket state check before proceeding
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.error(f"❌ Cannot send welcome message: WebSocket is disconnected for call {call_sid}")
                return
            
            # Log the exact Stream SID we're using to help debug
            logger.info(f"🔍 Using Stream SID: '{stream_sid}' for welcome message")
            
            # Initialize ElevenLabs client
            if not self.client:
                logger.error("❌ ELEVENLABS_API_KEY not found in environment variables")
                return

            eleven_labs_client = self.client
            
            # Welcome message content
            welcome_message = "Hello, how can I help you today?"
            
            logger.info(f"🔊 Generating audio for welcome message: '{welcome_message}'")
            
            # Generate audio without streaming - simple API call
            try:
                audio_generator = eleven_labs_client.text_to_speech.convert(
                    voice_id="ZF6FPAbjXT4488VcRRnw",  
                    text=welcome_message,
                    model_id="eleven_flash_v2",
                    output_format="ulaw_8000"  # Format compatible with Twilio
                )
                
                # Convert generator to bytes with timeout handling
                try:
                    # Set a timeout for collecting audio chunks
                    audio_chunks = list(audio_generator)
                    audio_data = b''.join(audio_chunks)
                    
                    # Strip WAV header if present on full audio
                    original = audio_data
                    audio_data = self._strip_wav_header(audio_data)
                    if audio_data != original:
                        logger.info("🗑️ Stripped WAV headers from ElevenLabs welcome response")
                    
                    logger.info(f"🔊 Welcome message audio generated successfully, size: {len(audio_data)} bytes")
                except Exception as audio_error:
                    logger.error(f"❌ Error collecting audio data: {str(audio_error)}")
                    return
            except Exception as tts_error:
                logger.error(f"❌ Error generating speech with ElevenLabs: {str(tts_error)}")
                return
            
            # Check WebSocket state before sending
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.error(f"❌ Cannot send welcome message: WebSocket disconnected during audio generation for call {call_sid}")
                return
                
            # Send a mark message to indicate the start of the welcome message
            logger.info(f"🔊 Sending start-welcome-message mark for stream '{stream_sid}'")
            mark_msg = json.dumps({
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {
                    "name": "start-welcome"
                }
            })
            # Log the exact message we're sending to Twilio
            logger.info(f"🔍 DEBUG: Sending mark message: {mark_msg}")
            await websocket.send_text(mark_msg)
            
            # Define maximum burst size (ulaw format, up to 400ms = 3200 bytes)
            max_chunk_size = 3200     # Twilio limit: under 4000 bytes per message
            
            # Send audio in smaller, regular chunks rather than all at once
            logger.info(f"🔊 Sending welcome audio in chunks for call {call_sid}")
            audio_buffer = audio_data
            chunk_count = 0
            
            # Process buffer in bursts
            while audio_buffer:
                # Take up to max_chunk_size bytes for this burst
                send_chunk = audio_buffer[:max_chunk_size]
                audio_buffer = audio_buffer[len(send_chunk):]
                chunk_count += 1
                
                # Encode the chunk
                audio_base64 = base64.b64encode(send_chunk).decode('utf-8')
                
                # Send a properly formatted media message
                media_msg = json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                })
                
                # Send the chunk
                logger.debug(f"🔍 DEBUG: Sending welcome audio chunk {chunk_count} with streamSid: {stream_sid}")
                await websocket.send_text(media_msg)
                # Sleep exactly for the duration of the burst: bytes/8000 = seconds of µ-law audio
                await asyncio.sleep(len(send_chunk) / 8000)
            
            logger.info(f"🔊 Finished sending welcome audio ({chunk_count} chunks) for call {call_sid}")
            
            # Send a mark message to indicate the end of the welcome message
            logger.info(f"🔊 Sending end-welcome-message mark for stream '{stream_sid}'")
            mark_msg = json.dumps({
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {
                    "name": "end-welcome"
                }
            })
            # Log the exact message we're sending to Twilio
            logger.info(f"🔍 DEBUG: Sending end mark message: {mark_msg}")
            await websocket.send_text(mark_msg)
            
            logger.info(f"✅ Welcome message sent successfully for call {call_sid} using stream '{stream_sid}'")
            
        except Exception as e:
            logger.error(f"❌ Error generating welcome message for call {call_sid}: {str(e)}")
            logger.error(f"Error details: {repr(e)}")
        finally:
            # clear speaking flag after welcome completes or errors
            interruption_manager.set_speaking(call_sid, False)

    def _strip_wav_header(self, data: bytes) -> bytes:
        """
        Remove RIFF/WAVE headers and return raw μ-law audio data. If no header, return data unchanged.
        """
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return data

        offset = 12
        length = len(data)
        while offset + 8 <= length:
            chunk_id = data[offset:offset+4]
            chunk_len = int.from_bytes(data[offset+4:offset+8], "little")
            offset += 8
            if chunk_id == b"data":
                return data[offset:offset+chunk_len]
            offset += chunk_len
        # Malformed header, return empty to avoid static
        return b""

    async def stream_response_to_user(self, call_sid, delta_generator, websocket, stream_sid):
        """
        Stream text deltas through ElevenLabs realtime TTS and pipe audio to Twilio WebSocket.
        """
        try:
            logger.info(f"⏩ PIPELINE: Streaming TTS response for call {call_sid}")
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.error(f"❌ PIPELINE ERROR: WebSocket disconnected before streaming for call {call_sid}")
                return
            if not self.client:
                logger.error("❌ PIPELINE ERROR: ELEVENLABS_API_KEY not found")
                return
            eleven_client = self.client
            # Sync queues: text input via queue.Queue; audio output via asyncio.Queue
            text_queue = queue.Queue()
            audio_q = asyncio.Queue()
            # create send queue and strict 20ms pacing sender
            send_q = asyncio.Queue()

            async def _safe_send(payload: str):
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(payload)
                    except Exception:
                        pass

            async def _sender():
                # maintain even 20ms intervals
                next_time = asyncio.get_event_loop().time()
                while True:
                    chunk = await send_q.get()
                    if chunk is None:
                        break
                    now = asyncio.get_event_loop().time()
                    if now < next_time:
                        await asyncio.sleep(next_time - now)
                    media_msg = json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode('utf-8')}
                    })
                    await _safe_send(media_msg)
                    next_time += 0.02

            sender_task = asyncio.create_task(_sender())

            # Register this streaming task and mark agent as speaking
            current_task = asyncio.current_task()
            interruption_manager.register_tts_task(call_sid, current_task)
            interruption_manager.set_speaking(call_sid, True)
            stop_event = threading.Event()
            loop = asyncio.get_running_loop()

            # Add pacing/backpressure parameters for Twilio media (max 200ms ahead)
            max_buffered_bytes = 1600  # allow up to ~200ms queued
            chunk_size = 160           # 20ms per chunk

            # Producer: feed text to ElevenLabs realtime TTS
            def _producer():
                def _text_iter():
                    while True:
                        txt = text_queue.get()
                        if txt is None:
                            break
                        yield txt
                try:
                    # Create custom voice settings
                    voice_id = "ZF6FPAbjXT4488VcRRnw"
                    custom_voice_settings = VoiceSettings(
                        stability=0.7,
                        similarity_boost=0.9,
                        style=0.1,
                        use_speaker_boost=True
                    )
                    # Stream with custom voice settings and support early closure
                    audio_gen = eleven_client.text_to_speech.convert_realtime(
                            voice_id=voice_id,
                            text=_text_iter(),
                            model_id="eleven_flash_v2",
                            output_format="ulaw_8000",
                            voice_settings=custom_voice_settings
                    )
                    for audio_chunk in audio_gen:
                        if stop_event.is_set():
                            # close generator early
                            close_method = getattr(audio_gen, "aclose", None)
                            if close_method:
                                try:
                                    audio_gen.aclose()
                                except Exception:
                                    pass
                            break
                        # push into asyncio queue for consumer with back-pressure (<=10 frames)
                        while audio_q.qsize() >= (max_buffered_bytes // chunk_size):
                            time.sleep(0.01)
                        logger.debug(f"[TTS] audio_q size before put: {audio_q.qsize()}")
                        loop.call_soon_threadsafe(audio_q.put_nowait, audio_chunk)
                except Exception as e:
                    logger.error(f"❌ PIPELINE ERROR: ElevenLabs TTS thread error: {e}")
                finally:
                    # signal end of stream to consumer
                    loop.call_soon_threadsafe(audio_q.put_nowait, None)

            # Define cleanup helper to stop producer thread, cancel feed_task, and send silent frame
            async def _cleanup():
                # clear speaking flag immediately on cleanup
                interruption_manager.set_speaking(call_sid, False)
                # cancel sender task
                if not sender_task.done():
                    sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender_task
                stop_event.set()
                if producer_thread.is_alive():
                    await loop.run_in_executor(None, producer_thread.join, 1.0)
                if not feed_task.done():
                    feed_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await feed_task
                # drain any residual audio and buffers
                while not audio_q.empty():
                    audio_q.get_nowait()
                while not send_q.empty():
                    send_q.get_nowait()
                # send 20ms of silence safely
                if websocket.client_state == WebSocketState.CONNECTED:
                    silence = self.generate_silence(20)
                    silent_msg = json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": base64.b64encode(silence).decode('utf-8')} 
                    })
                    await _safe_send(silent_msg)

            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()

            # Send start-response mark
            start_msg = json.dumps({"event":"mark","streamSid":stream_sid,"mark":{"name":"start-response"}})
            # skip send if socket already closed
            if websocket.client_state == WebSocketState.DISCONNECTED:
                return
            try:
                await websocket.send_text(start_msg)
            except WebSocketDisconnect:
                logger.error(f"❌ PIPELINE WARNING: WebSocket disconnected before TTS start for call {call_sid}")
                return
            
            # Async task to feed text deltas into text_queue
            async def _feed_text():
                try:
                    async for delta in delta_generator:
                        text_queue.put(delta)
                except Exception as e:
                    logger.error(f"❌ PIPELINE ERROR: delta generator failed for call {call_sid}: {e}")
                finally:
                    text_queue.put(None)
            feed_task = asyncio.create_task(_feed_text())

            # Produce to send_q with backpressure control
            buffer = bytearray()
            while True:
                try:
                    raw_chunk = await asyncio.wait_for(audio_q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if feed_task.done():
                        break
                    continue

                if raw_chunk is None:
                    break

                buffer.extend(raw_chunk)
                while len(buffer) >= chunk_size:
                    send_chunk = bytes(buffer[:chunk_size])
                    del buffer[:chunk_size]
                    # backpressure: no more than max_buffered_bytes queued (~10 frames)
                    while send_q.qsize() >= (max_buffered_bytes // chunk_size):
                        await asyncio.sleep(0.01)
                    send_q.put_nowait(send_chunk)
                    logger.debug(f"[TTS] send_q size after put: {send_q.qsize()}")
            # flush any remainder
            if buffer:
                while send_q.qsize() >= (max_buffered_bytes // chunk_size):
                    await asyncio.sleep(0.01)
                send_q.put_nowait(bytes(buffer))
                logger.debug(f"[TTS] send_q size after remainder put: {send_q.qsize()}")
            # signal sender end and wait for it
            send_q.put_nowait(None)
            await sender_task

            # Send end-response mark
            end_msg = json.dumps({"event":"mark","streamSid":stream_sid,"mark":{"name":"end-response"}})
            # skip send if socket already closed
            if websocket.client_state == WebSocketState.DISCONNECTED:
                return
            await _safe_send(end_msg)
            logger.info(f"✅ PIPELINE COMPLETE: Streamed TTS for call {call_sid}")
        except asyncio.CancelledError:
            logger.info(f"TTS stream for {call_sid} cancelled due to user barge-in")
            await _cleanup()
            return
        except Exception as e:
            logger.error(f"❌ PIPELINE ERROR in stream_response_to_user for call {call_sid}: {e}")
            logger.error(repr(e))
            await _cleanup()
        finally:
            await _cleanup()
            interruption_manager.set_speaking(call_sid, False)
            interruption_manager.register_tts_task(call_sid, None)
            interruption_manager.register_llm_handle(call_sid, None)

# Create a singleton instance
tts_service = TTSService() 
