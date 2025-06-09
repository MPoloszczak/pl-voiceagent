import time
import asyncio
import webrtcvad
import audioop
import statistics
from collections import deque

from utils import logger

# Constants for VAD and turn-taking
FRAME_MS = 20            # 20 ms frames
# Require a bit more speech before triggering barge-in to avoid
# spurious detections from brief noise or echo
# 120 ms of voiced frames to trigger barge-in
CONSEC_VOICED_FRAMES = 6  # 120 ms of voiced frames to trigger barge-in

ENERGY_WINDOW = 8         # median over 8 frames (160 ms) for adaptive threshold
NOISE_FLOOR = 300         # minimum RMS threshold
RESUME_SILENCE_MS = 600    # wait 600 ms of silence before resuming agent
HUMAN_GAP_SEC = 0.20       # additional 200 ms safety gap before speaking
BARGE_IN_THROTTLE_SEC = 0.1  # throttle barge-in to once every 100 ms

class InterruptionDetector:
    def __init__(self, call_sid):
        self.call_sid = call_sid
        self.vad = webrtcvad.Vad(3)  # VERY_AGGRESSIVE mode
        self.energy = deque(maxlen=ENERGY_WINDOW)
        self.voiced = 0
        self.last_at = 0.0

    def is_speech(self, ulaw: bytes) -> bool:
        pcm16 = audioop.ulaw2lin(ulaw, 2)
        rms = audioop.rms(pcm16, 2)
        self.energy.append(rms)
        thresh = max(NOISE_FLOOR, statistics.median(self.energy))
        if rms < thresh:
            self.voiced = 0
            return False
        if self.vad.is_speech(pcm16, 8000):
            self.voiced += 1
            if self.voiced >= CONSEC_VOICED_FRAMES:
                now = time.time()
                if now - self.last_at > BARGE_IN_THROTTLE_SEC:
                    self.last_at = now
                    logger.info(f"[VAD] barge-in on {self.call_sid}")
                    self.voiced = 0
                    return True
        else:
            self.voiced = 0
        return False

class InterruptionManager:
    def __init__(self):
        self.detectors = {}            # call_sid -> InterruptionDetector
        self.awaiting_user_end = {}    # call_sid -> bool
        self.last_barge_in = {}        # call_sid -> float
        self.last_user_end = {}        # call_sid -> float
        self.tts_tasks = {}            # call_sid -> asyncio.Task
        self.llm_handles = {}          # call_sid -> handle with .cancel()
        self.speaking_flag = {}        # call_sid -> bool
        self.silence_tasks = {}        # call_sid -> asyncio.Task
        self.resume_callbacks = {}     # call_sid -> coroutine callback

    def process_frame(self, call_sid, ulaw: bytes):
        # only run VAD when the agent is speaking or waiting for user to finish
        if not (self.speaking_flag.get(call_sid, False) or
                self.awaiting_user_end.get(call_sid, False)):
            return
        det = self.detectors.get(call_sid)
        if not det:
            det = InterruptionDetector(call_sid)
            self.detectors[call_sid] = det
        if det.is_speech(ulaw):
            # Barge-in detected
            self._cancel_tts(call_sid)
            self._cancel_llm(call_sid)
            self.awaiting_user_end[call_sid] = True
            self.last_barge_in[call_sid] = time.time()
            existing = self.silence_tasks.get(call_sid)
            if existing and not existing.done():
                existing.cancel()
            task = asyncio.create_task(self._wait_for_silence(call_sid))
            self.silence_tasks[call_sid] = task

    async def _wait_for_silence(self, call_sid):
        while True:
            await asyncio.sleep(RESUME_SILENCE_MS / 1000)
            last = self.last_barge_in.get(call_sid, 0)
            if time.time() - last >= RESUME_SILENCE_MS / 1000:
                if self.awaiting_user_end.get(call_sid):
                    await self._clear_barge_in(call_sid)
                break
        self.silence_tasks.pop(call_sid, None)

    async def _clear_barge_in(self, call_sid):
        self.awaiting_user_end.pop(call_sid, None)
        self.last_barge_in.pop(call_sid, None)
        self.last_user_end[call_sid] = time.time()
        callback = self.resume_callbacks.pop(call_sid, None)
        if callback:
            try:
                await callback()
            except Exception:
                logger.exception("Error executing resume callback")

    async def clear_barge_in_now(self, call_sid):
        """Immediately clear any active barge-in state for the call."""
        task = self.silence_tasks.pop(call_sid, None)
        if task and not task.done():
            task.cancel()
        await self._clear_barge_in(call_sid)

    def register_tts_task(self, call_sid, task):
        if task:
            self.tts_tasks[call_sid] = task
        else:
            self.tts_tasks.pop(call_sid, None)

    # NEW: register or clear the OpenAI LLM streaming handle
    def register_llm_handle(self, call_sid, handle):
        if handle:
            self.llm_handles[call_sid] = handle
        else:
            self.llm_handles.pop(call_sid, None)

    def set_speaking(self, call_sid, is_speaking: bool):
        # debug speaking flag changes
        logger.debug(f"[SPEAK] {call_sid}: speaking={is_speaking}")
        self.speaking_flag[call_sid] = is_speaking

    def can_agent_speak(self, call_sid) -> bool:
        # Agent may speak only after silence + human gap and not while already speaking
        if self.speaking_flag.get(call_sid) or self.awaiting_user_end.get(call_sid):
            return False
        last_end = self.last_user_end.get(call_sid, 0)
        return (time.time() - last_end) > HUMAN_GAP_SEC

    def _cancel_tts(self, call_sid):
        task = self.tts_tasks.get(call_sid)
        if task and not task.done():
            task.cancel()
        self.tts_tasks.pop(call_sid, None)
        # ensure speaking flag cleared immediately so further transcripts are processed
        self.set_speaking(call_sid, False)

    # NEW: cancel any active LLM stream handle
    def _cancel_llm(self, call_sid):
        handle = self.llm_handles.get(call_sid)
        if handle and hasattr(handle, "cancel"):
            try:
                handle.cancel()
            except Exception:
                pass
        self.llm_handles.pop(call_sid, None)

    def register_resume_callback(self, call_sid, callback):
        if callback:
            self.resume_callbacks[call_sid] = callback
        else:
            self.resume_callbacks.pop(call_sid, None)

    def cleanup(self, call_sid):
        # Remove all call-specific data to avoid memory leaks
        self.detectors.pop(call_sid, None)
        self.awaiting_user_end.pop(call_sid, None)
        self.last_barge_in.pop(call_sid, None)
        self.last_user_end.pop(call_sid, None)
        self.tts_tasks.pop(call_sid, None)
        self.llm_handles.pop(call_sid, None)
        self.speaking_flag.pop(call_sid, None)
        self.resume_callbacks.pop(call_sid, None)
        task = self.silence_tasks.pop(call_sid, None)
        if task and not task.done():
            task.cancel()

# Singleton instance for use in Twilio and TTS modules
interruption_manager = InterruptionManager() 