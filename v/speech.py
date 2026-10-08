"""Talking and listening.

- Recorder: grabs one utterance from the default microphone. It calibrates
  the room's noise floor, starts when your voice rises above it, and stops
  after a short pause — no push-to-talk needed.
- Transcriber: local speech-to-text with faster-whisper (nothing leaves the
  machine).
- Speaker: speaks replies sentence by sentence on a background thread, so
  the first sentence plays while the model is still writing the rest. Uses
  Kokoro when installed, else the OS voices via pyttsx3, else just prints.

Heavy imports (numpy, sounddevice, faster_whisper, kokoro) happen lazily so
text mode works without any of them installed.
"""

from __future__ import annotations

import io
import queue
import re
import threading
import wave
from typing import Callable, Iterator, Optional

SAMPLE_RATE = 16_000
BLOCK_S = 0.1


class Recorder:
    def __init__(
        self,
        silence_s: float = 0.8,
        max_s: float = 30.0,
        min_speech_s: float = 0.3,
        calibrate_s: float = 0.5,
    ):
        import numpy as np
        import sounddevice as sd

        self._np = np
        self._sd = sd
        self.silence_s = silence_s
        self.max_s = max_s
        self.min_speech_s = min_speech_s
        self.calibrate_s = calibrate_s
        self.threshold: Optional[float] = None

    def _rms(self, block) -> float:
        return float(self._np.sqrt(self._np.mean(block**2)))

    def record(self, paused: Callable[[], bool] = lambda: False):
        """Block until one utterance is captured; returns float32 mono audio
        at 16 kHz. While `paused()` is true (v itself is talking) audio is
        discarded so v never transcribes its own voice."""
        np = self._np
        block = int(SAMPLE_RATE * BLOCK_S)
        with self._sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block) as stream:
            if self.threshold is None:
                frames, _ = stream.read(int(SAMPLE_RATE * self.calibrate_s))
                noise = self._rms(frames[:, 0])
                self.threshold = max(0.01, noise * 3.0)

            preroll: list = []
            speech: list = []
            silent_blocks = 0
            needed_silence = int(self.silence_s / BLOCK_S)
            max_blocks = int(self.max_s / BLOCK_S)
            while True:
                data, _ = stream.read(block)
                chunk = data[:, 0].copy()
                if paused():
                    preroll.clear()
                    speech.clear()
                    silent_blocks = 0
                    continue
                loud = self._rms(chunk) > self.threshold
                if not speech:
                    preroll = (preroll + [chunk])[-3:]
                    if loud:
                        speech = preroll + []
                    continue
                speech.append(chunk)
                silent_blocks = 0 if loud else silent_blocks + 1
                if silent_blocks >= needed_silence or len(speech) >= max_blocks:
                    voiced = len(speech) - silent_blocks
                    if voiced * BLOCK_S < self.min_speech_s:
                        # a cough or a door — keep listening
                        speech, preroll, silent_blocks = [], [], 0
                        continue
                    return np.concatenate(speech)


class Transcriber:
    def __init__(self, model_size: str = "base.en"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="auto", compute_type="default")
        self.language = "en" if model_size.endswith(".en") else None

    def transcribe(self, audio) -> str:
        """`audio` is float32 samples at 16 kHz, or the bytes of a recorded
        file in any format ffmpeg/PyAV can decode (the phone sends webm or mp4)."""
        if isinstance(audio, (bytes, bytearray)):
            audio = io.BytesIO(audio)
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


# --- text-to-speech -------------------------------------------------------


class KokoroEngine:
    rate = 24_000

    def __init__(self, voice: str = "af_heart"):
        from kokoro import KPipeline

        # lang_code "a" = American English; voice names starting with "b"
        # (e.g. bf_emma) are British and need lang_code "b".
        self.pipeline = KPipeline(lang_code=voice[0] if voice[:1] in "ab" else "a")
        self.voice = voice

    def synthesize(self, text: str):
        """Float32 samples at 24 kHz for the whole text."""
        import numpy as np

        parts = []
        for result in self.pipeline(text, voice=self.voice):
            audio = result.audio
            if audio is not None:
                parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        return np.concatenate(parts) if parts else np.zeros(0, dtype="float32")

    def speak(self, text: str) -> None:
        import sounddevice as sd  # only needed when playing through this computer's speakers

        audio = self.synthesize(text)
        if len(audio):
            sd.play(audio, self.rate)
            sd.wait()

    def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass


class Pyttsx3Engine:
    def __init__(self):
        import pyttsx3

        self.engine = pyttsx3.init()

    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self) -> None:
        self.engine.stop()


class SilentEngine:
    def speak(self, text: str) -> None:
        pass

    def stop(self) -> None:
        pass


def make_engine(voice: str, log=print):
    try:
        return KokoroEngine(voice)
    except Exception as e:  # ImportError, missing model download, no audio device...
        log(f"(Kokoro voice unavailable: {e.__class__.__name__}: {e})")
    try:
        return Pyttsx3Engine()
    except Exception as e:
        log(f"(pyttsx3 voice unavailable: {e.__class__.__name__}: {e}); replies will be text only)")
    return SilentEngine()


def to_wav(samples, rate: int) -> bytes:
    """Float samples in [-1, 1] -> 16-bit mono WAV bytes."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


_MARKUP = re.compile(r"[*_#`>|]+")
_URL = re.compile(r"https?://\S+")


def clean_for_speech(text: str) -> str:
    text = _URL.sub("the link", text)
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    """Queue of sentences spoken one after another on a worker thread."""

    def __init__(self, engine):
        self.engine = engine
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._busy = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            self._busy.set()
            try:
                self.engine.speak(text)
            except Exception as e:
                print(f"(speech failed: {e})")
            finally:
                self._q.task_done()
                if self._q.empty():
                    self._busy.clear()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if text:
            self._busy.set()
            self._q.put(text)

    def is_speaking(self) -> bool:
        return self._busy.is_set()

    def wait(self) -> None:
        self._q.join()

    def stop(self) -> None:
        """Drop everything queued and cut off the current sentence."""
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass
        self.engine.stop()

    def close(self) -> None:
        self._q.put(None)


class SentenceChunker:
    """Turns a stream of text deltas into whole sentences for the Speaker."""

    _END = re.compile(r"(.+?[.!?](?:[\"')\]]*)(?:\s+|$)|.+?\n+)", re.S)

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> Iterator[str]:
        self.buf += delta
        while True:
            m = self._END.match(self.buf)
            # A match that runs to the very end of the buffer may be cut mid
            # sentence ("v1." of "v1.2") — wait for more text unless it ended
            # on whitespace.
            if not m or (m.end() == len(self.buf) and not self.buf[-1].isspace()):
                return
            sentence = m.group(0).strip()
            self.buf = self.buf[m.end():]
            if sentence:
                yield sentence

    def flush(self) -> Iterator[str]:
        rest, self.buf = self.buf.strip(), ""
        if rest:
            yield rest
