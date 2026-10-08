import threading

from v.speech import SentenceChunker, Speaker, clean_for_speech


def chunk(deltas):
    c = SentenceChunker()
    out = []
    for d in deltas:
        out += list(c.feed(d))
    return out, list(c.flush())


def test_sentences_are_emitted_as_soon_as_complete():
    out, rest = chunk(["Hello the", "re. How are", " you? I'm fi", "ne"])
    assert out == ["Hello there.", "How are you?"]
    assert rest == ["I'm fine"]


def test_does_not_split_inside_version_numbers():
    out, rest = chunk(["Upgrade to v1.", "2 now. Done"])
    assert out == ["Upgrade to v1.2 now."]
    assert rest == ["Done"]


def test_newlines_end_a_chunk():
    out, rest = chunk(["First line\nSecond line\n"])
    assert out == ["First line", "Second line"]
    assert rest == []


def test_clean_for_speech():
    assert clean_for_speech("**Done** — see `main.py` at https://x.io/a") == "Done — see main.py at the link"


class RecordingEngine:
    def __init__(self):
        self.spoken = []
        self.gate = threading.Event()
        self.gate.set()

    def speak(self, text):
        self.gate.wait(2)
        self.spoken.append(text)

    def stop(self):
        pass


def test_speaker_speaks_in_order_and_reports_busy():
    engine = RecordingEngine()
    engine.gate.clear()
    speaker = Speaker(engine)
    speaker.say("one")
    speaker.say("two")
    assert speaker.is_speaking()
    engine.gate.set()
    speaker.wait()
    assert engine.spoken == ["one", "two"]
    assert not speaker.is_speaking()
    speaker.close()


def test_speaker_ignores_empty_text():
    engine = RecordingEngine()
    speaker = Speaker(engine)
    speaker.say("  ** ")
    speaker.wait()
    assert engine.spoken == [] and not speaker.is_speaking()
    speaker.close()


class FakeInputStream:
    """Plays back a scripted signal in the blocks Recorder asks for."""

    def __init__(self, signal):
        self.signal = signal
        self.pos = 0

    def __call__(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        chunk = self.signal[self.pos : self.pos + frames]
        self.pos += frames
        if len(chunk) < frames:
            raise AssertionError("recorder read past the end of the signal")
        return chunk.reshape(-1, 1), False


def make_recorder(monkeypatch, signal, **kwargs):
    import sys
    import types

    stream = FakeInputStream(signal)
    sd = types.ModuleType("sounddevice")
    sd.InputStream = stream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    from v.speech import Recorder

    return Recorder(**kwargs), stream


def tone(seconds, amp):
    import numpy as np

    t = np.arange(int(16_000 * seconds)) / 16_000
    return (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")


def test_recorder_captures_one_utterance(monkeypatch):
    import numpy as np

    signal = np.concatenate([tone(0.5, 0.001), tone(1.0, 0.001), tone(1.2, 0.3), tone(2.0, 0.001)])
    recorder, stream = make_recorder(monkeypatch, signal, silence_s=0.8)
    audio = recorder.record()
    seconds = len(audio) / 16_000
    # ~1.2s of speech + up to 0.3s pre-roll + 0.8s trailing silence
    assert 1.9 <= seconds <= 2.4
    assert np.abs(audio).max() > 0.25


def test_recorder_ignores_short_blips_and_its_own_voice(monkeypatch):
    import numpy as np

    signal = np.concatenate(
        [
            tone(0.5, 0.001),  # calibration
            tone(0.1, 0.3),  # a click: too short to be speech
            tone(1.0, 0.001),
            tone(1.0, 0.3),  # v talking (paused) — must be ignored
            tone(0.5, 0.001),
            tone(1.0, 0.3),  # the user
            tone(1.5, 0.001),
        ]
    )
    recorder, stream = make_recorder(monkeypatch, signal, silence_s=0.8)
    v_talking = (int(16_000 * 1.6), int(16_000 * 2.6))
    audio = recorder.record(paused=lambda: v_talking[0] <= stream.pos <= v_talking[1])
    end = stream.pos / 16_000
    assert end > 3.6  # stopped after the user's utterance, not the blip or v's voice
    assert 1.6 <= len(audio) / 16_000 <= 2.2


def test_to_wav_roundtrip():
    import io
    import wave

    import numpy as np

    from v.speech import to_wav

    data = to_wav(np.array([0.0, 0.5, -0.5, 2.0], dtype="float32"), 24_000)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24_000, 4)
        frames = np.frombuffer(w.readframes(4), dtype="<i2")
    assert list(frames) == [0, 16383, -16383, 32767]  # clipped at full scale
