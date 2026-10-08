"""`v check`: does v really work on this computer, with the real model?

Runs a short real conversation and prints a pass/fail line for each step,
so a problem shows up as one clear line instead of a vague failure later.
Exit code 0 means the essentials work.
"""

from __future__ import annotations

import secrets
import tempfile
import time
from pathlib import Path
from typing import Callable

from .config import Config
from .confirm import Confirmer


class _Quiet:
    """A UI that records what happened instead of showing it."""

    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *args: self.events.append((name, *args))


def run_check(url=None, model=None, log: Callable[[str], None] = print) -> int:
    from .local import LocalClient, LocalError, ensure_server
    from .local_agent import LocalAgent
    from .quick import QuickCommands
    from .tools import Toolbox

    failures, essential_failed = [], False

    def report(ok: bool, label: str, detail: str = "", essential: bool = False) -> None:
        nonlocal essential_failed
        log(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
        if not ok:
            failures.append(label)
            essential_failed = essential_failed or essential

    log("v check: a short real conversation with your model\n")

    # 1. the model server
    try:
        server = ensure_server(lambda line: log(f"        {line}"), url=url, model=model)
    except LocalError as e:
        report(False, "free model is running", str(e), essential=True)
        return 1
    report(True, "free model is running", server.label)
    client = LocalClient(server, timeout=300)

    # 2. a plain reply, timed
    started, first, text = time.time(), None, ""
    try:
        for chunk in client.stream_chat(
            [{"role": "system", "content": f"Answer in one short sentence. {server.system_suffix}".strip()},
             {"role": "user", "content": "Say hello to the user."}],
            [],
        ):
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
            if delta and first is None:
                first = time.time()
            text += delta
        elapsed = time.time() - (first or started)
        speed = (len(text) / 4) / elapsed if elapsed > 0 else 0
        report(bool(text.strip()), "replies", f"first words after {(first or time.time()) - started:.1f}s, ~{speed:.0f} tokens/s",
               essential=True)
    except LocalError as e:
        report(False, "replies", str(e), essential=True)
        return 1

    # 3. tools: read a file it can't guess, then write one
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp)
        word = f"{secrets.choice(['pineapple', 'lighthouse', 'saxophone', 'origami'])}-{secrets.randbelow(9000) + 1000}"
        (project / "secret.txt").write_text(f"The secret word is {word}.\n")
        cfg = Config(project_dir=project, confirm="none")
        ui, spoken = _Quiet(), []
        agent = LocalAgent(client, cfg, spoken.append, Toolbox(cfg, Confirmer("none", lambda q: "yes"), ui), ui=ui)

        t0 = time.time()
        agent.turn("What is the secret word in the file secret.txt? Read the file, then tell me the word.")
        used = [m for m in agent.history if m["role"] == "tool"]
        answer = " ".join(m.get("content") or "" for m in agent.history if m["role"] == "assistant")
        report(bool(used) and word in answer, "uses tools (reads a file)",
               f"{time.time() - t0:.1f}s" if word in answer else
               f"tools used: {[m.get('name') for m in used] or 'none'}; answer: {answer[:120]!r}", essential=True)

        t0 = time.time()
        agent.turn("Create a file called hello.txt that contains the text: hi from v")
        made = project / "hello.txt"
        ok = made.exists() and "hi from v" in made.read_text().lower()
        report(ok, "uses tools (writes a file)", f"{time.time() - t0:.1f}s" if ok else "hello.txt wasn't written as asked")

    # 4. instant commands (no model)
    quick = QuickCommands(lambda t: None, _Quiet(), launcher=lambda target: None)
    math_ok = quick.handle("what's 12 times 12") == "That's 144."
    time_ok = (quick.handle("what time is it") or "").startswith("It's ")
    report(math_ok and time_ok, "instant commands", "math and time")

    # 5. optional pieces
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from rapidocr_onnxruntime import RapidOCR

        image = Image.new("RGB", (480, 120), "white")
        draw = ImageDraw.Draw(image)
        font = None
        for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"):
            try:
                font = ImageFont.truetype(name, 40)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=40)  # Pillow 10.1+: a scalable built-in font
            except TypeError:
                font = ImageFont.load_default()
        draw.text((20, 30), "Subscribe", fill="black", font=font)
        found, _ = RapidOCR()(np.array(image))
        texts = [t for _, t, _ in found or []]
        report(any("subscribe" in t.lower() for t in texts), "reads the screen (OCR)", f"saw {texts}")
    except ImportError:
        log("  skip  reads the screen (OCR) — install with: pip install rapidocr_onnxruntime")
    for module, label in (("faster_whisper", "speech recognition"), ("kokoro", "Kokoro voice"), ("pyttsx3", "built-in voice")):
        try:
            __import__(module)
            report(True, label, "installed")
        except ImportError:
            log(f"  skip  {label} — not installed")

    log("")
    if not failures:
        log("Everything works.")
    elif essential_failed:
        log(f"v can't work properly yet: {', '.join(failures)} failed.")
    else:
        log(f"The essentials work; {', '.join(failures)} didn't.")
    return 1 if essential_failed else 0
