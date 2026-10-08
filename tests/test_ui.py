"""The page's two layouts, driven in a real browser: a computer (wide, keyboard
and mouse) and a phone (narrow, touch), against v running with a fake model.

Needs Playwright: pip install playwright && playwright install chromium.
Skipped when it isn't installed.
"""

import base64
import glob
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from PIL import Image, ImageDraw  # noqa: E402

from tests.test_local import FakeModelServer, text_chunks, tool_chunks  # noqa: E402
from v import phone  # noqa: E402
from v.check import sample_font  # noqa: E402

DESKTOP = {"viewport": {"width": 1280, "height": 800}}
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True, "device_scale_factor": 2}
FAKE_MIC = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]

# Speech recognition that listens until stopped, then hears a phrase (headless
# browsers have no microphone, so the page's own handling is what's tested).
FAKE_SPEECH = """
window.SpeechRecognition = window.webkitSpeechRecognition = class {
  start() { setTimeout(() => this.onstart && this.onstart(), 10); }
  stop() {
    if (this.onresult) this.onresult({ results: [[{ transcript: "what's 12 times 12" }]] });
    setTimeout(() => this.onend && this.onend(), 10);
  }
};
"""


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch(args=FAKE_MIC)
        except Exception:
            # a Chromium that came with the machine (e.g. a CI image) instead of Playwright's own download
            found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
            if not found:
                pytest.skip("Playwright's Chromium isn't installed: playwright install chromium")
            b = pw.chromium.launch(executable_path=found[-1], args=FAKE_MIC)
        yield b
        b.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def receipt_photo(path):
    img = Image.new("RGB", (700, 360), "white")
    draw = ImageDraw.Draw(img)
    for y, line in ((30, "Corner Cafe"), (130, "Latte   4.50"), (230, "Total   4.50")):
        draw.text((40, y), line, fill="black", font=sample_font(60))
    img.save(path)


@pytest.fixture
def running_v(tmp_path):
    """The real v (`v --phone`, plain HTTP) with a scripted fake model."""
    home = tmp_path / "home"
    home.mkdir()
    inbox = home / ".v" / "inbox"
    model = FakeModelServer([
        tool_chunks("read_file", {"path": str(inbox / "receipt.png")}),
        tool_chunks("write_file", {"path": "receipt summary.txt", "content": "Corner Cafe: 4.50"}, call_id="call_b"),
        text_chunks("Your Corner Cafe receipt comes to $4.50. I saved a note of it in receipt summary.txt."),
        text_chunks("That's the same receipt again: Corner Cafe, $4.50."),
    ])
    port = free_port()
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), NO_PROXY="*", no_proxy="*")
    env.pop("ANTHROPIC_API_KEY", None)
    out = open(tmp_path / "v.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "v", "--no-computer", "--local-url", model.url, "--port", str(port), "--phone", "--http"],
        cwd=home, env=env, stdout=out, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        try:
            opener.open(base + "/health", timeout=1).read()
            break
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    token = (home / ".v" / "phone" / "token").read_text(encoding="utf-8").strip()
    receipt_photo(tmp_path / "receipt.png")
    yield {"url": f"{base}/?t={token}", "home": home, "inbox": inbox, "model": model, "proc": proc,
           "receipt": tmp_path / "receipt.png", "log": tmp_path / "v.log"}
    if proc.poll() is None:
        proc.kill()
    out.close()
    model.close()


def visible(page, selector) -> bool:
    return page.eval_on_selector(selector, """el => { const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden'; }""")


def box(page, selector) -> dict:
    return page.eval_on_selector(selector, "el => { const r = el.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; }")


def wait_for_reply(page, text, timeout=30000):
    page.wait_for_function("t => [...document.querySelectorAll('.msg.v')].some(m => m.textContent.includes(t))", arg=text, timeout=timeout)


def test_computer_and_phone_layouts(browser, running_v):
    v = running_v
    errors = []

    # ---------------- computer: wide, keyboard and mouse ----------------
    desk_ctx = browser.new_context(**DESKTOP)
    desk_ctx.add_init_script(FAKE_SPEECH)
    desk = desk_ctx.new_page()
    desk.on("pageerror", lambda e: errors.append(str(e)))
    desk.goto(v["url"])
    desk.wait_for_selector("#status", state="hidden", timeout=30000)
    assert visible(desk, "#panel"), "the side panel shows on a computer"
    assert not visible(desk, "#phoneBtn"), "phone pairing lives in the panel, not behind a header button"
    desk.wait_for_function("() => document.getElementById('qrSide').naturalWidth > 0")
    mic, field, send = box(desk, "#mic"), box(desk, "#input"), box(desk, "#send")
    assert abs((mic["y"] + mic["h"]) - (field["y"] + field["h"])) < 4 and field["x"] < mic["x"] < send["x"] and mic["w"] <= 48
    assert "Hold Space" in desk.text_content("#hint")

    # the project goal, edited from the panel
    desk.click("#editGoal")
    desk.fill("#goalInput", "Launch the bakery website by June")
    desk.click("#saveGoal")
    desk.wait_for_function("() => document.getElementById('goalText').textContent.includes('bakery')")
    assert "bakery" in desk.text_content("#goal")

    # an instant-command timer counts down in the panel
    desk.fill("#input", "set a 10 minute timer")
    desk.press("#input", "Enter")
    desk.wait_for_selector("#timerList li")
    first = desk.text_content("#timerList .left")
    assert first in ("10:00",) or first.startswith("9:5"), first
    desk.wait_for_timeout(1300)
    assert desk.text_content("#timerList .left") != first, "the countdown ticks"

    # Shift+Enter is a new line; the box grows
    desk.fill("#input", "line one")
    desk.press("#input", "Shift+Enter")
    desk.type("#input", "line two")
    assert desk.input_value("#input") == "line one\nline two"
    assert box(desk, "#input")["h"] > field["h"] + 10
    desk.fill("#input", "")

    # drop a photo onto the window, ask about it: v reads the words in it
    data = base64.b64encode(v["receipt"].read_bytes()).decode()
    drag = desk.evaluate_handle("""d => { const t = new DataTransfer();
        t.items.add(new File([Uint8Array.from(atob(d), c => c.charCodeAt(0))], 'receipt.png', {type: 'image/png'}));
        return t; }""", data)
    desk.dispatch_event("main", "dragenter", {"dataTransfer": drag})
    assert visible(desk, "#dropzone")
    desk.dispatch_event("main", "drop", {"dataTransfer": drag})
    desk.wait_for_selector("#attachments .chip.ready")
    assert not visible(desk, "#dropzone")
    desk.fill("#input", "what does this receipt say?")
    desk.press("#input", "Enter")
    wait_for_reply(desk, "$4.50")
    assert "receipt.png" in desk.text_content(".msg.you .files")
    desk.wait_for_selector("#activityList li")
    assert "wrote receipt summary.txt" in desk.text_content("#activityList"), "project paths are shortened"
    assert not desk.eval_on_selector_all(".line.activity", "ls => ls.some(l => l.offsetWidth > 0)"), "not repeated in the chat"

    # hold Space to talk; let go to send
    desk.click("main")
    desk.keyboard.down(" ")
    desk.wait_for_timeout(600)
    assert desk.eval_on_selector("#mic", "m => m.classList.contains('recording')")
    assert desk.input_value("#input") == "", "holding Space doesn't type spaces"
    desk.keyboard.up(" ")
    wait_for_reply(desk, "That's 144.")
    # a tap keeps listening; Esc cancels without sending
    sent = desk.eval_on_selector_all(".msg.you", "m => m.length")
    desk.keyboard.press(" ")
    desk.wait_for_timeout(200)
    assert desk.eval_on_selector("#mic", "m => m.classList.contains('recording')")
    desk.keyboard.press("Escape")
    desk.wait_for_timeout(300)
    assert not desk.eval_on_selector("#mic", "m => m.classList.contains('recording')")
    assert desk.eval_on_selector_all(".msg.you", "m => m.length") == sent
    desk.keyboard.press("/")
    assert desk.evaluate("() => document.activeElement.id") == "input"

    # ---------------- phone: narrow, touch ----------------
    phone_page = browser.new_context(**PHONE).new_page()
    phone_page.on("pageerror", lambda e: errors.append(str(e)))
    phone_page.goto(v["url"])
    phone_page.wait_for_function("() => document.querySelectorAll('.msg.v').length >= 1")
    assert not visible(phone_page, "#panel")
    assert box(phone_page, "#mic")["w"] >= 70, "a big talk button"
    assert visible(phone_page, "#attach")
    assert visible(phone_page, "#timerBar .pill"), "the running timer shows as a pill"
    assert phone_page.eval_on_selector_all(".line.activity", "ls => ls.filter(l => l.offsetWidth > 0).length") > 0
    assert not phone_page.evaluate("() => document.documentElement.scrollWidth > innerWidth")
    with phone_page.expect_file_chooser() as chooser:
        phone_page.tap("#attach")
    chooser.value.set_files(str(v["receipt"]))
    phone_page.wait_for_selector("#attachments .chip.ready")
    phone_page.fill("#input", "and this one?")
    phone_page.tap("#send")
    wait_for_reply(phone_page, "same receipt")
    phone_page.tap("#heading")
    assert phone_page.eval_on_selector("#goalDialog", "d => d.open")
    assert "bakery" in phone_page.input_value("#goalInput")
    phone_page.keyboard.press("Escape")
    phone_page.wait_for_timeout(200)
    # cancelling the timer on the phone updates the computer too
    phone_page.tap("#timerBar .pill button")
    phone_page.wait_for_selector("#timerBar", state="hidden")
    desk.wait_for_selector("#timersCard", state="hidden")
    assert desk.text_content("#goalText") == "Launch the bakery website by June", "Esc in the editor changes nothing"

    # quit from the computer's panel
    desk.click("#quitSide")
    desk.wait_for_function("() => document.getElementById('statusText').textContent.includes('stopped')")
    v["proc"].wait(timeout=15)
    assert not errors, errors

    # what happened on the computer
    home, inbox, model = v["home"], v["inbox"], v["model"]
    assert (home / ".v" / "goal.txt").read_text(encoding="utf-8").strip() == "Launch the bakery website by June"
    assert (inbox / "receipt.png").exists() and (inbox / "receipt (2).png").exists()
    first_request = json.dumps(model.requests[0]["body"]["messages"])
    assert "receipt.png" in first_request and "Launch the bakery website by June" in first_request
    tool_result = json.dumps(model.requests[1]["body"]["messages"][-1])
    assert "Corner" in tool_result and "4.50" in tool_result, tool_result  # read_file read the photo's words
    assert (home / "receipt summary.txt").exists()


def test_every_panel_fits_every_screen(browser):
    """A busy session (screenshot, timers, approval, activity) in phone, tablet
    and computer sizes, light and dark: nothing spills off the side."""
    hub, media = phone.Hub(), phone.MediaStore()
    project = "/Users/sam/bakery-site"
    hello = {"project": "bakery-site", "project_path": project, "goal": "Launch the bakery website by June, with online cake orders",
             "stt": True, "voice": "server", "computer": True, "confirm": "risky", "brain": "Qwen3 8B (free, on this computer)",
             "ready": True, "status": "Ready", "phone_url": "https://192.168.1.20:8765/?t=x", "timers": []}
    srv = phone.PhoneServer(("127.0.0.1", 0), hub, media, phone.Bridge(hub, log=lambda *a: None), "tok-0123456789abcdef",
                            lambda: hello)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ui = phone.PhoneUI(hub, media)
    buf = io.BytesIO()
    Image.new("RGB", (1280, 800), (30, 30, 36)).save(buf, format="PNG")
    hub.publish({"type": "user", "text": "Add an order form for custom cakes to the website"})
    ui.activity(f"wrote {project}/order.html")
    ui.activity("opened http://localhost:8000/a-very-long-address/that/keeps/going/and/going/order.html?with=query")
    ui.image(base64.b64encode(buf.getvalue()).decode())
    ui.text("I added an order page and linked it from the home page. ")
    ui.text_end()
    ui.timers([{"id": "a", "label": "25 minute timer", "kind": "timer", "ends": round((time.time() + 1500) * 1000)},
               {"id": "b", "label": "reminder to call the printer about the flyers for the opening",
                "kind": "reminder", "ends": round((time.time() + 3 * 3600) * 1000)}])
    hub.publish({"type": "user", "text": "Here's the logo", "files": ["rosies-logo-final-version-2.png"]})
    hub.publish({"type": "busy", "busy": True})
    hub.publish({"type": "ask", "id": "q1", "question": "Run this command: git commit -am \"Add order page and logo\"?"})
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t=tok-0123456789abcdef"
    tablet = {"viewport": {"width": 1024, "height": 768}, "is_mobile": True, "has_touch": True}
    try:
        for name, opts, wide in (("computer", DESKTOP, True), ("phone", PHONE, False), ("tablet", tablet, True)):
            for scheme in ("light", "dark"):
                ctx = browser.new_context(**opts, color_scheme=scheme)
                page = ctx.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(url)
                page.wait_for_selector("#ask:not([hidden])")
                page.wait_for_selector("#screenCard:not([hidden])" if wide else "#screen:not([hidden])")
                assert not page.evaluate("() => document.documentElement.scrollWidth > innerWidth"), f"{name}/{scheme} scrolls sideways"
                assert visible(page, "#panel") == wide, name
                assert visible(page, "#screenCard" if wide else "#screen"), f"{name}: the screenshot shows"
                assert visible(page, "#timersCard" if wide else "#timerBar"), f"{name}: the timers show"
                assert visible(page, "#yes") and visible(page, "#no"), f"{name}: the approval buttons show"
                assert not errors, errors
                ctx.close()
    finally:
        srv.shutdown()
        srv.server_close()
