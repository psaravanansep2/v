"""Serving v's web interface: to phones on the local network (`v --phone`),
and to this computer's own app window as well (`v app`).

The web page comes up right away; the model, speech recognition and voice
get ready in the background with progress shown on the page, so the first
launch (which downloads a free model) never looks frozen.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

from .config import Config

STATE_DIR = Path.home() / ".v"
APP_FILE = STATE_DIR / "app.json"


def running_app_url() -> str | None:
    """The window address of a v app that's already running, if any."""
    try:
        info = json.loads(APP_FILE.read_text())
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            info["local_url"].split("?")[0] + "health", timeout=1.5
        ) as resp:
            if json.load(resp).get("v"):
                return info["local_url"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def run(cfg: Config, args, window: bool) -> int:
    from . import phone
    from .cli import _computer, _make_brain, _stop_sessions
    from .confirm import Confirmer
    from .speech import Speaker
    from .ui import TeeUI, TerminalUI

    hub, media = phone.Hub(), phone.MediaStore()
    status = {"text": "Starting…", "ready": False, "brain": ""}

    def set_status(text: str) -> None:
        text = text.strip()
        if text:
            status["text"] = text
            hub.publish({"type": "status", "text": text})
            print(text, flush=True)

    voice = phone.PhoneVoice(hub, media, None)
    speaker = Speaker(voice)
    bridge = phone.Bridge(hub, None, say=speaker.say)
    computer_box: dict = {}

    ip = phone.lan_ip()
    ssl_ctx, scheme = None, "http"
    if args.cert or args.key:
        if not (args.cert and args.key):
            raise SystemExit("--cert and --key go together")
        ssl_ctx, scheme = phone.ssl_context(args.cert, args.key), "https"
    elif not args.http:
        try:
            ssl_ctx, scheme = phone.ssl_context(*phone.self_signed_cert(ip)), "https"
        except ImportError:
            print("(cryptography isn't installed, so phones get plain HTTP: typing and keyboard dictation work, "
                  "the talk button needs HTTPS)")
    token = phone.load_token(new=args.new_token)
    phone_url = f"{scheme}://{ip}:{args.port}/?t={token}"

    def hello() -> dict:
        return {
            "project": cfg.project_dir.name or str(cfg.project_dir),
            "goal": cfg.goal,
            "stt": bridge.transcriber is not None,
            "voice": voice.mode,
            "computer": computer_box.get("computer") is not None,
            "confirm": cfg.confirm,
            "brain": status["brain"],
            "ready": status["ready"],
            "status": status["text"],
            "phone_url": phone_url,
        }

    servers = []
    try:
        servers.append(phone.PhoneServer(("0.0.0.0", args.port), hub, media, bridge, token, hello, ssl_ctx))
    except OSError as e:
        if not window:
            raise SystemExit(f"Can't listen on port {args.port}: {e}. Try --port with another number.")
        print(f"(phones can't connect: port {args.port} is busy)")
        phone_url = ""
    local_url = ""
    if window:
        # The app window talks to this computer directly: plain HTTP on
        # 127.0.0.1 is a secure page to the browser, so no certificate
        # warning, and the microphone works.
        for port in range(args.port + 1, args.port + 20):
            try:
                servers.append(phone.PhoneServer(("127.0.0.1", port), hub, media, bridge, token, hello, None))
                local_url = f"http://127.0.0.1:{port}/?t={token}"
                break
            except OSError:
                continue
        if not local_url:
            raise SystemExit("Couldn't start v's window: no free port.")

    stop = threading.Event()
    for srv in servers:
        srv.on_quit = stop.set
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    def prepare() -> None:
        try:
            from .speech import Transcriber

            set_status("Getting speech recognition ready…")
            bridge.transcriber = Transcriber(cfg.stt_model)
        except Exception as e:
            print(f"(Whisper unavailable: {e.__class__.__name__}; the browser's own speech recognition will be used)")
        try:
            from .speech import KokoroEngine

            voice.synth = KokoroEngine(cfg.voice)
        except Exception as e:
            print(f"(Kokoro voice unavailable: {e.__class__.__name__}; the browser's own voice will be used)")
        computer_box["computer"] = _computer(cfg)
        set_status("Getting the free AI model ready…")
        try:
            assistant, sessions, label = _make_brain(
                cfg, args, say=speaker.say, confirmer=Confirmer(cfg.confirm, bridge.ask),
                ui=TeeUI(TerminalUI(), phone.PhoneUI(hub, media)), stop_speaking=speaker.stop,
                computer=computer_box["computer"], log=set_status,
            )
        except SystemExit as e:
            bridge.not_ready = f"I couldn't finish setting up: {e}"
            set_status(f"Setup didn't finish: {e}")
            hub.publish({"type": "setup_failed", "text": f"Setup didn't finish: {e}"})
            return
        computer_box["sessions"] = sessions
        bridge.agent = assistant
        status.update(ready=True, brain=label, text="Ready")
        hub.publish({"type": "ready", "brain": label})
        print(f"Ready: {label}", flush=True)

    threading.Thread(target=prepare, daemon=True).start()
    threading.Thread(target=bridge.run_worker, daemon=True).start()

    print(f"\nv · {cfg.project_dir}")
    print(f"goal: {cfg.goal or '(none yet)'}\n")
    if phone_url:
        print("Open this on your phone (same Wi-Fi), or scan the code:\n")
        if not phone.print_qr(phone_url):
            print("(`pip install qrcode` to show a QR code here)")
        print(f"\n  {phone_url}\n")
        if scheme == "https" and not args.cert:
            print("The first time, your phone warns about the certificate (it's v's own, made on this computer).")
            print("Tap Show Details / Advanced, then visit the site. Then Share → Add to Home Screen.\n")
        print("Anyone with this link can control this computer; keep it private.")
    if window:
        from .desktop import open_window

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        APP_FILE.write_text(json.dumps({"local_url": local_url}))
        open_window(local_url)
        print(f"v's window: {local_url}")
    print("Ctrl+C to stop.\n", flush=True)

    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        _stop_sessions(computer_box.get("sessions"))
        for srv in servers:
            srv.stopping.set()
            srv.shutdown()
            srv.server_close()
        speaker.close()
        if window:
            APP_FILE.unlink(missing_ok=True)
    return 0


def app_main(cfg: Config, args) -> int:
    """`v app`: reuse a running v (just show its window), or start one."""
    url = running_app_url()
    if url:
        from .desktop import open_window

        open_window(url)
        return 0
    if args.background or sys.stdout is None:
        # started from the icon: no terminal, so keep a log instead
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log = open(STATE_DIR / "app.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = log
    return run(cfg, args, window=True)
