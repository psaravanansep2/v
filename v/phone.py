"""Phone remote: talk to v from any iPhone or Android phone.

v keeps running on your computer (that's where it controls the screen and
runs Claude Code sessions); the phone becomes its microphone, speaker and
screen. Open the link (or scan the QR code) that `v --phone` prints and the
browser does the rest. No app to install, nothing to approve in an app store.

    phone ──POST /message (voice or text)──▶ v ──▶ Claude, tools, sessions
    phone ◀──GET /events (live stream)────── v    replies, progress, approvals,
                                                  spoken audio, screenshots

Security: this page can control your computer, so every request must carry
the pairing token from the printed link. The token is kept in
~/.v/phone/token so a home-screen shortcut keeps working across restarts;
`--new-token` replaces it (and locks out every phone that had the old one).
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import queue
import secrets
import socket
import ssl
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit

from .speech import to_wav
from .ui import Interrupted

STATE_DIR = Path.home() / ".v" / "phone"
DEFAULT_PORT = 8765
MAX_UPLOAD = 25 * 1024 * 1024
ASK_TIMEOUT_S = 300


# --- events -----------------------------------------------------------------


class Hub:
    """Fans events out to every connected phone. Recent events are kept so a
    phone that connects (or reconnects after the screen locked) catches up on
    the conversation."""

    def __init__(self, history: int = 1000):
        self._clients: list[queue.Queue] = []
        self._history: deque = deque(maxlen=history)
        self._lock = threading.Lock()
        self._seq = 0

    def publish(self, event: dict) -> None:
        with self._lock:
            self._seq += 1
            event = {**event, "seq": self._seq}
            self._history.append(event)
            for q in self._clients:
                q.put(event)

    def subscribe(self) -> tuple[queue.Queue, list]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._clients.append(q)
            return q, list(self._history)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


class MediaStore:
    """Recent spoken audio and screenshots, fetched by the phone by id."""

    def __init__(self, keep: int = 64):
        self._items: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
        self._keep = keep
        self._lock = threading.Lock()

    def put(self, data: bytes, content_type: str) -> str:
        media_id = uuid.uuid4().hex
        with self._lock:
            self._items[media_id] = (data, content_type)
            while len(self._items) > self._keep:
                self._items.popitem(last=False)
        return media_id

    def get(self, media_id: str) -> Optional[tuple[bytes, str]]:
        with self._lock:
            return self._items.get(media_id)


class PhoneUI:
    """Agent UI that streams everything to the phone."""

    def __init__(self, hub: Hub, media: MediaStore):
        self.hub = hub
        self.media = media

    def user(self, text: str) -> None:
        self.hub.publish({"type": "user", "text": text})

    def text(self, delta: str) -> None:
        self.hub.publish({"type": "text", "delta": delta})

    def text_end(self) -> None:
        self.hub.publish({"type": "text_end"})

    def progress(self, note: str) -> None:
        self.hub.publish({"type": "progress", "text": note})

    def activity(self, line: str) -> None:
        self.hub.publish({"type": "activity", "text": line})

    def notice(self, line: str) -> None:
        self.hub.publish({"type": "notice", "text": line})

    def error(self, message: str) -> None:
        self.hub.publish({"type": "error", "text": message})

    def image(self, png_b64: str) -> None:
        import base64

        media_id = self.media.put(base64.b64decode(png_b64), "image/png")
        self.hub.publish({"type": "image", "id": media_id})

    def busy(self, busy: bool) -> None:
        self.hub.publish({"type": "busy", "busy": busy})

    def goal(self, goal: str) -> None:
        self.hub.publish({"type": "goal", "text": goal})


class PhoneVoice:
    """Speaker engine for phone mode. With Kokoro installed, speech is made
    on this computer and sent to the phone as audio; otherwise the phone's
    own text-to-speech reads the text (works on iOS and Android browsers)."""

    def __init__(self, hub: Hub, media: MediaStore, synth=None):
        self.hub = hub
        self.media = media
        self.synth = synth

    @property
    def mode(self) -> str:
        return "server" if self.synth else "browser"

    def speak(self, text: str) -> None:
        if self.synth is None:
            self.hub.publish({"type": "speak", "text": text})
            return
        samples = self.synth.synthesize(text)
        if len(samples):
            media_id = self.media.put(to_wav(samples, self.synth.rate), "audio/wav")
            self.hub.publish({"type": "audio", "id": media_id, "text": text})

    def stop(self) -> None:
        self.hub.publish({"type": "stop_audio"})


# --- phone input -> agent ---------------------------------------------------


class Bridge:
    """Feeds what the phone sends to the agent, one turn at a time, and asks
    approval questions on the phone. While a question is open, the next thing
    the user says or types is taken as the answer."""

    def __init__(self, hub: Hub, transcriber=None, say: Callable[[str], None] = lambda t: None, log=print):
        self.hub = hub
        self.transcriber = transcriber
        self.say = say
        self.log = log
        self.agent = None
        self.not_ready = "I'm still getting ready (setting up the free model). Try again in a moment."
        self.inbox: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending: Optional[dict] = None

    # called from HTTP handler threads

    def submit_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "empty"
        if self._answer_pending(text):
            return "answered"
        self.inbox.put(("text", text))
        return "queued"

    def submit_audio(self, data: bytes) -> str:
        if self.transcriber is None:
            raise RuntimeError("speech recognition isn't installed on the computer")
        with self._lock:
            pending = self._pending is not None
        if pending:
            text = self.transcriber.transcribe(data)
            self.hub.publish({"type": "user", "text": text or "(didn't catch that)"})
            if text and self._answer_pending(text, echo=False):
                return "answered"
            return "empty"
        self.inbox.put(("audio", data))
        return "queued"

    def answer(self, ask_id: str, reply: str) -> bool:
        with self._lock:
            pending = self._pending
            if pending is None or pending["id"] != ask_id:
                return False
            pending["answer"] = reply
            pending["done"].set()
            return True

    def interrupt(self) -> None:
        if self.agent is not None and hasattr(self.agent, "cancel"):
            self.agent.cancel()
        with self._lock:
            if self._pending is not None:
                self._pending["cancelled"] = True
                self._pending["done"].set()

    def _answer_pending(self, text: str, echo: bool = True) -> bool:
        with self._lock:
            pending = self._pending
            if pending is None:
                return False
            pending["answer"] = text
            pending["done"].set()
        if echo:
            self.hub.publish({"type": "user", "text": text})
        return True

    # called on the worker thread (by the agent's Confirmer)

    def ask(self, question: str) -> str:
        pending = {"id": uuid.uuid4().hex[:12], "done": threading.Event(), "answer": "", "cancelled": False}
        with self._lock:
            self._pending = pending
        self.log(f"v? {question}")
        self.hub.publish({"type": "ask", "id": pending["id"], "question": question})
        self.say(question)
        answered = pending["done"].wait(ASK_TIMEOUT_S)
        with self._lock:
            self._pending = None
        reply = pending["answer"] if answered else ""
        self.hub.publish({"type": "ask_done", "id": pending["id"], "answer": reply})
        if pending["cancelled"]:
            raise Interrupted()
        return reply if answered else "no"

    def run_worker(self) -> None:
        import anthropic

        while True:
            kind, payload = self.inbox.get()
            if kind == "stop":
                return
            if self.agent is None:
                self.hub.publish({"type": "notice", "text": self.not_ready})
                continue
            try:
                if kind == "audio":
                    self.hub.publish({"type": "busy", "busy": True})
                    text = self.transcriber.transcribe(payload)
                    if not text:
                        self.hub.publish({"type": "notice", "text": "Didn't catch that. Try again?"})
                        self.hub.publish({"type": "busy", "busy": False})
                        continue
                else:
                    text = payload
                self.log(f"phone> {text}")
                self.hub.publish({"type": "user", "text": text})
                self.agent.turn(text)
            except anthropic.AuthenticationError:
                self.hub.publish({"type": "error", "text": "Claude API authentication failed. Check ANTHROPIC_API_KEY on the computer."})
                self.hub.publish({"type": "busy", "busy": False})
            except Exception as e:  # keep serving the phone whatever happens in one turn
                self.hub.publish({"type": "error", "text": f"{e.__class__.__name__}: {e}"})
                self.hub.publish({"type": "busy", "busy": False})

    def stop(self) -> None:
        self.interrupt()
        self.inbox.put(("stop", None))


# --- HTTP ---------------------------------------------------------------


def _page() -> bytes:
    return resources.files("v").joinpath("phone.html").read_bytes()


ICON_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<rect width="512" height="512" rx="112" fill="#111827"/>
<path d="M146 170l110 200 110-200" fill="none" stroke="#a5b4fc" stroke-width="56" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


class Handler(BaseHTTPRequestHandler):
    server: "PhoneServer"
    server_version = "v"

    def log_message(self, fmt, *args):  # keep the terminal for the conversation
        pass

    # --- helpers ---

    def _token_ok(self) -> bool:
        query = parse_qs(urlsplit(self.path).query)
        supplied = self.headers.get("X-V-Token") or (query.get("t") or [""])[0]
        return bool(supplied) and hmac.compare_digest(supplied.encode(), self.server.token.encode())

    def _send(self, status: int, body: bytes, content_type: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _locked_out(self) -> None:
        self._send(
            401,
            b"<!doctype html><meta name=viewport content='width=device-width'>"
            b"<body style='font-family:system-ui;padding:24px'><h2>Scan the QR code</h2>"
            b"<p>Open v with the link (or QR code) shown on your computer. It contains the pairing code.</p>",
            "text/html; charset=utf-8",
        )

    # --- routes ---

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/icon.svg":
            return self._send(200, ICON_SVG, "image/svg+xml")
        if path == "/health":  # lets a second launch find the running v
            return self._json(200, {"ok": True, "v": True})
        if not self._token_ok():
            return self._locked_out()
        if path == "/manifest.webmanifest":
            manifest = {
                "name": "v",
                "short_name": "v",
                "start_url": f"/?t={self.server.token}",
                "display": "standalone",
                "background_color": "#0b0d12",
                "theme_color": "#0b0d12",
                "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
            }
            return self._send(200, json.dumps(manifest).encode(), "application/manifest+json")
        if path == "/":
            csp = (
                "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'"
            )
            return self._send(200, _page(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
        if path == "/events":
            return self._events()
        if path == "/qr.svg":
            url = self.server.hello().get("phone_url")
            svg = qr_svg(url) if url else None
            if svg is None:
                return self._json(404, {"error": "no phone link"})
            return self._send(200, svg, "image/svg+xml")
        if path.startswith("/media/"):
            item = self.server.media.get(path.rsplit("/", 1)[-1])
            if item is None:
                return self._json(404, {"error": "gone"})
            return self._send(200, item[0], item[1])
        self._json(404, {"error": "not found"})

    def do_POST(self):
        # The token travels in a custom header, which a page on another site
        # can't add to a request without the browser asking this server first.
        if not self.headers.get("X-V-Token") or not self._token_ok():
            return self._json(401, {"error": "pairing token required"})
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            return self._json(413, {"error": "too large"})
        body = self.rfile.read(length)
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        bridge = self.server.bridge
        try:
            if path == "/message":
                if ctype.startswith("audio/") or ctype == "video/mp4" or ctype == "application/octet-stream":
                    return self._json(200, {"status": bridge.submit_audio(body)})
                data = json.loads(body or b"{}")
                return self._json(200, {"status": bridge.submit_text(str(data.get("text", "")))})
            if path == "/answer":
                data = json.loads(body or b"{}")
                ok = bridge.answer(str(data.get("id", "")), str(data.get("answer", "")))
                return self._json(200 if ok else 409, {"ok": ok})
            if path == "/interrupt":
                bridge.interrupt()
                return self._json(200, {"ok": True})
            if path == "/quit" and self.server.on_quit is not None:
                self._json(200, {"ok": True})
                threading.Thread(target=self.server.on_quit, daemon=True).start()
                return
        except (json.JSONDecodeError, RuntimeError) as e:
            return self._json(400, {"error": str(e)})
        self._json(404, {"error": "not found"})

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.connection.settimeout(30)  # a phone that stops reading gets dropped, not buffered forever
        q, history = self.server.hub.subscribe()

        def send(event: dict) -> None:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        try:
            send({"type": "hello", **self.server.hello()})
            for event in history:
                send({**event, "replay": True})
            send({"type": "replay_end"})
            while not self.server.stopping.is_set():
                try:
                    send(q.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (OSError, ssl.SSLError):
            pass  # phone went away (screen locked, network change); it reconnects
        finally:
            self.server.hub.unsubscribe(q)


class PhoneServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, hub: Hub, media: MediaStore, bridge: Bridge, token: str,
                 hello: Callable[[], dict], ssl_context: Optional[ssl.SSLContext] = None):
        super().__init__(address, Handler)
        self.hub = hub
        self.media = media
        self.bridge = bridge
        self.token = token
        self.hello = hello
        self.ssl_context = ssl_context
        self.stopping = threading.Event()
        self.on_quit: Optional[Callable[[], None]] = None

    def finish_request(self, request, client_address):
        # TLS handshake on the per-connection thread, so one slow or broken
        # client can't stall the accept loop.
        if self.ssl_context is not None:
            request.settimeout(15)
            try:
                request = self.ssl_context.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError):
                return  # e.g. the phone refusing our self-signed cert until the user accepts it
            request.settimeout(None)
            try:
                super().finish_request(request, client_address)
            finally:
                request.close()
            return
        super().finish_request(request, client_address)

    def shutdown(self):
        self.stopping.set()
        super().shutdown()


# --- setup helpers ----------------------------------------------------------


def lan_ip() -> str:
    """This computer's address on the local network (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def load_token(new: bool = False, state_dir: Path = STATE_DIR) -> str:
    path = state_dir / "token"
    if not new:
        try:
            token = path.read_text().strip()
            if len(token) >= 16:
                return token
        except OSError:
            pass
    state_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(18)
    path.write_text(token + "\n")
    os.chmod(path, 0o600)
    return token


def self_signed_cert(ip: str, state_dir: Path = STATE_DIR) -> tuple[Path, Path]:
    """A certificate for this computer's LAN address, made once and reused.
    Phones show a warning for it the first time; after the user accepts,
    the page counts as secure and the microphone works."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_path, key_path = state_dir / "cert.pem", state_dir / "key.pem"
    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            fresh = cert.not_valid_after_utc > datetime.now(timezone.utc) + timedelta(days=7)
            if fresh and ipaddress.ip_address(ip) in sans.get_values_for_type(x509.IPAddress):
                return cert_path, key_path
        except Exception:
            pass  # unreadable or for an old address: make a new one

    state_dir.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"v on {socket.gethostname()}")])
    now = datetime.now(timezone.utc)
    alt_names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    if ip != "127.0.0.1":
        alt_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    os.chmod(key_path, 0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx


def qr_svg(url: str) -> Optional[bytes]:
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    image = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, border=2)
    return image.to_string()


def print_qr(url: str) -> bool:
    try:
        import qrcode
    except ImportError:
        return False
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)
    return True
