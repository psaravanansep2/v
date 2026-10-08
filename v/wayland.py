"""Screen control on Wayland (KDE Plasma, GNOME, and others).

On X11 any app could look at the screen and press keys. Wayland doesn't
allow that, so v asks the desktop properly, through the remote-control
permission that screen-sharing apps use (xdg-desktop-portal). The first time,
the desktop asks whether v may control the computer; v keeps the desktop's
answer, so it doesn't ask again every time (where the desktop supports that).

    seeing the screen:   KDE's Spectacle, grim (Sway, Hyprland), gnome-screenshot,
                         or the desktop's screenshot permission
    clicking and typing: the RemoteDesktop portal

The result looks like pyautogui to the rest of v, so `Computer` and the free
brain's `Screen` work unchanged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

TOKEN_FILE = Path.home() / ".v" / "wayland-permission"
DESKTOP = "/org/freedesktop/portal/desktop"
PORTAL_NAME = "org.freedesktop.portal.Desktop"
REMOTE = "org.freedesktop.portal.RemoteDesktop"
CAST = "org.freedesktop.portal.ScreenCast"
SHOT = "org.freedesktop.portal.Screenshot"


def is_wayland(env=os.environ, platform: str = sys.platform) -> bool:
    return platform.startswith("linux") and (env.get("XDG_SESSION_TYPE") == "wayland" or bool(env.get("WAYLAND_DISPLAY")))


class PortalError(RuntimeError):
    pass


class PortalDenied(PortalError):
    pass


# --- talking to the desktop's portal over D-Bus ---------------------------------------------


class Portal:
    """Calls to xdg-desktop-portal. Most of its methods answer later, with a Response
    signal on a Request object whose path v can work out in advance."""

    def __init__(self, conn=None):
        from jeepney.io.blocking import open_dbus_connection

        self.conn = conn or open_dbus_connection(bus="SESSION")
        self.sender = self.conn.unique_name.lstrip(":").replace(".", "_")
        self._n = 0

    def _address(self, interface: str, path: str = DESKTOP):
        from jeepney import DBusAddress

        return DBusAddress(path, bus_name=PORTAL_NAME, interface=interface)

    def call(self, interface: str, method: str, signature: str = "", *args, path: str = DESKTOP):
        """A method that answers straight away."""
        from jeepney import new_method_call
        from jeepney.wrappers import DBusErrorResponse, unwrap_msg

        try:  # an error reply comes back like any other message unless unwrapped
            return unwrap_msg(self.conn.send_and_get_reply(
                new_method_call(self._address(interface, path), method, signature, args), timeout=10))
        except DBusErrorResponse as e:
            raise PortalError(f"{interface.rsplit('.', 1)[-1]}.{method}: {e}") from None

    def request(self, interface: str, method: str, signature: str, *args, options: Optional[dict] = None,
                timeout: float = 60) -> dict:
        """A method that answers with a Response signal (e.g. after the user decides in a dialog)."""
        from jeepney.bus_messages import MatchRule, message_bus

        self._n += 1
        token = f"v{os.getpid()}_{self._n}"
        path = f"{DESKTOP}/request/{self.sender}/{token}"
        rule = MatchRule(type="signal", interface="org.freedesktop.portal.Request", member="Response", path=path)
        from jeepney.wrappers import unwrap_msg

        unwrap_msg(self.conn.send_and_get_reply(message_bus.AddMatch(rule), timeout=10))  # listen first: no answer is missed
        with self.conn.filter(rule) as answers:
            self.call(interface, method, signature + "a{sv}", *args, {**(options or {}), "handle_token": ("s", token)})
            try:
                code, results = self.conn.recv_until_filtered(answers, timeout=timeout).body
            except TimeoutError:
                raise PortalError(f"no answer from the desktop to {method} within {timeout:.0f}s") from None
        if code == 1:
            raise PortalDenied("the request wasn't allowed")
        if code != 0:
            raise PortalError(f"{method} didn't work (code {code})")
        return {k: v[1] for k, v in results.items()}  # unwrap the (signature, value) variants


# --- keys -----------------------------------------------------------------------------------------

# pyautogui key names -> X keysyms (what the portal takes)
KEYSYMS = {
    "enter": 0xFF0D, "return": 0xFF0D, "esc": 0xFF1B, "escape": 0xFF1B, "tab": 0xFF09, "backspace": 0xFF08,
    "delete": 0xFFFF, "del": 0xFFFF, "insert": 0xFF63, "home": 0xFF50, "end": 0xFF57, "pageup": 0xFF55,
    "pagedown": 0xFF56, "up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53, "space": 0x20,
    "shift": 0xFFE1, "shiftleft": 0xFFE1, "shiftright": 0xFFE2, "ctrl": 0xFFE3, "ctrlleft": 0xFFE3,
    "ctrlright": 0xFFE4, "alt": 0xFFE9, "altleft": 0xFFE9, "altright": 0xFFEA, "win": 0xFFEB, "winleft": 0xFFEB,
    "winright": 0xFFEC, "command": 0xFFEB, "super": 0xFFEB, "capslock": 0xFFE5, "printscreen": 0xFF61,
    "volumeup": 0x1008FF13, "volumedown": 0x1008FF11, "volumemute": 0x1008FF12, "playpause": 0x1008FF14,
    "nexttrack": 0x1008FF17, "prevtrack": 0x1008FF16,
    **{f"f{n}": 0xFFBE + n - 1 for n in range(1, 25)},
}

# The same keys as Linux key codes, for desktops that only take those (a US keyboard layout).
_ROWS = {2: "1234567890-=", 16: "qwertyuiop[]", 30: "asdfghjkl;'`", 44: "zxcvbnm,./"}
KEYCODES = {ch: start + i for start, row in _ROWS.items() for i, ch in enumerate(row)}
KEYCODES.update({"\\": 43, " ": 57, "space": 57, "enter": 28, "return": 28, "esc": 1, "escape": 1, "tab": 15,
                 "backspace": 14, "delete": 111, "del": 111, "insert": 110, "home": 102, "end": 107, "pageup": 104,
                 "pagedown": 109, "up": 103, "down": 108, "left": 105, "right": 106, "shift": 42, "shiftleft": 42,
                 "shiftright": 54, "ctrl": 29, "ctrlleft": 29, "ctrlright": 97, "alt": 56, "altleft": 56,
                 "altright": 100, "win": 125, "winleft": 125, "winright": 126, "command": 125, "super": 125,
                 "capslock": 58, "printscreen": 99, "volumeup": 115, "volumedown": 114, "volumemute": 113,
                 **{f"f{n}": 58 + n for n in range(1, 11)}, "f11": 87, "f12": 88})
SHIFTED = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))


def keysym(key: str) -> int:
    if key in KEYSYMS:
        return KEYSYMS[key]
    if len(key) == 1:
        code = ord(key)
        return code if 0x20 <= code <= 0x7E or 0xA0 <= code <= 0xFF else 0x01000000 + code
    raise ValueError(f"unknown key {key!r}")


# --- control: pointer and keyboard through the portal ---------------------------------------------


class RemoteControl:
    """A remote-desktop session with the desktop: move, click, scroll, press keys."""

    BUTTONS = {"left": 0x110, "right": 0x111, "middle": 0x112}  # Linux BTN_LEFT, BTN_RIGHT, BTN_MIDDLE

    def __init__(self, portal: Optional[Portal] = None, token_file: Optional[Path] = None,
                 log: Callable[[str], None] = print, timeout: float = 120):
        token_file = token_file or TOKEN_FILE
        self.portal = portal or Portal()
        self.use_keycodes = False
        p = self.portal
        self.session = p.request(REMOTE, "CreateSession", "", options={"session_handle_token": ("s", "v")})["session_handle"]
        devices = {"types": ("u", 3), "persist_mode": ("u", 2)}  # keyboard and pointer; remember the answer
        try:
            saved = token_file.read_text(encoding="utf-8").strip()
        except OSError:
            saved = ""
        if saved:
            devices["restore_token"] = ("s", saved)
        p.request(REMOTE, "SelectDevices", "o", self.session, options=devices)
        # A shared screen gives clicks their coordinates; v doesn't watch the stream itself.
        p.request(CAST, "SelectSources", "o", self.session, options={"types": ("u", 1), "multiple": ("b", False)})
        if not saved:
            log("Your desktop is asking whether v may control the screen. Choose your screen, then Allow.")
        started = p.request(REMOTE, "Start", "os", self.session, "", options={}, timeout=timeout)
        streams = started.get("streams") or []
        if not streams:
            raise PortalError("the desktop didn't share a screen, so v can't point at things on it")
        self.stream, props = streams[0]
        unwrap = lambda name, default: (props.get(name) or (None, default))[1]  # noqa: E731
        self.x0, self.y0 = unwrap("position", (0, 0))
        self.width, self.height = unwrap("size", (0, 0))
        token = started.get("restore_token")
        if token:
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(token, encoding="utf-8")
            try:
                token_file.chmod(0o600)
            except OSError:
                pass
        self.pos = (self.width // 2, self.height // 2)

    def _notify(self, method: str, signature: str, *args) -> None:
        self.portal.call(REMOTE, method, "oa{sv}" + signature, self.session, {}, *args)

    def move(self, x: float, y: float) -> None:
        x, y = min(max(x, 0), max(self.width - 1, 0)), min(max(y, 0), max(self.height - 1, 0))
        self._notify("NotifyPointerMotionAbsolute", "udd", self.stream, float(x), float(y))
        self.pos = (int(x), int(y))

    def button(self, name: str, down: bool) -> None:
        self._notify("NotifyPointerButton", "iu", self.BUTTONS[name], 1 if down else 0)

    def scroll(self, steps: int, horizontal: bool = False) -> None:
        self._notify("NotifyPointerAxisDiscrete", "ui", 1 if horizontal else 0, int(steps))

    def key(self, key: str, down: bool) -> None:
        state = 1 if down else 0
        if not self.use_keycodes:
            try:
                return self._notify("NotifyKeyboardKeysym", "iu", keysym(key), state)
            except PortalError:
                self.use_keycodes = True  # some desktops only take key codes
        code = KEYCODES.get(key.lower() if len(key) == 1 else key)
        if code is None:
            raise PortalError(f"this desktop can't type {key!r}")
        self._notify("NotifyKeyboardKeycode", "iu", code, state)

    def close(self) -> None:
        try:
            self.portal.call("org.freedesktop.portal.Session", "Close", path=self.session)
        except PortalError:
            pass


class WaylandGui:
    """The parts of pyautogui v uses, done through the portal."""

    def __init__(self, control: RemoteControl):
        self.c = control

    def size(self):
        return self.c.width, self.c.height

    def position(self):
        return self.c.pos

    def moveTo(self, x=None, y=None, duration=0.0, **kwargs):
        if x is not None and y is not None:
            self.c.move(x, y)

    def mouseDown(self, x=None, y=None, button="left", **kwargs):
        self.moveTo(x, y)
        self.c.button(button, True)

    def mouseUp(self, x=None, y=None, button="left", **kwargs):
        self.moveTo(x, y)
        self.c.button(button, False)

    def click(self, x=None, y=None, clicks=1, interval=0.0, button="left", **kwargs):
        if x is not None and y is not None:
            self.c.move(x, y)
            time.sleep(0.03)
        for i in range(clicks):
            self.c.button(button, True)
            time.sleep(0.02)
            self.c.button(button, False)
            if i < clicks - 1:
                time.sleep(interval or 0.05)

    def dragTo(self, x, y, duration=0.3, button="left", **kwargs):
        (sx, sy), steps = self.c.pos, 12
        self.c.button(button, True)
        for i in range(1, steps + 1):
            self.c.move(sx + (x - sx) * i / steps, sy + (y - sy) * i / steps)
            time.sleep(duration / steps)
        self.c.button(button, False)

    def scroll(self, clicks, **kwargs):
        self.c.scroll(-int(clicks))  # pyautogui: positive is up; the portal: positive is down

    def hscroll(self, clicks, **kwargs):
        self.c.scroll(int(clicks), horizontal=True)

    def keyDown(self, key, **kwargs):
        self.c.key(key, True)

    def keyUp(self, key, **kwargs):
        self.c.key(key, False)

    def press(self, key, presses=1, interval=0.0, **kwargs):
        for _ in range(presses):
            self.c.key(key, True)
            self.c.key(key, False)
            if interval:
                time.sleep(interval)

    def hotkey(self, *keys, **kwargs):
        for key in keys:
            self.c.key(key, True)
        for key in reversed(keys):
            self.c.key(key, False)

    def write(self, text, interval=0.0, **kwargs):
        named = {"\n": "enter", "\t": "tab"}
        for ch in text:
            if ch in named:
                self.press(named[ch])
                continue
            shift = ch.isupper() or ch in SHIFTED
            if shift:
                self.c.key("shift", True)
            # with key codes, a shifted symbol is its unshifted key plus shift
            key = SHIFTED.get(ch, ch) if self.c.use_keycodes else ch
            self.c.key(key, True)
            self.c.key(key, False)
            if shift:
                self.c.key("shift", False)
            if interval:
                time.sleep(interval)


# --- seeing the screen ----------------------------------------------------------------------------------

SCREENSHOT_TOOLS = [
    ("spectacle", ["spectacle", "--background", "--nonotify", "--current", "--output", "{file}"]),  # KDE
    ("grim", ["grim", "{file}"]),  # Sway, Hyprland and other wlroots desktops
    ("gnome-screenshot", ["gnome-screenshot", "--file", "{file}"]),
]


def grab(portal_factory: Callable[[], Portal] = Portal):
    """The screen at full resolution, as a PIL image."""
    from PIL import Image

    errors = []
    for name, template in SCREENSHOT_TOOLS:
        if not shutil.which(name):
            continue
        fd, path = tempfile.mkstemp(suffix=".png", prefix="v-screen-")
        os.close(fd)
        try:
            subprocess.run([a.replace("{file}", path) for a in template], check=True, timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            errors.append(f"{name}: {e.__class__.__name__}")
        finally:
            Path(path).unlink(missing_ok=True)
    try:
        uri = portal_factory().request(SHOT, "Screenshot", "s", "", options={"interactive": ("b", False)})["uri"]
    except (PortalError, KeyError, OSError) as e:
        raise RuntimeError("couldn't take a screenshot (" + "; ".join(errors + [f"portal: {e}"]) + ")") from None
    path = Path(unquote(urlparse(uri).path))
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    finally:
        path.unlink(missing_ok=True)  # the portal saves it in Pictures; it was only for v


def wayland_computer(log: Callable[[str], None] = print):
    """A Computer that works on Wayland. Asks the desktop for permission (once, where it remembers)."""
    from .computer import Computer

    control = RemoteControl(log=log)
    if not (control.width and control.height):  # the desktop didn't say how big the shared screen is
        control.width, control.height = grab().size
    computer = Computer(gui=WaylandGui(control), grab=grab, platform="linux")
    computer.control = control
    return computer
