"""Executes Claude's computer toolset calls (`computer_toolset_20260801`) on
this machine's primary display.

Coordinates: Claude sees screenshots scaled down to fit the API's image
limits and answers in that scaled pixel space. Everything here converts
between three spaces:

- model:    the scaled screenshot Claude sees (what its coordinates mean)
- logical:  pyautogui's screen coordinates (what mouse calls take)
- physical: the captured bitmap (2x logical on a Retina display)

Scaling is computed from the logical size, so a Retina capture is simply
downscaled further and clicks still land in the right place.
"""

from __future__ import annotations

import base64
import io
import math
import sys
import time
from typing import Any, Callable, Optional

TOOLSET_NAME = "computer"

# Members that only look at the screen or pause; they never need approval.
READ_ONLY = {"screenshot", "zoom", "cursor_position", "wait", "mouse_move"}

CLICKS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}

# The computer-use docs' scaling limits for the long edge and total pixels.
# Newer models accept larger images, but smaller screenshots mean fewer
# tokens per step and faster replies — and `zoom` covers fine detail.
MAX_LONG_EDGE = 1568
MAX_PIXELS = 1_150_000

NOT_EXECUTED = "Not executed: an earlier computer action in this turn failed."


def scale_factor(width: int, height: int) -> float:
    long_edge = max(width, height)
    return min(1.0, MAX_LONG_EDGE / long_edge, math.sqrt(MAX_PIXELS / (width * height)))


# xdotool-style names Claude uses -> pyautogui key names
_KEYS = {
    "return": "enter",
    "enter": "enter",
    "kp_enter": "enter",
    "backspace": "backspace",
    "escape": "esc",
    "esc": "esc",
    "tab": "tab",
    "space": "space",
    "delete": "delete",
    "del": "delete",
    "insert": "insert",
    "home": "home",
    "end": "end",
    "page_up": "pageup",
    "pageup": "pageup",
    "prior": "pageup",
    "page_down": "pagedown",
    "pagedown": "pagedown",
    "next": "pagedown",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "caps_lock": "capslock",
    "print": "printscreen",
    "ctrl": "ctrl",
    "control": "ctrl",
    "control_l": "ctrlleft",
    "control_r": "ctrlright",
    "alt": "alt",
    "alt_l": "altleft",
    "alt_r": "altright",
    "shift": "shift",
    "shift_l": "shiftleft",
    "shift_r": "shiftright",
    "minus": "-",
    "plus": "+",
    "equal": "=",
    "comma": ",",
    "period": ".",
    "slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "apostrophe": "'",
    "grave": "`",
    "bracketleft": "[",
    "bracketright": "]",
}
_SUPER = {"super", "super_l", "super_r", "meta", "cmd", "command", "win", "windows"}


def _super_key(platform: str) -> str:
    if platform == "darwin":
        return "command"
    if platform.startswith("win"):
        return "win"
    return "winleft"


def to_keys(combo: str, platform: str = sys.platform) -> list[str]:
    """'ctrl+shift+Escape' -> ['ctrl', 'shift', 'esc']. A lone '+' is the plus key."""
    if combo.strip() == "+":
        return ["+"]
    keys = []
    for raw in combo.split("+"):
        name = raw.strip()
        if not name:
            continue
        low = name.lower()
        if low in _SUPER:
            keys.append(_super_key(platform))
        elif low in _KEYS:
            keys.append(_KEYS[low])
        elif len(low) > 1 and low[0] == "f" and low[1:].isdigit():
            keys.append(low)  # F1..F24
        elif len(name) == 1:
            keys.append(name.lower())
        else:
            keys.append(low)
    return keys


def import_pyautogui():
    """pyautogui imports mouseinfo, which calls sys.exit() on Linux when
    tkinter isn't installed — taking the whole program down. mouseinfo only
    powers pyautogui.mouseInfo(), which v never uses, so in that case make
    its import fail normally; pyautogui handles an ImportError there."""
    if "mouseinfo" not in sys.modules:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            sys.modules["mouseinfo"] = None
    import pyautogui

    return pyautogui


def _default_grab():
    """Full-resolution capture of the primary monitor as a PIL RGB image."""
    from PIL import Image

    try:
        import mss

        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except ImportError:
        return import_pyautogui().screenshot().convert("RGB")


class Computer:
    def __init__(self, gui: Any = None, grab: Optional[Callable[[], Any]] = None, platform: str = sys.platform):
        if gui is None:
            gui = import_pyautogui()
            # Slam the mouse into a screen corner to abort (pyautogui raises
            # FailSafeException on the next action). Keep it on.
            gui.FAILSAFE = True
            gui.PAUSE = 0.05
        self.gui = gui
        self.grab = grab or _default_grab
        self.platform = platform
        self.width, self.height = gui.size()
        self.scale = scale_factor(self.width, self.height)
        self.model_width = round(self.width * self.scale)
        self.model_height = round(self.height * self.scale)

    # --- coordinate conversion ---

    def to_logical(self, coord) -> tuple[int, int]:
        x, y = coord
        lx = min(max(round(x / self.scale), 0), self.width - 1)
        ly = min(max(round(y / self.scale), 0), self.height - 1)
        return lx, ly

    def to_model(self, x: float, y: float) -> tuple[int, int]:
        return round(x * self.scale), round(y * self.scale)

    # --- screenshots ---

    def _png_block(self, image) -> dict:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        data = base64.standard_b64encode(buf.getvalue()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    def screenshot(self) -> dict:
        from PIL import Image

        image = self.grab().resize((self.model_width, self.model_height), Image.LANCZOS)
        return self._png_block(image)

    def zoom(self, region) -> dict:
        from PIL import Image

        image = self.grab()
        # physical pixels per model pixel (covers Retina's 2x capture)
        ratio = image.width / self.model_width
        x0, y0, x1, y1 = region
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"zoom region must have x0 < x1 and y0 < y1, got {list(region)}")
        box = tuple(round(v * ratio) for v in (x0, y0, x1, y1))
        crop = image.crop(box)
        fit = min(self.model_width / crop.width, self.model_height / crop.height)
        crop = crop.resize((max(1, round(crop.width * fit)), max(1, round(crop.height * fit))), Image.LANCZOS)
        return self._png_block(crop)

    # --- input helpers ---

    def _modifiers(self, text: Optional[str]) -> list[str]:
        return to_keys(text, self.platform) if text else []

    def _with_modifiers(self, text: Optional[str], action: Callable[[], None]) -> None:
        held = self._modifiers(text)
        for key in held:
            self.gui.keyDown(key)
        try:
            action()
        finally:
            for key in reversed(held):
                self.gui.keyUp(key)

    def _type(self, text: str) -> None:
        if text.isascii():
            self.gui.write(text, interval=0.005)
            return
        # pyautogui.write silently drops non-ASCII characters; paste instead.
        import pyperclip

        pyperclip.copy(text)
        self.gui.hotkey("command" if self.platform == "darwin" else "ctrl", "v")

    # --- dispatch ---

    def run(self, name: str, args: dict) -> Any:
        """Perform one member action. Returns a string or a list of content
        blocks for the tool_result. Raises on failure."""
        gui = self.gui
        if name == "screenshot":
            return [self.screenshot()]
        if name == "zoom":
            return [self.zoom(args["region"])]
        if name == "cursor_position":
            x, y = gui.position()
            mx, my = self.to_model(x, y)
            return f"[{mx}, {my}]"
        if name in CLICKS:
            button, clicks = CLICKS[name]
            coord = args.get("coordinate")

            def click():
                if coord is not None:
                    x, y = self.to_logical(coord)
                    gui.click(x=x, y=y, clicks=clicks, interval=0.05, button=button)
                else:
                    gui.click(clicks=clicks, interval=0.05, button=button)

            self._with_modifiers(args.get("text"), click)
            return "OK"
        if name == "left_click_drag":
            sx, sy = self.to_logical(args["start_coordinate"])
            ex, ey = self.to_logical(args["coordinate"])

            def drag():
                gui.moveTo(sx, sy)
                gui.dragTo(ex, ey, duration=0.3, button="left")

            self._with_modifiers(args.get("text"), drag)
            return "OK"
        if name == "mouse_move":
            gui.moveTo(*self.to_logical(args["coordinate"]))
            return "OK"
        if name == "left_mouse_down":
            gui.mouseDown(button="left")
            return "OK"
        if name == "left_mouse_up":
            gui.mouseUp(button="left")
            return "OK"
        if name == "scroll":
            direction = args["scroll_direction"]
            amount = int(args["scroll_amount"])
            coord = args.get("coordinate")
            if coord is not None:
                gui.moveTo(*self.to_logical(coord))

            def scroll():
                if direction in ("up", "down"):
                    gui.scroll(amount if direction == "up" else -amount)
                elif direction in ("left", "right"):
                    clicks = amount if direction == "right" else -amount
                    if hasattr(gui, "hscroll") and not self.platform.startswith("win"):
                        gui.hscroll(clicks)
                    else:  # Windows: shift+wheel scrolls sideways
                        gui.keyDown("shift")
                        try:
                            gui.scroll(-clicks)
                        finally:
                            gui.keyUp("shift")
                else:
                    raise ValueError(f"unknown scroll_direction {direction!r}")

            self._with_modifiers(args.get("text"), scroll)
            return "OK"
        if name == "type":
            self._type(args["text"])
            return "OK"
        if name == "key":
            keys = to_keys(args["text"], self.platform)
            repeat = max(1, min(int(args.get("repeat") or 1), 100))
            for _ in range(repeat):
                if len(keys) == 1:
                    gui.press(keys[0])
                else:
                    gui.hotkey(*keys)
            return "OK"
        if name == "hold_key":
            keys = to_keys(args["text"], self.platform)
            duration = min(float(args["duration"]), 300.0)
            for key in keys:
                gui.keyDown(key)
            try:
                time.sleep(duration)
            finally:
                for key in reversed(keys):
                    gui.keyUp(key)
            return "OK"
        if name == "wait":
            time.sleep(min(float(args["duration"]), 300.0))
            return "OK"
        raise ValueError(f"unsupported computer action {name!r}")


def describe(name: str, args: dict) -> str:
    """Short spoken description of an action, for confirmation prompts."""
    coord = args.get("coordinate")
    where = f" at {coord[0]}, {coord[1]}" if coord else ""
    if name in CLICKS:
        return f"{name.replace('_', ' ')}{where}"
    if name == "type":
        text = args.get("text", "")
        return f"type “{text[:60]}{'…' if len(text) > 60 else ''}”"
    if name in ("key", "hold_key"):
        return f"press {args.get('text', '')}"
    if name == "scroll":
        return f"scroll {args.get('scroll_direction', '')}{where}"
    if name == "left_click_drag":
        return f"drag to {coord[0]}, {coord[1]}" if coord else "drag"
    return name.replace("_", " ")


def run_batch(computer: Optional[Computer], blocks: list, approve: Callable[[list], bool]) -> list[dict]:
    """Run a turn's computer tool_use blocks in order, stopping at the first
    failure (later actions were planned assuming earlier ones worked).
    `approve(blocks)` is asked once for the state-changing actions in the
    batch; a refusal declines all of them. Returns one tool_result per block,
    each echoing toolset_name as the API requires."""

    def result(block, content, is_error=False) -> dict:
        r = {"type": "tool_result", "tool_use_id": block.id, "toolset_name": TOOLSET_NAME, "content": content}
        if is_error:
            r["is_error"] = True
        return r

    if computer is None:
        return [result(b, "Computer control is turned off for this session.", True) for b in blocks]

    changing = [b for b in blocks if b.name not in READ_ONLY]
    if changing and not approve(changing):
        return [result(b, "The user declined this action. Ask them what to do instead.", True) for b in blocks]

    results = []
    failed = False
    for block in blocks:
        if failed:
            results.append(result(block, NOT_EXECUTED, True))
            continue
        try:
            results.append(result(block, computer.run(block.name, block_input(block))))
        except Exception as e:
            failed = True
            results.append(result(block, f"Error: {e.__class__.__name__}: {e}", True))
    return results


def block_input(block) -> dict:
    inp = block.input
    if isinstance(inp, dict):
        return inp
    if hasattr(inp, "model_dump"):
        return inp.model_dump(exclude_none=True)
    return dict(inp or {})
