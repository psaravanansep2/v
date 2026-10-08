"""Everyday tools for the free brain.

Files, apps, websites, web search, the clipboard, shell commands and the
screen. Every tool is free and runs on this computer; web search and page
reading are the only ones that touch the internet.

Screen control works without a vision model: the screen is read with
on-device OCR (RapidOCR), and the model clicks things by the number shown
next to each piece of text. That keeps it usable by small open models.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional

from .config import Config, save_goal
from .confirm import COMMAND, COMPUTER, Confirmer
from .shell import run_command

MAX_RESULT = 8_000
MAX_LISTING = 200
USER_AGENT = "Mozilla/5.0 (compatible; v-assistant/0.1)"
WRITE = "write"  # confirmation kind for file changes outside the project


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _clip(text: str, limit: int = MAX_RESULT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[...cut {len(text) - limit} more characters]"


# --- web helpers --------------------------------------------------------------


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "pre"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)


def html_to_text(page: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(page)
    return parser.title.strip(), parser.text()


class _DuckResults(HTMLParser):
    """Parses DuckDuckGo's no-JavaScript results page."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._field: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": _real_url(a.get("href", "")), "snippet": ""})
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag in ("a", "div", "td"):
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            self.results[-1][self._field] += data


def _real_url(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect: //duckduckgo.com/l/?uddg=<url>."""
    href = html.unescape(href)
    parsed = urllib.parse.urlparse(href if "://" in href else "https:" + href if href.startswith("//") else href)
    if parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def parse_search_results(page: str, limit: int = 6) -> list[dict]:
    parser = _DuckResults()
    parser.feed(page)
    out = []
    for r in parser.results:
        title = re.sub(r"\s+", " ", r["title"]).strip()
        if title and r["url"].startswith("http"):
            out.append({"title": title, "url": r["url"], "snippet": re.sub(r"\s+", " ", r["snippet"]).strip()})
        if len(out) >= limit:
            break
    return out


def _http_get(url: str, timeout: int = 20, max_bytes: int = 3_000_000) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes)
        ctype = resp.headers.get("Content-Type", "")
        charset = resp.headers.get_content_charset() or "utf-8"
    return data.decode(charset, errors="replace"), ctype


def _http_download(url: str, timeout: int = 30, max_bytes: int = 15_000_000) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(max_bytes), resp.headers.get("Content-Type", "")


# --- checking web pages v writes --------------------------------------------------
# Small models often link a style sheet by the wrong name, or write animations
# that never run. Saying so right after the write lets them fix it at once.

_LINK = re.compile(r"""(?:href|src)\s*=\s*["']([^"'#?]+)""", re.I)
_URL = re.compile(r"""url\(\s*["']?([^"')#?]+)""", re.I)
_KEYFRAMES = re.compile(r"@(?:-webkit-)?keyframes\s+([\w-]+)", re.I)
_ANIMATION = re.compile(r"(?<![\w-])animation(-name)?\s*:\s*([^;}]+)", re.I)
_NOT_NAMES = {"infinite", "alternate", "alternate-reverse", "reverse", "normal", "forwards", "backwards", "both", "none",
              "running", "paused", "linear", "ease", "ease-in", "ease-out", "ease-in-out", "step-start", "step-end",
              "initial", "inherit", "unset"}


class _Saved(Exception):
    """A picture was saved as it is (an animated GIF)."""


def _external(ref: str) -> bool:
    return bool(re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", ref, re.I))


def _animation_names(value: str, name_only: bool) -> list:
    names = []
    for part in value.split(","):
        tokens = re.sub(r"\w+\([^)]*\)", " ", part).split()  # drop cubic-bezier(...), steps(...)
        for t in tokens:
            if name_only or not (t.lower() in _NOT_NAMES or re.fullmatch(r"-?[\d.]+(m?s|%)?", t, re.I)):
                names.append(t)
                break
    return names


def web_problems(path: Path) -> list:
    """What would stop a web page (or its style sheet) from showing as meant."""
    suffix = path.suffix.lower()
    if suffix not in (".html", ".htm", ".css"):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    problems, refs = [], []
    if suffix != ".css":
        refs += [r for r in _LINK.findall(text) if not _external(r) and not r.startswith(("javascript:", "mailto:"))]
    refs += [r for r in _URL.findall(text) if not _external(r)]
    here = [f.relative_to(path.parent).as_posix() for f in path.parent.rglob("*") if f.is_file()]
    for ref in dict.fromkeys(refs):
        target = (path.parent / ref.lstrip("/")).resolve()
        if not target.exists():
            import difflib

            near = difflib.get_close_matches(ref, here, n=1, cutoff=0.5)
            problems.append(f"it links to {ref}, which doesn't exist" + (f" (there is {near[0]})" if near else ""))
    css = text if suffix == ".css" else " ".join(re.findall(r"<style[^>]*>(.*?)</style>", text, re.S | re.I))
    defined = set(_KEYFRAMES.findall(css))
    used = set()
    for prop, value in _ANIMATION.findall(css):
        used.update(_animation_names(value, bool(prop)))
    for name in sorted(used - defined - _NOT_NAMES):
        problems.append(f"an animation uses {name}, but there's no @keyframes {name}")
    for name in sorted(defined - used):
        problems.append(f"@keyframes {name} isn't used by any animation, so it never runs")
    return problems


def _with_problems(message: str, path: Path) -> str:
    problems = web_problems(path)
    return message + (" Problems found: " + "; ".join(problems) + ". Fix them." if problems else "")


# --- reading text in pictures ---------------------------------------------------

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
_ocr_lock = threading.Lock()
_ocr = None


def ocr_engine():
    """RapidOCR, loaded once (its models ship with the package: no download)."""
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            _ocr = RapidOCR()
    return _ocr


def read_image(p: Path, engine=None) -> str:
    """The text in a photo, scan or screenshot, line by line."""
    try:
        import numpy as np
        from PIL import Image, ImageOps

        engine = engine or ocr_engine()
    except ImportError:
        return f"{p} is a picture. Reading text in pictures needs the free OCR add-on: pip install 'v[free]'."
    if p.suffix.lower() in (".heic", ".heif"):  # iPhone photos
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            return (f"{p.name} is an iPhone photo (HEIC), and the add-on that opens those isn't installed: "
                    "pip install pillow-heif. Or send it as a JPEG (iPhone Settings > Camera > Formats > Most Compatible).")
    with Image.open(p) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")  # phone photos are often stored sideways
    image.thumbnail((2400, 2400))
    result, _ = engine(np.array(image))
    words = []
    for box, text, score in result or []:
        if float(score) < 0.5 or not text.strip():
            continue
        ys = [pt[1] for pt in box]
        words.append({"text": text.strip(), "x": min(pt[0] for pt in box), "y": sum(ys) / 4, "h": max(ys) - min(ys)})
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["y"]):
        if rows and abs(w["y"] - rows[-1][0]["y"]) < max(rows[-1][0]["h"], w["h"]) * 0.5:
            rows[-1].append(w)
        else:
            rows.append([w])
    lines = ["  ".join(w["text"] for w in sorted(row, key=lambda w: w["x"])) for row in rows]
    head = f"{p.name} is a {image.width}x{image.height} picture."
    if not lines:
        return head + " There's no readable text in it."
    return head + " Its text, read with OCR (may contain small mistakes):\n" + "\n".join(lines)


# --- screen ---------------------------------------------------------------------


class Screen:
    """Reads the screen with OCR and clicks/types through pyautogui."""

    def __init__(self, computer, ocr=None):
        self.computer = computer  # v.computer.Computer: gui + grab + logical size
        self._ocr = ocr
        self.elements: list[dict] = []

    def _engine(self):
        if self._ocr is None:
            self._ocr = ocr_engine()
        return self._ocr

    def look(self) -> tuple[str, str]:
        """Returns (description for the model, PNG base64 for the phone preview)."""
        import numpy as np

        image = self.computer.grab()
        ratio = image.width / self.computer.width  # physical pixels per screen point (2 on Retina)
        result, _ = self._engine()(np.array(image))
        elements = []
        for box, text, score in result or []:
            if float(score) < 0.5 or not text.strip():
                continue
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            elements.append({"text": text.strip(), "x": round(sum(xs) / 4 / ratio), "y": round(sum(ys) / 4 / ratio)})
        # reading order: rows of ~12 points, then left to right
        elements.sort(key=lambda e: (round(e["y"] / 12), e["x"]))
        self.elements = elements[:150]
        lines = [f"Screen is {self.computer.width}x{self.computer.height}. Text on screen (number: text @ x,y):"]
        lines += [f"{i}: {e['text']} @ {e['x']},{e['y']}" for i, e in enumerate(self.elements, 1)]
        if not self.elements:
            lines.append("(no readable text found)")
        preview = image.copy()
        preview.thumbnail((1280, 1280))
        buf = io.BytesIO()
        preview.save(buf, format="PNG", optimize=True)
        return "\n".join(lines), base64.standard_b64encode(buf.getvalue()).decode()

    def point(self, element: Optional[int], x: Optional[int], y: Optional[int]) -> tuple[int, int]:
        if element is not None:
            if not 1 <= element <= len(self.elements):
                raise ValueError(f"there is no element {element}; call look_at_screen first")
            e = self.elements[element - 1]
            return e["x"], e["y"]
        if x is None or y is None:
            raise ValueError("give an element number, or both x and y")
        return (min(max(int(x), 0), self.computer.width - 1), min(max(int(y), 0), self.computer.height - 1))

    def click(self, element=None, x=None, y=None, button="left", double=False) -> str:
        px, py = self.point(element, x, y)
        self.computer.gui.click(x=px, y=py, clicks=2 if double else 1, interval=0.05, button=button)
        return f"Clicked at {px},{py}. Call look_at_screen to see the result."

    def type_text(self, text: str) -> str:
        self.computer._type(text)
        return "Typed."

    def press_keys(self, keys: str) -> str:
        from .computer import to_keys

        names = to_keys(keys, self.computer.platform)
        if len(names) == 1:
            self.computer.gui.press(names[0])
        else:
            self.computer.gui.hotkey(*names)
        return f"Pressed {keys}."

    def scroll(self, direction: str, amount: int = 5) -> str:
        clicks = max(1, min(int(amount), 50))
        self.computer.gui.scroll(clicks if direction == "up" else -clicks)
        return f"Scrolled {direction}."


# --- the toolbox ------------------------------------------------------------------


def open_with_system(target: str) -> None:
    if sys.platform == "darwin":
        is_app = not (target.startswith(("http://", "https://", "/", "~")) or os.path.exists(os.path.expanduser(target)))
        cmd = ["open", "-a", target] if is_app else ["open", os.path.expanduser(target)]
    elif sys.platform.startswith("win"):
        os.startfile(os.path.expanduser(target))  # type: ignore[attr-defined]
        return
    else:
        cmd = ["xdg-open", os.path.expanduser(target)]
        if not (target.startswith(("http://", "https://")) or os.path.exists(os.path.expanduser(target))):
            exe = subprocess.run(["which", target.lower()], capture_output=True, text=True).stdout.strip()
            if exe:
                cmd = [exe]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


# Small models often get a tool's name or argument names slightly wrong.
# Rather than fail, v maps the common variants to what was meant.
TOOL_ALIASES = {
    "open_app": "open", "open_application": "open", "launch_app": "open", "launch": "open", "start_app": "open",
    "open_url": "open", "open_website": "open", "open_webpage": "open", "open_file": "open", "open_folder": "open",
    "search": "web_search", "search_web": "web_search", "google": "web_search", "internet_search": "web_search",
    "websearch": "web_search", "search_internet": "web_search", "search_online": "web_search",
    "fetch_url": "read_webpage", "browse": "read_webpage", "get_webpage": "read_webpage", "read_url": "read_webpage",
    "fetch": "read_webpage", "fetch_webpage": "read_webpage", "get_url": "read_webpage",
    "shell": "run_command", "bash": "run_command", "execute": "run_command", "run": "run_command",
    "terminal": "run_command", "exec": "run_command", "run_shell": "run_command", "execute_command": "run_command",
    "ls": "list_files", "list_dir": "list_files", "list_directory": "list_files", "find_files": "list_files",
    "search_files": "list_files", "glob": "list_files", "find_file": "list_files",
    "cat": "read_file", "read": "read_file", "read_document": "read_file", "view_file": "read_file",
    "create_file": "write_file", "save_file": "write_file", "write": "write_file",
    "replace_in_file": "edit_file", "modify_file": "edit_file", "update_file": "edit_file",
    "screenshot": "look_at_screen", "take_screenshot": "look_at_screen", "see_screen": "look_at_screen",
    "read_screen": "look_at_screen", "get_screen": "look_at_screen", "view_screen": "look_at_screen",
    "type": "type_text", "write_text": "type_text", "keyboard_type": "type_text", "input_text": "type_text",
    "press_key": "press_keys", "key": "press_keys", "hotkey": "press_keys", "keypress": "press_keys",
    "mouse_click": "click", "left_click": "click", "double_click": "click", "right_click": "click",
    "copy_to_clipboard": "clipboard", "set_clipboard": "clipboard", "read_clipboard": "clipboard", "get_clipboard": "clipboard",
    "set_goal": "set_project_goal",
    "find_images": "get_images", "search_images": "get_images", "download_images": "get_images",
    "get_pictures": "get_images", "find_photos": "get_images", "get_photos": "get_images", "image_search": "get_images",
}
# arguments a wrong name implies, e.g. double_click -> double=True
NAME_DEFAULTS = {
    "double_click": {"double": True}, "right_click": {"button": "right"},
    "copy_to_clipboard": {"action": "write"}, "set_clipboard": {"action": "write"},
    "read_clipboard": {"action": "read"}, "get_clipboard": {"action": "read"},
}
ARG_ALIASES = {
    "open": {"app": "target", "app_name": "target", "application": "target", "url": "target", "name": "target",
             "path": "target", "file": "target", "website": "target", "folder": "target"},
    "web_search": {"q": "query", "search": "query", "search_query": "query", "text": "query", "term": "query", "keywords": "query"},
    "read_webpage": {"link": "url", "address": "url", "page": "url", "website": "url"},
    "run_command": {"cmd": "command", "shell": "command", "script": "command", "code": "command"},
    "list_files": {"directory": "path", "dir": "path", "folder": "path", "glob": "pattern", "query": "pattern", "name": "pattern"},
    "read_file": {"file": "path", "filename": "path", "file_path": "path", "filepath": "path", "name": "path"},
    "write_file": {"file": "path", "filename": "path", "file_path": "path", "text": "content", "contents": "content", "data": "content"},
    "edit_file": {"file": "path", "file_path": "path", "filename": "path", "old": "old_text", "new": "new_text", "find": "old_text",
                  "replace": "new_text", "old_string": "old_text", "new_string": "new_text", "search": "old_text", "replacement": "new_text"},
    "type_text": {"content": "text", "value": "text", "string": "text"},
    "press_keys": {"key": "keys", "combo": "keys", "shortcut": "keys", "hotkey": "keys", "text": "keys"},
    "click": {"id": "element", "element_id": "element", "number": "element", "index": "element", "target": "element"},
    "scroll": {"amount_lines": "amount", "clicks": "amount", "dir": "direction"},
    "clipboard": {"content": "text", "value": "text", "mode": "action"},
    "set_project_goal": {"text": "goal", "project_goal": "goal"},
    "get_images": {"q": "query", "search": "query", "topic": "query", "subject": "query", "keywords": "query",
                   "dir": "folder", "directory": "folder", "path": "folder", "save_to": "folder", "destination": "folder",
                   "number": "count", "n": "count", "num": "count", "limit": "count"},
}


def resolve_call(name: str, args: dict, known: set) -> tuple[str, dict]:
    """The intended tool and arguments for a possibly-misnamed call."""
    original = name
    name = name.strip()
    if name not in known:
        name = TOOL_ALIASES.get(name.lower().replace("-", "_").replace(" ", "_"), name)
    fixed = dict(NAME_DEFAULTS.get(original.lower(), {}))
    aliases = ARG_ALIASES.get(name, {})
    for key, value in args.items():
        fixed[aliases.get(key, key)] = value
    return name, fixed


def _coerce(handler, args: dict) -> dict:
    """Drop arguments the tool doesn't take; turn "3" into 3 and "true" into True where a number or flag is expected."""
    import inspect

    params = inspect.signature(handler).parameters
    out = {}
    for key, value in args.items():
        if key not in params:
            continue
        annotation = str(params[key].annotation)
        if isinstance(value, str) and "int" in annotation and re.fullmatch(r"-?\d+", value.strip()):
            value = int(value)
        elif isinstance(value, str) and "bool" in annotation and value.strip().lower() in ("true", "false", "yes", "no"):
            value = value.strip().lower() in ("true", "yes")
        elif isinstance(value, float) and "int" in annotation and value.is_integer():
            value = int(value)
        out[key] = value
    return out


class Toolbox:
    def __init__(self, cfg: Config, confirmer: Confirmer, ui, screen: Optional[Screen] = None,
                 opener: Callable[[str], None] = open_with_system, fetch=_http_get, download=_http_download):
        self.cfg = cfg
        self.confirmer = confirmer
        self.ui = ui
        self.screen = screen
        self.opener = opener
        self.fetch = fetch
        self.download = download

    # --- definitions ---

    def definitions(self) -> list[dict]:
        path = {"type": "string", "description": "File or folder path. Relative paths are inside the project folder; ~ is the home folder."}
        tools = [
            _fn("run_command", "Run a shell command in the project folder and get its output.",
                {"command": {"type": "string"}}, ["command"]),
            _fn("list_files", "List a folder, or find files whose names match a pattern (like *.pdf) in it and its subfolders.",
                {"path": path, "pattern": {"type": "string", "description": "Optional glob pattern, e.g. *.pdf or report*"}}, []),
            _fn("read_file", "Read a file: text, PDF, Word, or the words in a picture.", {"path": path}, ["path"]),
            _fn("write_file", "Create or overwrite a text file.", {"path": path, "content": {"type": "string"}}, ["path", "content"]),
            _fn("edit_file", "Replace one exact piece of text in a file with new text.",
                {"path": path, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
            _fn("open", "Open an app (by name), a file, a folder, or a web address with the computer's default program.",
                {"target": {"type": "string", "description": "e.g. Spotify, ~/Downloads, https://example.com"}}, ["target"]),
            _fn("web_search", "Search the web and get the top results (title, address, snippet).",
                {"query": {"type": "string"}}, ["query"]),
            _fn("read_webpage", "Get the readable text of a web page.", {"url": {"type": "string"}}, ["url"]),
            _fn("get_images", "Find free, openly licensed photos online and save them in a folder (for a web page or "
                "document). Returns the file names and the credits to show.",
                {"query": {"type": "string", "description": "what the photos should show, e.g. pizza"},
                 "folder": {"type": "string", "description": "where to save them, e.g. pizza_site/images"},
                 "count": {"type": "integer"}}, ["query"]),
            _fn("clipboard", "Read the clipboard, or put text on it.",
                {"action": {"type": "string", "enum": ["read", "write"]}, "text": {"type": "string"}}, ["action"]),
            _fn("set_project_goal", "Save the user's overall goal for this project; it's remembered next time.",
                {"goal": {"type": "string"}}, ["goal"]),
        ]
        if self.screen is not None:
            tools += [
                _fn("look_at_screen", "Read what's on the computer screen: every piece of visible text, numbered, with its position.", {}, []),
                _fn("click", "Click something on screen: by its number from look_at_screen, or at x,y.",
                    {"element": {"type": "integer"}, "x": {"type": "integer"}, "y": {"type": "integer"},
                     "button": {"type": "string", "enum": ["left", "right"]}, "double": {"type": "boolean"}}, []),
                _fn("type_text", "Type text into whatever is focused on screen.", {"text": {"type": "string"}}, ["text"]),
                _fn("press_keys", "Press a key or shortcut, e.g. Return, Escape, ctrl+s, cmd+space, alt+Tab.",
                    {"keys": {"type": "string"}}, ["keys"]),
                _fn("scroll", "Scroll the screen up or down.",
                    {"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer"}}, ["direction"]),
            ]
        return tools

    # --- helpers ---

    def _path(self, raw: str) -> Path:
        p = Path(os.path.expanduser(raw or "."))
        return (p if p.is_absolute() else self.cfg.project_dir / p).resolve()

    def _inside_project(self, p: Path) -> bool:
        try:
            p.relative_to(self.cfg.project_dir.resolve())
            return True
        except ValueError:
            return False

    def _may_write(self, p: Path, verb: str) -> bool:
        needs = self.confirmer.needs(COMPUTER) or (self.confirmer.needs(COMMAND) and not self._inside_project(p))
        return not needs or self.confirmer.confirm(f"{verb} {p}?")

    def _may_act_on_screen(self, what: str) -> bool:
        return not self.confirmer.needs(COMPUTER) or self.confirmer.confirm(f"I'm about to {what}. Okay?")

    # --- dispatch ---

    def names(self) -> set:
        return {t["function"]["name"] for t in self.definitions()}

    def run(self, name: str, args: dict) -> str:
        name, args = resolve_call(name, args, self.names())
        handler = getattr(self, f"t_{name}", None)
        if handler is None or (self.screen is None and name in ("look_at_screen", "click", "type_text", "press_keys", "scroll")):
            return f"Error: there is no tool named {name}. Available: {', '.join(sorted(self.names()))}."
        try:
            return _clip(handler(**_coerce(handler, args)))
        except TypeError as e:
            return f"Error: wrong arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {e.__class__.__name__}: {e}"

    def t_run_command(self, command: str) -> str:
        if self.confirmer.needs(COMMAND) and not self.confirmer.confirm(f"Run this command: {command}?"):
            return "The user declined to run this command."
        self.ui.activity(f"$ {command}")
        out, _ = run_command(command, self.cfg.project_dir, self.cfg.command_timeout_s)
        return out

    def t_list_files(self, path: str = ".", pattern: Optional[str] = None) -> str:
        root = self._path(path)
        if not root.exists():
            return f"Error: {root} doesn't exist."
        if root.is_file():
            return f"{root} is a file ({root.stat().st_size} bytes)."
        if pattern:
            matches = []
            for p in root.rglob(pattern):
                if any(part.startswith(".") for part in p.relative_to(root).parts):
                    continue
                matches.append(p)
                if len(matches) >= MAX_LISTING:
                    break
            lines = [p.relative_to(root).as_posix() for p in sorted(matches)]
            return f"{len(lines)} match(es) for {pattern} in {root}:\n" + "\n".join(lines) if lines else f"Nothing matches {pattern} in {root}."
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for p in entries[:MAX_LISTING]:
            if p.name.startswith("."):
                continue
            lines.append(f"{p.name}/" if p.is_dir() else f"{p.name}  ({p.stat().st_size:,} bytes)")
        more = f"\n…and {len(entries) - MAX_LISTING} more" if len(entries) > MAX_LISTING else ""
        return f"{root}:\n" + ("\n".join(lines) or "(empty)") + more

    def t_read_file(self, path: str) -> str:
        p = self._path(path)
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return read_pdf(p)
        if suffix == ".docx":
            return read_docx(p)
        if suffix in IMAGE_SUFFIXES:
            return read_image(p, self.screen._ocr if self.screen is not None else None)
        data = p.read_bytes()[:400_000]
        if b"\x00" in data[:2000]:
            return f"{p} isn't a text file ({p.stat().st_size:,} bytes)."
        return data.decode("utf-8", errors="replace")

    def t_write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        if not self._may_write(p, "Write the file"):
            return "The user declined this change."
        p.parent.mkdir(parents=True, exist_ok=True)
        write_exact(p, content)
        self.ui.activity(f"wrote {p}")
        return _with_problems(f"Wrote {len(content):,} characters to {p}.", p)

    def t_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p = self._path(path)
        text = read_exact(p)
        if old_text not in text and "\r\n" in text:
            old_text, new_text = old_text.replace("\n", "\r\n"), new_text.replace("\n", "\r\n")  # a Windows-style file
        count = text.count(old_text)
        if count == 0:
            return "Error: old_text wasn't found in the file. Read the file and copy the text exactly."
        if count > 1:
            return f"Error: old_text appears {count} times; include more surrounding text so it's unique."
        if not self._may_write(p, "Edit the file"):
            return "The user declined this change."
        write_exact(p, text.replace(old_text, new_text, 1))
        self.ui.activity(f"edited {p}")
        return _with_problems(f"Edited {p}.", p)

    def t_open(self, target: str) -> str:
        if not target.startswith(("http://", "https://")):
            path = self._path(target)
            if path.exists():  # "documents/report.pdf" means the one in the project, wherever v was started
                target = str(path)
        self.opener(target)
        self.ui.activity(f"opened {target}")
        return f"Opened {target}."

    def t_web_search(self, query: str) -> str:
        page, _ = self.fetch("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query))
        results = parse_search_results(page)
        if not results:
            return "No results (the search page may have changed or blocked the request)."
        return "\n\n".join(f"{i}. {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1))

    def t_read_webpage(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        page, ctype = self.fetch(url)
        if "html" not in ctype and "<html" not in page[:500].lower():
            return page
        title, text = html_to_text(page)
        return f"{title}\n\n{text}" if title else text

    def _image_results(self, query: str, n: int, animated: bool = False) -> list:
        """[(url, title, creator, license, page)] from Openverse, or Wikimedia Commons if that fails."""
        q = urllib.parse.quote_plus(query)
        gif = "&extension=gif" if animated else ""
        try:
            data = json.loads(self.fetch(f"https://api.openverse.org/v1/images/?q={q}&page_size={n}&mature=false{gif}")[0])
            found = [(r["url"], r.get("title") or query, r.get("creator") or "unknown",
                      f"CC {r.get('license', '').upper()} {r.get('license_version', '')}".strip(),
                      r.get("foreign_landing_url") or r["url"]) for r in data.get("results", []) if r.get("url")]
            if found:
                return found
        except (OSError, ValueError, KeyError):
            pass
        wiki_q = urllib.parse.quote_plus(query + (" filetype:gif" if animated else ""))
        url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6"
               f"&gsrlimit={n}&gsrsearch={wiki_q}&prop=imageinfo&iiprop=url|extmetadata"
               + ("" if animated else "&iiurlwidth=1600"))  # a resized GIF is a still picture
        pages = json.loads(self.fetch(url)[0]).get("query", {}).get("pages", {})
        found = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip() or "unknown"
            if info.get("thumburl") or info.get("url"):
                found.append((info.get("thumburl") or info["url"], page.get("title", query).replace("File:", ""), artist,
                              meta.get("LicenseShortName", {}).get("value", "free license"), info.get("descriptionurl", "")))
        return found

    def t_get_images(self, query: str, folder: str = "images", count: int = 3) -> str:
        count = max(1, min(int(count), 8))
        dest = self._path(folder)
        if not self._may_write(dest, "Save pictures into"):
            return "The user declined this change."
        dest.mkdir(parents=True, exist_ok=True)
        animated = bool(re.search(r"\b(gifs?|animat\w*|moving|moves)\b", query, re.I))
        search = re.sub(r"\b(an? |gifs?|animat\w*|moving|moves)\b", " ", query, flags=re.I).strip() or query
        slug = re.sub(r"[^a-z0-9]+", "-", search.lower()).strip("-")[:40] or "photo"
        saved, credits = [], []
        for url, title, creator, license_, page in self._image_results(search, count * 2, animated):
            if len(saved) >= count:
                break
            try:
                data, ctype = self.download(url)
                name = f"{slug}-{len(saved) + 1}.jpg"
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(data)) as image:
                        if animated:
                            if not getattr(image, "is_animated", False):
                                continue  # asked for an animation: a still picture won't do
                            name = name[:-4] + ".gif"
                            (dest / name).write_bytes(data)  # kept as it is: resizing would stop it moving
                            raise _Saved()
                        image = image.convert("RGB")
                        image.thumbnail((1600, 1600))  # web-sized
                        image.save(dest / name, format="JPEG", quality=85)
                except ImportError:
                    ext = {"image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(ctype.split(";")[0], ".jpg")
                    name = name[:-4] + ext
                    (dest / name).write_bytes(data)
            except _Saved:
                pass
            except Exception:  # a broken or unreachable picture: try the next one
                continue
            saved.append(name)
            credits.append(f"{name}: “{title}” by {creator}, {license_}" + (f" ({page})" if page else ""))
        if not saved:
            what = "animations (GIFs)" if animated else "pictures"
            return f"Couldn't find or download {what} of {search} (check the internet connection)."
        with open(dest / "CREDITS.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(credits) + "\n")
        self.ui.activity(f"saved {len(saved)} pictures of {query} to {dest}")
        where = dest.relative_to(self.cfg.project_dir.resolve()).as_posix() if self._inside_project(dest) else str(dest)
        what = "animated GIFs" if animated else "pictures"
        return (f"Saved {len(saved)} {what} in {where}: {', '.join(saved)}. Credits (show them on the page, e.g. in the "
                f"footer): " + "; ".join(credits))

    def t_clipboard(self, action: str, text: Optional[str] = None) -> str:
        import pyperclip

        if action == "write":
            pyperclip.copy(text or "")
            return "Copied to the clipboard."
        return pyperclip.paste() or "(the clipboard is empty)"

    def t_set_project_goal(self, goal: str) -> str:
        goal = goal.strip()
        self.cfg.goal = goal
        save_goal(self.cfg.project_dir, goal)
        self.ui.goal(goal)
        return f"Project goal saved: {goal}"

    def t_look_at_screen(self) -> str:
        description, preview = self.screen.look()
        self.ui.image(preview)
        return description

    def t_click(self, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
                button: str = "left", double: bool = False) -> str:
        px, py = self.screen.point(element, x, y)
        label = f"“{self.screen.elements[element - 1]['text']}”" if element else f"{px},{py}"
        if not self._may_act_on_screen(f"{'double-click' if double else 'click'} {label}"):
            return "The user declined this action."
        return self.screen.click(element, x, y, button, double)

    def t_type_text(self, text: str) -> str:
        if not self._may_act_on_screen(f"type “{text[:60]}”"):
            return "The user declined this action."
        return self.screen.type_text(text)

    def t_press_keys(self, keys: str) -> str:
        if not self._may_act_on_screen(f"press {keys}"):
            return "The user declined this action."
        return self.screen.press_keys(keys)

    def t_scroll(self, direction: str, amount: int = 5) -> str:
        return self.screen.scroll(direction, amount)


def read_exact(p: Path) -> str:
    """A text file exactly as it is: UTF-8, line endings untouched."""
    return p.read_bytes().decode("utf-8", errors="replace")


def write_exact(p: Path, text: str) -> None:
    """Write UTF-8 without translating line endings (Windows would turn \\n into \\r\\n)."""
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def read_pdf(p: Path, max_pages: int = 50) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Reading PDFs needs the pypdf package (pip install pypdf)."
    reader = PdfReader(str(p))
    pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    text = "\n\n".join(t.strip() for t in pages if t.strip())
    more = f"\n[...{len(reader.pages) - max_pages} more pages not read]" if len(reader.pages) > max_pages else ""
    return (text or "(no text found; the PDF may be scanned images)") + more


def read_docx(p: Path) -> str:
    """Text of a Word document, straight from its XML (no extra packages)."""
    import zipfile

    with zipfile.ZipFile(p) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml)).strip() or "(the document is empty)"


def parse_arguments(raw) -> dict:
    """Tool arguments from a model: a JSON string (usually) or an object."""
    if isinstance(raw, dict):
        return raw
    raw = (raw or "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value
