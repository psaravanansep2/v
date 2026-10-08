"""Instant commands: the everyday basics, handled without the AI model.

Opening apps and websites, web search, time and date, timers, reminders,
notes, volume, screenshots, math and weather are answered directly: instant,
the same every time, and right even on the smallest free model. Matching is
deliberately strict. Anything that isn't clearly one of these goes to the
model, which can work it out with its tools.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import threading
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from . import apps

STATE_DIR = Path.home() / ".v"

# --- spoken numbers, durations and times --------------------------------------

_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def _word_value(word: str) -> Optional[int]:
    """25 for 'twenty-five', 7 for 'seven', None if it isn't a number word."""
    parts = word.split("-")
    if not all(p in _SMALL or p in _TENS for p in parts) or not word:
        return None
    return sum(_SMALL.get(p, _TENS.get(p, 0)) for p in parts)


def words_to_numbers(text: str) -> str:
    """'twenty five minutes' -> '25 minutes'; 'one hundred and twenty' -> '120'.
    Numbers combine only the way people say them, so 'five fifteen' stays
    two numbers ('5 15', a time) instead of adding up to 20."""
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        core = words[i].lower().strip(",.?!")
        if _word_value(core) is None:
            out.append(words[i])
            i += 1
            continue
        total, current, j = 0, 0, i
        while j < len(words):
            word = words[j].lower().strip(",.?!")
            value = _word_value(word)
            if value is not None:
                if j == i or (current == 0 and total):
                    current = value  # first word, or the next group after "thousand"
                elif value < 10 and current % 10 == 0 and current % 100 != 0 and current >= 20:
                    current += value  # twenty + five
                elif current % 100 == 0 and current and value < 100:
                    current += value  # (one hundred) + twenty
                else:
                    break
            elif word == "hundred" and current:
                current *= 100
            elif word == "thousand" and current:
                total, current = total + current * 1000, 0
            elif word == "and" and ((current % 100 == 0 and current) or (current == 0 and total)) and j + 1 < len(words) and \
                    _word_value(words[j + 1].lower().strip(",.?!")) is not None:
                pass  # "one hundred and twenty"
            else:
                break
            j += 1
        suffix = re.sub(r"^[\w-]+", "", words[j - 1])  # keep trailing punctuation
        out.append(f"{total + current}{suffix}")
        i = j
    return " ".join(out)


_UNIT_SECONDS = {"h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
                 "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
                 "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1}


def parse_duration(text: str) -> Optional[float]:
    t = words_to_numbers(text.lower())
    t = t.replace("half an hour", "30 minutes").replace("half a minute", "30 seconds").replace("quarter of an hour", "15 minutes")
    t = re.sub(r"\ban? (hour|minute|second)", r"1 \1", t)
    t = re.sub(r"(\d+) and a half (hour|minute|second)s?", lambda m: f"{int(m.group(1)) + 0.5} {m.group(2)}s", t)
    t = re.sub(r"(\d+(?:\.\d+)?) (hour|minute|second)s? and a half", lambda m: f"{float(m.group(1)) + 0.5} {m.group(2)}s", t)
    total, found = 0.0, False
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", t):
        total += float(m.group(1)) * _UNIT_SECONDS[m.group(2)]
        found = True
    if not found or total <= 0:
        return None
    rest = re.sub(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b|\band\b|,", "", t).strip()
    return total if not rest else None  # only a duration, nothing else


def format_duration(seconds: float, adjective: bool = False) -> str:
    """'1 hour and 30 minutes'; with adjective=True, '1 hour and 30 minute' (as in "a 90 minute timer")."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    plural = lambda n: "s" if n != 1 and not adjective else ""  # noqa: E731
    parts = []
    if h:
        parts.append(f"{h} hour{plural(h)}")
    if m:
        parts.append(f"{m} minute{plural(m)}")
    if s and not h:
        parts.append(f"{s} second{plural(s)}")
    return " and ".join(parts) or "0 seconds"


def parse_clock(text: str, now: datetime) -> Optional[datetime]:
    t = words_to_numbers(text.lower()).replace(".", "").strip()
    t = re.sub(r"\b(o'? ?clock|today|tonight|this)\b", "", t).strip()
    if t in ("noon", "midday"):
        h, mi, ap = 12, 0, "pm"
    elif t == "midnight":
        h, mi, ap = 12, 0, "am"
    else:
        m = re.fullmatch(r"(\d{1,2})(?::(\d{2})|\s(\d{2}))?\s*(am|pm|a m|p m)?\s*(in the (morning|afternoon|evening))?", t)
        if not m:
            return None
        h, mi = int(m.group(1)), int(m.group(2) or m.group(3) or 0)
        ap = (m.group(4) or "").replace(" ", "") or {"morning": "am", "afternoon": "pm", "evening": "pm"}.get(m.group(6) or "")
        if h > 23 or mi > 59:
            return None
    if ap:
        if h > 12:
            return None
        h = h % 12 + (12 if ap == "pm" else 0)
        hours = [h]
    else:
        hours = [h] if h > 12 or h == 0 else [h, (h + 12) % 24]
    best = None
    for hour in hours:
        candidate = now.replace(hour=hour, minute=mi, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        if best is None or candidate < best:
            best = candidate
    return best


def say_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


# --- math -----------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def spoken_math(text: str) -> Optional[str]:
    t = words_to_numbers(text.lower())
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,240 -> 1240
    t = re.sub(r"square root of\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(percent|%)\s*of", r"(\1/100)*", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(percent|%)", r"(\1/100)", t)
    for words, sym in (("multiplied by", "*"), ("times", "*"), ("divided by", "/"), ("over", "/"), ("plus", "+"),
                       ("minus", "-"), ("to the power of", "**"), ("squared", "**2"), ("cubed", "**3"), ("mod", "%")):
        t = t.replace(words, f" {sym} ")
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t)
    t = re.sub(r"\b(equals?|equal to|make|is|the answer to|exactly|the|a)\b|\?", " ", t)  # after the phrases above use them
    t = re.sub(r"\s+", " ", t).strip()
    if not re.fullmatch(r"[\d.\s+\-*/%()sqrt]+", t) or not re.search(r"[+\-*/%]|sqrt", t) or not re.search(r"\d", t):
        return None
    return t


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("too big")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
            return math.sqrt(ev(node.args[0]))
        raise ValueError("not arithmetic")

    return ev(ast.parse(expr, mode="eval"))


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


# --- weather -------------------------------------------------------------------

_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow", 80: "light showers", 81: "showers",
        82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
        96: "thunderstorms with hail", 99: "thunderstorms with hail"}


# --- the commands ----------------------------------------------------------------


def normalize(text: str) -> str:
    """Strips the wake word, politeness and end punctuation; keeps the user's capitals."""
    t = text.strip().replace("’", "'")
    t = re.sub(r"^(hey |ok |okay )?v\b[,.!]?\s+", "", t, flags=re.I)
    t = re.sub(r"^(please |can you |could you |would you |will you |i want you to |i'd like you to |go ahead and )+", "", t, flags=re.I)
    t = re.sub(r"[\s.!?,]+$", "", t)
    t = re.sub(r"\s+(please|for me)$", "", t, flags=re.I)
    return t.strip()


HELP = (
    "Instantly, I can open apps and websites, search the web, set timers and reminders, take and read notes, "
    "check the weather, do math, tell the time and date, take screenshots, and change the volume. "
    "For anything else, like finding and reading your files, writing or editing documents, answering questions "
    "from the web, or using an app on your screen, just ask and I'll work it out."
)


class QuickCommands:
    def __init__(self, say: Callable[[str], None], ui, stop_speaking: Callable[[], None] = lambda: None,
                 finder: Optional[apps.AppFinder] = None, launcher: Callable[[str], None] = apps.launch,
                 fetch=None, gui=None, grab=None, now: Callable[[], datetime] = datetime.now,
                 state_dir: Path = STATE_DIR, home: Optional[Path] = None):
        self.say = say
        self.ui = ui
        self.stop_speaking = stop_speaking
        self.finder = finder or apps.AppFinder()
        self.launcher = launcher
        self._fetch = fetch
        self._gui = gui
        self._grab = grab
        self.now = now
        self.state_dir = state_dir
        self.home = home or Path.home()
        self.timers: list[dict] = []
        self._lock = threading.Lock()
        self.rules = [
            (r"(what can you do|help|what do you do|how can you help( me)?|what are you able to do)", self._help),
            (r"(stop|stop talking|be quiet|quiet|shut up|never ?mind|cancel that|that's enough|enough)", self._stop),
            (r"(what('s| is) the time|what time is it( now| right now)?|tell me the time|current time|the time)", self._time),
            (r"(what('s| is) (the date|today's date|the date today|today)|what day is (it|today)( today)?|today's date)", self._date),
            (r"(my city is|i live in|i'm in|i am in|set my (city|location) to|my location is) (?P<city>.+)", self._set_city),
            (r"((what's|what is|how's|how is) the (weather|forecast)( like)?|weather|forecast|the weather)"
             r"(( in| for| at) (?P<city>.+?))?( (?P<when>today|tomorrow|now|right now|outside|this morning|tonight))?",
             self._weather),
            (r"((will|is) it (going to )?rain|do i need an umbrella)( (?P<when>today|tomorrow|tonight))?(( in| at) (?P<city>.+?))?"
             r"( (?P<when2>today|tomorrow|tonight))?", self._rain),
            (r"(set|start) (a |an )?(?P<dur>.+?) timer", self._timer),
            (r"(set|start) (a |an )?timer (for )?(?P<dur>.+)", self._timer),
            (r"timer (for )?(?P<dur>.+)", self._timer),
            (r"(cancel|stop|clear|delete|remove) (the |my |all |all my |all the )?(timers?|reminders?|alarms?)", self._cancel_timers),
            (r"(how (much time|long) is left|how much time is left on (the|my) timer|time left|how long until my timer)",
             self._time_left),
            (r"remind me (?P<rest>.+)", self._remind),
            (r"(read|show|tell me|what are|what's in|list) (me )?(my|the) notes", self._read_notes),
            (r"(read back my notes|my notes)", self._read_notes),
            (r"(take a note|make a note|add a note|note down|note|write down|jot down|add to my notes)( that|:)?\s+(?P<note>.+)",
             self._note),
            (r"(turn (the )?volume up|volume up|louder|turn it up|increase (the )?volume|raise (the )?volume)", self._volume_up),
            (r"(turn (the )?volume down|volume down|quieter|turn it down|decrease (the )?volume|lower (the )?volume)",
             self._volume_down),
            (r"(mute|unmute|mute (the )?(sound|volume|audio)|unmute (the )?(sound|volume|audio))", self._mute),
            (r"(take|grab|capture|save) (a )?screenshot( of (the|my) screen)?|screenshot", self._screenshot),
            (r"(search (the web |google |online |the internet )?for|google|look up|search) (?P<q>.+)", self._search),
            (r"(open|launch|start|go to|bring up|pull up|show me) (?P<target>.+)", self._open),
            (r"((what's|what is|calculate|compute|how much is|work out) )?(?P<expr>.+)", self._math),
        ]

    # --- entry point ---

    def handle(self, text: str) -> Optional[str]:
        """The reply ("" = handled silently), or None to let the model handle it."""
        t = normalize(text)
        if not t:
            return None
        for pattern, handler in self.rules:
            m = re.fullmatch(pattern, t, re.I)
            if m:
                try:
                    reply = handler(m)
                except Exception as e:  # a broken basic must not break the conversation
                    reply = None
                    self.ui.error(f"instant command failed: {e.__class__.__name__}: {e}")
                if reply is not None:
                    return reply
        return None

    # --- helpers ---

    def fetch_json(self, url: str):
        if self._fetch is None:
            from .tools import _http_get

            self._fetch = _http_get
        body, _ = self._fetch(url)
        return json.loads(body)

    def _settings_path(self) -> Path:
        return self.state_dir / "settings.json"

    def settings(self) -> dict:
        try:
            return json.loads(self._settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_setting(self, key: str, value) -> None:
        data = self.settings()
        data[key] = value
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._settings_path().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def gui(self):
        if self._gui is None:
            from .computer import import_pyautogui

            self._gui = import_pyautogui()
        return self._gui

    def notes_path(self) -> Path:
        docs = self.home / "Documents"
        return (docs / "v notes.md") if docs.is_dir() else (self.home / "v-notes.md")

    # --- handlers ---

    def _help(self, m):
        return HELP

    def _stop(self, m):
        self.stop_speaking()
        return ""

    def _time(self, m):
        return f"It's {say_time(self.now())}."

    def _date(self, m):
        now = self.now()
        return f"Today is {now:%A}, {now:%B} {now.day}, {now.year}."

    def _set_city(self, m):
        city = m.group("city").strip().title()
        if len(city.split()) > 4:
            return None
        self.save_setting("city", city)
        return f"Got it, I'll use {city} for the weather."

    def _city(self, m) -> Optional[str]:
        city = (m.groupdict().get("city") or "").strip()
        if city.lower() in ("here", "my area", "my city", "outside"):
            city = ""
        return city or self.settings().get("city")

    def _forecast(self, city: str):
        geo = self.fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=" + urllib.parse.quote(city)
        )
        results = geo.get("results") or []
        if not results:
            return None, None, None
        place = results[0]
        fahrenheit = place.get("country_code") in ("US", "LR", "MM")
        url = (
            "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max&timezone=auto&forecast_days=2{unit}"
        ).format(lat=place["latitude"], lon=place["longitude"], unit="&temperature_unit=fahrenheit" if fahrenheit else "")
        return place, self.fetch_json(url), fahrenheit

    def _weather(self, m):
        city = self._city(m)
        if not city:
            return "Which city should I use for the weather? Say, for example: my city is Chicago."
        try:
            place, data, _ = self._forecast(city)
        except Exception:
            return "I couldn't get the weather right now. Check the internet connection."
        if place is None:
            return f"I couldn't find a place called {city}."
        daily = data["daily"]
        name = place["name"]
        if (m.groupdict().get("when") or "").lower() == "tomorrow":
            return (f"Tomorrow in {name}: {_WMO.get(daily['weather_code'][1], 'mixed')}, high of {round(daily['temperature_2m_max'][1])}, "
                    f"low of {round(daily['temperature_2m_min'][1])}, {daily['precipitation_probability_max'][1] or 0} percent chance of rain.")
        cur = data["current"]
        return (f"It's {round(cur['temperature_2m'])} degrees and {_WMO.get(cur['weather_code'], 'mixed')} in {name}. "
                f"Today's high is {round(daily['temperature_2m_max'][0])} with a low of {round(daily['temperature_2m_min'][0])}, "
                f"and a {daily['precipitation_probability_max'][0] or 0} percent chance of rain.")

    def _rain(self, m):
        city = self._city(m)
        if not city:
            return "Which city should I check? Say, for example: my city is Chicago."
        try:
            place, data, _ = self._forecast(city)
        except Exception:
            return "I couldn't get the forecast right now. Check the internet connection."
        if place is None:
            return f"I couldn't find a place called {city}."
        when = (m.groupdict().get("when") or m.groupdict().get("when2") or "today").lower()
        day = 1 if when == "tomorrow" else 0
        chance = data["daily"]["precipitation_probability_max"][day] or 0
        label = "tomorrow" if day else "today"
        if chance >= 60:
            return f"Yes, rain is likely {label} in {place['name']}: {chance} percent chance."
        if chance >= 30:
            return f"Maybe. There's a {chance} percent chance of rain {label} in {place['name']}."
        return f"Probably not. Only a {chance} percent chance of rain {label} in {place['name']}."

    def _start_timer(self, seconds: float, message: str, label: str) -> None:
        due = self.now() + timedelta(seconds=seconds)
        entry = {"label": label, "due": due}

        def fire():
            with self._lock:
                if entry in self.timers:
                    self.timers.remove(entry)
            self.ui.notice(message)
            self.say(message)

        entry["thread"] = threading.Timer(seconds, fire)
        entry["thread"].daemon = True
        with self._lock:
            self.timers.append(entry)
        entry["thread"].start()

    def _timer(self, m):
        seconds = parse_duration(m.group("dur"))
        if seconds is None or seconds > 24 * 3600:
            return None
        label = f"{format_duration(seconds, adjective=True)} timer"
        self._start_timer(seconds, f"Time's up! Your {label} is done.", label)
        return f"Timer set for {format_duration(seconds)}."

    def _cancel_timers(self, m):
        with self._lock:
            entries, self.timers = self.timers, []
        for entry in entries:
            entry["thread"].cancel()
        if not entries:
            return "You don't have any timers or reminders."
        return "Cancelled your timer." if len(entries) == 1 else f"Cancelled all {len(entries)} timers and reminders."

    def _time_left(self, m):
        with self._lock:
            entries = sorted(self.timers, key=lambda e: e["due"])
        if not entries:
            return "You don't have any timers running."
        left = (entries[0]["due"] - self.now()).total_seconds()
        return f"{format_duration(max(left, 0))} left on your {entries[0]['label']}."

    def _when(self, kind: str, text: str):
        """(seconds from now, spoken when) for 'in <duration>' or 'at <clock time>'."""
        now = self.now()
        if kind == "in":
            seconds = parse_duration(text)
            return (seconds, f"in {format_duration(seconds)}") if seconds else None
        due = parse_clock(text, now)
        if due is None:
            return None
        day = "tomorrow " if due.date() != now.date() else ""
        return (due - now).total_seconds(), f"{day}at {say_time(due)}"

    def _remind(self, m):
        rest = m.group("rest").strip()
        found = None
        lead = re.fullmatch(r"(in|at) (?P<when>.+?)(?: to (?P<task>.+))?", rest, re.I)  # "in 20 minutes to …"
        if lead:
            when = self._when(lead.group(1).lower(), lead.group("when"))
            if when:
                found = when, (lead.group("task") or "").strip()
        if found is None:
            # "to <task> in|at <when>": try each " in "/" at ", last first, since the task can contain them
            body = re.sub(r"^to ", "", rest, flags=re.I)
            for split in reversed(list(re.finditer(r" (in|at) ", body, re.I))):
                when = self._when(split.group(1).lower(), body[split.end():])
                if when:
                    found = when, body[: split.start()].strip()
                    break
        if found is None:
            return None
        (seconds, when_text), task = found
        message = f"Reminder: {task}." if task else "This is your reminder."
        self._start_timer(seconds, message, f"reminder{' to ' + task if task else ''}")
        return f"Okay, I'll remind you {when_text}{' to ' + task if task else ''}."

    def _note(self, m):
        note = m.group("note").strip()
        path = self.notes_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- {self.now():%Y-%m-%d %H:%M} — {note}\n")
        return "Noted."

    def _read_notes(self, m):
        try:
            lines = [ln for ln in self.notes_path().read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]
        except OSError:
            lines = []
        if not lines:
            return "You don't have any notes yet. Say: take a note, then what to write."
        latest = [ln.split(" — ", 1)[-1].rstrip(".") for ln in lines[-5:]]
        intro = "Your latest notes: " if len(lines) > 5 else "Your notes: "
        return intro + ". ".join(latest) + "."

    def _volume(self, key: str, presses: int, reply: str):
        try:
            self.gui().press(key, presses=presses)
        except Exception:
            return "I can't change the volume on this computer."
        return reply

    def _volume_up(self, m):
        return self._volume("volumeup", 5, "Turned it up.")

    def _volume_down(self, m):
        return self._volume("volumedown", 5, "Turned it down.")

    def _mute(self, m):
        return self._volume("volumemute", 1, "Done.")

    def _screenshot(self, m):
        if self._grab is None:
            from .computer import _default_grab

            self._grab = _default_grab
        image = self._grab()
        folder = next((p for p in (self.home / "Desktop", self.home / "Pictures") if p.is_dir()), self.home)
        path = folder / f"Screenshot {self.now():%Y-%m-%d at %H.%M.%S}.png"
        image.save(path)
        return f"Saved a screenshot to your {folder.name if folder != self.home else 'home'} folder."

    def _search(self, m):
        q = m.group("q").strip()
        if m.group(0).lower().startswith("google ") and f"google {q.lower()}" in apps.SITES:
            return self._open_target(f"google {q}")  # "google docs" means the site, not a search for "docs"
        if re.match(r"(my |the |this |for )?(files?|folders?|computer|documents?|downloads|desktop|notes)\b", q, re.I) or re.search(
            r"\b(in|on) (my|this|the) (computer|files?|folders?|documents?|downloads|desktop|drive)\b", q, re.I
        ):
            return None  # searching the computer, not the web: the model does that with its file tools
        self.launcher("https://www.google.com/search?q=" + urllib.parse.quote_plus(q))
        self.ui.activity(f"searched the web for {q}")
        return f"Here are the results for {q}."

    def _open(self, m):
        return self._open_target(m.group("target"))

    def _open_target(self, spoken: str):
        found = apps.resolve(spoken, self.finder, self.home)
        if found is None:
            return None
        kind, target, label = found
        self.launcher(target)
        self.ui.activity(f"opened {target}")
        return f"Opening {label}."

    def _math(self, m):
        expr = spoken_math(m.group("expr"))
        if expr is None:
            return None
        try:
            value = safe_eval(expr)
        except ZeroDivisionError:
            return "That's undefined. You can't divide by zero."
        except (ValueError, SyntaxError, OverflowError, TypeError):
            return None
        return f"That's {format_number(value)}."
