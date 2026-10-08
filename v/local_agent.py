"""The conversation loop for the free brain (an open model on this computer).

Same job as agent.Agent (which uses Claude), adapted to small local models:
a short system prompt, plain function tools, a history window that fits a
16K-token context, and a filter that hides "thinking" text some open models
write before answering.
"""

from __future__ import annotations

import json
import platform
import re
import threading
from datetime import date
from typing import Callable

from .config import Config
from .local import LocalClient, LocalError, StreamAccumulator
from .speech import SentenceChunker
from .tools import Toolbox, parse_arguments
from .ui import Interrupted, TerminalUI

HISTORY_CHARS = 30_000  # ~8K tokens of conversation, leaving room for tools and the reply

_HIDDEN = re.compile(r"<(think|tool_call)>.*?(</\1>|$)", re.S)


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest end of `text` that could be the start of `tag`."""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class ThinkFilter:
    """Drops <think>…</think> and <tool_call>…</tool_call> sections from
    streamed text, so reasoning and tool calls written as text are never
    shown or spoken."""

    TAGS = ("think", "tool_call")

    def __init__(self):
        self.buf = ""
        self.inside: str = ""  # the tag we're inside, if any

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            if self.inside:
                close = f"</{self.inside}>"
                end = self.buf.find(close)
                if end == -1:
                    keep = _partial_suffix(self.buf, close)
                    self.buf = self.buf[len(self.buf) - keep:] if keep else ""
                    break
                self.buf = self.buf[end + len(close):]
                self.inside = ""
            else:
                starts = [(self.buf.find(f"<{t}>"), t) for t in self.TAGS]
                starts = [(i, t) for i, t in starts if i != -1]
                if not starts:
                    keep = max(_partial_suffix(self.buf, f"<{t}>") for t in self.TAGS)
                    out.append(self.buf[: len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                start, tag = min(starts)
                out.append(self.buf[:start])
                self.buf = self.buf[start + len(tag) + 2:]
                self.inside = tag
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        self.inside = ""
        return rest


def _json_objects(text: str) -> list:
    """Every top-level JSON value in text (handles ```json fences and several objects)."""
    decoder = json.JSONDecoder()
    found, i = [], 0
    text = re.sub(r"```(?:json)?", "", text)
    while i < len(text):
        brace = min((j for j in (text.find("{", i), text.find("[", i)) if j != -1), default=-1)
        if brace == -1:
            break
        try:
            value, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            i = brace + 1
            continue
        found.append(value)
        i = end
    return found


def text_tool_calls(content: str, known: Callable[[str], bool]) -> tuple[list[dict], str]:
    """Tool calls a model wrote as text instead of real tool calls: Hermes/Qwen
    <tool_call> tags, or a bare JSON object like {"name": ..., "arguments": ...}.
    Returns (calls in OpenAI format, the remaining text)."""
    blocks = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", content, re.S)
    rest = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", content, flags=re.S)
    candidates = []
    for block in blocks:
        candidates += _json_objects(block)
    if not blocks and rest.lstrip().startswith(("{", "[", "```")):
        candidates = _json_objects(rest)
        if candidates:
            rest = ""
    calls = []
    for value in candidates:
        for obj in value if isinstance(value, list) else [value]:
            if not isinstance(obj, dict):
                continue
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            name = fn.get("name") or fn.get("tool") or (obj.get("function") if isinstance(obj.get("function"), str) else None)
            args = next((fn[k] for k in ("arguments", "parameters", "args", "input") if k in fn), {})
            if not isinstance(name, str) or not known(name):
                continue
            calls.append({
                "id": f"text_call_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
            })
    if not calls:
        return [], content
    return calls, rest.strip()


def _plain(text: str) -> str:
    return text.replace("**", "").replace("`", "")


def system_prompt(cfg: Config, has_screen: bool, suffix: str = "") -> str:
    goal = cfg.goal.strip() or "not set yet; if the user mentions what they're working toward, save it with set_project_goal"
    screen = (
        "To use an app on screen: look_at_screen, then click things by their number, type_text, press_keys. "
        "Look again afterwards to check it worked."
        if has_screen
        else "Screen control is off."
    )
    return "\n".join(
        [
            f"You are v, a helpful voice assistant running on the user's own computer ({platform.system()}).",
            "The user talks to you out loud and hears your replies, so:",
            "- Answer in one to three short, plain sentences. No markdown, lists, code blocks or links.",
            "- Use your tools to get things done instead of explaining how. Check the result before saying it worked.",
            "- You can build things: web pages, documents, scripts, small programs. Write the files in the project "
            "folder, then open what you made so the user sees it. get_images finds free photos for it.",
            "- Ask at most one short question before starting; if details are missing, pick sensible ones and say what "
            "you chose.",
            "- If the user says something you made doesn't work, read your files to find out why and fix it.",
            "- You can't see pictures; read_file only gives the words in them. Don't guess what a picture shows.",
            "- If a tool result says the user declined, don't try again; ask what they'd like instead.",
            f"Project folder: {cfg.project_dir}",
            f"Project goal: {goal}",
            screen,
            f"Today is {date.today():%A, %B %d, %Y}.",
            suffix,
        ]
    ).strip()


class LocalAgent:
    def __init__(
        self,
        client: LocalClient,
        cfg: Config,
        say: Callable[[str], None],
        toolbox: Toolbox,
        ui=None,
        stop_speaking: Callable[[], None] = lambda: None,
    ):
        self.client = client
        self.cfg = cfg
        self.say = say
        self.toolbox = toolbox
        self.ui = ui or TerminalUI()
        self.stop_speaking = stop_speaking
        self.tools = toolbox.definitions()
        self.history: list[dict] = []
        self._turn_start = 0
        self._cancel = threading.Event()

    # same surface as agent.Agent, so the terminal and phone front ends work with either brain

    def cancel(self) -> None:
        self._cancel.set()

    def on_session_finish(self, session) -> None:  # the free brain has no background sessions
        pass

    def remember(self, user_text: str, reply: str) -> None:
        """Record an exchange handled without the model (an instant command)."""
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Interrupted()

    def turn(self, user_text: str) -> None:
        self._cancel.clear()
        self.ui.busy(True)
        try:
            self._turn(user_text)
        except (KeyboardInterrupt, Interrupted):
            self.stop_speaking()
            self._close_dangling()
            self.ui.notice("interrupted")
        finally:
            self.ui.busy(False)

    def _turn(self, user_text: str) -> None:
        self._turn_start = len(self.history)
        self.history.append({"role": "user", "content": user_text})
        for _ in range(self.cfg.max_tool_rounds):
            msg = self._stream_once()
            if msg is None:
                return
            self.history.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return
            for call in calls:
                name = call["function"]["name"]
                if self._cancel.is_set():
                    result = "Interrupted by the user before this ran."
                else:
                    try:
                        result = self.toolbox.run(name, parse_arguments(call["function"]["arguments"]))
                    except ValueError as e:
                        result = f"Error: the arguments weren't valid JSON ({e}). Try again."
                self.history.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})
            self._check_cancel()
        self.say("I've taken a lot of steps on this. Say continue if you want me to keep going.")

    def _messages(self) -> list[dict]:
        system = {"role": "system", "content": system_prompt(self.cfg, self.toolbox.screen is not None, self.client.server.system_suffix)}
        return [system] + self._window()

    def _window(self) -> list[dict]:
        """The most recent history that fits, never starting on a tool result
        (which would lose the call it answers)."""
        size = lambda m: len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))  # noqa: E731
        total, start = 0, len(self.history)
        for i in range(len(self.history) - 1, -1, -1):
            total += size(self.history[i])
            if total > HISTORY_CHARS:
                break
            start = i
        if start <= self._turn_start:
            while start < len(self.history) and self.history[start]["role"] != "user":
                start += 1
            return self.history[start:]
        # This turn alone is too long: keep its request, then the latest steps
        # starting at an assistant message.
        while start < len(self.history) and self.history[start]["role"] != "assistant":
            start += 1
        return [self.history[self._turn_start]] + self.history[start:]

    def _stream_once(self):
        acc, think, chunker = StreamAccumulator(), ThinkFilter(), SentenceChunker()
        printed = False
        held = ""  # text that starts like JSON: maybe a tool call written as text, so don't speak it yet
        holding = None  # decided on the first visible character

        def show(text: str) -> None:
            nonlocal printed, held, holding
            if not text:
                return
            if holding is None and text.strip():
                holding = text.lstrip()[:1] in ("{", "[", "`")
            if holding:
                held += text
                return
            text = _plain(text)  # replies are heard: no `code` or **bold** marks
            self.ui.text(text)
            printed = True
            for sentence in chunker.feed(text):
                self.say(sentence)

        try:
            for chunk in self.client.stream_chat(self._messages(), self.tools):
                self._check_cancel()
                show(think.feed(acc.feed(chunk)))
            show(think.flush())
        except LocalError as e:
            self.ui.error(str(e))
            self.say("My local model isn't responding. Check that it's still running on the computer.")
            return None
        msg = acc.message()
        content = msg["content"]
        if not msg.get("tool_calls"):
            calls, content = text_tool_calls(content, self._is_tool)
            if calls:
                msg["tool_calls"] = calls
                held = ""
        if held:  # it was ordinary text after all
            holding = False
            show(held)
        for sentence in chunker.flush():
            self.say(sentence)
        if printed:
            self.ui.text_end()
        msg["content"] = _HIDDEN.sub("", content).strip()
        return msg

    def _is_tool(self, name: str) -> bool:
        from .tools import resolve_call

        known = self.toolbox.names()
        return resolve_call(name, {}, known)[0] in known

    def _close_dangling(self) -> None:
        """Answer tool calls left without results, so the history stays valid."""
        if not self.history or self.history[-1]["role"] not in ("assistant", "tool"):
            return
        # find the last assistant message with tool calls and the results it got
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            if msg["role"] == "assistant":
                answered = {m.get("tool_call_id") for m in self.history[i + 1:]}
                for call in msg.get("tool_calls") or []:
                    if call["id"] not in answered:
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call["function"]["name"],
                            "content": "Interrupted by the user before this ran.",
                        })
                return
