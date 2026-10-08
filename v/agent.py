"""The conversation with Claude and the tool loop behind each spoken turn.

One `turn()` = the user said something; Claude may answer straight away or
work through many tool calls (looking at the screen, clicking, running a
command, starting a Claude Code session) before it replies. Text is spoken
as it streams in, and so are the short progress notes Claude writes between
tool calls, so the user hears what's happening during long actions.

The message history is append-only: earlier turns are never edited or
pruned, because Claude Opus 5.5 ties its thinking blocks to the exact
conversation that produced them. Old screenshots are cleared server-side by
context editing instead.
"""

from __future__ import annotations

import platform
import re
import threading
from typing import Callable, Optional

import anthropic

from . import computer as computer_mod
from .computer import Computer, describe, run_batch
from .config import Config, save_goal
from .confirm import COMMAND, COMPUTER, SESSION, Confirmer
from .sessions import Session, SessionManager
from .shell import run_command
from .speech import SentenceChunker

BETAS = [
    "thinking-display-updates-2026-08-18",  # progress notes between tool calls, as text
    "server-side-fallback-2026-07-01",  # fallbacks="default" if a request is declined
    "context-management-2025-06-27",  # clear old tool results (screenshots) server-side
]

MAX_TOKENS = 32_000


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        # Inputs stream as they're generated; validate() checks them before use.
        "eager_input_streaming": True,
    }


SUMMARY_PROP = {
    "type": "string",
    "description": "A short spoken summary of the task, under 15 words, used when asking the user to approve it.",
}

CUSTOM_TOOLS = [
    _tool(
        "run_command",
        "Run one shell command in a fresh shell in the project directory and return its output. "
        "For quick checks and small actions: git status, listing files, opening an app. "
        "cd and environment variables do not carry over between calls. "
        "Use start_session instead for coding work.",
        {"command": {"type": "string", "description": "The command line to run."}},
        ["command"],
    ),
    _tool(
        "start_session",
        "Start a Claude Code session in the project directory, running in the background. Use it for "
        "coding work: writing or changing code, fixing bugs, running and fixing tests, investigating the "
        "codebase. The session can read and edit files and run commands. It does not see this conversation, "
        "so the task must be complete and self-contained. Returns a session id immediately; the result "
        "arrives later as a [Session update] note.",
        {
            "task": {"type": "string", "description": "Full, self-contained instructions for the session."},
            "summary": SUMMARY_PROP,
        },
        ["task", "summary"],
    ),
    _tool(
        "session_status",
        "Get the status and latest result of one Claude Code session, or of all sessions if no id is given.",
        {"session_id": {"type": "integer"}},
        [],
    ),
    _tool(
        "continue_session",
        "Send a follow-up to a finished Claude Code session. It keeps everything it learned last time.",
        {
            "session_id": {"type": "integer"},
            "message": {"type": "string", "description": "The follow-up instructions."},
            "summary": SUMMARY_PROP,
        },
        ["session_id", "message", "summary"],
    ),
    _tool(
        "stop_session",
        "Stop a running Claude Code session. It can be continued later.",
        {"session_id": {"type": "integer"}},
        ["session_id"],
    ),
    _tool(
        "set_project_goal",
        "Save a new overall goal for the project. Future sessions are given this goal, and it is remembered "
        "the next time v starts. Use it when the user states or changes what the project is trying to achieve.",
        {"goal": {"type": "string"}},
        ["goal"],
    ),
]

_TYPES = {"string": str, "integer": int}


def validate(tool: dict, args) -> Optional[str]:
    """Error message if `args` doesn't match the tool's schema, else None.
    Needed because eager input streaming skips server-side validation."""
    if not isinstance(args, dict):
        return "Tool input must be a JSON object."
    schema = tool["input_schema"]
    for key in schema["required"]:
        if key not in args:
            return f"Missing required field {key!r}."
    for key, value in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            return f"Unknown field {key!r}."
        expected = _TYPES[spec["type"]]
        if not isinstance(value, expected) or isinstance(value, bool):
            return f"Field {key!r} must be a {spec['type']}."
        if expected is str and not value.strip():
            return f"Field {key!r} must not be empty."
    return None


def build_system_prompt(cfg: Config, computer: Optional[Computer]) -> str:
    goal = cfg.goal.strip() or "(not set yet: ask the user what they're trying to achieve, then save it with set_project_goal)"
    lines = [
        "You are v, a voice assistant running on the user's own computer. The user talks to you out loud, "
        "and everything you write as a reply is read aloud by a text-to-speech voice.",
        "",
        "How to talk:",
        "- Keep replies short and conversational, usually one to three sentences.",
        "- No markdown, bullet lists, code blocks, tables or URLs in replies; they can't be read aloud. "
        "Say file names and commands in plain words.",
        "- Before a multi-step action, say in one short sentence what you're about to do. When it's done, "
        "say briefly what happened.",
        "- If a request is ambiguous and the wrong reading would waste effort or change something, ask one short "
        "question first.",
        "- Speech recognition makes mistakes. If a request sounds garbled, check what the user meant.",
        "",
        "The project:",
        f"- Directory: {cfg.project_dir}",
        f"- Goal when this conversation started: {goal}",
        "Keep your suggestions and the sessions you start pointed at this goal. If the user changes it, save the "
        "new one with set_project_goal.",
        "",
        "Your tools:",
        "- start_session runs Claude Code in the project directory in the background. Delegate coding work to it "
        "rather than doing it through the screen or shell yourself. Write the task so it stands on its own. You can "
        "keep talking with the user while it runs. Its result arrives in a [Session update] note at the start of a "
        "later user message; these notes come from v, not from the user. Tell the user the outcome briefly.",
        "- run_command runs one quick shell command in the project directory.",
    ]
    if computer is not None:
        lines.append(
            "- The computer tools let you see the screen and use the mouse and keyboard, for anything in a "
            "graphical app. Take a screenshot before acting and again afterwards to check the result. "
            f"Screenshots are {computer.model_width}x{computer.model_height} pixels; use coordinates in that space."
        )
    else:
        lines.append("- Screen control is turned off for this session.")
    lines += [
        "- Some actions need the user's spoken approval. If the result says the user declined, don't retry the "
        "same thing; ask what they'd like instead.",
        "",
        f"Environment: {platform.system()} {platform.release()}.",
    ]
    return "\n".join(lines)


def spoken_summary(result: str, limit: int = 300) -> str:
    """The closing paragraph of a session's result (where the summary goes),
    trimmed to a couple of sentences."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", result) if p.strip()]
    if not paragraphs:
        return ""
    text = paragraphs[-1]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences:
        if out and len(out) + len(s) > limit:
            break
        out = f"{out} {s}".strip()
    return out[:limit]


class Agent:
    def __init__(
        self,
        client: anthropic.Anthropic,
        cfg: Config,
        say: Callable[[str], None],
        confirmer: Confirmer,
        computer: Optional[Computer],
        sessions: SessionManager,
        out: Callable[..., None] = print,
        stop_speaking: Callable[[], None] = lambda: None,
    ):
        self.client = client
        self.cfg = cfg
        self.say = say
        self.confirmer = confirmer
        self.computer = computer
        self.sessions = sessions
        self.out = out
        self.stop_speaking = stop_speaking
        self.messages: list[dict] = []
        self._notes: list[str] = []
        self._notes_lock = threading.Lock()
        self.system = build_system_prompt(cfg, computer)
        self.tools = list(CUSTOM_TOOLS)
        if computer is not None:
            self.tools.append({"type": "computer_toolset_20260801"})
        self._tool_specs = {t["name"]: t for t in CUSTOM_TOOLS}

    # --- background session updates ---

    def on_session_finish(self, session: Session) -> None:
        with self._notes_lock:
            self._notes.append(f"[Session update] {session.describe()}")
        verb = {"done": "finished", "failed": "failed", "stopped": "stopped"}.get(session.status, session.status)
        line = f"Session {session.id} {verb}."
        summary = spoken_summary(session.result) if session.status != "stopped" else ""
        self.out(f"\n[{line}]")
        self.say(f"{line} {summary}".strip())

    def _take_notes(self) -> list[str]:
        with self._notes_lock:
            notes, self._notes = self._notes, []
        return notes

    # --- the turn loop ---

    def _params(self) -> dict:
        return dict(
            model=self.cfg.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": self.cfg.effort},
            cache_control={"type": "ephemeral"},
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
            fallbacks="default",
            betas=BETAS,
        )

    def turn(self, user_text: str) -> None:
        notes = self._take_notes()
        content = "\n\n".join(notes + [user_text]) if notes else user_text
        self.messages.append({"role": "user", "content": content})
        rounds = 0
        try:
            while True:
                msg = self._stream_once()
                if msg is None:
                    return
                self.messages.append({"role": "assistant", "content": msg.content})
                tool_uses = [b for b in msg.content if b.type == "tool_use"]

                if msg.stop_reason == "pause_turn":
                    continue
                if msg.stop_reason == "refusal":
                    self._close_dangling("Not executed: the response was declined.")
                    self.say("Sorry, I can't help with that one.")
                    return
                if msg.stop_reason == "max_tokens" and tool_uses:
                    # The last tool call's input may be cut off; don't run any of them.
                    self._close_dangling("Not executed: the response hit the length limit. Try again in smaller steps.")
                elif not tool_uses:
                    return
                else:
                    self.messages.append({"role": "user", "content": self._run_tools(tool_uses)})

                rounds += 1
                if rounds >= self.cfg.max_tool_rounds:
                    self.say("I've taken a lot of steps on this. Say continue if you want me to keep going.")
                    return
        except KeyboardInterrupt:
            self.stop_speaking()
            self._close_dangling("Interrupted by the user before this ran.")
            self.out("\n(interrupted)")

    def _stream_once(self):
        chunker = SentenceChunker()
        json_retries = 0
        while True:
            printed = False
            try:
                with self.client.beta.messages.stream(**self._params()) as stream:
                    for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            self.out(event.delta.text, end="", flush=True)
                            printed = True
                            for sentence in chunker.feed(event.delta.text):
                                self.say(sentence)
                        elif event.type == "content_block_stop":
                            block = event.content_block
                            if block.type == "text":
                                for sentence in chunker.flush():
                                    self.say(sentence)
                            elif block.type == "thinking" and block.thinking.strip():
                                # a progress note (display="updates"); reasoning itself stays hidden
                                self.out(f"\n… {block.thinking.strip()}", flush=True)
                                self.say(block.thinking)
                    msg = stream.get_final_message()
                if printed:
                    self.out("")
                return msg
            except ValueError:
                # Tool-input JSON the SDK couldn't parse at all (eager input
                # streaming). The tool_use never completed, so re-issue the turn.
                json_retries += 1
                if json_retries > 2:
                    self.say("Something went wrong reading my own tool call. Please try again.")
                    return None
            except anthropic.AuthenticationError:
                raise
            except anthropic.RateLimitError:
                self.say("I'm being rate limited right now. Give it a moment and ask again.")
                return None
            except anthropic.APIStatusError as e:
                self.out(f"\n(API error {e.status_code}: {e.message})")
                self.say("I hit an error talking to Claude. The details are in the terminal.")
                return None
            except anthropic.APIConnectionError:
                self.say("I can't reach the Claude API. Check the internet connection.")
                return None

    def _close_dangling(self, reason: str) -> None:
        """If the last message is an assistant turn with unanswered tool
        calls, answer each with an error so the history stays valid."""
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return
        tool_uses = [b for b in self.messages[-1]["content"] if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            return
        results = []
        for b in tool_uses:
            r = {"type": "tool_result", "tool_use_id": b.id, "content": reason, "is_error": True}
            if getattr(b, "toolset_name", None):
                r["toolset_name"] = b.toolset_name
            results.append(r)
        self.messages.append({"role": "user", "content": results})

    # --- tools ---

    def _approve_computer(self, blocks: list) -> bool:
        if not self.confirmer.needs(COMPUTER):
            return True
        steps = ", then ".join(describe(b.name, computer_mod.block_input(b)) for b in blocks[:4])
        if len(blocks) > 4:
            steps += f", and {len(blocks) - 4} more steps"
        return self.confirmer.confirm(f"I'm about to {steps}. Okay?")

    def _run_tools(self, tool_uses: list) -> list[dict]:
        computer_blocks = [b for b in tool_uses if getattr(b, "toolset_name", None) == computer_mod.TOOLSET_NAME]
        results: list[dict] = []
        computer_done = False
        for block in tool_uses:
            if getattr(block, "toolset_name", None) == computer_mod.TOOLSET_NAME:
                # The whole batch runs in order where the first one appears.
                if not computer_done:
                    results += run_batch(self.computer, computer_blocks, self._approve_computer)
                    computer_done = True
                continue
            results.append(self._run_custom(block))
        return results

    def _run_custom(self, block) -> dict:
        def result(content: str, is_error: bool = False) -> dict:
            r = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                r["is_error"] = True
            return r

        spec = self._tool_specs.get(block.name)
        if spec is None:
            return result(f"Unknown tool {block.name!r}.", True)
        args = computer_mod.block_input(block)
        error = validate(spec, args)
        if error:
            return result(error, True)
        try:
            return result(*self._dispatch(block.name, args))
        except (KeyError, RuntimeError) as e:
            return result(str(e).strip("'\""), True)
        except Exception as e:  # report to Claude rather than ending the conversation
            return result(f"Error: {e.__class__.__name__}: {e}", True)

    def _dispatch(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "run_command":
            command = args["command"]
            if self.confirmer.needs(COMMAND) and not self.confirmer.confirm(f"Run this command: {command}?"):
                return "The user declined to run this command.", True
            self.out(f"\n$ {command}")
            return run_command(command, self.cfg.project_dir, self.cfg.command_timeout_s)

        if name == "start_session":
            if self.confirmer.needs(SESSION) and not self.confirmer.confirm(
                f"Start a Claude Code session to {args['summary']}?"
            ):
                return "The user declined to start this session.", True
            session = self.sessions.start(args["task"], args["summary"])
            if session.status == "failed":
                return session.result, True
            self.out(f"\n[session {session.id} started: {session.summary}]")
            return f"Started session {session.id}. It runs in the background; its result will arrive as a [Session update].", False

        if name == "session_status":
            if "session_id" in args:
                return self.sessions.get(args["session_id"]).describe(), False
            sessions = self.sessions.all()
            if not sessions:
                return "No sessions have been started yet.", False
            return "\n\n".join(s.describe() for s in sessions), False

        if name == "continue_session":
            sid = args["session_id"]
            if self.confirmer.needs(SESSION) and not self.confirmer.confirm(
                f"Send session {sid} a follow-up to {args['summary']}?"
            ):
                return "The user declined this follow-up.", True
            self.sessions.resume(sid, args["message"])
            self.out(f"\n[session {sid} continued: {args['summary']}]")
            return f"Session {sid} is running again with the follow-up.", False

        if name == "stop_session":
            session = self.sessions.stop(args["session_id"])
            return f"Session {session.id} is {'stopping' if session.status == 'running' else session.status}.", False

        if name == "set_project_goal":
            goal = args["goal"].strip()
            self.cfg.goal = goal
            save_goal(self.cfg.project_dir, goal)
            return f"Project goal saved: {goal}", False

        raise KeyError(f"unhandled tool {name}")
