"""Background Claude Code sessions working toward the project goal.

Each session is a `claude -p` run in the project directory. It runs on its
own thread so the voice conversation carries on; when it finishes, the
`on_finish` callback fires so v can tell the user. Follow-ups resume the
same Claude Code conversation with `--resume <session_id>`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MAX_RESULT_CHARS = 6_000

SUMMARY_INSTRUCTION = (
    "When you finish, end with a short plain-language summary of two or three "
    "sentences: what you did, whether it worked, and anything the user needs to "
    "decide. It will be read aloud, so no markdown, lists or code in the summary."
)


@dataclass
class Session:
    id: int
    task: str
    summary: str
    status: str = "running"  # running | done | failed | stopped
    claude_session_id: Optional[str] = None
    result: str = ""
    cost_usd: Optional[float] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    stop_requested: bool = False

    def describe(self) -> str:
        elapsed = (self.finished_at or time.time()) - self.started_at
        lines = [f"Session {self.id} ({self.summary}): {self.status}, {elapsed / 60:.1f} min"]
        if self.cost_usd is not None:
            lines[0] += f", ~${self.cost_usd:.2f}"
        if self.result:
            # the summary is asked for at the end, so keep the tail
            result = self.result
            if len(result) > MAX_RESULT_CHARS:
                result = f"[...earlier output cut...]\n{result[-MAX_RESULT_CHARS:]}"
            lines.append(result)
        return "\n".join(lines)


class SessionManager:
    def __init__(
        self,
        project_dir: Path,
        goal: Callable[[], str],
        permission_mode: str = "acceptEdits",
        claude_bin: str = "claude",
        on_finish: Optional[Callable[[Session], None]] = None,
    ):
        self.project_dir = project_dir
        self.goal = goal
        self.permission_mode = permission_mode
        self.claude_bin = claude_bin
        self.on_finish = on_finish
        self._sessions: dict[int, Session] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    def _prompt(self, task: str) -> str:
        goal = self.goal().strip()
        parts = []
        if goal:
            parts.append(f"Overall project goal: {goal}")
        parts.append(f"Your task now: {task}")
        parts.append(SUMMARY_INSTRUCTION)
        return "\n\n".join(parts)

    def _command(self, prompt: str, resume_id: Optional[str]) -> list[str]:
        cmd = [self.claude_bin, "-p", prompt, "--output-format", "json", "--permission-mode", self.permission_mode]
        if resume_id:
            cmd += ["--resume", resume_id]
        return cmd

    def start(self, task: str, summary: str) -> Session:
        with self._lock:
            session = Session(id=self._next_id, task=task, summary=summary)
            self._next_id += 1
            self._sessions[session.id] = session
        self._launch(session, self._prompt(task), None)
        return session

    def resume(self, session_id: int, message: str) -> Session:
        session = self.get(session_id)
        if session.status == "running":
            raise RuntimeError(f"session {session_id} is still running; wait for it or stop it first")
        if not session.claude_session_id:
            raise RuntimeError(f"session {session_id} has no Claude Code session id to resume")
        session.status = "running"
        session.result = ""
        session.stop_requested = False
        session.started_at = time.time()
        session.finished_at = None
        self._launch(session, f"{message}\n\n{SUMMARY_INSTRUCTION}", session.claude_session_id)
        return session

    def _launch(self, session: Session, prompt: str, resume_id: Optional[str]) -> None:
        try:
            session.proc = subprocess.Popen(
                self._command(prompt, resume_id),
                cwd=self.project_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as e:
            self._finish(session, "failed", f"Could not start Claude Code ({self.claude_bin}): {e}")
            return
        threading.Thread(target=self._wait, args=(session,), daemon=True).start()

    def _wait(self, session: Session) -> None:
        out, err = session.proc.communicate()
        code = session.proc.returncode
        data = None
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            # The last line is the result if anything printed before it.
            for line in reversed((out or "").strip().splitlines()):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if isinstance(data, dict):
            session.claude_session_id = data.get("session_id") or session.claude_session_id
            session.cost_usd = data.get("total_cost_usd", session.cost_usd)
            result = str(data.get("result") or "").strip()
            failed = bool(data.get("is_error")) or code != 0
        else:
            result = (out or err or "").strip()[-2000:] or f"Claude Code exited with code {code}"
            failed = True
        if session.stop_requested:
            status = "stopped"
        else:
            status = "failed" if failed else "done"
        self._finish(session, status, result)

    def _finish(self, session: Session, status: str, result: str) -> None:
        session.status = status
        session.result = result
        session.finished_at = time.time()
        session.proc = None
        if self.on_finish:
            self.on_finish(session)

    def stop(self, session_id: int) -> Session:
        session = self.get(session_id)
        proc = session.proc
        if session.status != "running" or proc is None:
            return session
        session.stop_requested = True
        # SIGINT lets Claude Code end its turn cleanly so the session can be
        # resumed later; escalate if it doesn't exit.
        try:
            if os.name == "posix":
                proc.send_signal(signal.SIGINT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
        return session

    def get(self, session_id: int) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"no session {session_id}")
            return self._sessions[session_id]

    def all(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())

    def stop_all(self) -> None:
        for session in self.all():
            if session.status == "running":
                self.stop(session.id)
