"""Settings for one v run, plus the per-project goal file.

The goal lives in <project>/.v/goal.txt so it survives restarts — every new
run (and every Claude Code session it starts) picks up where the last left
off instead of starting from a blank slate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"
CONFIRM_POLICIES = ("all", "risky", "none")


@dataclass
class Config:
    project_dir: Path
    goal: str = ""
    model: str = DEFAULT_MODEL
    # Opus 5.5 defaults to medium; it keeps spoken replies snappy while still
    # handling multi-step computer use. Raise it for harder autonomous work.
    effort: str = "medium"
    # all   = ask before every action that changes something (clicks, typing,
    #         commands, sessions)
    # risky = ask before shell commands and Claude Code sessions; screen
    #         control runs freely (default)
    # none  = never ask
    confirm: str = "risky"
    computer: bool = True
    # Permission mode passed to `claude -p` for background sessions.
    session_permission_mode: str = "acceptEdits"
    claude_bin: str = "claude"
    stt_model: str = "base.en"
    voice: str = "af_heart"
    max_tool_rounds: int = 60
    command_timeout_s: int = 120


def goal_path(project_dir: Path) -> Path:
    return project_dir / ".v" / "goal.txt"


def load_goal(project_dir: Path) -> str:
    try:
        return goal_path(project_dir).read_text().strip()
    except OSError:
        return ""


def save_goal(project_dir: Path, goal: str) -> None:
    path = goal_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(goal.strip() + "\n")
