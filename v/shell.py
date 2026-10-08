"""One-shot shell commands in the project directory."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MAX_OUTPUT = 20_000


def run_command(command: str, cwd: Path, timeout_s: int = 120) -> tuple[str, bool]:
    """Returns (output, is_error). Each call is a fresh shell, so `cd` and
    variables don't carry over between calls."""
    shell = ["bash", "-lc", command] if os.name == "posix" else ["cmd", "/c", command]
    try:
        proc = subprocess.run(
            shell,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") if isinstance(e.stdout, str) else ""
        return f"Timed out after {timeout_s}s.\n{partial[-MAX_OUTPUT:]}", True
    except OSError as e:
        return f"Could not run command: {e}", True
    out = proc.stdout
    if proc.stderr:
        out += ("\n" if out else "") + proc.stderr
    if len(out) > MAX_OUTPUT:
        out = f"[...{len(out) - MAX_OUTPUT} characters cut...]\n" + out[-MAX_OUTPUT:]
    out = out.strip() or "(no output)"
    return f"exit code {proc.returncode}\n{out}", proc.returncode != 0
