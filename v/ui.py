"""Where the agent's output goes: the terminal, the phone, or both.

The agent reports what happens as typed events (reply text, progress notes,
commands it runs, session notices, screenshots) rather than printing, so the
same conversation can drive the terminal and the phone page at once.
"""

from __future__ import annotations


class Interrupted(Exception):
    """The user asked v to stop what it's doing (Ctrl+C or the phone's Stop)."""


class TerminalUI:
    def __init__(self, out=print):
        self.out = out
        self._mid_line = False

    def user(self, text: str) -> None:
        pass  # the terminal already shows what was typed or heard

    def text(self, delta: str) -> None:
        self.out(delta, end="", flush=True)
        self._mid_line = True

    def text_end(self) -> None:
        if self._mid_line:
            self.out("")
            self._mid_line = False

    def _line(self, line: str) -> None:
        if self._mid_line:
            self.out("")
            self._mid_line = False
        self.out(line, flush=True)

    def progress(self, note: str) -> None:
        self._line(f"… {note}")

    def activity(self, line: str) -> None:
        self._line(line)

    def notice(self, line: str) -> None:
        # notices can arrive while the terminal sits at a "you>" prompt
        self._line(f"\n[{line}]")

    def error(self, message: str) -> None:
        self._line(f"({message})")

    def image(self, png_b64: str) -> None:
        pass

    def busy(self, busy: bool) -> None:
        pass

    def goal(self, goal: str) -> None:
        self._line(f"[goal: {goal}]")


class TeeUI:
    """Sends every event to several UIs."""

    def __init__(self, *uis):
        self.uis = uis

    def __getattr__(self, name):
        def call(*args, **kwargs):
            for ui in self.uis:
                getattr(ui, name)(*args, **kwargs)

        return call
