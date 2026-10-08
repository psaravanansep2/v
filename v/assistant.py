"""What the front ends (terminal, phone) talk to: instant commands first,
the AI model for everything else."""

from __future__ import annotations

from typing import Callable, Optional

from .quick import QuickCommands


class Assistant:
    def __init__(self, agent, quick: Optional[QuickCommands], ui, say: Callable[[str], None]):
        self.agent = agent
        self.quick = quick
        self.ui = ui
        self.say = say

    def turn(self, text: str) -> None:
        reply = self.quick.handle(text) if self.quick else None
        if reply is None:
            self.agent.turn(text)
            return
        if reply:
            self.ui.text(reply)
            self.ui.text_end()
            self.say(reply)
        # the model sees what happened, so a follow-up like "make it 15 minutes" makes sense
        self.agent.remember(text, reply or "(stopped talking)")

    def cancel(self) -> None:
        self.agent.cancel()

    def on_session_finish(self, session) -> None:
        self.agent.on_session_finish(session)
