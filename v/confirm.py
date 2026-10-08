"""Asking the user before v does something with consequences.

Which actions need a yes depends on the policy (see Config.confirm). The
question is spoken (or printed in text mode) and the answer is whatever the
user says next; anything that isn't a clear yes counts as no.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# kinds of action, from the agent's point of view
COMPUTER = "computer"  # mouse/keyboard input
COMMAND = "command"  # shell command
SESSION = "session"  # starting or continuing a Claude Code session

_NEEDS = {
    "all": {COMPUTER, COMMAND, SESSION},
    "risky": {COMMAND, SESSION},
    "none": set(),
}

_NO = re.compile(r"\b(no|nope|nah|don'?t|do not|stop|cancel|wait|never|negative|hold on)\b")
_YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|go for it|confirm(ed)?|proceed|affirmative|please do|y|absolutely|run it|sounds good)\b"
)


def parse_yes_no(reply: str) -> Optional[bool]:
    text = reply.lower().strip()
    if not text:
        return None
    # negatives win: "yes, wait, no" and "don't do it" are both a no
    if _NO.search(text):
        return False
    if _YES.search(text):
        return True
    return None


class Confirmer:
    def __init__(self, policy: str, ask: Callable[[str], str]):
        if policy not in _NEEDS:
            raise ValueError(f"unknown confirm policy {policy!r}; choose from {sorted(_NEEDS)}")
        self.policy = policy
        self.ask = ask

    def needs(self, kind: str) -> bool:
        return kind in _NEEDS[self.policy]

    def confirm(self, question: str) -> bool:
        reply = self.ask(question)
        answer = parse_yes_no(reply)
        if answer is None:
            answer = parse_yes_no(self.ask("Sorry, was that a yes or a no?"))
        return bool(answer)
