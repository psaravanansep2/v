import pytest

from v.confirm import COMMAND, COMPUTER, SESSION, Confirmer, parse_yes_no


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Yes.", True),
        ("yeah go ahead", True),
        ("Sure, do it!", True),
        ("okay", True),
        ("y", True),
        ("No.", False),
        ("don't do that", False),
        ("yes, wait, no", False),
        ("hold on", False),
        ("cancel", False),
        ("", None),
        ("what was that?", None),
    ],
)
def test_parse_yes_no(reply, expected):
    assert parse_yes_no(reply) is expected


def test_policies():
    ask = lambda q: "yes"  # noqa: E731
    assert Confirmer("all", ask).needs(COMPUTER)
    risky = Confirmer("risky", ask)
    assert not risky.needs(COMPUTER) and risky.needs(COMMAND) and risky.needs(SESSION)
    none = Confirmer("none", ask)
    assert not any(none.needs(k) for k in (COMPUTER, COMMAND, SESSION))
    with pytest.raises(ValueError):
        Confirmer("sometimes", ask)


def test_unclear_answer_asks_again_then_defaults_to_no():
    replies = iter(["hmm", "yes"])
    assert Confirmer("all", lambda q: next(replies)).confirm("Run it?") is True
    replies = iter(["hmm", "banana"])
    assert Confirmer("all", lambda q: next(replies)).confirm("Run it?") is False
