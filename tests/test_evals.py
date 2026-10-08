"""v eval's tasks: every known-good solution passes its check, and a model
that doesn't do the job fails, so the scores mean something."""

import json

import pytest

from tests.test_local import FakeModelServer, text_chunks
from v import evals


@pytest.mark.parametrize("split", ["eval", "train"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_known_good_solution_passes(seed, split):
    tasks = evals.make_tasks(seed=seed, split=split)
    assert {t.family for t in tasks} >= {"read a file", "find a file", "write a file", "answer from the web", "respect a no"}
    for task in tasks:
        result = evals.run_task(task, evals.ScriptedClient(task.solution))
        assert result.passed, f"{task.family}: {task.prompt} -> {result.reason}"
        assert result.style_ok, task.family


class Unhelpful:
    """A model that answers without doing anything."""

    def __init__(self, reply="Sorry, I can't help with that."):
        self.reply = reply
        self.server = type("S", (), {"system_suffix": ""})()

    def stream_chat(self, messages, tools):
        yield {"choices": [{"delta": {"content": self.reply}, "finish_reason": "stop"}]}


def test_a_model_that_does_nothing_fails_every_kind_of_task():
    for task in evals.make_tasks(seed=5):
        assert not evals.run_task(task, Unhelpful()).passed, task.family


def test_wrong_actions_fail():
    rng_tasks = {t.family: t for t in evals.make_tasks(seed=11)}
    click = rng_tasks["click a button"]
    target = next(step for step in click.solution if step[0] == "say")[1].split("clicked ")[1].rstrip(".")
    wrong = next(n for n in ("Save", "Don't Save", "Cancel") if n != target)
    steps = [evals.call("look_at_screen"), evals.call("click", element=evals._element(wrong)), evals.say("Done.")]
    assert not evals.run_task(click, evals.ScriptedClient(steps)).passed

    refuse = rng_tasks["respect a no"]
    pushy = refuse.solution[:1] * 4 + [evals.say("I've deleted it.")]
    result = evals.run_task(refuse, evals.ScriptedClient(pushy))
    assert not result.passed and "asked 4 times" in result.reason


def test_training_never_sees_the_eval_wordings():
    options = list(range(9))
    eval_seen = {evals.pick(__import__("random").Random(i), "eval", options) for i in range(200)}
    train_seen = {evals.pick(__import__("random").Random(i), "train", options) for i in range(200)}
    assert eval_seen == {2, 5, 8} and not (eval_seen & train_seen)


@pytest.mark.parametrize("command,ok", [
    ("ls -la photos", True), ("cat notes/todo.txt", True), ("wc -l data.csv", True), ("find . -name '*.pdf'", True),
    ("rm -rf old-logs", False), ("ls > out.txt", False), ("ls; rm x", False), ("cat a && rm b", False),
    ("find . -delete", False), ("python script.py", False), ("echo $HOME", False),
])
def test_only_commands_that_look_are_allowed_during_an_eval(command, ok):
    assert evals._looks_read_only(f"Run this command: {command}?") is ok
    assert not evals._looks_read_only("Write the file /etc/hosts?")


def test_spoken_style():
    assert evals.spoken_style("The total is $4.50.")
    assert not evals.spoken_style("**Total:** $4.50")
    assert not evals.spoken_style("Here you go:\n- eggs\n- milk")
    assert not evals.spoken_style("")
    assert not evals.spoken_style("word " * 120)


def test_v_eval_command(tmp_path):
    from v.cli import main

    everything = " ".join(answer for _, answer, _ in evals.QA)  # whichever question comes up, the answer is in here
    srv = FakeModelServer([text_chunks(everything)])
    out = tmp_path / "results.json"
    try:
        assert main(["eval", "--families", "question", "--local-url", srv.url, "--json", str(out)]) == 0
    finally:
        srv.close()
    data = json.loads(out.read_text())
    assert data["total"] == 1 and data["passed"] == 1 and data["tasks"][0]["family"] == "answer a question"
    with pytest.raises(SystemExit, match="unknown task kinds: nope"):
        main(["eval", "--families", "nope"])
