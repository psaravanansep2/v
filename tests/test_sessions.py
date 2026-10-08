import json
import sys
import threading
import time

import os

import pytest

from v.sessions import SessionManager

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the fake claude is a POSIX shell script")

FAKE_CLAUDE = r'''
import json, os, sys, time, signal
args = sys.argv[1:]
prompt = args[args.index("-p") + 1]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"args": args, "cwd": os.getcwd()}) + "\n")
if "SLEEP" in prompt:
    def done(*_):
        print(json.dumps({"type": "result", "result": "Interrupted.", "session_id": "s-1", "is_error": False}))
        sys.exit(0)
    signal.signal(signal.SIGINT, done)
    time.sleep(30)
if "FAIL" in prompt:
    print(json.dumps({"type": "result", "result": "Tests still failing.", "session_id": "s-1", "is_error": True}))
    sys.exit(1)
print("some log line")
print(json.dumps({"type": "result", "result": "Did the work.\n\nAdded the endpoint and the tests pass.",
                  "session_id": "s-1", "is_error": False, "total_cost_usd": 0.42}))
'''


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE)
    launcher = tmp_path / "claude"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    project = tmp_path / "project"
    project.mkdir()
    return launcher, log, project


def manager(fake_claude, goal="Ship the API"):
    launcher, log, project = fake_claude
    finished = []
    done = threading.Event()

    def on_finish(s):
        finished.append((s.id, s.status))
        done.set()

    m = SessionManager(project, goal=lambda: goal, claude_bin=str(launcher), on_finish=on_finish)
    return m, finished, done, log, project


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_session_runs_in_project_with_goal_and_reports_back(fake_claude):
    m, finished, done, log, project = manager(fake_claude)
    s = m.start("Add a health endpoint", "add a health endpoint")
    assert s.status == "running"
    assert done.wait(10)
    assert finished == [(1, "done")]
    assert s.claude_session_id == "s-1"
    assert s.cost_usd == 0.42
    assert s.result.endswith("the tests pass.")
    call = calls(log)[0]
    assert call["cwd"] == str(project)
    prompt = call["args"][call["args"].index("-p") + 1]
    assert "Overall project goal: Ship the API" in prompt
    assert "Your task now: Add a health endpoint" in prompt
    assert call["args"][call["args"].index("--permission-mode") + 1] == "acceptEdits"
    assert "--resume" not in call["args"]


def test_follow_up_resumes_the_same_claude_session(fake_claude):
    m, finished, done, log, _ = manager(fake_claude)
    s = m.start("first", "first")
    assert done.wait(10)
    done.clear()
    m.resume(s.id, "now add docs")
    assert done.wait(10)
    args = calls(log)[1]["args"]
    assert args[args.index("--resume") + 1] == "s-1"
    assert finished == [(1, "done"), (1, "done")]


def test_cannot_resume_while_running(fake_claude):
    m, _, done, _, _ = manager(fake_claude)
    s = m.start("SLEEP", "sleep")
    time.sleep(0.5)
    with pytest.raises(RuntimeError):
        m.resume(s.id, "more")
    m.stop(s.id)
    assert done.wait(10)


def test_failed_run(fake_claude):
    m, finished, done, _, _ = manager(fake_claude)
    s = m.start("FAIL", "fail")
    assert done.wait(10)
    assert finished == [(1, "failed")]
    assert s.result == "Tests still failing."


def test_stop_interrupts_cleanly(fake_claude):
    m, finished, done, _, _ = manager(fake_claude)
    s = m.start("SLEEP please", "sleep")
    time.sleep(0.5)
    m.stop(s.id)
    assert done.wait(10)
    assert finished == [(1, "stopped")]
    assert s.claude_session_id == "s-1"  # resumable later


def test_missing_claude_binary(tmp_path):
    finished = []
    m = SessionManager(tmp_path, goal=lambda: "", claude_bin=str(tmp_path / "nope"), on_finish=lambda s: finished.append(s.status))
    s = m.start("anything", "anything")
    assert s.status == "failed" and finished == ["failed"]
    assert "Could not start Claude Code" in s.result


def test_unknown_session():
    m = SessionManager(None, goal=lambda: "")
    with pytest.raises(KeyError):
        m.get(7)


def test_long_results_keep_the_closing_summary():
    from v.sessions import MAX_RESULT_CHARS, Session

    s = Session(id=1, task="t", summary="s", status="done", result="x" * 50_000 + "\n\nFinal summary.")
    text = s.describe()
    assert len(text) < MAX_RESULT_CHARS + 200 and text.endswith("Final summary.")
