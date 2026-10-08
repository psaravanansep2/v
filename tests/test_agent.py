"""Agent tests against the real Anthropic SDK, with the HTTP layer replaced
by a fake server that replays canned streaming (SSE) responses. This checks
both what v sends (request body + beta headers) and that it parses real
stream events correctly."""

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from tests.test_computer import make as make_computer
from v.agent import BETAS, CUSTOM_TOOLS, Agent, spoken_summary, validate
from v.config import Config, load_goal
from v.confirm import Confirmer
from v.sessions import SessionManager


def sse(blocks, stop_reason="end_turn"):
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    ]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            for piece in (b["text"][: len(b["text"]) // 2], b["text"][len(b["text"]) // 2 :]):
                events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": piece}})
        elif b["type"] == "thinking":
            events.append(
                {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}
            )
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": b["thinking"]}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": "sig"}})
        elif b["type"] == "tool_use":
            start = {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}
            if "toolset_name" in b:
                start["toolset_name"] = b["toolset_name"]
            events.append({"type": "content_block_start", "index": i, "content_block": start})
            events.append(
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])},
                }
            )
        events.append({"type": "content_block_stop", "index": i})
    events.append(
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 5}}
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


class FakeAPI:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        status, body = self.responses.pop(0)
        if status != 200:
            return httpx2.Response(status, json=body)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def reply(blocks, stop_reason="end_turn"):
    return (200, sse(blocks, stop_reason))


class Harness:
    def __init__(self, tmp_path, *responses, answers=(), confirm="risky", computer=True):
        self.api = FakeAPI(*responses)
        client = anthropic.Anthropic(
            api_key="test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self.api)),
        )
        self.spoken = []
        self.printed = []
        self.questions = []
        answers = list(answers)

        def ask(q):
            self.questions.append(q)
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        self.cfg = Config(project_dir=tmp_path, goal="Ship the signup flow", confirm=confirm)
        self.computer, self.gui = make_computer() if computer else (None, None)
        self.sessions = SessionManager(tmp_path, goal=lambda: self.cfg.goal)
        self.agent = Agent(
            client,
            self.cfg,
            say=self.spoken.append,
            confirmer=Confirmer(confirm, ask),
            computer=self.computer,
            sessions=self.sessions,
            out=lambda *a, **k: self.printed.append(" ".join(map(str, a))),
        )

    def body(self, i):
        return self.api.requests[i]["body"]


def test_plain_reply_is_spoken_sentence_by_sentence(tmp_path):
    h = Harness(tmp_path, reply([{"type": "text", "text": "Sure thing. The build is green."}]))
    h.agent.turn("how's the build?")
    assert h.spoken == ["Sure thing.", "The build is green."]
    assert [m["role"] for m in h.agent.messages] == ["user", "assistant"]


def test_request_shape(tmp_path):
    h = Harness(tmp_path, reply([{"type": "text", "text": "Hi."}]))
    h.agent.turn("hello")
    req = h.api.requests[0]
    assert set(req["headers"]["anthropic-beta"].split(",")) == set(BETAS)
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive", "display": "updates"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert body["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919"}]}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["stream"] is True
    assert {"type": "computer_toolset_20260801"} in body["tools"]
    assert {t.get("name") for t in body["tools"]} >= {t["name"] for t in CUSTOM_TOOLS}
    assert "Ship the signup flow" in body["system"]
    assert "betas" not in body


def test_no_computer_toolset_when_disabled(tmp_path):
    h = Harness(tmp_path, reply([{"type": "text", "text": "Hi."}]), computer=False)
    h.agent.turn("hello")
    assert all(t.get("type") != "computer_toolset_20260801" for t in h.body(0)["tools"])
    assert "Screen control is turned off" in h.body(0)["system"]


def test_computer_batch_round_trip(tmp_path):
    h = Harness(
        tmp_path,
        reply(
            [
                {"type": "thinking", "thinking": "Checking what's on screen first."},
                {"type": "tool_use", "id": "tu_1", "name": "screenshot", "input": {}, "toolset_name": "computer"},
                {"type": "tool_use", "id": "tu_2", "name": "left_click", "input": {"coordinate": [100, 50]}, "toolset_name": "computer"},
            ],
            "tool_use",
        ),
        reply([{"type": "text", "text": "Done, the editor is focused."}]),
    )
    h.agent.turn("click into the editor")
    assert h.spoken[0] == "Checking what's on screen first."  # progress note spoken
    assert h.spoken[-1] == "Done, the editor is focused."
    assert [c[0] for c in h.gui.calls] == ["click"]

    second = h.body(1)["messages"]
    assistant = second[1]["content"]
    tool_uses = [b for b in assistant if b["type"] == "tool_use"]
    assert [b.get("toolset_name") for b in tool_uses] == ["computer", "computer"]
    assert any(b["type"] == "thinking" and b["signature"] == "sig" for b in assistant)  # replayed unchanged
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
    assert all(r["toolset_name"] == "computer" for r in results)
    assert results[0]["content"][0]["type"] == "image"
    assert results[1]["content"] == "OK"


def test_command_runs_after_spoken_yes(tmp_path):
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_1", "name": "run_command", "input": {"command": "echo hello-from-v"}}], "tool_use"),
        reply([{"type": "text", "text": "It printed hello."}]),
        answers=["yes please"],
    )
    h.agent.turn("run echo")
    assert h.questions == ["Run this command: echo hello-from-v?"]
    result = h.body(1)["messages"][2]["content"][0]
    assert result["tool_use_id"] == "tu_1"
    assert "hello-from-v" in result["content"] and "exit code 0" in result["content"]
    assert "toolset_name" not in result


def test_declined_command_is_reported_to_claude(tmp_path):
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_1", "name": "run_command", "input": {"command": "rm -rf build"}}], "tool_use"),
        reply([{"type": "text", "text": "Okay, I won't."}]),
        answers=["no"],
    )
    h.agent.turn("clean up")
    result = h.body(1)["messages"][2]["content"][0]
    assert result["is_error"] and "declined" in result["content"]


def test_invalid_tool_input_is_rejected_before_running(tmp_path):
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_1", "name": "run_command", "input": {"cmd": "ls"}}], "tool_use"),
        reply([{"type": "text", "text": "Oops."}]),
        confirm="none",
    )
    h.agent.turn("list files")
    result = h.body(1)["messages"][2]["content"][0]
    assert result["is_error"] and "Missing required field 'command'" in result["content"]


def test_set_goal_persists(tmp_path):
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_1", "name": "set_project_goal", "input": {"goal": "Launch beta"}}], "tool_use"),
        reply([{"type": "text", "text": "Saved."}]),
    )
    h.agent.turn("our goal is launching the beta")
    assert h.cfg.goal == "Launch beta"
    assert load_goal(tmp_path) == "Launch beta"


def test_refusal_closes_open_tool_calls(tmp_path):
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_1", "name": "run_command", "input": {"command": "x"}}], "refusal"),
    )
    h.agent.turn("do the thing")
    last = h.agent.messages[-1]
    assert last["role"] == "user" and last["content"][0]["tool_use_id"] == "tu_1"
    assert last["content"][0]["is_error"]
    assert h.spoken[-1].startswith("Sorry")


def test_interrupt_leaves_history_valid(tmp_path):
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_9", "name": "run_command", "input": {"command": "sleep 100"}}], "tool_use"),
        answers=[KeyboardInterrupt()],
    )
    h.agent.turn("wait a while")
    last = h.agent.messages[-1]
    assert last["content"][0]["tool_use_id"] == "tu_9" and "Interrupted" in last["content"][0]["content"]


def test_session_updates_reach_claude_on_next_turn(tmp_path):
    h = Harness(tmp_path, reply([{"type": "text", "text": "Nice, it worked."}]))
    session = SimpleNamespace(
        id=3, status="done", result="Lots of detail.\n\nAdded login and all tests pass.", describe=lambda: "Session 3: done"
    )
    h.agent.on_session_finish(session)
    assert h.spoken == ["Session 3 finished. Added login and all tests pass."]
    h.agent.turn("anything new?")
    first = h.body(0)["messages"][0]["content"]
    assert first.startswith("[Session update] Session 3: done") and first.endswith("anything new?")


def test_api_error_is_spoken_not_raised(tmp_path):
    h = Harness(tmp_path, (500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}))
    h.agent.turn("hello")
    assert "error" in h.spoken[-1]


def test_validate():
    spec = {t["name"]: t for t in CUSTOM_TOOLS}
    assert validate(spec["stop_session"], {"session_id": 2}) is None
    assert validate(spec["stop_session"], {"session_id": "2"}) == "Field 'session_id' must be a integer."
    assert validate(spec["stop_session"], {"session_id": True}) is not None
    assert validate(spec["run_command"], {"command": "  "}) == "Field 'command' must not be empty."
    assert validate(spec["session_status"], {}) is None


def test_spoken_summary_takes_the_closing_paragraph():
    assert spoken_summary("Long log...\n\nFixed the bug. Tests pass. Also refactored a lot of things here.", 40) == (
        "Fixed the bug. Tests pass."
    )
    assert spoken_summary("") == ""


def test_unexpected_tool_error_goes_back_to_claude(tmp_path, monkeypatch):
    import v.agent

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(v.agent, "save_goal", broken)
    h = Harness(
        tmp_path,
        reply([{"type": "tool_use", "id": "tu_1", "name": "set_project_goal", "input": {"goal": "x"}}], "tool_use"),
        reply([{"type": "text", "text": "I couldn't save that."}]),
    )
    h.agent.turn("save the goal")
    result = h.body(1)["messages"][2]["content"][0]
    assert result["is_error"] and "OSError: disk full" in result["content"]
