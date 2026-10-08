"""The free brain: local model client, server discovery, and the agent loop,
tested against fake OpenAI-compatible and Ollama servers over real HTTP."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from v import local
from v.config import Config
from v.confirm import Confirmer
from v.local import LocalClient, LocalServer, StreamAccumulator
from v.local_agent import HISTORY_CHARS, LocalAgent, ThinkFilter
from v.tools import Toolbox


class FakeModelServer:
    """Serves canned replies. Each reply is a list of stream chunks (OpenAI
    style) or of NDJSON events (Ollama style); records request bodies."""

    def __init__(self, replies=(), tags=None):
        self.replies = list(replies)
        self.requests = []
        self.tags = tags  # Ollama /api/tags models, or None if not an Ollama server
        self.pulled = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _write(self, body: bytes, ctype="application/json"):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags" and outer.tags is not None:
                    return self._write(json.dumps({"models": [{"name": n} for n in outer.tags]}).encode())
                if self.path == "/v1/models" and outer.tags is None:
                    return self._write(json.dumps({"data": [{"id": "test-model"}]}).encode())
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                if self.path == "/api/pull":
                    outer.pulled.append(body["model"])
                    outer.tags.append(body["model"])
                    events = [{"status": "pulling", "total": 100, "completed": 50}, {"status": "success"}]
                    return self._write("\n".join(json.dumps(e) for e in events).encode())
                reply = outer.replies.pop(0)
                if self.path == "/api/chat":
                    return self._write("\n".join(json.dumps(e) for e in reply).encode(), "application/x-ndjson")
                payload = "".join(f"data: {json.dumps(c)}\n\n" for c in reply) + "data: [DONE]\n\n"
                self._write(payload.encode(), "text/event-stream")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def text_chunks(text):
    half = len(text) // 2
    return [{"choices": [{"delta": {"content": piece}}]} for piece in (text[:half], text[half:])] + [
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    ]


def tool_chunks(name, args, call_id="call_a"):
    raw = json.dumps(args)
    return [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[:5]}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[5:]}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]


class RecordingUI:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *args: self.events.append((name, *args))


def make_agent(tmp_path, server_url, kind="llama.cpp", model="qwen3-8b", answers=(), confirm="risky"):
    answers = list(answers)
    cfg = Config(project_dir=tmp_path, goal="Organize my files", confirm=confirm)
    ui, spoken = RecordingUI(), []
    confirmer = Confirmer(confirm, lambda q: answers.pop(0))
    base = server_url + "/v1"
    client = LocalClient(LocalServer(base, model, kind, local.model_for_tag(model)))
    agent = LocalAgent(client, cfg, spoken.append, Toolbox(cfg, confirmer, ui), ui=ui)
    return agent, spoken, ui


@pytest.fixture
def server():
    servers = []

    def make(*args, **kwargs):
        s = FakeModelServer(*args, **kwargs)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_turn_with_streamed_tool_call(tmp_path, server):
    (tmp_path / "notes.txt").write_text("buy milk")
    srv = server([tool_chunks("list_files", {"path": "."}), text_chunks("You have one file, notes dot txt. Want me to read it?")])
    agent, spoken, ui = make_agent(tmp_path, srv.url)
    agent.turn("what's in my project folder?")

    first = srv.requests[0]["body"]
    assert first["stream"] is True and first["model"] == "qwen3-8b"
    assert {t["function"]["name"] for t in first["tools"]} >= {"list_files", "read_file", "web_search", "open"}
    assert first["messages"][0]["role"] == "system" and first["messages"][0]["content"].endswith("/no_think")
    assert "Organize my files" in first["messages"][0]["content"]

    second = srv.requests[1]["body"]["messages"]
    assert second[2]["tool_calls"][0]["function"] == {"name": "list_files", "arguments": '{"path": "."}'}
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "call_a"
    assert "notes.txt" in second[3]["content"] and "name" not in second[3]  # "name" is Ollama-only
    assert spoken == ["You have one file, notes dot txt.", "Want me to read it?"]
    assert ("busy", True) in ui.events and ui.events[-1] == ("busy", False)


def test_ollama_native_api_with_context_size(tmp_path, server):
    srv = server(
        [
            [
                {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 1200, "prompt_eval_duration": 2_000_000_000},
            ],
            [
                {"message": {"role": "assistant", "content": "It says hello."}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 40, "prompt_eval_duration": 100_000_000},
            ],
        ],
        tags=["qwen3:8b"],
    )
    (tmp_path / "a.txt").write_text("hello")
    agent, spoken, _ = make_agent(tmp_path, srv.url, kind="ollama", model="qwen3:8b")
    agent.turn("read a.txt")
    first = srv.requests[0]
    assert first["path"] == "/api/chat"
    assert first["body"]["options"]["num_ctx"] == local.CONTEXT_TOKENS
    assert first["body"]["think"] is False
    second = srv.requests[1]["body"]["messages"]
    assert second[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}  # object, not a string
    assert second[3] == {"role": "tool", "content": "hello", "tool_name": "read_file"}
    assert spoken == ["It says hello."]
    # how much prompt Ollama had to read (v check shows it: slow CPUs choke on re-reading long prompts)
    assert agent.client.prompt_tokens == 1240 and abs(agent.client.prompt_seconds - 2.1) < 1e-9


def test_llama_server_prompt_timings(tmp_path, server):
    chunks = text_chunks("Hi there.")
    chunks[-1]["timings"] = {"prompt_n": 37, "prompt_ms": 412.5, "cache_n": 1500}
    srv = server([chunks])
    agent, spoken, _ = make_agent(tmp_path, srv.url)
    agent.turn("hi")
    assert spoken == ["Hi there."]
    assert agent.client.prompt_tokens == 37 and abs(agent.client.prompt_seconds - 0.4125) < 1e-9


def test_thinking_text_is_never_spoken_or_kept(tmp_path, server):
    srv = server([[
        {"choices": [{"delta": {"content": "<thi"}}]},
        {"choices": [{"delta": {"content": "nk>let me plan this</th"}}]},
        {"choices": [{"delta": {"content": "ink>Done. "}}]},
        {"choices": [{"delta": {"content": "All set."}, "finish_reason": "stop"}]},
    ]])
    agent, spoken, _ = make_agent(tmp_path, srv.url)
    agent.turn("hi")
    assert spoken == ["Done.", "All set."]
    assert agent.history[-1]["content"] == "Done. All set."


def test_bad_tool_arguments_go_back_to_the_model(tmp_path, server):
    bad = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{oops"}}]}}]}]
    srv = server([bad, text_chunks("Sorry, let me try again.")])
    agent, _, _ = make_agent(tmp_path, srv.url)
    agent.turn("read it")
    tool_msg = srv.requests[1]["body"]["messages"][3]
    assert tool_msg["content"].startswith("Error: the arguments weren't valid JSON")


def test_command_needs_a_yes(tmp_path, server):
    srv = server([tool_chunks("run_command", {"command": "echo hi"}), text_chunks("Okay.")])
    agent, _, ui = make_agent(tmp_path, srv.url, answers=["no"])
    agent.turn("say hi in the shell")
    assert srv.requests[1]["body"]["messages"][3]["content"] == "The user declined to run this command."


def test_unreachable_server_is_reported_not_raised(tmp_path):
    agent, spoken, ui = make_agent(tmp_path, "http://127.0.0.1:9")
    agent.turn("hello")
    assert spoken[-1].startswith("My local model isn't responding")
    assert any(e[0] == "error" for e in ui.events)


def test_cancel_answers_remaining_calls(tmp_path, server):
    two_calls = [{"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "set_project_goal", "arguments": '{"goal": "A"}'}},
        {"index": 1, "id": "c2", "function": {"name": "set_project_goal", "arguments": '{"goal": "B"}'}},
    ]}}]}]
    srv = server([two_calls])
    agent, _, ui = make_agent(tmp_path, srv.url)
    original = agent.toolbox.t_set_project_goal

    def set_then_stop(goal):
        out = original(goal)
        agent.cancel()
        return out

    agent.toolbox.t_set_project_goal = set_then_stop
    agent.turn("set goals")
    results = [m for m in agent.history if m["role"] == "tool"]
    assert "Project goal saved: A" in results[0]["content"]
    assert results[1]["content"].startswith("Interrupted")
    assert ("notice", "interrupted") in ui.events
    assert len(srv.requests) == 1


def test_history_window_fits_and_starts_cleanly(tmp_path):
    agent, _, _ = make_agent(tmp_path, "http://127.0.0.1:9")
    for i in range(40):
        agent.history += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "name": "x", "content": "y" * 2000},
            {"role": "assistant", "content": f"a{i}"},
        ]
    agent._turn_start = len(agent.history) - 4
    window = agent._window()
    assert window[0]["role"] == "user"
    assert sum(len(str(m.get("content") or "")) for m in window) <= HISTORY_CHARS + 2000
    assert window[-1]["content"] == "a39"


def test_think_filter_handles_split_tags():
    f = ThinkFilter()
    out = "".join(f.feed(p) for p in ["Hi <", "think>sec", "ret</thi", "nk> there<", "b>"]) + f.flush()
    assert out == "Hi  there<b>"


def test_accumulator_separates_calls_sent_without_index():
    acc = StreamAccumulator()
    acc.feed({"choices": [{"delta": {"tool_calls": [{"id": "x1", "function": {"name": "open", "arguments": {"target": "a"}}}]}}]})
    acc.feed({"choices": [{"delta": {"tool_calls": [{"id": "x2", "function": {"name": "open", "arguments": {"target": "b"}}}]}}]})
    calls = acc.message()["tool_calls"]
    assert [c["id"] for c in calls] == ["x1", "x2"]
    assert [json.loads(c["function"]["arguments"])["target"] for c in calls] == ["a", "b"]


def test_finds_ollama_and_its_best_installed_model(server, monkeypatch):
    srv = server(tags=["llama3.2:3b", "qwen3:8b", "nomic-embed-text:latest"])
    monkeypatch.setattr(local, "OLLAMA_URL", srv.url)
    monkeypatch.setattr(local, "V_SERVER_PORT", 9)
    monkeypatch.setattr(local, "LMSTUDIO_URL", "http://127.0.0.1:9")
    found = local.find_running(lambda *_: None)
    assert found.kind == "ollama" and found.model == "qwen3:8b" and found.choice.name == "Qwen3 8B"


def test_ensure_server_pulls_a_model_into_ollama(server, monkeypatch):
    srv = server(tags=[])
    monkeypatch.setattr(local, "OLLAMA_URL", srv.url)
    monkeypatch.setattr(local, "V_SERVER_PORT", 9)
    monkeypatch.setattr(local, "LMSTUDIO_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(local, "choose_model", lambda budget=None: local.MODELS[-1])
    logs = []
    found = local.ensure_server(logs.append)
    assert srv.pulled == ["qwen3:4b"]
    assert found.kind == "ollama" and found.model == "qwen3:4b"
    assert any("50%" in line for line in logs)


def test_custom_server_url(server):
    srv = server()
    found = local.custom_server(srv.url, None)
    assert found.base_url == srv.url + "/v1" and found.model == "test-model"


def test_no_server_and_no_runner_explains_the_fix(monkeypatch):
    monkeypatch.setattr(local, "ollama_models", lambda: None)
    monkeypatch.setattr(local, "_start_ollama", lambda log: False)
    monkeypatch.setattr(local, "find_running", lambda log: None)
    import builtins

    real_import = builtins.__import__

    def no_cheapstack(name, *args, **kwargs):
        if name.startswith("cheapstack"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cheapstack)
    with pytest.raises(local.LocalError, match="install Ollama"):
        local.ensure_server(lambda *_: None)


@pytest.mark.parametrize(
    "ram,vram,apple,expected",
    [
        (8, 0, False, "Qwen3 4B"),  # ordinary laptop
        (16, 0, False, "Qwen3 8B"),
        (16, 0, True, "Qwen3 8B"),  # 16 GB MacBook Air
        (32, 0, True, "gpt-oss 20B"),
        (64, 0, True, "Qwen3 30B-A3B"),
        (16, 24, False, "Qwen3 30B-A3B"),  # gaming PC with a 24 GB card
        (16, 12, False, "Qwen3 8B"),
    ],
)
def test_model_choice_by_machine(ram, vram, apple, expected):
    assert local.choose_model(local.memory_budget_gb(ram, vram, apple)).name == expected


def test_free_brain_drives_the_phone(tmp_path, server):
    from v import phone

    srv = server([tool_chunks("set_project_goal", {"goal": "Plan the trip"}), text_chunks("Saved. Where to first?")])
    hub, media = phone.Hub(), phone.MediaStore()
    q, _ = hub.subscribe()
    cfg = Config(project_dir=tmp_path, confirm="risky")
    ui = phone.PhoneUI(hub, media)
    agent = LocalAgent(LocalClient(LocalServer(srv.url + "/v1", "qwen3-8b", "llama.cpp")), cfg, lambda t: None,
                       Toolbox(cfg, Confirmer("risky", lambda q: "yes"), ui), ui=ui)
    agent.turn("my goal is planning the trip")
    kinds = []
    while not q.empty():
        kinds.append(q.get_nowait())
    assert {"type": "goal", "text": "Plan the trip"} in [{k: e[k] for k in ("type", "text")} for e in kinds if e["type"] == "goal"]
    assert "".join(e["delta"] for e in kinds if e["type"] == "text") == "Saved. Where to first?"
    assert [e["busy"] for e in kinds if e["type"] == "busy"] == [True, False]


@pytest.mark.parametrize(
    "written",
    [
        'Let me check. <tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>',
        '{"name": "read_file", "arguments": {"path": "a.txt"}}',
        '```json\n{"name": "cat", "parameters": {"file": "a.txt"}}\n```',  # fenced, misnamed tool and argument
        '[{"function": {"name": "read_file", "arguments": "{\\"path\\": \\"a.txt\\"}"}}]',
    ],
)
def test_tool_calls_written_as_text_still_run_and_are_not_spoken(tmp_path, server, written):
    (tmp_path / "a.txt").write_text("hello there")
    srv = server([text_chunks(written), text_chunks("It says hello there.")])
    agent, spoken, _ = make_agent(tmp_path, srv.url)
    agent.turn("what's in a.txt")
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["content"] == "hello there"
    assert all("{" not in s and "tool_call" not in s for s in spoken)
    assert spoken[-1] == "It says hello there."


def test_text_that_only_looks_like_json_is_still_spoken(tmp_path, server):
    srv = server([text_chunks("[1] is the first step. Then save the file.")])
    agent, spoken, _ = make_agent(tmp_path, srv.url)
    agent.turn("what's first")
    assert spoken == ["[1] is the first step.", "Then save the file."]
    assert "tool_calls" not in agent.history[-1]


def test_unknown_names_in_text_are_not_treated_as_calls(tmp_path, server):
    srv = server([text_chunks('{"name": "Ada", "role": "engineer"}')])
    agent, spoken, _ = make_agent(tmp_path, srv.url)
    agent.turn("show me the record")
    assert "tool_calls" not in agent.history[-1]
    assert spoken == ['{"name": "Ada", "role": "engineer"}']


OLD_STYLE = ["llama-b6000-bin-ubuntu-x64.zip", "llama-b6000-bin-ubuntu-vulkan-x64.zip", "llama-b6000-bin-macos-arm64.zip",
             "llama-b6000-bin-macos-x64.zip", "llama-b6000-bin-win-cpu-x64.zip", "llama-b6000-bin-win-cuda-12.4-x64.zip",
             "llama-b6000-bin-win-vulkan-x64.zip", "llama-b6000-bin-win-cpu-arm64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
             "llama-b6000-xcframework.zip"]
NEW_STYLE = ["llama-b7000-bin-linux-x64.tar.gz", "llama-b7000-bin-linux-vulkan-x64.tar.gz", "llama-b7000-bin-linux-arm64.tar.gz",
             "llama-b7000-bin-macos-arm64.tar.gz", "llama-b7000-bin-win-cpu-x64.zip", "llama-b7000-bin-win-hip-radeon-x64.zip",
             "llama-b7000-bin-ubuntu-rocm-7.0-x64.tar.gz"]


@pytest.mark.parametrize(
    "names,platform,machine,vulkan,expected",
    [
        (OLD_STYLE, "linux", "x86_64", False, "llama-b6000-bin-ubuntu-x64.zip"),
        (OLD_STYLE, "linux", "x86_64", True, "llama-b6000-bin-ubuntu-vulkan-x64.zip"),
        (OLD_STYLE, "darwin", "arm64", False, "llama-b6000-bin-macos-arm64.zip"),
        (OLD_STYLE, "win32", "AMD64", False, "llama-b6000-bin-win-cpu-x64.zip"),
        (OLD_STYLE, "win32", "ARM64", False, "llama-b6000-bin-win-cpu-arm64.zip"),
        (NEW_STYLE, "linux", "x86_64", False, "llama-b7000-bin-linux-x64.tar.gz"),
        (NEW_STYLE, "linux", "aarch64", False, "llama-b7000-bin-linux-arm64.tar.gz"),
        (NEW_STYLE, "darwin", "arm64", False, "llama-b7000-bin-macos-arm64.tar.gz"),
    ],
)
def test_picks_the_right_llama_cpp_download(names, platform, machine, vulkan, expected):
    assets = [{"name": n} for n in names]
    assert local.pick_llama_asset(assets, platform, machine, vulkan)["name"] == expected


def test_no_matching_download_lists_what_was_there():
    with pytest.raises(local.LocalError, match="assets: llama-b1-bin-plan9-x64.zip"):
        local.pick_llama_asset([{"name": "llama-b1-bin-plan9-x64.zip"}], "linux", "x86_64")


def test_skips_a_release_without_builds():
    releases = [
        {"tag_name": "nightly", "assets": [{"name": "nightly-tag.txt"}]},
        {"tag_name": "b7001", "draft": True, "assets": [{"name": "llama-b7001-bin-ubuntu-x64.zip"}]},
        {"tag_name": "b7000", "assets": [{"name": n} for n in OLD_STYLE]},
    ]
    assert local.pick_llama_download(releases, platform="linux", machine="x86_64")["name"] == "llama-b6000-bin-ubuntu-x64.zip"
    with pytest.raises(local.LocalError):
        local.pick_llama_download(releases[:1], platform="linux", machine="x86_64")
