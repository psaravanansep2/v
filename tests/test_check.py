import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from v.check import run_check


def stand_in_model(messages):
    """Behaves like a cooperative model: decides from the conversation."""
    last = messages[-1]
    if last["role"] == "tool":
        found = re.search(r"secret word is (\S+?)\.", last["content"])
        return {"content": f"The secret word is {found.group(1)}." if found else "Done, I created hello.txt."}
    text = last["content"]
    if "secret.txt" in text:
        return {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": json.dumps({"path": "secret.txt"})}}]}
    if "hello.txt" in text:
        return {"tool_calls": [{"index": 0, "id": "c2", "function": {"name": "write_file",
                                "arguments": json.dumps({"path": "hello.txt", "content": "hi from v"})}}]}
    return {"content": "Hello! How can I help?"}


@pytest.fixture
def model_url():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "stand-in"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            delta = stand_in_model(req["messages"])
            chunk = {"choices": [{"delta": delta, "finish_reason": "stop"}]}
            body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_check_passes_with_a_working_model(model_url):
    lines = []
    assert run_check(url=model_url, log=lines.append) == 0
    text = "\n".join(lines)
    for step in ("free model is running", "replies", "uses tools (reads a file)", "uses tools (writes a file)", "instant commands"):
        assert f"PASS  {step}" in text
    assert "FAIL" not in text


def test_check_fails_clearly_without_a_model():
    lines = []
    assert run_check(url="http://127.0.0.1:9", log=lines.append) == 1
    assert any(line.startswith("  FAIL  free model is running") for line in lines)
