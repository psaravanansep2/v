import http.client
import json
import os
import ssl
import threading
import time

import pytest

from v import phone
from v.ui import Interrupted

TOKEN = "test-token-0123456789abcdef"


class FakeTranscriber:
    def __init__(self, text="open my editor"):
        self.text = text
        self.calls = []

    def transcribe(self, data):
        self.calls.append(data)
        return self.text


class FakeAgent:
    def __init__(self):
        self.turns = []
        self.cancelled = 0
        self.skipped_quick = []

    def turn(self, text, quick=True):
        self.turns.append(text)
        if not quick:
            self.skipped_quick.append(text)

    def cancel(self):
        self.cancelled += 1


@pytest.fixture
def server(tmp_path):
    hub, media = phone.Hub(), phone.MediaStore()
    bridge = phone.Bridge(hub, FakeTranscriber(), log=lambda *a: None)
    bridge.agent = FakeAgent()
    srv = phone.PhoneServer(("127.0.0.1", 0), hub, media, bridge, TOKEN, lambda: {"project": "app", "goal": "Ship it", "stt": True})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def auth(extra=None):
    return {"X-V-Token": TOKEN, **(extra or {})}


class EventReader:
    def __init__(self, srv):
        self.conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        self.conn.request("GET", f"/events?t={TOKEN}")
        self.resp = self.conn.getresponse()
        assert self.resp.status == 200

    def next(self):
        while True:
            line = self.resp.fp.readline().decode()
            if line.startswith("data: "):
                return json.loads(line[6:])

    def until(self, kind):
        while True:
            e = self.next()
            if e["type"] == kind:
                return e

    def close(self):
        self.conn.close()


def test_everything_needs_the_token(server):
    assert request(server, "GET", "/")[0] == 401
    assert request(server, "GET", "/?t=wrong")[0] == 401
    assert request(server, "GET", "/events")[0] == 401
    assert request(server, "GET", "/manifest.webmanifest")[0] == 401  # its start_url contains the token
    assert request(server, "POST", "/message", b'{"text":"hi"}', {"Content-Type": "application/json"})[0] == 401
    # the token in the query string isn't enough for POSTs: they need the custom header
    status, _ = request(server, "POST", f"/message?t={TOKEN}", b'{"text":"hi"}', {"Content-Type": "application/json"})
    assert status == 401
    assert request(server, "GET", "/icon.svg")[0] == 200  # harmless and public


def test_page_and_manifest(server):
    status, body = request(server, "GET", f"/?t={TOKEN}")
    assert status == 200 and b"<title>v</title>" in body
    status, body = request(server, "GET", f"/manifest.webmanifest?t={TOKEN}")
    assert status == 200 and json.loads(body)["start_url"] == f"/?t={TOKEN}"


def test_text_message_is_queued_for_the_agent(server):
    status, body = request(server, "POST", "/message", b'{"text":"  run the tests "}', auth({"Content-Type": "application/json"}))
    assert status == 200 and json.loads(body)["status"] == "queued"
    assert server.bridge.inbox.get_nowait() == ("text", "run the tests", ())


def test_audio_message_is_transcribed_on_the_worker(server):
    status, body = request(server, "POST", "/message", b"\x1a\x45webm...", auth({"Content-Type": "audio/webm"}))
    assert json.loads(body)["status"] == "queued"
    reader = EventReader(server)
    reader.until("replay_end")
    threading.Thread(target=server.bridge.run_worker, daemon=True).start()
    assert reader.until("user")["text"] == "open my editor"
    time.sleep(0.1)
    assert server.bridge.agent.turns == ["open my editor"]
    assert server.bridge.transcriber.calls == [b"\x1a\x45webm..."]
    server.bridge.stop()
    reader.close()


def test_event_stream_replays_history_then_streams_live(server):
    server.hub.publish({"type": "user", "text": "earlier"})
    reader = EventReader(server)
    hello = reader.next()
    assert hello["type"] == "hello" and hello["goal"] == "Ship it"
    replayed = reader.next()
    assert replayed == {"type": "user", "text": "earlier", "seq": 1, "replay": True}
    assert reader.next()["type"] == "replay_end"
    server.hub.publish({"type": "text", "delta": "Hi"})
    assert reader.next() == {"type": "text", "delta": "Hi", "seq": 2}
    reader.close()


def ask_in_background(bridge):
    out = {}

    def run():
        try:
            out["reply"] = bridge.ask("Run this command: git push?")
        except Interrupted:
            out["interrupted"] = True

    t = threading.Thread(target=run)
    t.start()
    return t, out


def test_approval_from_the_phone_buttons(server):
    reader = EventReader(server)
    reader.until("replay_end")
    t, out = ask_in_background(server.bridge)
    ask = reader.until("ask")
    assert ask["question"] == "Run this command: git push?"
    assert request(server, "POST", "/answer", json.dumps({"id": "nope", "answer": "yes"}).encode(), auth())[0] == 409
    assert request(server, "POST", "/answer", json.dumps({"id": ask["id"], "answer": "yes"}).encode(), auth())[0] == 200
    t.join(5)
    assert out == {"reply": "yes"}
    assert reader.until("ask_done")["answer"] == "yes"
    reader.close()


def test_spoken_or_typed_reply_answers_an_open_question(server):
    reader = EventReader(server)
    reader.until("replay_end")
    t, out = ask_in_background(server.bridge)
    reader.until("ask")
    server.bridge.transcriber.text = "yes go ahead"
    status, body = request(server, "POST", "/message", b"audio", auth({"Content-Type": "audio/mp4"}))
    assert json.loads(body)["status"] == "answered"
    t.join(5)
    assert out == {"reply": "yes go ahead"}
    assert server.bridge.inbox.empty()  # not treated as a new request
    reader.close()


def test_stop_cancels_the_agent_and_an_open_question(server):
    t, out = ask_in_background(server.bridge)
    time.sleep(0.2)
    assert request(server, "POST", "/interrupt", b"{}", auth())[0] == 200
    t.join(5)
    assert out == {"interrupted": True}
    assert server.bridge.agent.cancelled == 1


def test_phone_voice_sends_audio_or_text():
    import numpy as np

    hub, media = phone.Hub(), phone.MediaStore()
    q, _ = hub.subscribe()

    class Synth:
        rate = 24_000

        def synthesize(self, text):
            return np.zeros(2400, dtype="float32")

    phone.PhoneVoice(hub, media, Synth()).speak("Hello there.")
    event = q.get_nowait()
    assert event["type"] == "audio" and event["text"] == "Hello there."
    data, ctype = media.get(event["id"])
    assert ctype == "audio/wav" and data[:4] == b"RIFF"

    phone.PhoneVoice(hub, media, None).speak("Hello there.")
    assert q.get_nowait()["type"] == "speak"


def test_media_store_keeps_recent_items():
    media = phone.MediaStore(keep=2)
    a = media.put(b"a", "image/png")
    media.put(b"b", "image/png")
    media.put(b"c", "image/png")
    assert media.get(a) is None


def test_token_is_persisted_and_can_be_replaced(tmp_path):
    first = phone.load_token(state_dir=tmp_path)
    assert len(first) >= 16
    assert phone.load_token(state_dir=tmp_path) == first
    if os.name == "posix":
        assert (tmp_path / "token").stat().st_mode & 0o077 == 0  # private to the user
    assert phone.load_token(new=True, state_dir=tmp_path) != first


def test_https_with_generated_certificate(tmp_path):
    cert, key = phone.self_signed_cert("127.0.0.1", state_dir=tmp_path)
    assert phone.self_signed_cert("127.0.0.1", state_dir=tmp_path) == (cert, key)  # reused
    hub, media = phone.Hub(), phone.MediaStore()
    bridge = phone.Bridge(hub, log=lambda *a: None)
    srv = phone.PhoneServer(("127.0.0.1", 0), hub, media, bridge, TOKEN, lambda: {}, phone.ssl_context(cert, key))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        # a client that trusts nothing but our certificate can connect: the
        # certificate is valid for the address it's served on
        ctx = ssl.create_default_context(cafile=str(cert))
        conn = http.client.HTTPSConnection("127.0.0.1", srv.server_address[1], context=ctx, timeout=5)
        conn.request("GET", f"/?t={TOKEN}")
        assert conn.getresponse().status == 200
        conn.close()
        # a client that rejects the certificate doesn't take the server down
        bad = http.client.HTTPSConnection("127.0.0.1", srv.server_address[1], context=ssl.create_default_context(), timeout=5)
        with pytest.raises(ssl.SSLError):
            bad.request("GET", "/")
        bad.close()
        conn = http.client.HTTPSConnection("127.0.0.1", srv.server_address[1], context=ctx, timeout=5)
        conn.request("GET", "/icon.svg")
        assert conn.getresponse().status == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_certificate_is_remade_when_the_address_changes(tmp_path):
    cert, _ = phone.self_signed_cert("192.168.1.20", state_dir=tmp_path)
    first = cert.read_bytes()
    phone.self_signed_cert("10.0.0.5", state_dir=tmp_path)
    assert cert.read_bytes() != first


def test_a_second_launch_finds_the_running_app(server, tmp_path, monkeypatch):
    from v import serve

    app_file = tmp_path / "app.json"
    monkeypatch.setattr(serve, "APP_FILE", app_file)
    assert serve.running_app_url() is None  # nothing recorded
    url = f"http://127.0.0.1:{server.server_address[1]}/?t={TOKEN}"
    app_file.write_text(json.dumps({"local_url": url}))
    assert serve.running_app_url() == url
    app_file.write_text(json.dumps({"local_url": "http://127.0.0.1:9/?t=x"}))  # stale: v isn't running
    assert serve.running_app_url() is None


def test_messages_before_setup_finishes_get_a_clear_answer(server):
    reader = EventReader(server)
    reader.until("replay_end")
    server.bridge.agent = None
    server.bridge.not_ready = "I couldn't finish setting up: no internet."
    threading.Thread(target=server.bridge.run_worker, daemon=True).start()
    request(server, "POST", "/message", b'{"text":"hi"}', auth({"Content-Type": "application/json"}))
    assert reader.until("notice")["text"] == "I couldn't finish setting up: no internet."
    server.bridge.stop()
    reader.close()


def test_certificate_with_a_very_long_computer_name(tmp_path, monkeypatch):
    monkeypatch.setattr(phone.socket, "gethostname", lambda: "x" * 80 + ".local")
    cert, key = phone.self_signed_cert("127.0.0.1", state_dir=tmp_path)
    assert cert.exists() and key.exists()


def test_dropped_files_are_saved_and_handed_to_v(server, tmp_path):
    server.upload_dir = tmp_path / "inbox"
    headers = auth({"Content-Type": "image/png", "X-Filename": "my%20photo.png"})
    status, body = request(server, "POST", "/upload", b"\x89PNG...", headers)
    assert status == 200
    saved = json.loads(body)
    assert saved["name"] == "my photo.png" and (tmp_path / "inbox" / "my photo.png").read_bytes() == b"\x89PNG..."
    again = json.loads(request(server, "POST", "/upload", b"two", headers)[1])
    assert again["name"] == "my photo (2).png"  # never overwrites
    assert request(server, "POST", "/upload", b"x", {"X-Filename": "a.txt"})[0] == 401

    reader = EventReader(server)
    reader.until("replay_end")
    threading.Thread(target=server.bridge.run_worker, daemon=True).start()
    message = json.dumps({"text": "what does this say?", "files": [saved["path"]]}).encode()
    request(server, "POST", "/message", message, auth({"Content-Type": "application/json"}))
    user = reader.until("user")
    assert user["text"] == "what does this say?" and user["files"] == ["my photo.png"]
    time.sleep(0.1)
    told = server.bridge.agent.turns[-1]
    assert told.startswith("what does this say?") and saved["path"] in told
    assert server.bridge.agent.skipped_quick == [told]  # "open this" with a file isn't an instant command
    request(server, "POST", "/message", json.dumps({"files": [saved["path"]]}).encode(), auth())
    reader.until("user")
    time.sleep(0.1)
    assert server.bridge.agent.turns[-1].startswith("Take a look at this file.")
    server.bridge.stop()
    reader.close()


def test_uploads_cant_escape_the_folder(tmp_path):
    for name in ("../../etc/passwd", "..\\..\\boot.ini", "/abs/path.txt", "..", "CON.txt", "a<b>:c?.txt"):
        saved = phone.save_upload(tmp_path, name, b"x")
        assert saved.parent == tmp_path, name
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "passwd" in names and "boot.ini" in names and "path.txt" in names and "_CON.txt" in names
    assert (tmp_path / ".gitignore").read_text() == "*\n"  # copies don't end up in the project's git history


def test_goal_and_timer_controls(server):
    goals, cancelled = [], []
    server.on_goal = goals.append
    server.on_cancel_timer = lambda timer_id: cancelled.append(timer_id) or timer_id == "abc"
    assert request(server, "POST", "/goal", json.dumps({"goal": "  Launch the shop  "}).encode(), auth())[0] == 200
    assert goals == ["Launch the shop"]
    assert request(server, "POST", "/timer/cancel", b'{"id": "abc"}', auth())[0] == 200
    assert request(server, "POST", "/timer/cancel", b'{"id": "zzz"}', auth())[0] == 404
    assert cancelled == ["abc", "zzz"]
    assert request(server, "POST", "/goal", b'{"goal": "x"}', {"Content-Type": "application/json"})[0] == 401


def test_hello_carries_the_server_clock(server):
    reader = EventReader(server)
    hello = reader.next()
    assert abs(hello["now"] - time.time() * 1000) < 5000  # the page corrects countdowns for clock drift
    reader.close()


def test_timers_and_activity_reach_the_page():
    hub, media = phone.Hub(), phone.MediaStore()
    q, _ = hub.subscribe()
    ui = phone.PhoneUI(hub, media)
    ui.timers([{"id": "a", "label": "10 minute timer", "kind": "timer", "ends": 1}])
    assert q.get_nowait()["items"][0]["label"] == "10 minute timer"
    ui.activity("opened Spotify")
    event = q.get_nowait()
    assert event["text"] == "opened Spotify" and abs(event["at"] - time.time() * 1000) < 5000
