"""Screen control on Wayland, against a pretend desktop: a fake
xdg-desktop-portal on a private D-Bus, answering like KDE's and GNOME's do."""

import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

jeepney = pytest.importorskip("jeepney")
if not shutil.which("dbus-daemon"):
    pytest.skip("needs dbus-daemon", allow_module_level=True)

from jeepney import DBusAddress, HeaderFields, MessageType, new_error, new_method_return, new_signal  # noqa: E402
from jeepney.bus_messages import message_bus  # noqa: E402
from jeepney.io.blocking import open_dbus_connection  # noqa: E402
from PIL import Image  # noqa: E402

from v import wayland  # noqa: E402


class FakePortal:
    """Answers the portal calls v makes; records them."""

    def __init__(self, tmp: Path, keysyms=True, allow=True, size=(1920, 1080), token="restore-1"):
        self.tmp, self.keysyms, self.allow, self.size, self.token = tmp, keysyms, allow, size, token
        self.calls = []
        self.daemon = subprocess.Popen(["dbus-daemon", "--session", "--nofork", "--print-address=1"],
                                       stdout=subprocess.PIPE, text=True)
        self.address = self.daemon.stdout.readline().strip()
        self.conn = open_dbus_connection(bus=self.address)
        self.conn.send_and_get_reply(message_bus.RequestName(wayland.PORTAL_NAME))
        self.stopping = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self.stopping:
            try:
                msg = self.conn.receive(timeout=0.1)
            except TimeoutError:
                continue
            except OSError:
                return
            if msg.header.message_type != MessageType.method_call:
                continue
            fields = msg.header.fields
            iface, member = fields.get(HeaderFields.interface), fields[HeaderFields.member]
            self.calls.append((member, msg.body))
            sender = fields[HeaderFields.sender].lstrip(":").replace(".", "_")
            if member.startswith("Notify"):
                if member == "NotifyKeyboardKeysym" and not self.keysyms:
                    self.conn.send_message(new_error(msg, "org.freedesktop.DBus.Error.NotSupported", "s", ("keysyms",)))
                else:
                    self.conn.send_message(new_method_return(msg))
                continue
            if member == "Close":
                self.conn.send_message(new_method_return(msg))
                continue
            options = msg.body[-1]
            handle = f"{wayland.DESKTOP}/request/{sender}/{options['handle_token'][1]}"
            self.conn.send_message(new_method_return(msg, "o", (handle,)))
            code, results = 0, {}
            if member == "CreateSession":
                results = {"session_handle": ("s", f"{wayland.DESKTOP}/session/{sender}/v")}
            elif member == "Start":
                if not self.allow:
                    code = 1
                else:
                    results = {"devices": ("u", 3),
                               "streams": ("a(ua{sv})", [(42, {"position": ("(ii)", (0, 0)), "size": ("(ii)", self.size)})]),
                               "restore_token": ("s", self.token)}
            elif member == "Screenshot":
                path = self.tmp / "Screenshot from v.png"
                Image.new("RGB", (2560, 1440), "white").save(path)
                results = {"uri": ("s", path.as_uri())}
            self.conn.send_message(new_signal(DBusAddress(handle, interface="org.freedesktop.portal.Request"), "Response",
                                              "ua{sv}", (code, results)))

    def sent(self, member):
        return [body for name, body in self.calls if name == member]

    def close(self):
        self.stopping = True
        self.thread.join(2)
        self.conn.close()
        self.daemon.terminate()
        self.daemon.wait(5)


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    portals = []

    def make(**kwargs):
        portal = FakePortal(tmp_path, **kwargs)
        portals.append(portal)
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", portal.address)
        return portal

    yield make
    for p in portals:
        p.close()


def control(tmp_path, logs=None):
    return wayland.RemoteControl(token_file=tmp_path / "permission", log=(logs.append if logs is not None else lambda m: None))


def test_asks_once_then_remembers(desktop, tmp_path):
    portal = desktop()
    logs = []
    c = control(tmp_path, logs)
    assert (c.stream, c.width, c.height) == (42, 1920, 1080)
    devices = portal.sent("SelectDevices")[0][-1]
    assert devices["types"] == ("u", 3) and devices["persist_mode"] == ("u", 2) and "restore_token" not in devices
    assert any("Allow" in line for line in logs)
    token = tmp_path / "permission"
    assert token.read_text() == "restore-1"
    if os.name == "posix":
        assert token.stat().st_mode & 0o077 == 0
    logs.clear()
    control(tmp_path, logs)  # next time: the desktop's saved answer, no question
    assert portal.sent("SelectDevices")[1][-1]["restore_token"] == ("s", "restore-1") and logs == []


def test_clicking_typing_and_scrolling(desktop, tmp_path):
    portal = desktop()
    gui = wayland.WaylandGui(control(tmp_path))
    gui.click(x=100, y=200)
    session = portal.sent("NotifyPointerMotionAbsolute")[0][0]
    assert portal.sent("NotifyPointerMotionAbsolute")[-1] == (session, {}, 42, 100.0, 200.0)
    assert [b[2:] for b in portal.sent("NotifyPointerButton")] == [(0x110, 1), (0x110, 0)]
    gui.click(clicks=2, button="right")
    assert len(portal.sent("NotifyPointerButton")) == 6
    gui.scroll(3)  # pyautogui: up
    assert portal.sent("NotifyPointerAxisDiscrete")[-1][2:] == (0, -3)
    gui.write("Hi!\n")
    keys = [b[2:] for b in portal.sent("NotifyKeyboardKeysym")]
    shift = 0xFFE1
    assert keys == [(shift, 1), (ord("H"), 1), (ord("H"), 0), (shift, 0), (ord("i"), 1), (ord("i"), 0),
                    (shift, 1), (ord("!"), 1), (ord("!"), 0), (shift, 0), (0xFF0D, 1), (0xFF0D, 0)]
    gui.hotkey("ctrl", "s")
    assert [b[2:] for b in portal.sent("NotifyKeyboardKeysym")][-4:] == [(0xFFE3, 1), (ord("s"), 1), (ord("s"), 0), (0xFFE3, 0)]
    gui.click(x=5000, y=-20)  # off the screen: kept on it
    assert portal.sent("NotifyPointerMotionAbsolute")[-1][3:] == (1919.0, 0.0)


def test_desktops_that_only_take_key_codes(desktop, tmp_path):
    portal = desktop(keysyms=False)
    gui = wayland.WaylandGui(control(tmp_path))
    gui.write("a!")
    codes = [b[2:] for b in portal.sent("NotifyKeyboardKeycode")]
    assert codes == [(30, 1), (30, 0), (42, 1), (2, 1), (2, 0), (42, 0)]  # a, then shift+1


def test_saying_no_turns_screen_control_off(desktop, tmp_path):
    desktop(allow=False)
    with pytest.raises(wayland.PortalDenied):
        control(tmp_path)


def test_screenshot_through_the_desktop(desktop, tmp_path, monkeypatch):
    desktop()
    monkeypatch.setattr(wayland, "SCREENSHOT_TOOLS", [])  # no Spectacle, grim or gnome-screenshot
    image = wayland.grab()
    assert image.size == (2560, 1440)
    assert not (tmp_path / "Screenshot from v.png").exists()  # it was only for v: not left in Pictures


def test_screenshot_with_spectacle_first(monkeypatch, tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    tool = fake / "spectacle"
    tool.write_text('#!/bin/sh\nfor a; do last="$a"; done\n'
                    f'"{os.sys.executable}" -c "from PIL import Image; Image.new(\'RGB\', (800, 600)).save(\'$last\')"\n')
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(wayland, "SCREENSHOT_TOOLS", wayland.SCREENSHOT_TOOLS[:1])
    assert wayland.grab(portal_factory=lambda: pytest.fail("should not need the portal")).size == (800, 600)


def test_the_free_brain_clicks_through_it(desktop, tmp_path, monkeypatch):
    from v.tools import Screen

    portal = desktop(size=(1280, 720))
    monkeypatch.setattr(wayland, "TOKEN_FILE", tmp_path / "permission")
    monkeypatch.setattr(wayland, "grab", lambda: Image.new("RGB", (2560, 1440), "white"))  # 2x scaling

    class OCR:
        def __call__(self, image):
            return [([[1180, 1180], [1380, 1180], [1380, 1220], [1180, 1220]], "Save", 0.99)], 0.1

    computer = wayland.wayland_computer(log=lambda m: None)
    assert (computer.width, computer.height) == (1280, 720)
    screen = Screen(computer, ocr=OCR())
    assert "1: Save @ 640,600" in screen.look()[0]
    screen.click(element=1)
    assert portal.sent("NotifyPointerMotionAbsolute")[-1][3:] == (640.0, 600.0)


@pytest.mark.parametrize("env,expected", [
    ({"XDG_SESSION_TYPE": "wayland"}, True), ({"WAYLAND_DISPLAY": "wayland-0"}, True),
    ({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}, False), ({}, False),
])
def test_knows_when_it_is_on_wayland(env, expected):
    assert wayland.is_wayland(env, "linux") is expected
    assert wayland.is_wayland({"XDG_SESSION_TYPE": "wayland"}, "darwin") is False


def test_v_screen_test_on_wayland(desktop, tmp_path, monkeypatch, capsys):
    from v.cli import main

    portal = desktop(size=(1280, 720))
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "spectacle").write_text('#!/bin/sh\nfor a; do last="$a"; done\n'
                                    f'"{os.sys.executable}" -c "from PIL import Image; Image.new(\'RGB\', (2560, 1440)).save(\'$last\')"\n')
    (fake / "spectacle").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(wayland, "TOKEN_FILE", tmp_path / "permission")
    monkeypatch.setattr(wayland, "SCREENSHOT_TOOLS", wayland.SCREENSHOT_TOOLS[:1])
    assert main(["screen-test", "--project", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Wayland session (KDE)" in out and "screenshot: 2560x1440" in out and "control: a 1280x720 screen" in out
    assert len(portal.sent("NotifyPointerMotionAbsolute")) == 5 and not portal.sent("NotifyPointerButton")  # moves, never clicks
    assert (tmp_path / "permission").read_text() == "restore-1"
