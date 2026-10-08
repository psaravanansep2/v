import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from v.computer import (
    MAX_LONG_EDGE,
    MAX_PIXELS,
    NOT_EXECUTED,
    Computer,
    describe,
    run_batch,
    scale_factor,
    to_keys,
)


class FakeGui:
    def __init__(self, size=(1920, 1080)):
        self._size = size
        self.calls = []
        self.pos = (0, 0)

    def size(self):
        return self._size

    def position(self):
        return self.pos

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def make(size=(1920, 1080), physical=None, platform="linux"):
    physical = physical or size
    gui = FakeGui(size)
    grab = lambda: Image.new("RGB", physical, "white")  # noqa: E731
    return Computer(gui=gui, grab=grab, platform=platform), gui


def decode(block):
    return Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))


def block(name, id="t1", **input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input, toolset_name="computer")


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440), (3840, 2160), (1280, 800), (1024, 768)])
def test_scaled_screenshot_fits_limits(size):
    s = scale_factor(*size)
    w, h = round(size[0] * s), round(size[1] * s)
    assert max(w, h) <= MAX_LONG_EDGE
    assert w * h <= MAX_PIXELS * 1.01


def test_small_screens_are_not_upscaled():
    assert scale_factor(800, 600) == 1.0


def test_screenshot_is_model_sized_even_on_retina():
    computer, _ = make(size=(1512, 982), physical=(3024, 1964), platform="darwin")
    image = decode(computer.run("screenshot", {})[0])
    assert image.size == (computer.model_width, computer.model_height)


def test_click_maps_model_coordinates_to_screen():
    computer, gui = make(size=(1920, 1080))
    computer.run("left_click", {"coordinate": [500, 300]})
    name, _, kwargs = gui.calls[-1]
    assert name == "click"
    assert kwargs["x"] == round(500 / computer.scale)
    assert kwargs["y"] == round(300 / computer.scale)
    assert kwargs["button"] == "left" and kwargs["clicks"] == 1


def test_coordinates_are_clamped_to_screen():
    computer, _ = make(size=(1920, 1080))
    assert computer.to_logical([99999, -5]) == (1919, 0)


def test_double_click_and_modifiers_are_held_around_click():
    computer, gui = make()
    computer.run("double_click", {"coordinate": [10, 10], "text": "ctrl+shift"})
    names = [(c[0], c[1]) for c in gui.calls]
    assert names[0] == ("keyDown", ("ctrl",))
    assert names[1] == ("keyDown", ("shift",))
    assert names[2][0] == "click"
    assert gui.calls[2][2]["clicks"] == 2
    assert names[3:] == [("keyUp", ("shift",)), ("keyUp", ("ctrl",))]


def test_key_combo_and_repeat():
    computer, gui = make()
    computer.run("key", {"text": "ctrl+s"})
    computer.run("key", {"text": "Return", "repeat": 3})
    assert gui.calls[0] == ("hotkey", ("ctrl", "s"), {})
    assert gui.calls[1:] == [("press", ("enter",), {})] * 3


def test_scroll_directions():
    computer, gui = make()
    computer.run("scroll", {"scroll_direction": "down", "scroll_amount": 5, "coordinate": [100, 100]})
    assert gui.calls[0][0] == "moveTo"
    assert gui.calls[1] == ("scroll", (-5,), {})
    computer.run("scroll", {"scroll_direction": "right", "scroll_amount": 2})
    assert gui.calls[-1] == ("hscroll", (2,), {})


def test_cursor_position_reports_model_space():
    computer, gui = make(size=(1920, 1080))
    gui.pos = (960, 540)
    assert computer.run("cursor_position", {}) == f"[{round(960 * computer.scale)}, {round(540 * computer.scale)}]"


def test_zoom_returns_region_scaled_to_fit():
    computer, _ = make(size=(1920, 1080))
    image = decode(computer.run("zoom", {"region": [0, 0, 200, 100]})[0])
    assert image.width <= computer.model_width and image.height <= computer.model_height
    assert abs(image.width / image.height - 2.0) < 0.05


def test_zoom_rejects_empty_region():
    computer, _ = make()
    with pytest.raises(ValueError):
        computer.run("zoom", {"region": [50, 50, 10, 10]})


@pytest.mark.parametrize(
    "combo,platform,expected",
    [
        ("ctrl+shift+Escape", "linux", ["ctrl", "shift", "esc"]),
        ("Return", "linux", ["enter"]),
        ("super+space", "darwin", ["command", "space"]),
        ("super", "win32", ["win"]),
        ("Page_Down", "linux", ["pagedown"]),
        ("F5", "linux", ["f5"]),
        ("alt+Tab", "linux", ["alt", "tab"]),
        ("A", "linux", ["a"]),
        ("+", "linux", ["+"]),
    ],
)
def test_key_names(combo, platform, expected):
    assert to_keys(combo, platform) == expected


def test_batch_stops_at_first_failure_and_echoes_toolset_name():
    computer, _ = make()
    blocks = [
        block("left_click", "a", coordinate=[1, 1]),
        block("zoom", "b", region=[5, 5, 1, 1]),  # invalid -> fails
        block("type", "c", text="never typed"),
    ]
    results = run_batch(computer, blocks, approve=lambda b: True)
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert all(r["toolset_name"] == "computer" for r in results)
    assert "is_error" not in results[0]
    assert results[1]["is_error"] and results[1]["content"].startswith("Error:")
    assert results[2] == {
        "type": "tool_result",
        "tool_use_id": "c",
        "toolset_name": "computer",
        "content": NOT_EXECUTED,
        "is_error": True,
    }


def test_batch_declined_runs_nothing():
    computer, gui = make()
    blocks = [block("screenshot", "a"), block("left_click", "b", coordinate=[1, 1])]
    asked = []
    results = run_batch(computer, blocks, approve=lambda b: asked.append(b) or False)
    assert [b.name for b in asked[0]] == ["left_click"]  # only state-changing actions are put to the user
    assert gui.calls == []
    assert all(r["is_error"] for r in results)


def test_read_only_batch_never_asks():
    computer, _ = make()
    results = run_batch(computer, [block("screenshot")], approve=lambda b: pytest.fail("asked"))
    assert results[0]["content"][0]["type"] == "image"


def test_batch_without_computer():
    results = run_batch(None, [block("screenshot")], approve=lambda b: True)
    assert results[0]["is_error"] and results[0]["toolset_name"] == "computer"


def test_describe():
    assert describe("left_click", {"coordinate": [3, 4]}) == "left click at 3, 4"
    assert describe("key", {"text": "ctrl+s"}) == "press ctrl+s"


def test_pyautogui_import_survives_missing_tkinter(monkeypatch):
    # mouseinfo calls sys.exit() when tkinter is missing; v must block it instead.
    import sys
    import types

    from v.computer import import_pyautogui

    fake = types.ModuleType("pyautogui")
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    monkeypatch.setitem(sys.modules, "tkinter", None)  # import tkinter -> ImportError
    monkeypatch.delitem(sys.modules, "mouseinfo", raising=False)
    assert import_pyautogui() is fake
    assert sys.modules["mouseinfo"] is None
