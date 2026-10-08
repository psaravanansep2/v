import json
from datetime import datetime

import pytest
from PIL import Image

from v.apps import AppFinder, resolve
from v.assistant import Assistant
from v.quick import (
    QuickCommands,
    format_duration,
    normalize,
    parse_clock,
    parse_duration,
    safe_eval,
    spoken_math,
    words_to_numbers,
)

NOW = datetime(2026, 10, 8, 14, 20)  # a Thursday afternoon


class UI:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *a: self.events.append((name, *a))


class Gui:
    def __init__(self):
        self.presses = []

    def press(self, key, presses=1):
        self.presses.append((key, presses))


WEATHER = {
    "current": {"temperature_2m": 61.4, "weather_code": 2},
    "daily": {"weather_code": [2, 63], "temperature_2m_max": [66.2, 58], "temperature_2m_min": [50.1, 47],
              "precipitation_probability_max": [10, 80]},
}


@pytest.fixture
def quick(tmp_path):
    apps_dir = tmp_path / "Applications"
    for name in ("Spotify", "Google Chrome", "Visual Studio Code", "Maps"):
        (apps_dir / f"{name}.app").mkdir(parents=True)
    home = tmp_path / "home"
    for folder in ("Downloads", "Documents", "Desktop"):
        (home / folder).mkdir(parents=True)
    calls = {"opened": [], "spoken": [], "fetched": [], "stopped": 0}

    def fetch(url):
        calls["fetched"].append(url)
        if "geocoding" in url:
            name = url.rsplit("name=", 1)[-1]
            if name == "Atlantis":
                return json.dumps({"results": []}), "application/json"
            cc = "US" if name == "Chicago" else "FR"
            return json.dumps({"results": [{"name": name, "latitude": 1, "longitude": 2, "country_code": cc}]}), "application/json"
        return json.dumps(WEATHER), "application/json"

    def stop():
        calls["stopped"] += 1

    q = QuickCommands(calls["spoken"].append, UI(), stop, finder=AppFinder("darwin", home, [apps_dir]),
                      launcher=calls["opened"].append, fetch=fetch, gui=Gui(), grab=lambda: Image.new("RGB", (40, 30)),
                      now=lambda: NOW, state_dir=tmp_path / "state", home=home)
    q.calls = calls
    yield q
    q._cancel_timers(None)


@pytest.mark.parametrize(
    "said",
    [
        "open the file I was working on yesterday",
        "open the report from last week and summarize it",
        "what is the capital of France?",
        "remind me to email John about the project",  # no time given: the model asks
        "search my files for the tax return",
        "find the PDF invoice from last week",
        "write an email to my boss saying I'm sick",
        "summarize this webpage",
        "open Photoshop",  # not installed: the model can say so or find another way
        "run python",  # not an app to open; "run" goes to the model
        "",
    ],
)
def test_anything_unclear_goes_to_the_model(quick, said):
    assert quick.handle(said) is None
    assert quick.calls["opened"] == []


@pytest.mark.parametrize(
    "said,reply,opened",
    [
        ("Open Spotify.", "Opening Spotify.", "Spotify.app"),
        ("open chrome", "Opening Google Chrome.", "Google Chrome.app"),
        ("Hey v, open VS Code please.", "Opening Visual Studio Code.", "Visual Studio Code.app"),
        ("open maps", "Opening Maps.", "Maps.app"),  # the installed app wins over the website
        ("open youtube", "Opening YouTube.", "https://www.youtube.com"),
        ("open my downloads folder", "Opening your downloads folder.", "Downloads"),
        ("go to wikipedia dot org", "Opening wikipedia.org.", "https://wikipedia.org"),
        ("search for cheap flights to Lisbon", "Here are the results for cheap flights to Lisbon.",
         "https://www.google.com/search?q=cheap+flights+to+Lisbon"),
        ("google best pizza near me", "Here are the results for best pizza near me.", "search?q=best+pizza+near+me"),
        ("google docs", "Opening Google Docs.", "https://docs.google.com"),
    ],
)
def test_open_and_search(quick, said, reply, opened):
    assert quick.handle(said) == reply
    assert quick.calls["opened"][-1].endswith(opened)


@pytest.mark.parametrize(
    "said,reply",
    [
        ("What time is it?", "It's 2:20 PM."),
        ("what's the date today", "Today is Thursday, October 8, 2026."),
        ("What's 23 times 47?", "That's 1,081."),
        ("what is 15 percent of 240", "That's 36."),
        ("what's the square root of 144", "That's 12."),
        ("calculate two plus two", "That's 4."),
        ("what is 10 divided by 4", "That's 2.5."),
        ("what's 5 divided by 0", "That's undefined. You can't divide by zero."),
    ],
)
def test_time_date_math(quick, said, reply):
    assert quick.handle(said) == reply


def test_timers(quick):
    assert quick.handle("set a timer for 10 minutes") == "Timer set for 10 minutes."
    assert quick.handle("set a 90 second timer") == "Timer set for 1 minute and 30 seconds."
    assert quick.handle("how much time is left") == "1 minute and 30 seconds left on your 1 minute and 30 second timer."
    assert quick.handle("cancel all timers") == "Cancelled all 2 timers and reminders."
    assert quick.handle("cancel the timer") == "You don't have any timers or reminders."


def test_timer_goes_off(quick):
    import time

    assert quick.handle("timer 1 second") == "Timer set for 1 second."
    time.sleep(1.5)
    assert quick.calls["spoken"] == ["Time's up! Your 1 second timer is done."]
    assert ("notice", "Time's up! Your 1 second timer is done.") in quick.ui.events


def test_reminders(quick):
    assert quick.handle("remind me to call Mom in 20 minutes") == "Okay, I'll remind you in 20 minutes to call Mom."
    assert quick.handle("remind me at 5 pm to take out the trash") == "Okay, I'll remind you at 5:00 PM to take out the trash."
    assert quick.handle("remind me at noon to stretch") == "Okay, I'll remind you tomorrow at 12:00 PM to stretch."
    assert quick.handle("remind me in an hour and a half") == "Okay, I'll remind you in 1 hour and 30 minutes."
    assert quick.handle("remind me to check in on Sam in 2 hours") == "Okay, I'll remind you in 2 hours to check in on Sam."
    assert quick.handle("remind me in 5 minutes to go to the store") == "Okay, I'll remind you in 5 minutes to go to the store."
    assert quick.handle("remind me to meet Ana at the cafe at 6:30 pm") == "Okay, I'll remind you at 6:30 PM to meet Ana at the cafe."
    assert len(quick.timers) == 7


def test_notes_keep_the_users_words(quick):
    assert quick.handle("read my notes").startswith("You don't have any notes yet")
    assert quick.handle("take a note buy milk and eggs") == "Noted."
    assert quick.handle("note that the meeting moved to Thursday") == "Noted."
    assert quick.handle("read my notes") == "Your notes: buy milk and eggs. the meeting moved to Thursday."
    assert (quick.home / "Documents" / "v notes.md").read_text().count("\n") == 2


def test_weather(quick):
    assert quick.handle("what's the weather").startswith("Which city should I use")
    assert quick.handle("my city is Chicago") == "Got it, I'll use Chicago for the weather."
    assert quick.handle("what's the weather like") == (
        "It's 61 degrees and partly cloudy in Chicago. Today's high is 66 with a low of 50, and a 10 percent chance of rain."
    )
    assert "temperature_unit=fahrenheit" in quick.calls["fetched"][-1]  # US city
    assert quick.handle("weather tomorrow") == "Tomorrow in Chicago: rain, high of 58, low of 47, 80 percent chance of rain."
    assert quick.handle("will it rain tomorrow") == "Yes, rain is likely tomorrow in Chicago: 80 percent chance."
    assert quick.handle("what's the weather in Paris").endswith("in Paris. Today's high is 66 with a low of 50, and a 10 percent chance of rain.")
    assert "fahrenheit" not in quick.calls["fetched"][-1]  # Celsius outside the US
    assert quick.handle("weather in Atlantis") == "I couldn't find a place called Atlantis."


def test_weather_offline(quick):
    def offline(url):
        raise OSError("no network")

    quick._fetch = offline
    quick.save_setting("city", "Chicago")
    assert quick.handle("what's the weather") == "I couldn't get the weather right now. Check the internet connection."


def test_volume_screenshot_help_stop(quick):
    assert quick.handle("turn the volume up") == "Turned it up."
    assert quick.handle("mute") == "Done."
    assert quick._gui.presses == [("volumeup", 5), ("volumemute", 1)]
    assert quick.handle("take a screenshot") == "Saved a screenshot to your Desktop folder."
    assert len(list((quick.home / "Desktop").glob("Screenshot *.png"))) == 1
    assert quick.handle("what can you do").startswith("Instantly, I can open apps")
    assert quick.handle("stop") == "" and quick.calls["stopped"] == 1


def test_parsers():
    assert words_to_numbers("twenty five minutes") == "25 minutes"
    assert words_to_numbers("one hundred and twenty") == "120"
    assert words_to_numbers("five fifteen") == "5 15"  # a time, not 20
    assert words_to_numbers("three thousand two hundred") == "3200"
    assert words_to_numbers("salt and pepper") == "salt and pepper"
    assert parse_duration("half an hour") == 1800
    assert parse_duration("an hour and a half") == 5400
    assert parse_duration("ten minutes") == 600
    assert parse_duration("10 minutes to check the oven") is None
    assert parse_clock("five fifteen", NOW) == datetime(2026, 10, 8, 17, 15)
    assert parse_clock("8 in the morning", NOW) == datetime(2026, 10, 9, 8, 0)
    assert parse_clock("9 o'clock", NOW) == datetime(2026, 10, 8, 21, 0)
    assert parse_clock("noon", NOW) == datetime(2026, 10, 9, 12, 0)
    assert parse_clock("lunch time", NOW) is None
    assert spoken_math("the capital of France") is None
    assert safe_eval(spoken_math("2 to the power of 10")) == 1024
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")
    with pytest.raises(ValueError):
        safe_eval("9 ** 9999")  # no runaway numbers
    assert format_duration(90) == "1 minute and 30 seconds"
    assert format_duration(90, adjective=True) == "1 minute and 30 second"
    assert normalize("Hey v, could you please open Spotify?") == "open Spotify"


def test_linux_and_windows_app_finding(tmp_path):
    apps = tmp_path / "applications"
    apps.mkdir()
    (apps / "firefox.desktop").write_text("[Desktop Entry]\nName=Firefox Web Browser\nExec=firefox %u\n")
    (apps / "hidden.desktop").write_text("[Desktop Entry]\nName=Secret Tool\nNoDisplay=true\n")
    linux = AppFinder("linux", tmp_path, [apps])
    assert linux.find("firefox").endswith("firefox.desktop")
    assert linux.find("secret tool") is None
    assert resolve("the firefox app", linux, tmp_path)[2] == "Firefox Web Browser"

    start_menu = tmp_path / "Start Menu"
    (start_menu / "Microsoft Office").mkdir(parents=True)
    (start_menu / "Microsoft Office" / "Word.lnk").write_text("")
    (start_menu / "Uninstall Word.lnk").write_text("")
    windows = AppFinder("win32", tmp_path, [start_menu])
    assert windows.find("word").endswith("Word.lnk")
    assert windows.find("calculator") == "cmd:calc"


class FakeAgent:
    def __init__(self):
        self.turns, self.remembered = [], []

    def turn(self, text):
        self.turns.append(text)

    def remember(self, text, reply):
        self.remembered.append((text, reply))

    def cancel(self):
        pass


def test_assistant_routes_and_shares_history(quick):
    agent, spoken, ui = FakeAgent(), [], UI()
    assistant = Assistant(agent, quick, ui, spoken.append)
    assistant.turn("what time is it")
    assistant.turn("what is the capital of France")
    assert spoken == ["It's 2:20 PM."]
    assert agent.turns == ["what is the capital of France"]
    assert agent.remembered == [("what time is it", "It's 2:20 PM.")]
    assert ("text", "It's 2:20 PM.") in ui.events
