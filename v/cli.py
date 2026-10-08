"""`v` — talk to your computer. Free: it runs on an open model on your own machine.

    v setup                get a free model ready for this computer (one time)
    v                      voice conversation about the project in the current directory
    v --project ~/code/app --goal "Ship the signup flow by Friday"
    v --text               type instead of talking (add --speak to still hear replies)
    v app                  open v in its own window (and pair your phone from there)
    v --phone              talk to v from your phone (iPhone or Android): scan the QR code
    v install-shortcut     put a v icon on the desktop / in the app menu
    v --brain claude       use Claude instead (needs an Anthropic API key, paid per use)
    v say "hello"          test the voice
    v listen               test the microphone + speech recognition
    v doctor               check what's installed and what's missing
    v check                test v end to end with your real model
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import CONFIRM_POLICIES, DEFAULT_MODEL, Config, load_goal, save_goal

EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop listening", "goodbye v", "bye v"}


def _add_options(p: argparse.ArgumentParser, top: bool) -> None:
    """The options every command shares. They're accepted before or after the
    command name (`v --text check` and `v check --text`); on commands, the
    defaults are left to the top level so they don't overwrite it."""

    def opt(*names, default=None, **kwargs):
        p.add_argument(*names, default=default if top else argparse.SUPPRESS, **kwargs)

    opt("--project", type=Path, default=Path.cwd(), help="project directory (default: current directory)")
    opt("--goal", help="overall project goal; saved to <project>/.v/goal.txt for next time")
    opt("--text", action="store_true", default=False, help="type instead of using the microphone")
    opt("--speak", action="store_true", default=False, help="in --text mode, still speak replies")
    opt("--no-computer", action="store_true", default=False, help="don't let v see the screen or use mouse/keyboard")
    opt("--phone", action="store_true", default=False, help="use your phone (iPhone or Android) as v's mic, speaker and screen")
    opt("--port", type=int, default=8765, help="phone mode: port to serve on (default 8765)")
    opt("--http", action="store_true", default=False, help="phone mode: plain HTTP (typing and keyboard dictation still work)")
    opt("--cert", type=Path, help="phone mode: TLS certificate file, e.g. from `tailscale cert`")
    opt("--key", type=Path, help="phone mode: TLS private key file for --cert")
    opt("--new-token", action="store_true", default=False, help="phone mode: new pairing code (unpairs every phone)")
    opt("--confirm", choices=CONFIRM_POLICIES, default="risky",
        help="ask before: all = every click/keypress, commands and file changes; risky = commands and changes "
        "outside the project (default); none = never")
    opt("--brain", choices=["local", "claude"], default="local",
        help="local = free open model on this computer (default); claude = Anthropic API (paid)")
    opt("--local-url", help="use this OpenAI-compatible model server, e.g. http://127.0.0.1:1234")
    opt("--local-model", help="model name on the local server (e.g. qwen3:8b for Ollama)")
    opt("--model", default=DEFAULT_MODEL, help="Claude model, with --brain claude")
    opt("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude effort level, with --brain claude")
    opt("--session-mode", default="acceptEdits",
        help="permission mode for background Claude Code sessions (acceptEdits, auto, dontAsk, ...)")
    opt("--stt-model", default="base.en", help="faster-whisper model size (tiny.en, base.en, small.en, ...)")
    opt("--voice", default="af_heart", help="Kokoro voice name")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="v", description="Free voice assistant for your computer and your projects.")
    _add_options(p, top=True)
    sub = p.add_subparsers(dest="cmd")

    def command(name: str, help: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help, description=help)
        _add_options(c, top=False)
        return c

    command("say", "speak some text to test the voice").add_argument("words", nargs="+")
    command("listen", "record one utterance and print the transcript")
    command("doctor", "check dependencies, credentials and devices")
    command("setup", "get a free model ready for this computer (one time)")
    command("check", "test v end to end with your real model (a short conversation)")
    command("app", "open v in its own window (and for your phone)").add_argument(
        "--background", action="store_true", help=argparse.SUPPRESS)
    command("install-shortcut", "put a v icon on the desktop / in the app menu")
    return p


def _config(args) -> Config:
    project_dir = args.project.expanduser().resolve()
    if not project_dir.is_dir():
        raise SystemExit(f"project directory not found: {project_dir}")
    if args.goal:
        save_goal(project_dir, args.goal)
    return Config(
        project_dir=project_dir,
        goal=args.goal or load_goal(project_dir),
        brain=args.brain,
        model=args.model,
        effort=args.effort,
        confirm=args.confirm,
        computer=not args.no_computer,
        session_permission_mode=args.session_mode,
        stt_model=args.stt_model,
        voice=args.voice,
    )


def _has_credentials(client) -> bool:
    # API key, auth token, or an `ant auth login` profile / workload identity
    return any(getattr(client, attr, None) is not None for attr in ("api_key", "auth_token", "credentials"))


def _make_brain(cfg: Config, args, *, say, confirmer, ui, stop_speaking, computer, log=print):
    """Instant commands in front of the agent for the chosen brain, plus
    Claude Code sessions (Claude only)."""
    agent, sessions, label = _make_agent(cfg, args, say=say, confirmer=confirmer, ui=ui, stop_speaking=stop_speaking,
                                         computer=computer, log=log)
    from .assistant import Assistant
    from .quick import QuickCommands

    quick = QuickCommands(say, ui, stop_speaking,
                          gui=computer.gui if computer is not None else None,
                          grab=computer.grab if computer is not None else None)
    return Assistant(agent, quick, ui, say), sessions, label


def _make_agent(cfg: Config, args, *, say, confirmer, ui, stop_speaking, computer, log=print):
    """The agent for the chosen brain, plus Claude Code sessions (Claude only)."""
    if cfg.brain == "claude":
        import anthropic

        from .agent import Agent
        from .sessions import SessionManager

        client = anthropic.Anthropic()
        if not _has_credentials(client):
            raise SystemExit(
                "No Claude API credentials found. Set ANTHROPIC_API_KEY, or leave out --brain claude to use "
                "the free local model."
            )
        sessions = SessionManager(
            cfg.project_dir, goal=lambda: cfg.goal, permission_mode=cfg.session_permission_mode, claude_bin=cfg.claude_bin
        )
        agent = Agent(client, cfg, say=say, confirmer=confirmer, computer=computer, sessions=sessions, ui=ui,
                      stop_speaking=stop_speaking)
        sessions.on_finish = agent.on_session_finish
        return agent, sessions, f"Claude ({cfg.model})"

    from .local import LocalClient, LocalError, ensure_server
    from .local_agent import LocalAgent
    from .tools import Screen, Toolbox

    try:
        server = ensure_server(log, url=args.local_url, model=args.local_model)
    except LocalError as e:
        raise SystemExit(str(e))
    toolbox = Toolbox(cfg, confirmer, ui, Screen(computer) if computer is not None else None)
    agent = LocalAgent(LocalClient(server), cfg, say, toolbox, ui=ui, stop_speaking=stop_speaking)
    return agent, None, f"{server.label} (free, on this computer)"


def _stop_sessions(sessions) -> None:
    if sessions is None:
        return
    running = [s for s in sessions.all() if s.status == "running"]
    if running:
        print(f"Stopping {len(running)} running session(s); resume them later with `claude --resume`.")
        sessions.stop_all()


def _computer(cfg: Config):
    if not cfg.computer:
        return None
    try:
        from .computer import Computer

        return Computer()
    except Exception as e:
        print(f"(screen control unavailable: {e.__class__.__name__}: {e})")
        return None


def cmd_setup(args) -> int:
    from .local import LocalError, ensure_server, machine_report

    report = machine_report()
    gpu = f"{report.vram_gb:.0f} GB graphics memory" if report.vram_gb else ("Apple Silicon" if report.apple else "no graphics card")
    print(f"This computer: {report.ram_gb:.0f} GB RAM, {gpu}.")
    print(f"Best free model for it: {report.choice.name} (~{report.choice.download_gb:.0f} GB download).")
    for note in report.notes:
        print(f"  {note}")
    try:
        server = ensure_server(print, url=args.local_url, model=args.local_model)
    except LocalError as e:
        raise SystemExit(str(e))
    print(f"\nReady: {server.label}. Run `v` to start talking, or `v --phone` to use your phone.")
    return 0


def cmd_say(args) -> int:
    from .speech import Speaker, make_engine

    speaker = Speaker(make_engine(args.voice))
    speaker.say(" ".join(args.words))
    speaker.wait()
    return 0


def cmd_listen(args) -> int:
    from .speech import Recorder, Transcriber

    print("Loading speech recognition…")
    transcriber = Transcriber(args.stt_model)
    print("Listening — say something.")
    print("you>", transcriber.transcribe(Recorder().record()))
    return 0


def _check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  ok   {label}{f': {detail}' if detail else ''}")
        return True
    except Exception as e:
        print(f"  --   {label}: {e.__class__.__name__}: {e}")
        return False


def cmd_doctor(args) -> int:
    print("v doctor")

    def creds():
        import anthropic

        if not _has_credentials(anthropic.Anthropic()):
            raise RuntimeError("set ANTHROPIC_API_KEY (or log in with `ant auth login`)")
        return "found"

    def claude_cli():
        path = shutil.which("claude")
        if not path:
            raise RuntimeError("`claude` not on PATH; install Claude Code to use background sessions")
        return subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()

    def imports(*names):
        def check():
            for name in names:
                __import__(name)

        return check

    def screen_control():
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("no DISPLAY set; run v from a desktop session to use screen control")
        from .computer import import_pyautogui

        gui = import_pyautogui()
        imports("mss", "PIL")()
        w, h = gui.size()
        return f"{w}x{h} display"

    def microphone():
        import sounddevice as sd

        return sd.query_devices(kind="input")["name"]

    def local_brain():
        from .local import find_running

        server = find_running(lambda *_: None)
        if server is None:
            raise RuntimeError("no free model running yet; run `v setup`")
        return server.label

    def machine():
        from .local import machine_report

        r = machine_report()
        return f"{r.ram_gb:.0f} GB RAM, {r.vram_gb:.0f} GB GPU -> {r.choice.name}"

    print(" free brain (default)")
    _check("this computer", machine)
    _check("local model", local_brain)
    def ollama():
        path = shutil.which("ollama")
        if not path:
            raise RuntimeError("not installed (optional, free: ollama.com)")
        return path

    _check("Ollama", ollama)
    _check("screen reading (RapidOCR)", imports("rapidocr_onnxruntime"))
    _check("PDF reading (pypdf)", imports("pypdf"))
    print(" Claude brain (optional, paid)")
    _check("Claude API credentials", creds)
    _check("Claude Code CLI", claude_cli)
    print(" voice and screen")
    _check("speech recognition (faster-whisper)", imports("faster_whisper"))
    _check("microphone", microphone)
    if not _check("built-in voice (pyttsx3)", imports("pyttsx3")):
        print("        (in v's window and on phones, the browser speaks instead)")
    try:
        __import__("kokoro")
        print("  ok   natural voice (Kokoro)")
    except ImportError:
        print("  --   natural voice (Kokoro): optional; install with V_KOKORO=1")
    _check("screen control (pyautogui, mss, pillow)", screen_control)
    return 0


def cmd_run(args) -> int:
    from .confirm import Confirmer
    from .speech import Speaker, SilentEngine, make_engine
    from .ui import TerminalUI

    cfg = _config(args)
    text_mode = args.text
    speaker = Speaker(make_engine(cfg.voice) if (not text_mode or args.speak) else SilentEngine())

    recorder = transcriber = None
    if not text_mode:
        from .speech import Recorder, Transcriber

        print("Loading speech recognition…")
        try:
            transcriber = Transcriber(cfg.stt_model)
            recorder = Recorder()
        except Exception as e:
            raise SystemExit(f"Voice input isn't available ({e.__class__.__name__}: {e}). Try `v doctor` or `v --text`.")

    def hear(prompt: str = "you> ") -> str:
        if text_mode:
            return input(prompt).strip()
        speaker.wait()
        print("(listening…)", flush=True)
        text = transcriber.transcribe(recorder.record(paused=speaker.is_speaking))
        print(f"{prompt}{text}")
        return text

    def ask(question: str) -> str:
        print(f"\nv? {question}")
        speaker.say(question)
        return hear("(yes/no)> " if text_mode else "you> ")

    agent, sessions, brain = _make_brain(
        cfg, args, say=speaker.say, confirmer=Confirmer(cfg.confirm, ask), ui=TerminalUI(),
        stop_speaking=speaker.stop, computer=_computer(cfg),
    )

    print(f"v · {cfg.project_dir}")
    print(f"brain: {brain}")
    print(f"goal: {cfg.goal or '(none yet)'}")
    print("Say \"what can you do\" for ideas. Ctrl+C interrupts v; Ctrl+C while it's listening (or \"goodbye\") quits.\n")
    greeting = f"Ready. We're working toward: {cfg.goal}" if cfg.goal else "Ready. What can I do for you?"
    print(f"v> {greeting}")
    speaker.say(greeting)

    try:
        while True:
            try:
                text = hear()
            except (KeyboardInterrupt, EOFError):
                break
            if not text:
                continue
            if text.lower().strip(" .!?,") in EXIT_PHRASES:
                break
            print("v> ", end="", flush=True)
            agent.turn(text)
    except Exception as e:
        if e.__class__.__name__ == "AuthenticationError":
            raise SystemExit("Claude API authentication failed. Set ANTHROPIC_API_KEY and try again.")
        raise
    finally:
        _stop_sessions(sessions)
        speaker.say("Bye.")
        speaker.wait()
        speaker.close()
    return 0


def cmd_phone(args) -> int:
    from .serve import run

    return run(_config(args), args, window=False)


def cmd_app(args) -> int:
    from .serve import app_main

    return app_main(_config(args), args)


def cmd_install_shortcut(args) -> int:
    from .desktop import install_shortcut

    created = install_shortcut()
    for path in created:
        print(f"Created {path}")
    print("Start v from its icon from now on." if created else "Couldn't find where to put the icon.")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "say":
        return cmd_say(args)
    if args.cmd == "listen":
        return cmd_listen(args)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "setup":
        return cmd_setup(args)
    if args.cmd == "app":
        return cmd_app(args)
    if args.cmd == "install-shortcut":
        return cmd_install_shortcut(args)
    if args.cmd == "check":
        from .check import run_check

        return run_check(url=args.local_url, model=args.local_model, log=lambda line: print(line, flush=True))
    if args.phone:
        return cmd_phone(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
