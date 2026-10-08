"""`v` — talk to your computer. Free: it runs on an open model on your own machine.

    v setup                get a free model ready for this computer (one time)
    v                      voice conversation about the project in the current directory
    v --project ~/code/app --goal "Ship the signup flow by Friday"
    v --text               type instead of talking (add --speak to still hear replies)
    v --phone              talk to v from your phone (iPhone or Android): scan the QR code
    v --brain claude       use Claude instead (needs an Anthropic API key, paid per use)
    v say "hello"          test the voice
    v listen               test the microphone + speech recognition
    v doctor               check what's installed and what's missing
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="v", description="Free voice assistant for your computer and your projects.")
    p.add_argument("--project", type=Path, default=Path.cwd(), help="project directory (default: current directory)")
    p.add_argument("--goal", help="overall project goal; saved to <project>/.v/goal.txt for next time")
    p.add_argument("--text", action="store_true", help="type instead of using the microphone")
    p.add_argument("--speak", action="store_true", help="in --text mode, still speak replies")
    p.add_argument("--no-computer", action="store_true", help="don't let v see the screen or use mouse/keyboard")
    p.add_argument("--phone", action="store_true", help="use your phone (iPhone or Android) as v's mic, speaker and screen")
    p.add_argument("--port", type=int, default=8765, help="phone mode: port to serve on (default 8765)")
    p.add_argument("--http", action="store_true", help="phone mode: plain HTTP (typing and keyboard dictation still work)")
    p.add_argument("--cert", type=Path, help="phone mode: TLS certificate file, e.g. from `tailscale cert`")
    p.add_argument("--key", type=Path, help="phone mode: TLS private key file for --cert")
    p.add_argument("--new-token", action="store_true", help="phone mode: new pairing code (unpairs every phone)")
    p.add_argument(
        "--confirm",
        choices=CONFIRM_POLICIES,
        default="risky",
        help="ask before: all = every click/keypress, commands and sessions; "
        "risky = commands and sessions (default); none = never",
    )
    p.add_argument("--brain", choices=["local", "claude"], default="local",
                   help="local = free open model on this computer (default); claude = Anthropic API (paid)")
    p.add_argument("--local-url", help="use this OpenAI-compatible model server, e.g. http://127.0.0.1:1234")
    p.add_argument("--local-model", help="model name on the local server (e.g. qwen3:8b for Ollama)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Claude model, with --brain claude")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                   help="Claude effort level, with --brain claude")
    p.add_argument(
        "--session-mode",
        default="acceptEdits",
        help="permission mode for background Claude Code sessions (acceptEdits, auto, dontAsk, ...)",
    )
    p.add_argument("--stt-model", default="base.en", help="faster-whisper model size (tiny.en, base.en, small.en, ...)")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice name")
    sub = p.add_subparsers(dest="cmd")
    say = sub.add_parser("say", help="speak some text to test the voice")
    say.add_argument("words", nargs="+")
    sub.add_parser("listen", help="record one utterance and print the transcript")
    sub.add_parser("doctor", help="check dependencies, credentials and devices")
    sub.add_parser("setup", help="get a free model ready for this computer (one time)")
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
    if not _check("Kokoro voice", imports("kokoro")):
        _check("fallback voice (pyttsx3)", imports("pyttsx3"))
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
    import threading

    from . import phone
    from .confirm import Confirmer
    from .speech import Speaker
    from .ui import TeeUI, TerminalUI

    cfg = _config(args)
    hub, media = phone.Hub(), phone.MediaStore()

    synth = None
    try:
        from .speech import KokoroEngine

        synth = KokoroEngine(cfg.voice)
    except Exception as e:
        print(f"(Kokoro voice unavailable: {e.__class__.__name__}; the phone will use its own voice)")
    transcriber = None
    try:
        from .speech import Transcriber

        print("Loading speech recognition…")
        transcriber = Transcriber(cfg.stt_model)
    except Exception as e:
        print(f"(Whisper unavailable: {e.__class__.__name__}; the phone will use its own speech recognition)")

    voice = phone.PhoneVoice(hub, media, synth)
    speaker = Speaker(voice)
    bridge = phone.Bridge(hub, transcriber, say=speaker.say)

    computer = _computer(cfg)
    agent, sessions, brain = _make_brain(
        cfg, args, say=speaker.say, confirmer=Confirmer(cfg.confirm, bridge.ask),
        ui=TeeUI(TerminalUI(), phone.PhoneUI(hub, media)), stop_speaking=speaker.stop, computer=computer,
    )
    bridge.agent = agent

    ip = phone.lan_ip()
    ssl_ctx = None
    scheme = "http"
    if args.cert or args.key:
        if not (args.cert and args.key):
            raise SystemExit("--cert and --key go together")
        ssl_ctx, scheme = phone.ssl_context(args.cert, args.key), "https"
    elif not args.http:
        try:
            ssl_ctx, scheme = phone.ssl_context(*phone.self_signed_cert(ip)), "https"
        except ImportError:
            print("(cryptography not installed, so serving plain HTTP: typing and keyboard dictation work,")
            print(" the talk button needs HTTPS. `pip install cryptography` to enable it.)")

    token = phone.load_token(new=args.new_token)
    hello = lambda: {  # noqa: E731
        "project": cfg.project_dir.name,
        "goal": cfg.goal,
        "stt": transcriber is not None,
        "voice": voice.mode,
        "computer": computer is not None,
        "confirm": cfg.confirm,
        "brain": brain,
    }
    try:
        server = phone.PhoneServer(("0.0.0.0", args.port), hub, media, bridge, token, hello, ssl_ctx)
    except OSError as e:
        raise SystemExit(f"Can't listen on port {args.port}: {e}. Try --port with another number.")

    url = f"{scheme}://{ip}:{args.port}/?t={token}"
    print(f"\nv · {cfg.project_dir}")
    print(f"brain: {brain}")
    print(f"goal: {cfg.goal or '(none yet)'}\n")
    print("Open this on your phone (same Wi-Fi), or scan the code:\n")
    if not phone.print_qr(url):
        print("(`pip install qrcode` to show a QR code here)")
    print(f"\n  {url}\n")
    if scheme == "https" and not args.cert:
        print("The first time, your phone warns about the certificate (it's v's own, made on this computer).")
        print("Tap Show Details / Advanced, then visit the site. Then Share → Add to Home Screen.\n")
    print("Anyone with this link can control this computer; keep it private. Ctrl+C to stop.\n")

    worker = threading.Thread(target=bridge.run_worker, daemon=True)
    worker.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        _stop_sessions(sessions)
        server.stopping.set()
        server.server_close()
        speaker.close()
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
    if args.phone:
        return cmd_phone(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
