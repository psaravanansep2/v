# v

A voice assistant for your computer and your project. You talk, v answers
out loud, and it can act. It can look at your screen and use the mouse and
keyboard, run shell commands, and start Claude Code sessions that do the
coding work in the background. All of it is aimed at a project goal that v
remembers between runs.

```
you> the signup form still doesn't validate emails, can you get that fixed?
v>   I'll hand that to a coding session.
v?   Start a Claude Code session to add email validation to the signup form?
you> yes
v>   Started it. I'll tell you when it's done.
     … (you keep talking, or go get coffee)
v>   Session 1 finished. I added email validation with tests, and the suite passes.
you> great, open the signup page in the browser so I can see it
v>   Opening the browser now. … It's up on localhost 3000, the form is showing.
```

## How it works

| Part | What it does | Runs where |
|---|---|---|
| Listening | Mic capture with automatic start/stop (no push-to-talk), speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Your machine |
| Speaking | [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) neural voice, sentence by sentence while the reply is still streaming; OS voices via pyttsx3 as a fallback | Your machine |
| Thinking | Claude Opus 5.5 with a tool loop. It speaks short progress notes between steps so you hear what it's doing | Claude API |
| Screen control | Claude's computer toolset: screenshots, zoom, clicks, typing, keys, scrolling, dragging (pyautogui + mss) | Your machine |
| Coding work | Background `claude -p` sessions in your project directory, told the project goal, resumable for follow-ups | Your machine |
| Memory | The project goal is saved in `<project>/.v/goal.txt` and picked up on the next run | Your machine |
| Phone | Optional: any iPhone or Android phone becomes v's mic, speaker and screen through its web browser | Your phone + your machine |

## Install

```bash
pip install -e ".[all]"       # voice + screen control + phone
export ANTHROPIC_API_KEY=...   # or: ant auth login
v doctor                       # shows what's working and what's missing
```

- **Python 3.10+.** Kokoro needs Python 3.12 or older for now. On 3.13, v falls back to your OS voices automatically.
- **Claude Code** must be installed for background sessions (`claude` on your PATH).
- **macOS:** allow your terminal under System Settings → Privacy & Security → *Accessibility* (mouse and keyboard), *Screen Recording* (screenshots) and *Microphone*.
- **Linux:** needs an X11 session (pyautogui doesn't drive Wayland natively). Kokoro also wants `espeak-ng` installed for unusual words.

## Use

```bash
cd ~/code/my-app
v --goal "Ship the signup flow by Friday"   # first time: set the goal
v                                           # next time: it remembers
```

- Just talk; v starts listening when you speak and stops when you pause.
- **Ctrl+C** while v is working interrupts it. Ctrl+C while it's listening quits, and so does saying "goodbye".
- **Emergency stop for screen control:** slam the mouse into a screen corner.

## Use it from your phone (iPhone or Android)

```bash
v --phone
```

v prints a QR code. Scan it with your phone's camera (same Wi-Fi as the
computer) and v opens in the browser: no app to install, no app store.

- **Talk:** tap the big button and speak. It stops when you pause, and your words are transcribed on your computer.
- **Watch:** see v's replies and progress, plus the latest screenshot of your computer as v works.
- **Approve:** tap Yes or No when v asks before running something, or just say it.
- **Stop:** tap Stop to interrupt v, or mute its voice.
- **Install:** keep it one tap away with Share → *Add to Home Screen* (iPhone) or ⋮ → *Add to Home screen* (Android).

Things to know:

- **First visit:** your phone warns about the certificate. v makes its own HTTPS certificate on your computer, because phones only allow the microphone on secure pages. Tap *Show Details* → *visit this website* (iPhone) or *Advanced* → *Proceed* (Android) once.
- **Away from home:** use [Tailscale](https://tailscale.com) on both devices and pass its real certificate with `v --phone --cert host.crt --key host.key` (from `tailscale cert`). Don't expose the port to the internet.
- **The link is a key:** anyone with it can control your computer, so keep it private. `v --phone --new-token` makes a new one and unpairs every phone.
- **Missing pieces fall back gracefully:** without Whisper on the computer, the phone uses its own speech recognition. Without Kokoro, the phone reads replies in its own voice. With `--http`, typing and the keyboard's dictation key still work.
- **v can't control the phone itself.** The phone is a remote for your computer; iPhones don't allow apps to control other apps.

Other modes:

```bash
v --text                  # type instead of talking (add --speak to still hear replies)
v --no-computer           # no screen/mouse/keyboard access
v say "testing one two"   # test the voice
v listen                  # test the mic + speech recognition
```

### What v asks before doing

`--confirm` decides which actions need your spoken "yes" first:

| Policy | Asks before |
|---|---|
| `risky` (default) | shell commands, starting/continuing Claude Code sessions |
| `all` | the above, plus every click, keystroke and drag (batched into one question per step) |
| `none` | nothing |

Anything that isn't a clear yes counts as no, and v is told you declined.
Background sessions run with Claude Code's `acceptEdits` permission mode by
default: they can edit files in the project, but shell commands outside the
read-only set are refused. Use `--session-mode auto` to let Claude Code's
classifier approve commands instead.

### Options

| Flag | Default | |
|---|---|---|
| `--project DIR` | current dir | Project directory for commands and sessions |
| `--goal TEXT` | saved goal | Set and save the project goal |
| `--effort` | `medium` | Claude's effort level; `high` for harder autonomous work, `low` for snappier chat |
| `--model` | `claude-opus-5-5` | |
| `--stt-model` | `base.en` | faster-whisper size: `tiny.en` (fastest) … `small.en` / `medium.en` (most accurate) |
| `--voice` | `af_heart` | Kokoro voice (`am_michael`, `bf_emma`, …) |
| `--session-mode` | `acceptEdits` | Claude Code permission mode for sessions |
| `--phone` | off | Serve the phone page instead of using this computer's mic and speakers |
| `--port` | `8765` | Phone mode port |
| `--cert`, `--key` | self-signed | Phone mode TLS certificate (e.g. from Tailscale) |
| `--http` | off | Phone mode without HTTPS (no talk button; typing and keyboard dictation work) |
| `--new-token` | off | New phone pairing code |

## Notes on the Claude API usage

- Computer use goes through the `computer_toolset_20260801` toolset, the only form Opus 5.5 accepts on the Claude API. Screenshots are scaled to fit the image limits, and Claude's coordinates are mapped back to your screen, including on Retina displays.
- The conversation history is append-only. Opus 5.5 ties its thinking to the exact conversation, so old screenshots are cleared server-side with context editing (`clear_tool_uses_20250919`) instead of being pruned locally.
- `fallbacks: "default"` is on. If a request is declined by a safety classifier, the API retries it on a fallback model in the same call.
- Progress notes between tool calls come from `thinking.display: "updates"`, and v speaks them.

## License

MIT. See [LICENSE](LICENSE). v calls the Claude API, so each person running it needs their own Anthropic API key.

## Development

```bash
pip install -e ".[all,dev]"
pytest
```

The tests use fakes for the mic, speakers, screen and `claude` binary; the
phone server is tested over real HTTP and HTTPS connections. The
agent tests run the real Anthropic SDK against a fake HTTP server that
replays streaming responses. That checks both the request v sends and its
handling of real stream events.
