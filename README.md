# v

A free voice assistant that gets things done on your computer. Talk to it
from the computer or from your phone. It answers out loud, works with your
files, apps and the web, and can use apps on your screen.

**Free:** v runs on open AI models on your own machine. No account, no
subscription, no API key, nothing sent to an AI company.

```
you> what's the weather tomorrow
v>   Tomorrow in Chicago: rain, high of 58, low of 47, 80 percent chance of rain.
you> find the PDF invoice I downloaded last week and tell me the total
v>   Found invoice-0423.pdf in Downloads. The total is 1,240 dollars, due May 30th.
you> remind me to pay it at 5 pm
v>   Okay, I'll remind you at 5:00 PM to pay it.
you> open Gmail
v>   Opening Gmail.
```

## What it can do

### Instantly, and right every time

These basics don't use the AI model at all, so they work in a split second
on any computer:

| Say | v does |
|---|---|
| "Open Spotify" · "open YouTube" · "open my downloads folder" | Opens the app, site or folder. Installed apps first, then well-known websites |
| "Search for cheap flights to Lisbon" | Opens the search results in your browser |
| "Set a timer for 10 minutes" · "how much time is left" · "cancel the timer" | Timers, out loud when they go off |
| "Remind me to call Mom in 20 minutes" · "remind me at 5 pm to take out the trash" | Reminders (while v is running) |
| "Take a note: the meeting moved to Thursday" · "read my notes" | Notes, saved to *v notes.md* in your Documents |
| "What's the weather" · "will it rain tomorrow" · "my city is Chicago" | Weather from [Open-Meteo](https://open-meteo.com) (free, no account) |
| "What's 15 percent of 240" · "23 times 47" | Exact math (AI models are bad at arithmetic) |
| "What time is it" · "what's the date" | Time and date |
| "Turn the volume up" · "mute" · "take a screenshot" | Volume and screenshots |
| "Stop" · "what can you do" | Quiet, or a quick tour |

### Everything else, with the AI model

| | |
|---|---|
| **Files** | Find, read and summarize files: text, PDF, Word. Write notes, edit documents, organize folders |
| **The web** | Search the web and read pages, so you get the answer and not a list of links |
| **Your screen** | Read what's on screen and click, type, press shortcuts and scroll, to fill forms and use apps |
| **Clipboard and shell** | Copy and paste for you; run commands, asking first |
| **Projects** | Remembers what you're working toward (`--goal`) and keeps its help pointed at it |
| **Phone** | Use any iPhone or Android phone as v's microphone, speaker and screen |

v asks before running commands or changing files outside your project
folder. If it isn't sure what you meant by an instant command ("open the
file I was working on yesterday"), the AI works it out instead of guessing.
Small free models sometimes use a tool slightly wrong; v understands the
common mistakes instead of failing on them.

## What you need

A computer that can run a free model. v picks the best one for your machine automatically:

| Your computer | Model v uses | Download |
|---|---|---|
| 8 GB RAM laptop | Qwen3 4B | 2.5 GB |
| 16 GB RAM, or 16 GB Mac | Qwen3 8B | 5 GB |
| 32 GB Mac, or 16 GB graphics card | gpt-oss 20B | 12 GB |
| 64 GB Mac, or 24 GB graphics card | Qwen3 30B | 19 GB |

All of these are open models under the Apache 2.0 license. Apple Silicon Macs and computers with a graphics card answer fastest; ordinary laptops work, just slower.

## Install

One step. Nothing to set up first; the installer brings everything v needs,
and nothing needs an admin password.

**Windows:** download [Install v.cmd](https://github.com/psaravanansep2/v/raw/main/Install%20v.cmd) and double-click it
(if Windows warns about an unknown publisher, choose *More info* → *Run anyway*). Or paste this into PowerShell:

```powershell
irm https://raw.githubusercontent.com/psaravanansep2/v/main/install.ps1 | iex
```

**Mac and Linux:** paste this into Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/psaravanansep2/v/main/install.sh | sh
```

The installer gets [uv](https://docs.astral.sh/uv/) (which brings its own
Python), installs v, downloads the free model that fits your computer (a few
GB, one time), adds a **v icon** to your apps, and opens v.

Already use [Ollama](https://ollama.com) or LM Studio? v finds them and uses
their models. For the most natural voice, install with `V_KOKORO=1` in front
of the command (a bigger download).

Platform notes:

- **macOS:** the first time v uses the screen, allow it under System Settings → Privacy & Security → *Accessibility* and *Screen Recording*.
- **Linux:** screen control needs an X11 session.

Check that everything works with a short real conversation with your model:

```bash
v check
```

## Use

Click the **v icon**. v opens in its own window: tap the microphone and
talk, or type. The phone button shows a code to scan with your phone.

From a terminal:

```bash
v app                               # the v window (same as the icon)
v                                   # talk in the terminal (listens when you speak, stops when you pause)
v --goal "Plan the move to Lisbon"  # set what you're working toward; v remembers it for this folder
v --text                            # type in the terminal instead of talking
v --phone                           # phones only, no window
```

- **Interrupt:** Stop in the window, or Ctrl+C in the terminal. Saying "goodbye" ends a terminal session.
- **Emergency stop for screen control:** slam the mouse into a screen corner.

### From your phone (iPhone or Android)

In v's window, click the phone button and scan the code with your phone's
camera (same Wi-Fi as the computer). Or run `v --phone`, which prints the
code in the terminal. v opens in the phone's browser: no app to install, no
app store.

- **Talk:** tap the big button and speak.
- **Watch:** replies appear as they come, with a preview of your computer's screen while v works on it.
- **Approve:** tap Yes or No when v asks before doing something.
- **Stop:** tap Stop to interrupt v.
- **Install:** keep it one tap away with Share → *Add to Home Screen* (iPhone) or ⋮ → *Add to Home screen* (Android).

Things to know:

- **First visit:** your phone warns about the certificate. v makes its own HTTPS certificate on your computer, because phones only allow the microphone on secure pages. Tap *Show Details* → *visit this website* (iPhone) or *Advanced* → *Proceed* (Android) once.
- **Away from home:** use [Tailscale](https://tailscale.com) (free for personal use) and `v --phone --cert host.crt --key host.key` with its certificate (from `tailscale cert`). Don't expose the port to the internet.
- **The link is a key:** anyone with it can control your computer, so keep it private. `v --phone --new-token` makes a new one and unpairs every phone.
- **v can't control the phone itself.** The phone is a remote for your computer; iPhones don't let apps control other apps.

### What v asks before doing

`--confirm` decides which actions need your "yes" first:

| Policy | Asks before |
|---|---|
| `risky` (default) | commands, and changing files outside the project folder |
| `all` | the above, plus every file change, click, keystroke and drag |
| `none` | nothing |

Anything that isn't a clear yes counts as no.

### Options

| Flag | Default | |
|---|---|---|
| `--project DIR` | current folder | Folder v works in |
| `--goal TEXT` | saved goal | Set and save what you're working toward |
| `--local-model NAME` | best fit | A specific model, e.g. `qwen3:14b` with Ollama |
| `--local-url URL` | auto | Any OpenAI-compatible server, e.g. LM Studio at `http://127.0.0.1:1234` |
| `--no-computer` | off | Don't let v see the screen or use the mouse and keyboard |
| `--stt-model` | `base.en` | Speech recognition size: `tiny.en` (fastest) … `small.en` (most accurate) |
| `--voice` | `af_heart` | Kokoro voice (`am_michael`, `bf_emma`, …) |
| `--phone`, `--port`, `--cert`, `--key`, `--http`, `--new-token` | | Phone mode (above) |

## How it works

| Part | What runs it | Where |
|---|---|---|
| Listening | Microphone + [faster-whisper](https://github.com/SYSTRAN/faster-whisper) speech recognition | Your computer (or the phone's own) |
| Speaking | The browser's or computer's voice, or the [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) neural voice (`V_KOKORO=1`), sentence by sentence while the reply streams | Your computer or phone |
| Thinking | An open model through Ollama, LM Studio, or llama.cpp via cheapstack | Your computer |
| Seeing the screen | [RapidOCR](https://github.com/RapidAI/RapidOCR) reads the text on screen; v clicks things by number | Your computer |
| Web search | DuckDuckGo's plain HTML results | The internet |

Everything except web search and page reading stays on your machine.

**Honest limits:** free models are less capable than paid frontier models.
They handle everyday tasks well, but long multi-step jobs and complicated
screen work go wrong more often. Screen control reads text, so it can't
click icons that have no label.

## Optional: Claude

If you have an Anthropic API key, `v --brain claude` uses Claude Opus 5.5
instead (paid per use). It sees the screen directly, handles harder tasks,
and can run Claude Code sessions in the background for coding work. Set
`ANTHROPIC_API_KEY` first. Options: `--model`, `--effort`, `--session-mode`.

## License

MIT; see [LICENSE](LICENSE).

## Development

```bash
pip install -e ".[all,dev]"
pytest
v check     # end to end with a real local model
```

CI runs the tests on macOS, Windows and Linux, runs both installers, and
runs `v check` against a real free model (Qwen3 4B) through Ollama and
through v's own llama.cpp runner.

The tests use fakes for the mic, speakers, screen, OCR, model servers
(OpenAI-style and Ollama) and the `claude` binary, talking over real HTTP.
The Claude agent tests run the real Anthropic SDK against a fake streaming
server.
