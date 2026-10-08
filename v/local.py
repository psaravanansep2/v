"""The free brain: an open model running on this computer.

No account, no API key, no per-use cost. v finds a model server that's
already running, or sets one up:

1. Already running: Ollama (ollama.com), LM Studio, or v's own llama.cpp
   server from an earlier run. They all speak the same chat API.
2. Ollama installed but no model yet: v asks Ollama to download one that
   fits this computer.
3. Nothing installed: v uses cheapstack to fetch a prebuilt llama.cpp and a
   model file, then runs llama-server itself (with --jinja, which is what
   turns on tool calling).

The model is picked from a short list of free models that are good at
using tools, sized to this machine's graphics memory, or to its RAM on
laptops and Apple Silicon Macs.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

STATE_DIR = Path.home() / ".v" / "local"
OLLAMA_URL = "http://127.0.0.1:11434"
LMSTUDIO_URL = "http://127.0.0.1:1234"
V_SERVER_PORT = 8098
CONTEXT_TOKENS = 16384


class LocalError(RuntimeError):
    pass


class NotEnoughDisk(LocalError):
    def __init__(self, choice, need_gb: float, free_gb: float):
        super().__init__(f"There isn't enough free disk space for {choice.name}: it needs about {need_gb:.0f} GB, "
                         f"and {free_gb:.1f} GB is free. Free some space and try again.")
        self.choice = choice


# --- which model ---------------------------------------------------------


@dataclass(frozen=True)
class ModelChoice:
    name: str  # what the user sees
    ollama: str  # Ollama model tag
    hf_repos: tuple  # GGUF repos to try for llama.cpp, in order
    quant: str
    download_gb: float
    min_memory_gb: float  # weights + 16K-token context + slack
    system_suffix: str = ""  # per-model switches (e.g. no long "thinking" before answering)
    # Big mixture-of-experts models: their experts may sit in RAM next to a graphics card that holds the rest.
    # They're too slow on a processor alone, so they need a graphics card or an Apple Silicon Mac.
    big: bool = False


# Free, open-licensed (Apache 2.0) models with solid tool calling in llama.cpp
# and Ollama. Biggest first; the first one that fits is used.
MODELS = [
    ModelChoice(
        "Qwen3 235B-A22B",
        "qwen3:235b",
        ("unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF", "Qwen/Qwen3-235B-A22B-GGUF"),
        "Q4_K_M",
        142,
        150,  # e.g. a 256 GB Mac Studio
        "/no_think",
        big=True,
    ),
    ModelChoice(
        "gpt-oss 120B",
        "gpt-oss:120b",
        ("ggml-org/gpt-oss-120b-GGUF", "unsloth/gpt-oss-120b-GGUF"),
        "mxfp4",
        63,
        68,  # e.g. a 96-128 GB Mac, or a PC with 128 GB of RAM and an 8 GB+ graphics card
        "Reasoning: low",
        big=True,
    ),
    ModelChoice(
        "Qwen3 30B-A3B",
        "qwen3:30b",
        ("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "Qwen/Qwen3-30B-A3B-GGUF"),
        "Q4_K_M",
        18.6,
        21,  # fits a 24 GB card: 18.6 GB weights + ~1.6 GB for a 16K context
        "/no_think",
    ),
    ModelChoice(
        "gpt-oss 20B",
        "gpt-oss:20b",
        ("ggml-org/gpt-oss-20b-GGUF", "unsloth/gpt-oss-20b-GGUF"),
        "mxfp4",
        12.1,
        15,
        "Reasoning: low",
    ),
    ModelChoice(
        "Qwen3 8B",
        "qwen3:8b",
        ("Qwen/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        "Q4_K_M",
        5.0,
        7,
        "/no_think",
    ),
    ModelChoice(
        "Qwen3 4B",
        "qwen3:4b",
        ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen/Qwen3-4B-GGUF"),
        "Q4_K_M",
        2.5,
        4,
        "/no_think",
    ),
]


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30
    return 8.0


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def gpu_memory_gb() -> float:
    try:
        from cheapstack.detect import detect_all
    except ImportError:
        return 0.0
    try:
        return sum((g.vram_mb or 0) for g in detect_all()) / 1024
    except Exception:
        return 0.0


def memory_budget_gb(ram_gb: float, vram_gb: float, apple: bool) -> float:
    """Memory a model may use without starving everything else."""
    if apple:
        # unified memory: the GPU shares RAM (macOS lets it have about 3/4 of a big Mac's memory)
        return ram_gb * (0.75 if ram_gb >= 64 else 0.65)
    return max(vram_gb * 0.9, ram_gb * 0.5)


def split_budget_gb(ram_gb: float, vram_gb: float, apple: bool) -> float:
    """Memory for a big mixture-of-experts model: on a PC its experts go in RAM and the rest on the
    graphics card, which needs room for that (8 GB+). Without a graphics card it's too slow."""
    if apple:
        return memory_budget_gb(ram_gb, vram_gb, apple)
    return vram_gb * 0.9 + ram_gb * 0.5 if vram_gb >= 8 else 0.0


def choose_model(budget_gb: Optional[float] = None, split_gb: Optional[float] = None) -> ModelChoice:
    if budget_gb is None:
        ram, vram, apple = total_ram_gb(), gpu_memory_gb(), is_apple_silicon()
        budget_gb, split_gb = memory_budget_gb(ram, vram, apple), split_budget_gb(ram, vram, apple)
    for model in MODELS:
        limit = (split_gb or 0.0) if model.big else budget_gb
        if model.min_memory_gb <= limit:
            return model
    return MODELS[-1]


def _tokens(tag: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9.]+", tag.lower()) if t]


def _matches(tag: str, model: ModelChoice) -> bool:
    family, size = model.ollama.split(":")
    return family in tag.lower() and size in _tokens(tag)  # "4b" must not match "14b"


def model_for_tag(tag: str) -> Optional[ModelChoice]:
    return next((m for m in MODELS if _matches(tag, m)), None)


# --- talking to a server ---------------------------------------------------

# Local servers: never route through a proxy.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_json(url: str, timeout: float = 1.5):
    try:
        with _opener.open(url, timeout=timeout) as resp:
            return json.load(resp)
    except (OSError, ValueError, urllib.error.URLError):
        return None


@dataclass
class LocalServer:
    base_url: str  # ends in /v1
    model: str
    kind: str  # ollama | lmstudio | llama.cpp | custom
    choice: Optional[ModelChoice] = None

    @property
    def label(self) -> str:
        name = self.choice.name if self.choice else self.model
        return f"{name} via {self.kind}"

    @property
    def system_suffix(self) -> str:
        if self.choice:
            return self.choice.system_suffix
        low = self.model.lower()
        if "qwen3" in low and "instruct-2507" not in low:
            return "/no_think"
        if "gpt-oss" in low:
            return "Reasoning: low"
        return ""


class StreamAccumulator:
    """Builds one assistant message from streamed chat-completion chunks.
    Handles servers that stream tool calls in pieces (llama.cpp, LM Studio)
    and those that send each call whole (Ollama)."""

    def __init__(self):
        self.content = ""
        self.calls: dict[int, dict] = {}
        self.finish_reason: Optional[str] = None

    def feed(self, chunk: dict) -> str:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        self.content += text
        for i, call in enumerate(delta.get("tool_calls") or []):
            index = call.get("index", i)
            existing = self.calls.get(index)
            if existing and call.get("id") and existing["id"] and call["id"] != existing["id"]:
                index = max(self.calls) + 1  # a new call sent without an index
            slot = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                slot["id"] = call["id"]
            fn = call.get("function") or {}
            name = fn.get("name")
            if name and name != slot["name"]:
                slot["name"] += name
            args = fn.get("arguments")
            if isinstance(args, dict):  # some servers send parsed arguments
                slot["arguments"] = json.dumps(args)
            elif args:
                slot["arguments"] += args
        return text

    def message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.calls:
            msg["tool_calls"] = [
                {
                    "id": slot["id"] or f"call_{index}_{int(time.time() * 1000) % 100000}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
                }
                for index, slot in sorted(self.calls.items())
            ]
        return msg


class LocalClient:
    def __init__(self, server: LocalServer, timeout: float = 600):
        self.server = server
        self.timeout = timeout
        # prompt tokens the server actually had to read (not reused from its cache), when it reports them:
        # on a computer without a GPU, this is what makes a reply slow to start
        self.prompt_tokens = 0
        self.prompt_seconds = 0.0

    def _open(self, url: str, body: dict):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            return _opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if "jinja" in detail:
                detail += " (start llama-server with --jinja to enable tool calling)"
            raise LocalError(f"the local model server said {e.code}: {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise LocalError(f"can't reach the local model at {self.server.base_url} ({e})") from None

    def stream_chat(self, messages: list[dict], tools: list[dict]) -> Iterator[dict]:
        """Yields OpenAI-style streaming chunks, whatever the server."""
        if self.server.kind == "ollama":
            yield from self._ollama_chat(messages, tools)
            return
        body = {
            "model": self.server.model,
            # "name" on tool results is only for Ollama's native format
            "messages": [{k: v for k, v in m.items() if not (m["role"] == "tool" and k == "name")} for m in messages],
            "stream": True,
            "temperature": 0.3,
        }
        if tools:  # some servers reject an empty list
            body["tools"] = tools
        resp = self._open(self.server.base_url + "/chat/completions", body)
        with resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                timings = chunk.get("timings") if isinstance(chunk, dict) else None
                if isinstance(timings, dict):  # llama-server
                    self.prompt_tokens += int(timings.get("prompt_n") or 0)
                    self.prompt_seconds += float(timings.get("prompt_ms") or 0) / 1000
                yield chunk


    def _ollama_chat(self, messages: list[dict], tools: list[dict]) -> Iterator[dict]:
        """Ollama's native chat API. Unlike its OpenAI-compatible endpoint it
        lets v ask for a 16K context; the default (as small as 2-4K tokens)
        silently cuts off the start of the conversation, instructions included."""
        native = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                native.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"function": {"name": c["function"]["name"], "arguments": _as_object(c["function"]["arguments"])}}
                        for c in m["tool_calls"]
                    ],
                })
            elif m["role"] == "tool":
                native.append({"role": "tool", "content": m.get("content") or "", "tool_name": m.get("name", "")})
            else:
                native.append({"role": m["role"], "content": m.get("content") or ""})
        body = {
            "model": self.server.model,
            "messages": native,
            "stream": True,
            "options": {"num_ctx": CONTEXT_TOKENS, "temperature": 0.3},
        }
        if tools:
            body["tools"] = tools
        if "qwen3" in self.server.model.lower():
            body["think"] = False  # answer right away instead of reasoning out loud first
        resp = self._open(self.server.base_url[: -len("/v1")] + "/api/chat", body)
        count = 0
        with resp:
            for raw in resp:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise LocalError(f"Ollama: {event['error']}")
                if event.get("done"):
                    self.prompt_tokens += int(event.get("prompt_eval_count") or 0)
                    self.prompt_seconds += float(event.get("prompt_eval_duration") or 0) / 1e9
                msg = event.get("message") or {}
                delta: dict = {"content": msg.get("content") or ""}
                calls = []
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    calls.append({
                        "index": count,
                        "id": f"call_{count}",
                        "function": {"name": fn.get("name", ""), "arguments": json.dumps(fn.get("arguments") or {})},
                    })
                    count += 1
                if calls:
                    delta["tool_calls"] = calls
                yield {"choices": [{"delta": delta, "finish_reason": "stop" if event.get("done") else None}]}


def _as_object(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


# --- finding or starting a server ---------------------------------------------


def _first_model(base: str) -> Optional[str]:
    data = _get_json(base + "/v1/models")
    if isinstance(data, dict) and data.get("data"):
        return data["data"][0].get("id")
    return None


def ollama_models() -> Optional[list[str]]:
    """Installed Ollama models, or None if Ollama isn't running."""
    data = _get_json(OLLAMA_URL + "/api/tags")
    if data is None:
        return None
    return [m.get("name", "") for m in data.get("models", [])]


def pick_ollama_model(installed: list[str]) -> Optional[str]:
    for model in MODELS:
        for name in installed:
            if _matches(name, model):
                return name
    for name in installed:  # anything else from a tool-capable family
        if any(f in name for f in ("qwen3", "gpt-oss", "qwen2.5", "llama3.1", "llama3.2", "mistral")):
            return name
    return None


def find_running(log: Callable[[str], None] = print) -> Optional[LocalServer]:
    model = _first_model(f"http://127.0.0.1:{V_SERVER_PORT}")
    if model:
        return LocalServer(f"http://127.0.0.1:{V_SERVER_PORT}/v1", model, "llama.cpp", model_for_tag(model))
    installed = ollama_models()
    if installed:
        name = pick_ollama_model(installed)
        if name:
            return LocalServer(OLLAMA_URL + "/v1", name, "ollama", model_for_tag(name))
    model = _first_model(LMSTUDIO_URL)
    if model:
        return LocalServer(LMSTUDIO_URL + "/v1", model, "lmstudio", model_for_tag(model))
    return None


def custom_server(url: str, model: Optional[str]) -> LocalServer:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    model = model or _first_model(base[: -len("/v1")])
    if not model:
        raise LocalError(f"no model found at {url}; pass --local-model")
    return LocalServer(base, model, "custom", model_for_tag(model))


def _start_ollama(log) -> bool:
    exe = shutil.which("ollama")
    if not exe:
        return False
    log("Starting Ollama…")
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(40):
        if ollama_models() is not None:
            return True
        time.sleep(0.5)
    return False


def ollama_pull(tag: str, log) -> None:
    log(f"Downloading {tag} with Ollama (one time)…")
    req = urllib.request.Request(
        OLLAMA_URL + "/api/pull", data=json.dumps({"model": tag}).encode(), headers={"Content-Type": "application/json"}
    )
    last = -10
    with _opener.open(req, timeout=None) as resp:
        for raw in resp:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise LocalError(f"Ollama couldn't download {tag}: {event['error']}")
            total, done = event.get("total"), event.get("completed")
            if total and done:
                pct = int(done * 100 / total)
                if pct >= last + 10:
                    log(f"  {pct}%  ({done / 1e9:.1f} of {total / 1e9:.1f} GB)")
                    last = pct


def download_with_progress(url: str, dest: Path, log, expected_size: Optional[int] = None) -> Path:
    """Resumable download: an interrupted multi-GB download continues where it stopped."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "v"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored the range: start over
            have = 0
        total = expected_size or (int(resp.headers.get("Content-Length", 0)) + have) or None
        last = -10
        with open(part, "ab" if have else "wb") as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
                have += len(block)
                if total:
                    pct = int(have * 100 / total)
                    if pct >= last + 5:
                        log(f"  {pct}%  ({have / 1e9:.1f} of {total / 1e9:.1f} GB)")
                        last = pct
    part.rename(dest)
    return dest


HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"
_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def _hf_tree(repo: str) -> list:
    """Every file in a Hugging Face repo, subfolders included."""
    files, url = [], HF_TREE.format(repo=repo)
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "v"}), timeout=20) as resp:
            files += json.load(resp)
            links = re.findall(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link") or "")
        url = links[0] if links else None
    return files


def find_gguf_files(repo: str, quant: str, files: Optional[list] = None) -> list:
    """The GGUF file for `quant`, or all the parts of a model split into several files
    ("…-00001-of-00003.gguf"), in order. Big models come in parts, often in a folder."""
    files = _hf_tree(repo) if files is None else files
    want = re.compile(rf"(?<![a-z0-9]){re.escape(quant.lower())}(?![a-z0-9_])")
    sets: dict = {}
    for f in files:
        path = f.get("path", "")
        name = path.rsplit("/", 1)[-1].lower()
        if f.get("type") != "file" or not name.endswith(".gguf") or "mmproj" in name or not want.search(name):
            continue
        part = _PART.search(name)
        key = path[: part.start()] if part else path[: -len(".gguf")]
        sets.setdefault(key, {"total": int(part.group(2)) if part else 1, "parts": []})["parts"].append(
            (int(part.group(1)) if part else 1, {"path": path, "size": f.get("size") or (f.get("lfs") or {}).get("size")}))
    complete = {k: v for k, v in sets.items() if len(v["parts"]) == v["total"]}
    if not complete:
        raise LocalError(f"no {quant} GGUF in {repo}")
    best = min(complete, key=len)  # the plainest name; longer ones are variants
    return [info for _, info in sorted(complete[best]["parts"], key=lambda p: p[0])]


def free_disk_gb(path: Path) -> float:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def check_disk(choice: ModelChoice, need_gb: float, where: Path) -> None:
    free = free_disk_gb(where)
    if need_gb > 0 and free < need_gb * 1.02 + 3:  # leave a little room for everything else
        raise NotEnoughDisk(choice, need_gb, free)


def ollama_models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")


LLAMA_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
_OS_TOKENS = {"linux": {"ubuntu", "linux"}, "darwin": {"macos", "darwin", "osx", "mac"}, "win": {"win", "windows"}}
_ARCH_TOKENS = {"x64": {"x64", "x86", "amd64", "x86_64"}, "arm64": {"arm64", "aarch64"}}
# builds that need a particular GPU runtime installed; v uses the plain build (or Vulkan)
_RUNTIME_BUILDS = {"cuda", "cu11", "cu12", "cu13", "rocm", "hip", "sycl", "openvino", "opencl", "musa", "kompute",
                   "cann", "npu", "vulkan", "xcframework", "radeon", "intel"}


def _asset_tokens(name: str) -> set:
    return set(re.split(r"[-_.+]", name.lower()))


def pick_llama_asset(assets: list, platform: str = sys.platform, machine: str = platform.machine(),
                     prefer_vulkan: bool = False) -> dict:
    """The llama.cpp release download for this computer, matched by name parts
    so a renamed asset ("ubuntu" -> "linux", ".zip" -> ".tar.gz") still matches."""
    os_key = "darwin" if platform == "darwin" else "win" if platform.startswith("win") else "linux"
    arch = "arm64" if machine.lower() in ("arm64", "aarch64") else "x64"
    fitting = []
    for asset in assets:
        name = asset.get("name", "").lower()
        tokens = _asset_tokens(name)
        if not name.endswith((".zip", ".tar.gz", ".tgz")) or "bin" not in tokens:
            continue
        if not tokens & _OS_TOKENS[os_key] or not tokens & _ARCH_TOKENS[arch]:
            continue
        fitting.append((asset, tokens))
    if prefer_vulkan:
        for asset, tokens in fitting:
            if "vulkan" in tokens:
                return asset
    plain = [(a, t) for a, t in fitting if not t & _RUNTIME_BUILDS]
    if not plain:
        names = ", ".join(a.get("name", "") for a in assets[:40])
        raise LocalError(f"no llama.cpp download for {os_key}-{arch} in the latest release (assets: {names})")
    plain.sort(key=lambda at: ("cpu" not in at[1], len(at[0]["name"])))  # an explicit CPU build, else the plainest name
    return plain[0][0]


def _recent_llama_releases() -> list:
    headers = {"User-Agent": "v", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:  # optional: avoids GitHub's hourly limit for anonymous requests on shared machines
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(LLAMA_RELEASES, headers=headers), timeout=20) as resp:
        return json.load(resp)


def pick_llama_download(releases: list, prefer_vulkan: bool = False, **where) -> dict:
    """The newest release that has a build for this computer. The newest one
    can be incomplete while it uploads, or a nightly pointer with no builds."""
    error = None
    for release in releases:
        if release.get("draft"):
            continue
        try:
            return pick_llama_asset(release.get("assets", []), prefer_vulkan=prefer_vulkan, **where)
        except LocalError as e:
            error = error or e
    raise error or LocalError("no llama.cpp releases found")


def _server_info_path() -> Path:
    return STATE_DIR / "server.json"


def start_llama_server(choice: ModelChoice, log, smaller_ok: bool = False) -> LocalServer:
    try:
        return _start_llama_server(choice, log)
    except NotEnoughDisk as e:
        rest = MODELS[MODELS.index(choice) + 1:] if choice in MODELS else []
        if not (smaller_ok and rest):
            raise
        log(f"{e} Using {rest[0].name} instead.")
        return start_llama_server(rest[0], log, smaller_ok)
    except (urllib.error.URLError, OSError) as e:
        if isinstance(e, LocalError):
            raise
        raise LocalError(
            f"Couldn't download the free model ({e}). Check the internet connection and run `v setup` again "
            "(downloads resume where they stopped). Or install Ollama (free, ollama.com), which v uses automatically."
        ) from None


def _start_llama_server(choice: ModelChoice, log) -> LocalServer:
    try:
        from cheapstack import release
    except ImportError:
        raise LocalError(
            "No local model is running. The easiest fix: install Ollama (free, ollama.com) and run v again. "
            "Or install v's own model runner with: pip install 'v[local]'"
        ) from None

    bin_dir, model_dir = STATE_DIR / "bin", STATE_DIR / "models"
    binaries = list(bin_dir.rglob("llama-server.exe" if sys.platform.startswith("win") else "llama-server"))
    if binaries:
        binary = binaries[0]
    else:
        log("Fetching llama.cpp (one time)…")
        asset = pick_llama_download(_recent_llama_releases(), prefer_vulkan=gpu_memory_gb() > 0)
        log(f"  {asset['name']}")
        binary = release.download_and_extract(asset, bin_dir)

    model_path = None
    errors = []
    for repo in choice.hf_repos:
        try:
            parts = find_gguf_files(repo, choice.quant)
        except (LocalError, urllib.error.URLError, OSError, ValueError) as e:
            errors.append(f"{repo}: {e}")
            continue
        # parts keep their names, so llama.cpp finds the rest next to the first
        local = [(model_dir / info["path"].rsplit("/", 1)[-1], info) for info in parts]
        missing = [(path, info) for path, info in local if not (path.exists() and path.stat().st_size == info.get("size"))]
        part_gb = choice.download_gb / len(parts)
        check_disk(choice, sum((info.get("size") or part_gb * 1e9) / 1e9 for _, info in missing), model_dir)
        for path, info in missing:
            note = f", part {local.index((path, info)) + 1} of {len(local)}" if len(local) > 1 else ""
            log(f"Downloading {choice.name} ({choice.download_gb:.0f} GB{note}, one time)…")
            download_with_progress(HF_FILE.format(repo=repo, path=info["path"]), path, log, info.get("size"))
        model_path = local[0][0]
        break
    if model_path is None:
        raise LocalError(f"couldn't find a download for {choice.name}: " + "; ".join(errors))

    alias = choice.ollama.replace(":", "-")
    cmd = [
        str(binary),
        "-m", str(model_path),
        "--jinja",  # chat templates with tool calling
        "-c", str(CONTEXT_TOKENS),
        "-ngl", "999",  # use the GPU for as much as fits
        "--host", "127.0.0.1",
        "--port", str(V_SERVER_PORT),
        "--alias", alias,
    ]
    vram = gpu_memory_gb()
    if choice.big and not is_apple_silicon() and choice.min_memory_gb > vram * 0.9:
        cmd.append("--cpu-moe")  # experts in RAM, the rest on the graphics card
    log(f"Starting {choice.name}…")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    minutes = 10 if choice.big else 5  # big models take a while to load
    for _ in range(minutes * 120):
        if proc.poll() is not None:
            raise LocalError(f"llama-server exited while loading {choice.name} (code {proc.returncode})")
        health = _get_json(f"http://127.0.0.1:{V_SERVER_PORT}/health", timeout=1)
        if isinstance(health, dict) and health.get("status") == "ok":
            _server_info_path().write_text(json.dumps({"pid": proc.pid, "model": alias}), encoding="utf-8")
            return LocalServer(f"http://127.0.0.1:{V_SERVER_PORT}/v1", alias, "llama.cpp", choice)
        time.sleep(0.5)
    proc.terminate()
    raise LocalError(f"{choice.name} didn't finish loading in {minutes} minutes")


def ensure_server(log: Callable[[str], None] = print, url: Optional[str] = None, model: Optional[str] = None) -> LocalServer:
    """A ready-to-use local model server, setting one up if needed."""
    if url:
        return custom_server(url, model)
    if model is None:
        running = find_running(log)
        if running:
            return running
    choice = choose_model()
    if model:
        choice = model_for_tag(model) or choice  # e.g. --local-model qwen3:4b also picks the llama.cpp download
    installed = ollama_models()
    if installed is None and _start_ollama(log):
        installed = ollama_models()
    if installed is not None:
        tag = model or choice.ollama
        if not _ollama_has(installed, tag):
            if model is None:
                choice = _fit_disk(choice, ollama_models_dir(), log)
                tag = choice.ollama
            try:
                ollama_pull(tag, log)
            except (urllib.error.URLError, OSError) as e:
                raise LocalError(f"Ollama couldn't download {tag} ({e}). Check the internet connection and try again.") from None
        return LocalServer(OLLAMA_URL + "/v1", tag, "ollama", model_for_tag(tag))
    return start_llama_server(choice, log, smaller_ok=model is None)


def _ollama_has(installed: list, tag: str) -> bool:
    return any(name == tag or name.startswith(tag + "-") or name == tag + ":latest" for name in installed)


def _fit_disk(choice: ModelChoice, where: Path, log) -> ModelChoice:
    """The chosen model, or the biggest smaller one whose download fits on the disk (for Ollama)."""
    for candidate in MODELS[MODELS.index(choice):] if choice in MODELS else [choice]:
        try:
            check_disk(candidate, candidate.download_gb, where)
        except NotEnoughDisk as e:
            last = e
            continue
        if candidate is not choice:
            log(f"Not enough free disk space for {choice.name} ({free_disk_gb(where):.0f} GB free), "
                f"so using {candidate.name} instead.")
        return candidate
    raise last


def stop_own_server() -> bool:
    try:
        info = json.loads(_server_info_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        os.kill(info["pid"], 15)
    except OSError:
        pass
    _server_info_path().unlink(missing_ok=True)
    return True


@dataclass
class MachineReport:
    ram_gb: float
    vram_gb: float
    apple: bool
    budget_gb: float
    choice: ModelChoice
    notes: list = field(default_factory=list)


def machine_report() -> MachineReport:
    ram, vram, apple = total_ram_gb(), gpu_memory_gb(), is_apple_silicon()
    budget = memory_budget_gb(ram, vram, apple)
    choice = choose_model(budget, split_budget_gb(ram, vram, apple))
    report = MachineReport(ram, vram, apple, budget, choice)
    if budget < MODELS[-1].min_memory_gb:
        report.notes.append("This computer is below the comfortable minimum; replies will be slow.")
    if not vram and not apple:
        report.notes.append("No graphics card found, so the model runs on the processor (works, but slower).")
    return report
