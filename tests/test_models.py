"""Which free model a computer gets, and fetching it: single files, models
split into parts, big models split between graphics card and RAM, and
falling back to a smaller model when the disk is too full."""

import io
import json
import sys
import types

import pytest

from v import local


def pick(ram, vram, apple):
    budget, split = local.memory_budget_gb(ram, vram, apple), local.split_budget_gb(ram, vram, apple)
    return local.choose_model(budget, split).name


@pytest.mark.parametrize(
    "ram,vram,apple,expected",
    [
        (8, 0, False, "Qwen3 4B"),  # ordinary laptop
        (16, 0, False, "Qwen3 8B"),
        (16, 0, True, "Qwen3 8B"),  # 16 GB MacBook Air
        (32, 0, True, "gpt-oss 20B"),
        (64, 0, True, "Qwen3 30B-A3B"),
        (16, 24, False, "Qwen3 30B-A3B"),  # gaming PC with a 24 GB card
        (16, 12, False, "Qwen3 8B"),
        (96, 0, True, "gpt-oss 120B"),  # big Macs
        (128, 0, True, "gpt-oss 120B"),
        (192, 0, True, "gpt-oss 120B"),
        (256, 0, True, "Qwen3 235B-A22B"),
        (128, 8, False, "gpt-oss 120B"),  # PC: experts in 128 GB of RAM, the rest on an 8 GB card
        (128, 6, False, "Qwen3 30B-A3B"),  # card too small to hold the rest
        (128, 0, False, "Qwen3 30B-A3B"),  # no graphics card: the big ones would be too slow
        (64, 24, False, "Qwen3 30B-A3B"),
        (256, 32, False, "Qwen3 235B-A22B"),
    ],
)
def test_biggest_model_that_fits(ram, vram, apple, expected):
    assert pick(ram, vram, apple) == expected


def test_models_are_listed_biggest_first():
    sizes = [m.min_memory_gb for m in local.MODELS]
    assert sizes == sorted(sizes, reverse=True)
    for m in local.MODELS:
        assert m.download_gb < m.min_memory_gb


def files(*paths, size=100):
    return [{"type": "file", "path": p, "size": size} for p in paths]


def test_finds_the_plain_single_file():
    tree = files("README.md", "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                 "Qwen3-4B-Instruct-2507-IQ4_K_M.gguf", "mmproj-Q4_K_M.gguf", "Qwen3-4B-Instruct-2507-Q8_0.gguf")
    assert [f["path"] for f in local.find_gguf_files("r", "Q4_K_M", tree)] == ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"]


def test_finds_every_part_of_a_split_model_in_a_folder():
    tree = files("Q8_0/M-Q8_0-00001-of-00004.gguf", "Q4_K_M/M-Q4_K_M-00002-of-00003.gguf",
                 "Q4_K_M/M-Q4_K_M-00003-of-00003.gguf", "Q4_K_M/M-Q4_K_M-00001-of-00003.gguf",
                 "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf")
    assert [f["path"] for f in local.find_gguf_files("r", "Q4_K_M", tree)] == [
        "Q4_K_M/M-Q4_K_M-00001-of-00003.gguf", "Q4_K_M/M-Q4_K_M-00002-of-00003.gguf", "Q4_K_M/M-Q4_K_M-00003-of-00003.gguf"]
    gpt = files("gpt-oss-120b-mxfp4-00002-of-00003.gguf", "gpt-oss-120b-mxfp4-00001-of-00003.gguf",
                "gpt-oss-120b-mxfp4-00003-of-00003.gguf")
    assert len(local.find_gguf_files("r", "mxfp4", gpt)) == 3


def test_a_model_with_missing_parts_is_not_used():
    tree = files("M-Q4_K_M-00001-of-00003.gguf", "M-Q4_K_M-00002-of-00003.gguf")
    with pytest.raises(local.LocalError, match="no Q4_K_M GGUF"):
        local.find_gguf_files("r", "Q4_K_M", tree)


def test_reads_every_page_of_a_repo_listing(monkeypatch):
    pages = {
        "first": (files("a-Q4_K_M-00001-of-00002.gguf"), '<https://hf.example/next>; rel="next"'),
        "https://hf.example/next": (files("a-Q4_K_M-00002-of-00002.gguf"), ""),
    }

    class Resp(io.BytesIO):
        def __init__(self, url):
            body, link = pages["first" if "tree/main" in url else url]
            super().__init__(json.dumps(body).encode())
            self.headers = {"Link": link}

    monkeypatch.setattr(local.urllib.request, "urlopen", lambda req, timeout=None: Resp(req.full_url))
    assert len(local.find_gguf_files("org/repo", "Q4_K_M")) == 2


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """v's own llama.cpp runner with the downloads, the process and the hardware faked."""
    monkeypatch.setattr(local, "STATE_DIR", tmp_path)
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.write_text("")
    fake = types.ModuleType("cheapstack")
    fake.release = types.ModuleType("cheapstack.release")
    monkeypatch.setitem(sys.modules, "cheapstack", fake)
    state = {"downloads": [], "cmd": None, "vram": 8.0, "apple": False, "free": 10_000.0}

    def parts_for(repo, quant):
        if repo.startswith("ggml-org/gpt-oss-120b"):
            return [{"path": f"gpt-oss-120b-mxfp4-0000{i}-of-00003.gguf", "size": 21_000_000_000} for i in (1, 2, 3)]
        gb = next(m.download_gb for m in local.MODELS if repo in m.hf_repos)
        return [{"path": f"{repo.split('/')[-1]}-{quant}.gguf", "size": int(gb * 1e9)}]

    def download(url, dest, log, size=None):
        state["downloads"].append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")

    class Proc:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            state["cmd"] = cmd

        def poll(self):
            return None

    monkeypatch.setattr(local, "find_gguf_files", parts_for)
    monkeypatch.setattr(local, "download_with_progress", download)
    monkeypatch.setattr(local.subprocess, "Popen", Proc)
    monkeypatch.setattr(local, "_get_json", lambda url, timeout=1.5: {"status": "ok"})
    monkeypatch.setattr(local, "gpu_memory_gb", lambda: state["vram"])
    monkeypatch.setattr(local, "is_apple_silicon", lambda: state["apple"])
    monkeypatch.setattr(local, "free_disk_gb", lambda path: state["free"])
    return state


def model(name):
    return next(m for m in local.MODELS if m.name == name)


def test_downloads_every_part_and_splits_experts_into_ram(runner):
    server = local.start_llama_server(model("gpt-oss 120B"), lambda line: None)
    assert runner["downloads"] == [f"gpt-oss-120b-mxfp4-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    cmd = runner["cmd"]
    assert cmd[cmd.index("-m") + 1].endswith("gpt-oss-120b-mxfp4-00001-of-00003.gguf")  # llama.cpp finds the rest
    assert "--cpu-moe" in cmd and "--jinja" in cmd
    assert server.model == "gpt-oss-120b"


def test_experts_stay_on_the_gpu_when_there_is_room(runner):
    runner["apple"] = True  # unified memory: no need to split
    local.start_llama_server(model("gpt-oss 120B"), lambda line: None)
    assert "--cpu-moe" not in runner["cmd"]
    runner["apple"], runner["vram"] = False, 8
    local.start_llama_server(model("Qwen3 4B"), lambda line: None)  # small models never split
    assert "--cpu-moe" not in runner["cmd"]


def test_a_full_disk_means_a_smaller_model_not_a_failed_download(runner):
    runner["free"] = 20.0
    logs = []
    server = local.start_llama_server(model("gpt-oss 120B"), logs.append, smaller_ok=True)
    assert server.choice.name == "gpt-oss 20B"  # 120B (63 GB) and 30B (19 GB) don't fit in 20 GB
    assert any("Using Qwen3 30B-A3B instead" in line for line in logs)
    assert runner["downloads"] == ["gpt-oss-20b-GGUF-mxfp4.gguf"]


def test_a_model_the_user_asked_for_by_name_is_not_swapped(runner):
    runner["free"] = 20.0
    with pytest.raises(local.NotEnoughDisk, match="needs about 63 GB"):
        local.start_llama_server(model("gpt-oss 120B"), lambda line: None)
    assert runner["downloads"] == []


def test_already_downloaded_models_need_no_free_space(runner, tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    for i in (1, 2, 3):
        (tmp_path / "models" / f"gpt-oss-120b-mxfp4-0000{i}-of-00003.gguf").write_bytes(b"")
    runner["free"] = 1.0

    def same_size(repo, quant):
        return [{"path": f"gpt-oss-120b-mxfp4-0000{i}-of-00003.gguf", "size": 0} for i in (1, 2, 3)]

    monkeypatch.setattr(local, "find_gguf_files", same_size)
    local.start_llama_server(model("gpt-oss 120B"), lambda line: None)
    assert runner["downloads"] == []


def test_ollama_downloads_a_smaller_model_when_the_disk_is_full(monkeypatch):
    monkeypatch.setattr(local, "free_disk_gb", lambda path: 20.0)
    logs = []
    assert local._fit_disk(model("Qwen3 235B-A22B"), local.ollama_models_dir(), logs.append).name == "gpt-oss 20B"
    assert "using gpt-oss 20B instead" in logs[0]
    monkeypatch.setattr(local, "free_disk_gb", lambda path: 2.0)
    with pytest.raises(local.NotEnoughDisk):
        local._fit_disk(model("Qwen3 4B"), local.ollama_models_dir(), logs.append)
