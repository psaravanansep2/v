"""Every model v might download really exists, at about the size v tells
people, both as GGUF files on Hugging Face (v's own llama.cpp runner) and in
Ollama's library. Needs the internet: CI runs it with V_NETWORK_TESTS=1."""

import json
import os
import urllib.request
import warnings

import pytest

from v import local

pytestmark = pytest.mark.skipif(not os.environ.get("V_NETWORK_TESTS"), reason="needs the internet (V_NETWORK_TESTS=1)")


def near(actual_gb: float, expected_gb: float) -> bool:
    return abs(actual_gb - expected_gb) <= expected_gb * 0.2


@pytest.mark.parametrize("choice", local.MODELS, ids=lambda m: m.name)
def test_gguf_download_exists(choice):
    first, *fallbacks = choice.hf_repos
    parts = local.find_gguf_files(first, choice.quant)
    total = sum(p["size"] for p in parts) / 1e9
    assert near(total, choice.download_gb), f"{first}: {total:.1f} GB in {len(parts)} file(s)"
    for repo in fallbacks:  # only used if the first goes away; worth knowing, not worth failing over
        try:
            local.find_gguf_files(repo, choice.quant)
        except Exception as e:
            warnings.warn(f"fallback {repo} for {choice.name}: {e}")


@pytest.mark.parametrize("choice", local.MODELS, ids=lambda m: m.name)
def test_ollama_has_the_model(choice):
    name, tag = choice.ollama.split(":")
    req = urllib.request.Request(f"https://registry.ollama.ai/v2/library/{name}/manifests/{tag}",
                                 headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        manifest = json.load(resp)
    total = sum(layer.get("size", 0) for layer in manifest.get("layers", [])) / 1e9
    assert near(total, choice.download_gb), f"{choice.ollama}: {total:.1f} GB"
