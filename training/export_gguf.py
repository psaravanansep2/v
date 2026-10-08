"""Turn a trained model into one GGUF file that v runs (so do llama.cpp, LM
Studio and Ollama).

    python training/export_gguf.py --model runs/v-qwen3-4b/merged --out v-qwen3-4b-Q4_K_M.gguf
    v --local-model v-qwen3-4b-Q4_K_M.gguf

llama.cpp's converter makes a full-precision GGUF, then llama-quantize
shrinks it (Q4_K_M: about 2.5 GB for a 4B model, with little loss). Both
come from the same llama.cpp release v uses to run models.
"""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a checkout without installing

SOURCE = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.tar.gz"


def llama_release():
    """The newest llama.cpp release with a build for this computer: (tag, asset)."""
    from v.local import LocalError, _recent_llama_releases, pick_llama_asset

    for release in _recent_llama_releases():
        if release.get("draft") or not release.get("assets"):
            continue
        try:
            return release["tag_name"], pick_llama_asset(release["assets"])
        except LocalError:
            continue
    raise SystemExit("no llama.cpp release with a build for this computer")


def tools(work: Path, tag: str, asset: dict) -> Path:
    """llama-quantize from the release build."""
    from cheapstack import release

    exe = "llama-quantize.exe" if sys.platform.startswith("win") else "llama-quantize"
    bin_dir = work / f"bin-{tag}"
    found = list(bin_dir.rglob(exe))
    if not found:
        print(f"Fetching llama.cpp {tag} ({asset['name']})…", flush=True)
        release.download_and_extract(asset, bin_dir)
        found = list(bin_dir.rglob(exe))
    if not found:
        raise SystemExit(f"{exe} isn't in {asset['name']}")
    for f in found[0].parent.iterdir():  # zip archives don't keep the "can run" bit
        if f.is_file() and not sys.platform.startswith("win"):
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return found[0]


def converter(work: Path, tag: str) -> Path:
    """convert_hf_to_gguf.py (and the gguf library it uses) from the same release's source."""
    src = work / f"llama.cpp-{tag}"
    script = src / "convert_hf_to_gguf.py"
    if not script.exists():
        print(f"Fetching the llama.cpp {tag} converter…", flush=True)
        archive = work / f"llama.cpp-{tag}.tar.gz"
        req = urllib.request.Request(SOURCE.format(tag=tag), headers={"User-Agent": "v"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            archive.write_bytes(resp.read())
        with tarfile.open(archive) as tf:
            tf.extractall(work)
    if not script.exists():
        raise SystemExit(f"convert_hf_to_gguf.py isn't in llama.cpp {tag}")
    return script


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the trained model folder (finetune.py's merged/)")
    ap.add_argument("--out", required=True, help="the GGUF file to write")
    ap.add_argument("--quant", default="Q4_K_M", help="how small to make it (Q4_K_M, Q5_K_M, Q8_0, ...)")
    ap.add_argument("--work", help="folder for llama.cpp's tools (default: next to --out)")
    ap.add_argument("--keep-f16", action="store_true", help="keep the full-precision GGUF too")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else out.parent / ".llama-export"
    work.mkdir(parents=True, exist_ok=True)
    tag, asset = llama_release()
    quantize, script = tools(work, tag, asset), converter(work, tag)

    full = out.with_name(out.stem + "-f16.gguf")
    print("Converting to GGUF…", flush=True)
    subprocess.run([sys.executable, str(script), str(Path(args.model).resolve()), "--outfile", str(full),
                    "--outtype", "f16"], check=True)
    print(f"Quantizing to {args.quant}…", flush=True)
    subprocess.run([str(quantize), str(full), str(out), args.quant], check=True)
    if not args.keep_f16:
        full.unlink()
    print(f"\n{out} ({out.stat().st_size / 1e9:.2f} GB)\nUse it: v --local-model {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
