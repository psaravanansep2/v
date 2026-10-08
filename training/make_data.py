"""Training examples for teaching a small model v's tools.

Each example is one conversation exactly as v sends it to the model: v's
system prompt, the tool list, the user's request, the tool calls with their
real results, and the reply. They come from v eval's tasks (training
wordings only, and never the held-out kinds of task), solved two ways:

- known-good solutions played through v's real tools (always), and
- optionally, the model's own attempts that passed their check
  (--from-url), which keeps the model's own way of talking.

    python training/make_data.py --per-family 120 --out data/train.jsonl
    python training/make_data.py --from-url http://127.0.0.1:8000 --own-per-family 20 --out data/own.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a checkout without installing

from v import evals  # noqa: E402

# Never trained on: v eval's score on these shows whether the skills carry over to new kinds of job.
HOLD_OUT = ("todo", "open")
TRAIN_SEED = 7  # the eval uses its own seed (2026) and the other wordings
PROJECTS = ["bakery-site", "thesis", "trip-plans", "home", "shop", "notes", "family-photos", "taxes-2026"]


def system_suffix() -> str:
    """The switches v gives the model being trained (Qwen3)."""
    from v.local import MODELS

    return next(m.system_suffix for m in MODELS if m.name == "Qwen3 4B")


def anonymize(messages: list, scratch: str, rng: random.Random) -> list:
    """Swap the scratch folder for an ordinary-looking project folder, on macOS or Linux."""
    user = rng.choice(evals.NAMES).lower()
    system_name, root = rng.choice([("Darwin", f"/Users/{user}"), ("Linux", f"/home/{user}")])
    folder = f"{root}/{rng.choice(PROJECTS)}"
    today = date(2025, 1, 1) + timedelta(days=rng.randrange(3 * 365))
    # the folder as given and as resolved (on macOS /var/... is really /private/var/...), longest first
    names = sorted({scratch, str(Path(scratch).resolve())}, key=len, reverse=True)
    out = []
    for m in messages:
        text = json.dumps(m)
        for name in names:
            text = text.replace(json.dumps(name)[1:-1], json.dumps(folder)[1:-1])
        m = json.loads(text)
        if m["role"] == "system":
            m["content"] = m["content"].replace(f"computer ({platform.system()})", f"computer ({system_name})")
            m["content"] = re.sub(r"Today is [^\n]+\.", f"Today is {today:%A, %B %d, %Y}.", m["content"])
        out.append(m)
    return out


def to_chat(messages: list) -> list:
    """Chat-template form: tool call arguments as objects, nothing v keeps for itself."""
    chat = []
    for m in messages:
        if m["role"] == "assistant":
            item = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                item["tool_calls"] = [{"type": "function", "id": c["id"], "function": {
                    "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                    for c in m["tool_calls"]]
            chat.append(item)
        elif m["role"] == "tool":
            chat.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "name": m.get("name", ""),
                         "content": m.get("content") or ""})
        else:
            chat.append({"role": m["role"], "content": m.get("content") or ""})
    return chat


def example(result, rng: random.Random, source: str) -> dict:
    return {"messages": to_chat(anonymize(result.messages, result.project, rng)), "tools": result.tools,
            "family": result.family, "source": source}


def families(names=None, hold_out=HOLD_OUT) -> list:
    return [f for f in (names or evals.FAMILIES) if f not in hold_out and evals.available(f)]


def solutions(per_family: int, seed: int = TRAIN_SEED, names=None, hold_out=HOLD_OUT, log=print):
    rng, suffix = random.Random(seed), system_suffix()
    for name in families(names, hold_out):
        for _ in range(per_family):
            task = evals.FAMILIES[name](rng, "train")
            result = evals.run_task(task, evals.ScriptedClient(task.solution, suffix))
            if not result.passed:  # a broken solution must never become training data
                raise RuntimeError(f"known-good solution failed: {task.family}: {task.prompt}: {result.reason}")
            yield example(result, rng, "solution")
        log(f"  {name}: {per_family} solutions")


def own_attempts(client, per_family: int, seed: int = TRAIN_SEED + 1, names=None, hold_out=HOLD_OUT, tries: int = 3,
                 log=print):
    """The model's own runs that did the job in plain spoken style (up to `tries` attempts per kept example)."""
    rng = random.Random(seed)
    for name in families(names, hold_out):
        kept = attempts = 0
        while kept < per_family and attempts < per_family * tries:
            attempts += 1
            result = evals.run_task(evals.FAMILIES[name](rng, "train"), client)
            if result.passed and result.style_ok:
                kept += 1
                yield example(result, rng, "own")
        log(f"  {name}: kept {kept} of {attempts} of the model's own attempts")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the examples (JSON lines)")
    ap.add_argument("--per-family", type=int, default=120, help="known-good solutions per kind of task")
    ap.add_argument("--families", help="comma-separated kinds of task (default: all but the held-out ones)")
    ap.add_argument("--hold-out", default=",".join(HOLD_OUT), help="kinds of task never trained on")
    ap.add_argument("--from-url", help="also keep this model server's own successful attempts")
    ap.add_argument("--own-per-family", type=int, default=20)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = ap.parse_args(argv)
    names = args.families.split(",") if args.families else None
    hold_out = tuple(f for f in args.hold_out.split(",") if f)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        print("Known-good solutions:")
        for ex in solutions(args.per_family, args.seed, names, hold_out):
            f.write(json.dumps(ex) + "\n")
            count += 1
        if args.from_url:
            from v.local import LocalClient, custom_server

            print(f"The model's own attempts ({args.from_url}):")
            client = LocalClient(custom_server(args.from_url, None), timeout=600)
            for ex in own_attempts(client, args.own_per_family, args.seed + 1, names, hold_out):
                f.write(json.dumps(ex) + "\n")
                count += 1
    print(f"\n{count} examples in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
