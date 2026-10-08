"""The training pipeline's pieces that don't need PyTorch: the examples it
makes, and which tokens the model learns from. (CI also runs the whole
pipeline end to end with a tiny model.)"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import finetune  # noqa: E402
import make_data  # noqa: E402


def test_examples_are_conversations_as_v_sends_them():
    examples = list(make_data.solutions(per_family=2, log=lambda line: None))
    families = {ex["family"] for ex in examples}
    assert "read, then write" not in families and "open a file" not in families  # held out for the eval
    assert len(families) == len([f for f in make_data.families()])
    for ex in examples:
        msgs = ex["messages"]
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] and "tool_calls" not in msgs[-1]
        text = json.dumps(ex)
        assert "v-eval-" not in text and "/tmp/" not in text  # scratch folders replaced by ordinary ones
        assert "/no_think" in msgs[0]["content"]  # the same switches v gives Qwen3
        assert {t["function"]["name"] for t in ex["tools"]} >= {"read_file", "look_at_screen", "click"}
        for m in msgs:
            for c in m.get("tool_calls") or []:
                assert isinstance(c["function"]["arguments"], dict)
    systems = {ex["messages"][0]["content"].split("\n")[0] for ex in examples}
    assert len(systems) > 1  # macOS and Linux
    dates = {next(line for line in ex["messages"][0]["content"].split("\n") if line.startswith("Today")) for ex in examples}
    assert len(dates) > 5


class ChatMLTokenizer:
    """Stands in for Qwen's tokenizer: renders its chat format, one token per character."""

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        out = "<|im_start|>system\n" + messages[0]["content"] + "\n# Tools\n" + json.dumps(tools) + "<|im_end|>\n"
        for m in messages[1:]:
            if m["role"] == "tool":
                out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
            elif m["role"] == "assistant":
                calls = "".join(f"<tool_call>\n{json.dumps(c['function'])}\n</tool_call>" for c in m.get("tool_calls") or [])
                out += f"<|im_start|>assistant\n{m['content']}{calls}<|im_end|>\n"
            else:
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return out

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_the_model_learns_only_its_own_turns():
    example = next(make_data.solutions(per_family=1, names=["read"], log=lambda line: None))
    tok = ChatMLTokenizer()
    enc = finetune.encode(example, tok, max_len=100_000)
    text = tok.apply_chat_template(example["messages"], example["tools"])
    learned = "".join(chr(t) for t in enc["labels"] if t != -100)
    call_text = json.dumps(example["messages"][2]["tool_calls"][0]["function"])
    assert learned == f"<tool_call>\n{call_text}\n</tool_call><|im_end|>" + example["messages"][-1]["content"] + "<|im_end|>"
    assert "Password:" in text and "Password:" not in learned  # tool results are read, not learned
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"])


def test_conversations_too_long_are_left_out_not_cut():
    example = next(make_data.solutions(per_family=1, names=["read"], log=lambda line: None))
    assert finetune.encode(example, ChatMLTokenizer(), max_len=50) is None


def test_held_out_kinds_can_be_chosen():
    examples = list(make_data.solutions(per_family=1, names=["read", "write"], hold_out=("write",), log=lambda line: None))
    assert {ex["family"] for ex in examples} == {"read a file"}


@pytest.mark.parametrize("name", ["finetune.py", "serve_hf.py", "export_gguf.py", "make_data.py"])
def test_scripts_explain_themselves(name):
    import subprocess

    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "training" / name), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "usage:" in out.stdout
