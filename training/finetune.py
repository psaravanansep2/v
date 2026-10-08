"""Teach a small open model v's tools with LoRA: a small add-on trained on
top of the model, then folded into it.

The model learns only from the assistant's turns (which tool to call with
what, and what to say), never from the system prompt, the tool list or the
tool results, which it only reads.

Free GPUs that work: Google Colab or Kaggle (a 16 GB T4). Qwen3 4B on about
1,500 examples takes roughly an hour there.

    python training/finetune.py --data data/train.jsonl --out runs/v-qwen3-4b
    python training/finetune.py --base Qwen/Qwen3-0.6B --data data/train.jsonl --out runs/tiny --max-steps 6 --cpu
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"  # the model v runs on 8 GB laptops
ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)  # Qwen's chat format
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def assistant_spans(text: str) -> list:
    """Where the assistant's turns are in a rendered conversation (character offsets)."""
    return [(m.start(1), m.end(1)) for m in ASSISTANT.finditer(text)]


def encode(example: dict, tokenizer, max_len: int):
    """Token ids and labels for one conversation; labels are -100 (ignored) outside the assistant's turns.
    None if it's too long (cutting it would lose the end, which is what matters)."""
    text = tokenizer.apply_chat_template(example["messages"], tools=example.get("tools"), tokenize=False)
    spans = assistant_spans(text)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) > max_len or not spans:
        return None
    labels = []
    for token_id, (start, end) in zip(ids, enc["offset_mapping"]):
        inside = end > start and any(s <= start and end <= e for s, e in spans)
        labels.append(token_id if inside else -100)
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load_examples(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"the model to teach (default {DEFAULT_BASE})")
    ap.add_argument("--data", required=True, help="training examples from make_data.py (JSON lines)")
    ap.add_argument("--out", required=True, help="folder for the add-on (adapter/) and the finished model (merged/)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (quick checks)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096, help="longer conversations are left out")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--cpu", action="store_true", help="train on the processor (slow; for checking the pipeline)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(Path(args.data))
    random.Random(args.seed).shuffle(examples)
    encoded = [encode(ex, tokenizer, args.max_len) for ex in examples]
    data = [e for e in encoded if e is not None]
    print(f"{len(data)} examples ({len(examples) - len(data)} left out as too long); "
          f"{sum(sum(1 for t in e['labels'] if t != -100) for e in data):,} tokens to learn from", flush=True)
    if not data:
        raise SystemExit("no usable examples")

    gpu = torch.cuda.is_available() and not args.cpu
    bf16 = gpu and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16 if gpu else torch.float32
    dtype_arg = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.base, **{dtype_arg: dtype})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # lets gradients reach the add-on through checkpointed layers
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    class Examples(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return data[i]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=1,
            save_strategy="no", report_to=[], bf16=bf16, fp16=gpu and not bf16, use_cpu=not gpu,
            remove_unused_columns=False, seed=args.seed, dataloader_num_workers=0,
        ),
        train_dataset=Examples(),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    metrics = {"base": args.base, "examples": len(data), "steps": trainer.state.global_step,
               "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
               "losses": losses}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"loss {metrics['first_loss']:.3f} -> {metrics['last_loss']:.3f} over {metrics['steps']} steps", flush=True)

    model.save_pretrained(str(out / "adapter"))
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(str(out / "merged"), safe_serialization=True)
    tokenizer.save_pretrained(str(out / "merged"))
    print(f"The trained model is in {out / 'merged'}; make a GGUF for v with export_gguf.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
