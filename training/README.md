# Teaching a small model v's tools

Open models can be trained further. This folder trains a small add-on
(LoRA) that makes **Qwen3 4B**, the model v uses on 8 GB laptops, better at
v's everyday jobs, then turns the result into a GGUF file v runs.

The easy way: open [`v_finetune.ipynb`](v_finetune.ipynb) in Google Colab or
Kaggle with a free T4 GPU and run it top to bottom (about 1.5 to 2 hours).
It scores the model before and after with `v eval`, and gives you the file.

## How it works

1. **Examples** (`make_data.py`). v eval's tasks have known-good solutions.
   Played through v's real tools, each becomes a conversation exactly as v
   sends it to the model: v's system prompt, the tool list, the request,
   the tool calls with their real results, and the reply. With `--from-url`
   it also keeps the model's own attempts that passed their check, which
   keeps its natural way of talking.
2. **Training** (`finetune.py`). LoRA on all of the model's layers. The
   model learns only from the assistant's turns (which tool to call with
   what, and what to say), never from the prompt or tool results. The
   add-on is then folded into the model.
3. **Scoring** (`serve_hf.py` + `v eval`). A small server runs the model on
   the GPU, so the same tasks score it before and after.
4. **The file** (`export_gguf.py`). llama.cpp's converter and quantizer, from
   the same release v uses to run models, make a Q4_K_M GGUF (about 2.5 GB).

```bash
pip install -e ".[free,local]" -r training/requirements.txt   # plus PyTorch for your computer
python training/make_data.py --per-family 120 --out data/train.jsonl
python training/finetune.py --data data/train.jsonl --out runs/v-qwen3-4b
python training/export_gguf.py --model runs/v-qwen3-4b/merged --out v-qwen3-4b-Q4_K_M.gguf
v --local-model v-qwen3-4b-Q4_K_M.gguf
```

## Keeping the scores honest

- The eval uses wordings the examples never use, and different details
  (names, numbers, files).
- Two kinds of task, "read, then write" and "open a file", are never trained
  on. If those improve too, the skills carried over; if only the trained
  kinds improve, the model mostly learned the tasks.
- v eval's tasks are everyday jobs, but they're still a test. Try the
  trained model on your own work before you rely on it.

## Checked on every change

CI runs this whole pipeline with Qwen3 0.6B on a processor: examples,
training (the loss has to go down), scoring through the training server,
the GGUF, and v running it.
