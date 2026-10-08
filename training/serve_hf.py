"""A tiny chat server for a Hugging Face model, so `v eval` can score a model
before and after training on the same GPU, without converting it first.

    python training/serve_hf.py --model Qwen/Qwen3-4B-Instruct-2507 --port 8000 &
    v eval --local-url http://127.0.0.1:8000 --per-family 3 --json before.json

It speaks just enough of the OpenAI chat API for v: the model's reply goes
back as text, and v reads tool calls written in it (<tool_call>…</tool_call>,
as Qwen writes them). One request at a time, greedy decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a Hugging Face model name or folder (e.g. runs/v-qwen3-4b/merged)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu = torch.cuda.is_available()
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if gpu else torch.float32
    dtype_arg = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **{dtype_arg: dtype}).to("cuda" if gpu else "cpu").eval()
    name = args.model.rstrip("/").split("/")[-1]
    lock = threading.Lock()
    stops = [t for t in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")} if isinstance(t, int)]

    def reply(body: dict) -> tuple[str, int]:
        prompt = tokenizer.apply_chat_template(body["messages"], tools=body.get("tools") or None,
                                               add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with lock, torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=stops,
                                 pad_token_id=tokenizer.pad_token_id or stops[0])
        new = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new, skip_special_tokens=True), inputs["input_ids"].shape[1]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send(json.dumps({"data": [{"id": name}]}).encode(), "application/json")
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            started = time.time()
            text, prompt_tokens = reply(body)
            chunk = {"choices": [{"delta": {"content": text}, "finish_reason": "stop"}],
                     "timings": {"prompt_n": prompt_tokens, "prompt_ms": (time.time() - started) * 1000}}
            if body.get("stream"):
                return self._send(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(), "text/event-stream")
            self._send(json.dumps({"choices": [{"message": {"role": "assistant", "content": text},
                                                "finish_reason": "stop"}]}).encode(), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving {name} on http://127.0.0.1:{args.port} ({'GPU' if gpu else 'CPU'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
