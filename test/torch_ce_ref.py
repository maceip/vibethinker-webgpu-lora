#!/usr/bin/env python3
"""External cross-entropy reference for the WebGPU LoRA trainer (Rung 1).

Reads test/rung1_cases.json: {"cases":[{"id":..,"tokens":[...],"mask":[...]}...]}
where mask[t]==1 means "the prediction of tokens[t+1] from position t is trained".
Computes mean CE over the masked positions with the SAME shift convention as the
WebGPU trainer, using the base VibeThinker-3B in torch (independent implementation),
and writes test/rung1_ref.json. The WebGPU side runs int4-quantized weights, so an
exact match is not expected — closeness within quantization tolerance is.

Run with the torch venv:
  ~/vibethinker_conversion/venv-onnx/bin/python test/torch_ce_ref.py
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("VIBE_MODEL", "WeiboAI/VibeThinker-3B")
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "rung1_cases.json")
OUT = os.path.join(HERE, "rung1_ref.json")


def main():
    with open(CASES) as f:
        cases = json.load(f)["cases"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # f32 for a faithful reference; mps handles f32. CPU works too (slower).
    print(f"[ref] loading {MODEL} on {dev} (f32)…", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.to(dev).eval()
    print(f"[ref] loaded in {time.time()-t0:.1f}s", flush=True)

    out = []
    for c in cases:
        toks = c["tokens"]
        mask = c["mask"]
        ids = torch.tensor([toks], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(ids).logits[0].float()  # [T, vocab]
        logp = torch.log_softmax(logits, dim=-1)
        total = 0.0
        n = 0
        for t in range(len(toks) - 1):
            if mask[t]:
                total += -logp[t, toks[t + 1]].item()
                n += 1
        mean = total / max(1, n)
        out.append({"id": c["id"], "ref_loss": mean, "numActive": n})
        print(f"[ref] case {c['id']}: ref_loss={mean:.4f} active={n}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"model": MODEL, "device": dev, "cases": out}, f, indent=2)
    print(f"[ref] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
