<h1 align="center">🜂 EMBERGLASS</h1>
<p align="center"><em>A 3-billion-parameter model running inside a browser tab. No server, install, or upload.</em></p>

<p align="center">
<b>~35 tokens/sec decode · live LoRA hot-swap · bit-exact reference checks · 100% client-side WebGPU</b>
</p>

<p align="center"><a href="https://maceip.github.io/qwen-webgpu-lora/"><b>▶ Live demo</b></a> (bring your own model) · <a href="https://huggingface.co/macmacmacmac/qwen-webgpu-lora">model card</a></p>

---

## What this is

Emberglass is a hand-built inference engine for a fine-tuned **Qwen2.5-3B**
reasoning model that runs entirely on your own GPU from a static web page. The
browser streams model shards with range requests, quantizes weights to int4 on
upload, keeps the KV cache GPU-resident, and emits tokens without sending data
off-device.

Runtime LoRA hot-swap is built into the kernels: load the base once, parse a
PEFT/MLX adapter into GPU buffers, switch adapters live, and clear back to the
base without reloading or re-quantizing.

## Why it is fast

- **Custom WGSL kernels** for GEMV/GEMM, attention, RoPE, RMSNorm, SwiGLU,
  argmax, and top-k sampling.
- **int4 group-128 layer weights** plus int8 embeddings for compact GPU memory
  use while preserving reference-generation checks.
- **Split-K decode attention** and GPU-resident greedy decode batching to avoid
  one CPU/GPU synchronization per generated token.
- **Batched prefill** via tiled GEMM and flash-style causal attention for much
  faster time-to-first-token on medium and long prompts.
- **Static uniform / bind-group caching** and reusable readback buffers to reduce
  JavaScript/WebGPU object churn in hot paths.
- **GPU top-k sampling** so non-greedy generation reads back only `k` ids/logits
  instead of the full vocabulary every token.

## Current result

Measured runs show ~35 tok/s sustained app decode, exact base argmax/generation
checks against the HuggingFace reference, live LoRA swap/clear behavior, and
structured benchmark rows for load, prefill, decode, sampling, LoRA, and profile
regression tracking.

## Run it

```bash
npm install
npm run build
npx http-server . -p 8013 -c-1 --cors
# open http://localhost:8013 in a browser exposing WebGPU + the subgroups feature
```

Load weights from a Hugging Face repo id, same-origin `/model`, or a local
directory picker. Optional HF tokens are supported for gated/private repos.

## Verification and benchmarks

The Playwright harnesses expect the app to be served on port `8013` and a browser
with WebGPU `subgroups`. Set `CHROME_PATH` if Playwright should launch a specific
Chrome/Canary binary.

```bash
npm run test:correctness  # base argmax/generation + batched decode/prefill
npm run test:prefill      # sequential-vs-batched prefill differentials + long smoke
npm run test:lora         # adapter parse, hot-swap, restore, LoRA prefill, speed
npm run test:sampling     # GPU top-k sampler correctness + sampled decode smoke
npm run test:app          # full UI generation path
npm run bench:wgpu        # structured VWG_BENCH JSON rows for perf regression tracking
```

`bench:wgpu` reports time-to-ready, prefill by length, greedy decode by context,
batch candidate timings, sampling throughput, timestamp categories when available,
LoRA-on throughput when fixtures exist, and KV/prefill scratch estimates.

## Requirements

- Browser WebGPU with the **`subgroups`** device feature. The current fast kernels
  require subgroups and no fallback kernel set is bundled.
- Enough GPU memory for the selected context window. KV cache grows linearly with
  `maxCtx`; prefill scratch grows with prompt length.
- Bring your own Qwen2.5-compatible weights. This repository and the demo page do
  not host model weights.

## Project layout

- `src/qwgpu/` — WGSL kernels, runtime, quantization, schema/dispatch metadata,
  streaming safetensors loader, and model uploader.
- `src/services/` — app-facing model session, device, generation, prompt, and
  adapter registry services.
- `src/lora_gpu.js` — PEFT/MLX LoRA parser/uploader.
- `test/` — browser correctness, profiling, sampling, prefill, LoRA, and benchmark
  harnesses.
- `docs/` — GitHub Pages static demo bundle.

---

<p align="center"><sub>Built the hard way, on purpose. 🜂</sub></p>
