<h1 align="center">🜂 EMBERGLASS</h1>
<p align="center"><em>Optimized WebGPU inference for VibeThinker-3B — in your browser tab. No server, no upload.</em></p>

<p align="center">
<b>≥20 tok/s decode floor · live LoRA hot-swap · bit-exact reference checks · 100% client-side WebGPU</b>
</p>

<p align="center"><a href="https://maceip.github.io/qwen-webgpu-lora/"><b>▶ Live demo</b></a> · <a href="https://github.com/maceip/emberglass-tune">Training docs</a> · <a href="https://github.com/maceip/vibebounty">VibeBounty demo</a></p>

---

## Three repos

| Repo | Role | Train? | Run inference? |
|---|---|---|---|
| **[emberglass](https://github.com/maceip/qwen-webgpu-lora)** (this) | Custom **WebGPU** runtime — int4, fused kernels, LoRA hot-swap | **Experimental** (in-browser LoRA backward) | **Yes** (browser) |
| **[emberglass-tune](https://github.com/maceip/emberglass-tune)** | LoRA **training** — MLX + CUDA, Anthropic trace pipeline | **Yes** | No |
| **[vibebounty](https://github.com/maceip/vibebounty)** | Bug-bounty **demo** — tuned adapter, HackerOne UI, CPU/GPU serve | Uses emberglass-tune | Yes (server) |

**How the weights are made:** labeled reports → Anthropic teacher traces → LoRA SFT → `adapter_model.safetensors`. Full pipeline: **[emberglass-tune README](https://github.com/maceip/emberglass-tune)**.

**How to run them here:** load base weights + optional adapter into WebGPU for inference. The runtime is inference-first, but now also ships an **experimental in-browser LoRA trainer** (full backward pass + AdamW over the frozen-int4 base — see [docs/TRAINING_AND_LORA.md](docs/TRAINING_AND_LORA.md)). For production tuning, the canonical pipeline is still **emberglass-tune** (MLX/CUDA).

---

## What this repo is

Emberglass is an **inference-only** engine for Qwen2.5-class models (VibeThinker-3B):

- Custom **WGSL kernels** (GEMV/GEMM, attention, RoPE, sampling)
- **int4** layer weights on GPU, GPU-resident KV cache
- **Runtime LoRA hot-swap** — load PEFT/MLX `adapter_model.safetensors` without re-quantizing base (`src/lora_gpu.js`)
- Playwright correctness and throughput harnesses (`npm run test:*`)

| In emberglass | Elsewhere |
|---|---|
| WebGPU forward pass | Training → **emberglass-tune** |
| LoRA apply / swap / clear | Data + Anthropic traces → **emberglass-tune** + **vibebounty** |
| int4 load from `./model` or HF | HackerOne demo UI → **vibebounty** |
| | CPU/GPU serve for demos → **vibebounty** |

---

## Run it

```bash
cd ~/emberglass
npm install
npm run build
npm run serve    # http://localhost:8013
```

Open in Chrome/Edge with **WebGPU + `subgroups`**. Load base weights from `./model`, Hugging Face, or a directory picker. Optional LoRA adapter URL for hot-swap.

**Base model:** [WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)  
**Example adapter:** [macmacmacmac/vibebounty](https://huggingface.co/macmacmacmac/vibebounty) (train with emberglass-tune)

---

## Using a trained adapter

1. Train (or download) a PEFT adapter — see [emberglass-tune](https://github.com/maceip/emberglass-tune).
2. Serve adapter files same-origin (e.g. under `/adapters/my-run/`).
3. Load in the Emberglass UI or via VibeBounty's Emberglass bridge.

Tests: `npm run test:lora`, `npm run test:lora-path`.

---

## Verification

```bash
npm run test:correctness   # argmax / generation vs reference
npm run test:lora          # adapter parse, hot-swap, restore
npm run test:app           # full streaming UI path
npm run bench:wgpu         # structured throughput JSON
```

Requires port **8013**, WebGPU **`subgroups`**, and weights in `./model` (not bundled in repo).

---

## Performance

Throughput is hardware-dependent. Latest clean local browser run:
`npm run bench:wgpu` on Chrome Canary/WebGPU with `timestamp-query`, 3B-shaped mock weights, June 25, 2026.

| Measurement | Result |
|---|---:|
| Greedy decode @ ctx 128 | 124.6 tok/s |
| Greedy decode @ ctx 1,024 | 127.8 tok/s |
| Greedy decode @ ctx 4,096 | 109.7 tok/s |
| Greedy decode @ ctx 7,800 | 88.4 tok/s |
| GPU top-k sampling (`topK=40`, 4-byte readback) | 22.1 tok/s |
| Selected decode batch | 16 tokens |
| LoRA decode | skipped (`adapters_sel` fixture unavailable) |

Prefill latency:

| Prompt length | Latency |
|---:|---:|
| 64 tokens | 99.0 ms |
| 256 tokens | 240.9 ms |
| 1,024 tokens | 1,015.5 ms |
| 4,096 tokens | 6,177.7 ms |
| 8,192 tokens | 18,694.2 ms |

Single-token decode profile at ctx 18:

| Kernel category | GPU time |
|---|---:|
| `embed` | 9.8 us |
| `rmsNormQkvRope` | 427.9 us |
| `attnP` | 667.3 us |
| `attnC` | 106.8 us |
| `g4add:2048x2048` | 405.6 us |
| `rms` | 262.0 us |
| `gu:11008x2048` | 1,782.1 us |
| `g4add:2048x11008` | 1,364.1 us |
| `gemv:151936x2048` | 520.5 us |

Fused decode path: `fuseQKV` / `fuseRoPE` / `fuseMLP` / `fuseResidual`.

## Performance features added

- Hot kernels use `var<immediate>` + `setImmediates` for per-dispatch metadata; the benchmark completed without WebGPU validation errors.
- `shader-f16` paths are active for RMS normalization, RoPE, attention partial/combine, elementwise add, and SiLU.
- GPU-resident sampling keeps top-k selection and sampling on GPU; measured `topK=40` sampling was 22.1 tok/s with one token ID read back.
- Workgroup autotuning uses `timestamp-query`; clean-run winners were `add=64`, `rms=256`, `silu=256`.
- Specialization constants (`override`) are used for workgroup sizes on key kernels and are reflected in dispatch sizing.
- High-level `generate()` can use the GPU sampler when requested.
- Benchmark harness reports prefill latency, greedy decode tok/s, GPU top-k sampling tok/s, and decode sub-kernel timings.

---

## Requirements

- Browser WebGPU with **`subgroups`** (no fallback kernel set)
- GPU memory for chosen context window
- Bring your own weights — repo does not ship model files

---

## Layout

```
src/qwgpu/           WGSL kernels, runtime, int4 quantize
src/lora_gpu.js      PEFT/MLX adapter → GPU buffers
src/services/        App session, generation, adapter registry
test/                Browser harnesses
docs/                GitHub Pages demo + architecture notes
model/               BYO base weights (gitignored)
```

---

## Related docs

- **Training (MLX, CUDA, Anthropic traces):** [emberglass-tune README](https://github.com/maceip/emberglass-tune)
- **Bug-bounty demo:** [vibebounty](https://github.com/maceip/vibebounty)
- **Architecture map:** [`docs/REPO_ARCHITECTURE.md`](docs/REPO_ARCHITECTURE.md)

---

<p align="center"><sub>Built the hard way, on purpose. 🜂</sub></p>
