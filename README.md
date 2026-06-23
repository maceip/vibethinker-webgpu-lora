<h1 align="center">🜂 EMBERGLASS</h1>
<p align="center"><em>A 3-billion-parameter mind, running inside a browser tab. No server. No install. No upload. Just a page.</em></p>

<p align="center">
<b>~35 tokens/sec decode · live LoRA hot-swap · bit-exact to the reference · 100% client-side WebGPU</b>
</p>

<p align="center"><a href="https://maceip.github.io/emberglass/"><b>▶ Live demo</b></a> (bring your own model) · <a href="https://huggingface.co/macmacmacmac/emberglass">model card</a></p>

---

## What this is

Most "AI in the browser" is a thin client phoning home to someone else's GPU. **This isn't that.**

Emberglass is a hand-built inference engine that runs a fine-tuned **Qwen2.5-3B** reasoning model **entirely on your own machine's GPU**, from inside a single static web page — written from the metal up in raw WebGPU compute shaders. The model thinks for thousands of tokens, streams a verdict, and **never sends a single byte off your device.** You bring the weights; the page brings the engine.

And here's the part that shouldn't be possible at this speed: you can **swap the model's personality at runtime.** Load the base once, then hot-swap LoRA adapters *live* — no reload, no recompile, no re-quantization. The base weights never move. The output changes the instant you flip the adapter, and flips back **bit-for-bit identically** when you remove it. Specialize a generalist into a bug-bounty triage analyst, a code reviewer, a poet — mid-session, in milliseconds.

## Why it's hard (and why it's fast)

A browser tab is the most hostile environment imaginable for a 3B-parameter language model. No CUDA. No kernels from the vendor. No threads worth having. A 5.4 GB weight shard won't even fit in a single JavaScript array. Every fast path that exists on a server is closed.

So we closed the gap by hand:

- **Custom WGSL compute kernels** for every operation — GEMV, attention, RoPE, RMSNorm, SwiGLU — because we own every matmul, which is the *only* way LoRA could become a live, swappable thing instead of a baked-in constant.
- **int4 group-128 quantization** that is **numerically exact** on the reference decode — half the memory, zero quality lost.
- **Split-K flash-style decode attention** so the engine stays fast even when the model has thought itself out to thousands of tokens of context.
- **Subgroup-reduction GEMV** tuned to the GPU's actual memory behavior — the single change that turned a sluggish kernel into a 7× faster one.
- A **GPU-resident KV cache** and an in-shader RoPE that's free of the read/write race that quietly corrupts naïve implementations.

Every one of those wins was found by **measuring** — nanosecond GPU timestamp profiling — not by guessing. The engine went from 9 tokens/sec to ~35 over one focused push, and every step is reproducible.

## The result

| | |
|---|---|
| **Decode speed** | ~35 tok/s sustained across a full multi-thousand-token reasoning generation |
| **Correctness** | argmax + every generated token **exact** vs the HuggingFace reference; bit-exact run-to-run |
| **LoRA hot-swap** | load base once · swap adapters live · perfect restore on clear · no reload |
| **Footprint** | one static HTML page; weights supplied by the visitor (BYO-model) |
| **Privacy** | absolute — inference never leaves the device |

## How it works

The page asks your browser for a WebGPU device, streams the model's weight tensors in over HTTP range requests (because the whole shard won't fit in memory), quantizes them to int4 on the way to the GPU, and builds a resident runtime: 36 transformer layers of custom compute passes, a GPU KV cache, and a sampling loop. A drag-and-dropped PEFT/MLX LoRA adapter is parsed in pure JS into GPU buffers and handed to the kernels — which fold it into the math at decode time. Pull it out and the base reasserts itself, exactly.

## Run it

```bash
npx http-server . -p 8013 -c-1 --cors      # serve the page + your ./model
# open http://localhost:8013 in a WebGPU browser with the `subgroups` feature
```

Verification harnesses (Playwright + Chrome Canary):

```bash
node run_vwg.mjs      # base correctness + decode speed
node run_lorav.mjs    # LoRA hot-swap: 6/6 checks, bit-exact
node run_prof.mjs     # per-kernel GPU time breakdown
node run_app_e2e.mjs  # full app: load the model, triage a report, measure tok/s
```

Requires a browser exposing WebGPU **with the `subgroups` device feature** (Chrome Canary: `--enable-unsafe-webgpu --use-angle=metal`). Built and validated on an Apple M5 Max.

## What it is not

It does not host your weights, phone home, or need a build toolchain (no wasm, no native, no bazel). It is a single page and a pile of shaders.

---

<p align="center"><sub>Built the hard way, on purpose. 🜂</sub></p>
