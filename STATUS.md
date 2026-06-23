# B-hard: in-browser Qwen2.5-3B (bug-bounty triage) on WebGPU + runtime LoRA hot-swap

Goal: run our fine-tuned Qwen2.5-3B **client-side in the browser** via WebGPU,
with **LoRA adapters swappable at runtime** (no base reload), at **≥20 tok/s**.
BYO model file (visitor brings weights; we host only the static page).

## STATUS: BOTH HARD REQUIREMENTS MET ✅ (2026-06-22)

Custom pure-WebGPU runtime (WGSL compute kernels, no tf.js, no wasm/bazel).

| Metric | Result |
|---|---|
| Full thinking generation | **3062 tokens @ 35.4 tok/s** (valid triage JSON) |
| GPU decode @ ctx=3218 | 33 tok/s |
| Base correctness | argmax 4913 + 16/16 gen tokens EXACT vs HuggingFace; bit-exact deterministic |
| **LoRA hot-swap** | **6/6 checks pass** — load base once, swap live, `clearLora` restores logits bit-exact, no reload |
| LoRA active (180 modules) | 22.8 tok/s |

## Vehicle (final decision)
**Custom WGSL runtime** in `qwgpu/`. We own every kernel, so LoRA `A`/`B` are GPU
buffers consumed in-kernel → adapters hot-swap by swapping buffers (no requant, no
base reload). int4 group-128 weights (EXACT output), f32 norms/biases, GPU-resident
KV cache (ctx window 8192). tf.js path abandoned (1.6 tok/s + matMul transposeB bug).

Dead ends (closed, per user): A = build litert-lm wasm (Rust-atomics link wall);
LiteRT.js (`@litertjs/tensor` unpublished + needs unbuilt `@ml_drift`).

## Architecture
- `qwgpu/kernels.js` — WGSL: GEMV (int8, lm_head), GEMV4 (int4 group-128, layers),
  LORA_A (parallel subgroup GEMV), RMSNORM, ROPE (pair-wise, no race), ATTN_PARTIAL +
  ATTN_COMBINE (split-K decode attention), ADD, SILUMUL, EMBED, ARGMAX.
- `qwgpu/runtime.js` — `QwenWGPU`: build() loads+quantizes (int4 layers, int8 embed),
  KV cache, RoPE tables; token()/step()/argmaxLogits(); setLora()/clearLora() (live swap);
  gemv()/gemv4()/attn() (2-pass split-K); enableProf()/profToken() (GPU timestamp profiling).
- `qwgpu/quantize.js` — int8 per-channel + int4 group-128 (both preserve output EXACTLY).
- `weights.js` — per-tensor Range-fetch safetensors, BF16→F32.
- `lora_gpu.js` — tf-free PEFT/MLX adapter → GPU buffers (A transposed [rank][in], B [rank][out]).
- `main.js` + `index.html` — BYO-model app: load, LoRA dropdown (drag adapter files), triage.

## Perf engineering (this session: 9.1 → 35.4 tok/s, all via measurement not guessing)
1. GEMV workgroup 256→64: each thread reads >1 word → memory-level parallelism.
   The dominant `g4:11008x2048` kernel went 68.5ms → 9.3ms (7.4×).
2. LORA_A serial K-loop → parallel subgroup-reduction GEMV: LoRA path 2.6 → 23 tok/s (9×).
3. RoPE in-place cross-thread read/write RACE → per-pair kernel: fixed nondeterminism
   (was ~1.7 logit jitter run-to-run; now bit-exact).
4. Split-K (flash-style) decode attention: serial-softmax + 16-workgroup kernel →
   nHeads*nsplit workgroups. Attention at ctx=3218: 18ms → 5ms. Scales to 4k+ context.
5. App streaming: O(n^2) `textContent +=` → text-node appendData (O(n)).
6. GPU-resident batched decode: argmax->embed chained on GPU (EMBED_BUF reads the
   argmax id from a buffer), token ids read back once per 16-token batch instead of a
   blocking mapAsync every token. App 22.9 → 35.4 tok/s (measured: per-token gpu/dec/dom
   showed ALL cost was the per-token argmax sync + paint-during-await, not JS).

Profiling tool: `profile_wgpu.js` + `--disable-dawn-features=timestamp_quantization`
gives ns-resolution per-kernel GPU time. `prof.cap` query slots.

## Verify / run
```
cd ~/edge-thinker/qwen-webgpu
npx http-server . -p 8013 -c-1 --cors &              # serve app + ./model symlink
node run_vwg.mjs       # base correctness: argmax 4913, gen==ref, speed
node run_lorav.mjs     # LoRA hot-swap 6/6 + determinism + LoRA-active speed
node run_prof.mjs      # per-kernel GPU breakdown (edit WARM for ctx length)
node run_app_e2e.mjs   # full app: load 3B, run triage, measure tok/s
# or open http://localhost:8013/ in a WebGPU browser (needs 'subgroups' feature)
```
Requires: WebGPU + `subgroups` device feature. Tested in Chrome Canary
(`--enable-unsafe-webgpu --use-angle=metal`) on M5 Max.

## Remaining headroom (optional)
- Prefill is sequential T=1 (~ctx × per-token); a batched prefill kernel would cut TTFT.
- attnC combine runs 16 workgroups (low occupancy, ~4%); could fuse.
- A live "checkpoint selector": the UI dropdown lists adapters; selecting one hot-swaps it live.
