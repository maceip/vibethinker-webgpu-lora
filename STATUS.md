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
**Custom WGSL runtime** in `src/qwgpu/`. We own every kernel, so LoRA `A`/`B` are GPU
buffers consumed in-kernel → adapters hot-swap by swapping buffers (no requant, no
base reload). int4 group-128 weights (EXACT output), f32 norms/biases, GPU-resident
KV cache (ctx window 8192). tf.js path abandoned (1.6 tok/s + matMul transposeB bug).

Dead ends (closed, per user): A = build litert-lm wasm (Rust-atomics link wall);
LiteRT.js (`@litertjs/tensor` unpublished + needs unbuilt `@ml_drift`).

## Layout
- `src/` — runtime + app: `qwgpu/` (WGSL kernels, runtime, quantize), `config.js`, `weights.js`,
  `readers.js`, `lora_gpu.js`, `main.js`, `qwen25.js` (tf.js reference, used only by the accuracy gates).
- `test/` — Playwright harnesses (`run_*.mjs` + `*.html` + sources), accuracy gates, `ref.json`.
- `docs/` — GitHub Pages demo (`index.html` + built `bundle.js`). Root: `index.html` (local app), `package.json`.

## Architecture
- `src/qwgpu/kernels.js` — WGSL: GEMV (int8, lm_head), GEMV4 (int4 group-128, layers),
  LORA_A (parallel subgroup GEMV), RMSNORM, ROPE (pair-wise, no race), ATTN_PARTIAL +
  ATTN_COMBINE (split-K decode attention), ADD, SILUMUL, EMBED, ARGMAX, TOPK_SELECT.
- `src/qwgpu/runtime.js` — `QwenWGPU`: build() loads+quantizes (int4 layers, int8 embed),
  KV cache, RoPE tables; token()/step()/argmaxLogits(); setLora()/clearLora() (live swap);
  gemv()/gemv4()/attn() (2-pass split-K); GPU top-k sampling; decode-batch tuning;
  enableProf()/profToken() (GPU timestamp profiling).
- `src/qwgpu/quantize.js` — int8 per-channel + int4 group-128 (both preserve output EXACTLY).
- `src/weights.js` — per-tensor Range-fetch safetensors, BF16→F32.
- `src/lora_gpu.js` — tf-free PEFT/MLX adapter → GPU buffers (A transposed [rank][in], B [rank][out]).
- `src/main.js` + `index.html` — BYO-model app: load, LoRA dropdown (drag adapter files), triage.

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
   argmax id from a buffer), token ids read back once per selected batch instead of a
   blocking mapAsync every token. Batch size is now configurable/autotunable, starts small
   for interactivity/EOS, and grows for throughput. App 22.9 → 35.4 tok/s in the measured run.
7. Sampling mode no longer maps full vocab logits per token: TOPK_SELECT reads back only
   `k` ids/logits for CPU temperature/top-p sampling. Greedy argmax also reuses a persistent
   4-byte MAP_READ buffer.

Profiling tool: `profile_wgpu.js` + `--disable-dawn-features=timestamp_quantization`
gives ns-resolution per-kernel GPU time. `prof.cap` query slots.

## Verify / run
```
cd ~/edge-thinker/qwen-webgpu
npx http-server . -p 8013 -c-1 --cors &              # serve app + ./model symlink
node test/run_vwg.mjs       # base correctness: argmax 4913, gen==ref, speed
node test/run_lorav.mjs     # LoRA hot-swap 6/6 + determinism + LoRA-active speed
node test/run_sampling.mjs  # GPU top-k sampling correctness + smoke speed
node test/run_prof.mjs      # per-kernel GPU breakdown (edit WARM for ctx length)
node test/run_app_e2e.mjs   # full app: load 3B, run triage, measure tok/s
npm run bench:wgpu          # structured load/prefill/decode/sampling/LoRA benchmark rows
# or open http://localhost:8013/ in a WebGPU browser (needs 'subgroups' feature)
```
Requires: WebGPU + `subgroups` device feature. Tested in Chrome Canary
(`--enable-unsafe-webgpu --use-angle=metal`) on M5 Max.

## Loading (done)
- Model: from Hugging Face (`hfReader`, streams safetensors over CORS-enabled range
  requests, optional token for gated/private), same-origin URL, or a local folder picker.
- Adapter: from a Hugging Face LoRA repo (PEFT or MLX), or local files. Hot-swaps live.
- `readers.js` (tf-free) underpins all of it; `build()` takes a reader or a URL.
- Live demo: https://maceip.github.io/qwen-webgpu-lora/ (GitHub Pages, BYO model from HF).

## Batched prefill (done — scales to ctx 8192)
Prefill processes the whole prompt in one pass via a **tiled int4 GEMM** (`GEMM4`,
BM=16 tokens × BN=64 cols, A K-slice staged in shared memory so activations are reused
across columns — the naive per-column kernel re-reads activations N× and is *slower*
than sequential). T>1 kernels: `GEMM4`, `RMSNORM_T` (one wg/row), `ROPE_T`, `EMBED_T`,
`ATTN_PREFILL`; lm_head on the last row reuses the decode GEMV.
- **`ATTN_PREFILL` is FLASH / online-softmax**: streams keys in 256-wide blocks with a
  running (max, denom, weighted-V acc), so workgroup memory is O(block) not O(ctx). A full
  `sc[ctx]` array would be 32KB at ctx=8192 = the entire workgroup-storage budget — flash
  is what lets prefill reach 8192.
- **`ADD`/`SILUMUL`/`EMBED_T` are grid-stride** (use `num_workgroups`): at T=8192, n reaches
  T·I≈90M → 352k workgroups, far past the 65535/dim dispatch limit; grid-stride + a 65535
  cap covers any n. (Backward-compatible: decode's tiny n loops once.)
- Context, prefill, decode batching, sampling top-k, and scratch budget are configurable:
  `new QwenWGPU(dev, cfg, { maxCtx, maxPrefillT, decodeBatchSize, samplingTopK, maxPrefillScratchBytes })`
  (default 8192/8192; base VibeThinker-3B allows up to 128K — KV cache ~72KB/token is the cost).
  `prefillBatch(ids)`: lazy scratch sized to the prompt. Verified maxCtx=16384 prefills 9000 tokens.
- Validated in-browser: batched == sequential bit-exact at L=16/17/256/257/512/1024
  (incl. multi-block flash, ctx>256); 4096 prefill 6.6s, 8192 prefill 20s, both finite +
  valid argmax. GEMM verified vs CPU (err 2.5e-6). Decode/LoRA still match HF after the
  ADD/SILUMUL change (argmax 4913, hot-swap 5/5).
- LoRA-active batched prefill is validated against the sequential adapter path. Decode stops
  at maxCtx (KV-cache guard), and the benchmark harness reports KV/prefill scratch estimates.

## Verified against the VibeThinker-3B repo (no runtime surprises)
- **Config matches** `QWEN25_3B` exactly: hidden 2048 / 36 layers / 16 heads / 2 KV / inter 11008 /
  vocab 151936 / rope_theta 1e6 / rms_eps 1e-6 / head_dim 128 / tied embeddings / qkv-bias-only / silu.
- **`use_sliding_window: false`** → our FULL attention is correct at *every* context length (no SWA
  divergence past 32K). `sliding_window`/`max_window_layers` are inert.
- **Chat template** is standard Qwen ChatML; the app uses `apply_chat_template` when available and
  otherwise a faithful reimplementation (`chatML()`) that injects the default system prompt exactly
  like the template — identical output for the system+user prompts the app sends.
- **EOS/stop**: model eos = 151643 (`<|endoftext|>`); we stop on `[151645 (<|im_end|>), 151643]`. No BOS prepend (`add_bos_token: false`).
- **≥20 tok/s holds at all contexts**: raw decode measured 76/82/75/**68** tok/s at ctx ≈ 2k/4k/6k/7.8k;
  full app (with streaming) 35–65 tok/s. decodeBatch clamps K to maxCtx (can't overrun the KV cache).
- App `maxTokens` = `maxCtx` so long reasoning isn't truncated mid-thought (EOS / KV guard terminate).

## Not pursued (with reasons)
- **attnC fusion:** not possible. Split-K's combine needs every split's partial first,
  and WebGPU's only cross-workgroup barrier is a separate dispatch. The one skippable
  case (nsplit==1, ctx≤128) is where attnC is already near-free; the ~4% is at long
  context where the combine is mandatory — it's the cost of the occupancy win that took
  attention 18ms→5ms (net hugely positive).
- **WebNN backend:** `navigator.ml` not exposed in tested Canary; a future second backend
  (hot-swap survivable via graph inputs, potential ANE access). App/tokenizer/LoRA layers
  are already backend-agnostic, so it'd be a `QwenWebNN` sibling of `QwenWGPU`.
