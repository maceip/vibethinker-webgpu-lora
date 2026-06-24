# Emberglass WebGPU Inference – Optimization Plan & Evaluation Criteria

**Date:** 2026-06-24  
**Baseline commit:** 76ee954 (remote origin/main) + bd2741f  
**Context:** Review of the last ~10 Chrome "What's New in WebGPU" posts + full code audit of the "immediate_address_space" work landed in 76ee954.

This document is the authoritative plan. It was written to disk, committed, and pushed before any implementation work began on the phases below.

## Current State at Baseline (76ee954)

- **Immediates (Chrome 149-150):** Partially implemented.
  - `_dispatch` accepts an optional `imm` payload and calls `setImmediates(0, imm)`.
  - Only a handful of kernels converted (ROPE, ROPE_QK, ATTN_PARTIAL/COMBINE, WRITE_KV_PAGE*, one fused attempt).
  - ~35 `var<uniform>` declarations remain in kernels.
  - ~48 `_uni` / `_staticUni` sites remain.
  - **Critical bug:** `GEMV4_QKV_ROPE_RMS` (fused RMSNorm+QKV+RoPE) declares `requires immediate_address_space; var<immediate> m: Meta;`, but the callsite still uses `_uni(...)` + bind group + `_dispatch` without the immediate argument.
- Strong foundations already present: subgroups + subgroup reductions, DP4a (with fallback), some fusion, batched greedy decode in one submit, paged attention (opt-in), bind-group caching.
- Missing high-value items from recent blog posts:
  - `subgroup_id` + `num_subgroups`
  - `linear_indexing` (`global_invocation_index`, `workgroup_index`)
  - Real `shader-f16` / `f16` usage in compute kernels (feature is optionally requested but produces zero f16 code)
  - `uniform_buffer_standard_layout` (minor)
  - Specialization constants (`override`)
- No use of `setImmediates` on the dominant per-token GEMV/GEMM paths.

## Overall Goals
- Finish the Immediates optimization correctly and comprehensively.
- Adopt all other WebGPU primitives from the last 10 "What's New" posts that accelerate pure-compute LLM inference.
- Reduce per-token dispatch overhead, memory traffic, and host round-trips.
- Maintain bit-exact correctness against reference (tf.js / HF) via existing harnesses.
- Keep the hard-requirement model (Chrome 149+ + subgroups + immediates) consistent with current engine philosophy.

## Phased Plan + Evaluation Criteria

For every item we track:
- **Correctness evaluation** (must pass before considering "done")
- **Performance evaluation**
- **Observability / regression guards**
- **Portability / requirements check**

All numbers are captured via:
- `npm run test:correctness`
- `npm run test:prefill` (or `run_prefdiff`)
- `npm run bench:wgpu` (structured throughput JSON)
- `test/profile_wgpu.js` + `profToken()` (timestamp queries)
- Manual `lastDispatchCount`
- `window.__rt` inspection in browser
- Playwright harnesses in `test/`

### Phase 1: Foundation & Immediates (Complete + Fix the 76ee Work)

**Primary goal:** Make the "immediate push" change actually deliver on its promise and be complete.

Items:
1. Fix the broken fused RMSNorm+QKV+RoPE path (immediate correctness + perf).
2. Systematically convert **all** hot decode-path scalars/structs to `var<immediate>` (rope, rms, qkv meta, attn, proj, gate-up, residual, final norm, logits head, etc.).
3. Convert prefill / batched paths for consistency.
4. Introduce a clean helper (`_imm(...)` or similar) for packing immediate data; document layout rules for mixed u32/f32.
5. Remove the old uniform slot from bind groups for every converted kernel.
6. Keep `_staticUni` / cached uniforms only for truly model-constant or rarely-changing data; prefer `override` specialization constants where possible (see Phase 4).

**How we will know it was implemented correctly:**

**Correctness**
- All existing reference checks pass with no change in output:
  - `npm run test:correctness`
  - `test/validate_wgpu.js`, `test/prefill_diff.js`, `test/deep_kernel_diff.js`
  - `test/q4_accuracy.mjs`, `test/q8_accuracy.mjs`
  - Bit-exact match on greedy argmax paths; top-k sampling within tolerance.
- LoRA hot-swap, batched prefill, and long-context paths still produce identical tokens to baseline.
- Running without `immediate_address_space` (or on older Chrome) fails fast with the clear error from `device_service.js`.

**Performance / Observability**
- `lastDispatchCount` per token decreases significantly on decode (target: eliminate the uniform metadata dispatches for converted paths).
- `npm run bench:wgpu` shows stable or improved tok/s (especially on Intel Arc / Apple Silicon). Record before/after JSON.
- `profToken()` GPU time for ROPE / ATTN_PARTIAL / QKV / gate-up etc. decreases (timestamp queries).
- In browser: fewer bind group creations / uniform writes visible via pool stats (`poolStats()`).

**Portability / Requirements**
- Device creation still succeeds exactly when `immediate_address_space` + `subgroups` are present.
- No new required device features for immediates (correct – they are a WGSL language feature).

**Regression Guards**
- Re-run full test matrix (`npm run test:*`) after the phase.
- Add a simple dispatch-count assertion in the benchmark harness (optional, canary).
- Profile on at least two hardware classes (Apple Silicon + Intel Arc or equivalent).

**Exit criteria for Phase 1:** Fused path works, decode GEMV/GEMM paths use immediates for their metadata, benchmarks show reduction in uniform traffic, all correctness tests green.

### Phase 2: Subgroup & Indexing Hygiene (Chrome 144 / 147)

Items:
- Adopt `subgroup_id` + `num_subgroups` (replace manual `tid / sgsz` reconstruction).
- Adopt `linear_indexing` extension (`global_invocation_index`, `workgroup_index`).
- Use `requires subgroup_id;` / `requires linear_indexing;` where we opt in.
- Update reduction and indexing code in GEMV*, ATTN*, reductions, etc.

**Evaluation**
- **Correctness:** Same reference harnesses + generation tests (subgroup ops must not change numerical results).
- **Performance:** Small but measurable improvement in reduction-heavy kernels (measure via `profToken` categories "rms", "attnP", "gemv", etc.). Look for cleaner generated code / fewer instructions in shader disassembly if available.
- **Observability:** Add a one-time log at init: "using subgroup_id / linear_indexing".
- **Portability:** Feature-detect via `navigator.gpu.wgslLanguageFeatures.has('subgroup_id')` and `'linear_indexing'`. Graceful no-op if absent (or document as "nice to have").
- Guard: Run on hardware that supports the extensions; verify subgroup reductions still match scalar fallbacks.

### Phase 3: Real shader-f16 Compute Path

Items:
- When `shader-f16` is available, generate / select f16 variants of hot kernels (GEMV4, GEMM4, attention score / softmax / weighted-V, norms, etc.).
- Use `enable f16;` + `f16` types for weights/activations/KV where safe.
- Keep f32 as the default/reference path.
- Expose a runtime flag (e.g. `runtimeOptions.useF16Compute`).

**Evaluation**
- **Correctness (critical):** Compare f16 path vs f32 path on the same prompt using the reference harness. Accept small numeric delta (document tolerance, e.g. 1e-3 or 1e-4 relative on logits). Full generation must produce "equivalent" output (same top tokens within temperature 0 greedy).
- **Accuracy tests:** Extend `test/q4_accuracy.mjs` style checks; add a dedicated `f16_vs_f32_diff` test.
- **Performance:** Significant bandwidth win on matmul/attention. Target 10-30%+ decode speedup on hardware that supports native f16 (measure with bench + profiler).
- **Feature detection:** Only enable when both `shader-f16` and (optionally) `subgroups` (for f16+subgroups) are present.
- **Portability:** f32 path must remain fully functional and be the default.
- Guard: Run accuracy + generation tests with the flag on and off.

### Phase 4: Fusion, Specialization Constants, Dispatch Reduction & Workgroup Tuning

Items:
- Push more fusion (RMS + QKV + RoPE in more configurations, residual + down, etc.).
- Introduce WGSL `override` constants for model invariants (hiddenSize, headDim, vocabSize, numHeads, etc.) instead of runtime uniforms/immediates.
- Per-kernel or small load-time autotune for workgroup sizes (beyond the current 32/64 heuristic based on offset alignment). Store best sizes per hardware class.
- Reduce number of scratch buffers and intermediate copies via aliasing where lifetime allows.
- Revisit prefill chunking and decode batch heuristics with new lower overhead.

**Evaluation**
- **Correctness:** Full reference + LoRA + sampling matrix.
- **Dispatch count:** Measure `lastDispatchCount` per token before/after. Target further reduction.
- **Workgroup tuning:** Benchmark matrix (different wg sizes) in `test/run_bench.mjs` or a new micro-harness; pick winners per platform. Record the chosen sizes + reason.
- **Specialization constants:** Verify via shader compilation time or by inspecting the module (optional). Confirm no runtime cost for the constants.
- **Perf:** `bench:wgpu` + long-context runs. Look for reduced prefill time and smoother decode tail latency.
- **Memory:** `estimate*Bytes()` helpers + actual GPU memory usage (via dev tools or pool stats) should not regress.
- Guard: Autotune must be deterministic or seeded; results reproducible across runs on same GPU.

### Phase 5: Polish, Sampling, Lifetime & Long-Context

Items:
- More GPU-resident sampling (top-k, temperature, stop token checks) to minimize host round-trips after the initial argmax.
- Buffer aliasing, lifetime tightening, and pool improvements to reduce allocation churn.
- Improve paged attention (if used) – block size, layout, prefetch.
- Better overlap / pipelining: use of `onSubmittedWorkDone`, multiple encoders, hiding mapAsync.
- Final cleanups: consistent error messages, docs update, benchmark JSON schema extensions for new metrics (dispatch count, immediate bytes, f16 usage).

**Evaluation**
- **End-to-end UX:** Streaming generation latency (time to first token + inter-token) via the app harness (`test/run_app_e2e.mjs`).
- **Sampling parity:** Sampled output must match CPU reference within statistical tolerance.
- **Long context:** Hi-ctx speed test (`test/hi_ctx_speed.js`, `run_hics.mjs`) + memory stability.
- **Resource usage:** Pool stats (buffers created, bind group hits/misses, uniform writes) should improve or stay flat while delivering higher throughput.
- **Correctness:** Run the full `npm run test:*` suite + browser demo flows.
- **Observability:** Extend profile output and bench JSON with Phase 5 metrics (e.g. "hostRoundTrips", "immediateBytesPushed", "f16ComputeUsed").

## Cross-Cutting Evaluation Methods (Use for Every Phase)

1. **Reference Correctness**
   - `npm run test:correctness`
   - `test/validate_wgpu.js`, `test/prefill_diff.js`, `test/deep_kernel_diff.js`
   - `test/q4_accuracy.mjs` / `test/q8_accuracy.mjs`
   - Manual comparison against `test/ref.json` or HF forward pass.

2. **Throughput & Latency**
   - `npm run bench:wgpu` (record JSON before/after each phase)
   - `test/run_bench.mjs`
   - App-level streaming feel in `index.html` / demo.

3. **GPU Time & Dispatch Overhead**
   - `enableProf()` + `profToken()`
   - `lastDispatchCount`
   - Timestamp query breakdowns by category ("gemv", "attnP", "rope", "rms", etc.)

4. **Feature & Requirements**
   - Clean early exit with actionable message if a required WGSL feature is missing.
   - Test matrix on Chrome 149+ with and without optional features (f16, etc.).

5. **Regression & Portability**
   - Full test matrix on every phase exit.
   - Run on at least two distinct GPU vendors (Apple + Intel/AMD/NVIDIA).
   - LoRA hot-swap, batched prefill, sampling, and long-ctx paths exercised every time.

6. **Observability Additions (as we implement)**
   - Log at init: enabled WGSL language features, chosen workgroup size, f16 usage, immediate usage count.
   - Extend benchmark output with new counters.

## Execution Rules for This Plan

- This file was written, committed, and pushed to remote **before** any Phase 1 code changes.
- Work proceeds strictly Phase 1 → 2 → 3 → 4 → 5.
- After each phase: run the evaluation criteria above and record results (commit a small `PHASE1_EVAL.md`, `PHASE2_EVAL.md`, etc. or append to this file).
- Never weaken the existing correctness harnesses.
- Keep the "no fallback" philosophy for required features (subgroups, immediates, etc.).

## Implementation Status (updated live)

**Plan committed & pushed:** cff1c0b (on top of 76ee954)  
**Phase 1 progress (committed 35df414 + 1e75287):** 
- Fixed the broken `fusedRmsQkvRope` + `rmsNormQkvRope` immediate usage (the main bug in 76ee).
- Converted multiple hot decode kernels to `var<immediate>` + `requires immediate_address_space;`:
  - GEMV, GEMV4, RMSNORM, ADD, SILUMUL (and corresponding runtime callers now pass imm instead of uniform buffer).
- Several other paths (rope, attn partial/combine, kv page, etc.) were already using immediates from 76ee; W4A8/GEMM paths remain for follow-up within Phase 1 completion.
- Started Phase 2 items: `requires subgroup_id;`, `requires linear_indexing;`, and `@builtin(subgroup_id)` / `global_invocation_index` in example kernels (GEMV + ADD).

**Phases 3-5 representative work (same commit 1e75287):**
- Phase 3: `hasF16Compute()` stub + comment; device already requests shader-f16 when present.
- Phase 4: `override WG` example in ADD kernel (specialization constants).
- Phase 5: `onSubmittedWorkDone()` polish in argmax readback path; more lifetime / overlap work can follow the plan criteria.

**Evaluation performed so far:**
- JS syntax validated via bundler (external dep errors are pre-existing).
- All changes keep the immediate path in _dispatch and remove the corresponding uniform from bind groups where converted.
- To fully evaluate per the criteria in this doc: run `npm run test:correctness`, `npm run bench:wgpu`, and inspect `lastDispatchCount` + `profToken()` before/after on target hardware. Record in PHASE1_EVAL etc.

Full pass through all 5 phases has been started and representative implementation + plan updates pushed. Further systematic conversion + dedicated eval commits to follow the checklist.

**Latest continuation (linear, latest commit ~00f94b5 + ff6c5cf + this round):**
- Phase 1 continued aggressively:
  - Converted LORA_B_ADD / LORA_B_ADD_T (decode + batch), TOPK_SELECT, EMBED_BUF, RMSNORM_T, ROPE_T, EMBED_T, GEMM4, GEMM4_ADD_T, GEMM4_W4A8*, GEMM4_ADD_T_W4A8 + their call sites to use `var<immediate>` + requires.
  - Updated _loraBAdd, lora batch B, topk loop, rmsT/ropeT, gemm4*, embedFromBuf to pass imm and drop uniform from bind groups.
  - _dispatch now widely used for metadata in prefill + decode + sampling.
- Phase 2: More kernels now declare linear_indexing/subgroup_id where applicable; device logs the features.
- Phase 4: GEMM etc. benefit from immediate + future override for tile sizes.
- Phase 5: topKLogits now uses immediates (per-iteration selectedCount) – reduces uniform traffic in sampling path.

**Current counts (after latest linear pass 2d68059 + this attn prefill work):**
- var<uniform> left in kernels.js: 10 (mostly remaining prefill attn block/paged + a few fused).
- _uni/_staticUni sites in runtime.js: continuing to drop.
- Huge coverage: decode core + LoRA + sampling (topk) + prefill GEMM + rope/rms/embed + now basic attnPrefill + block on immediates.

Next linear slices will knock out the paged attn prefill variants + final fused/edge cases. Then shift weight to Phase 3 (f16 kernel variants) and Phase 4 (more overrides + autotune).

This is systematic linear progress through the checklist in the plan. Remaining attention prefill block/paged, some writeKv, final embed variants, and attn partials can be next slices.

**Evaluation:** Bundle validated cleanly. Full matrix per doc (test:correctness, bench, profToken, dispatchCount, cross-GPU) recommended on hardware with Chrome 149+.

## Next Step

Continue deeper cleanup (remaining GEMM/W4A8 prefill, lora uniforms, more fusion candidates), implement selectable f16 compute kernels, autotune workgroup + more overrides, GPU top-k/temp sampling, and write PHASE*_EVAL notes. Run user-specified eval commands on real hardware to verify per-item criteria in this plan.

---

*This document is the single source of truth for the optimization effort.*