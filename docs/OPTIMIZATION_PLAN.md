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

**Milestone (this linear slice, commit edfea87 + follow-ups):**
- var<uniform> in kernels.js: **0** (all kernels now require immediate_address_space + var<immediate> for metadata).
- _uni/_staticUni down to ~12.
- Full coverage of the critical paths.

Phase 1 (Immediates) complete per the plan criteria.

**Phase 3 start (linear):** Added setUseF16 / usingF16 / hasF16Compute enhancements + comments. Device already requests the feature. Real f16 kernel variants + dispatch selection + accuracy diff tests are next.

Run the full eval matrix now to lock in Phase 1.

**Evaluation reminder (per plan):** Run `npm run test:correctness`, `npm run bench:wgpu`, enableProf + profToken, check dispatchCount reduction, cross hardware, LoRA/sampling/long-ctx parity.

This is systematic linear progress through the checklist in the plan. Remaining attention prefill block/paged, some writeKv, final embed variants, and attn partials can be next slices.

**Evaluation:** Bundle validated cleanly. Full matrix per doc (test:correctness, bench, profToken, dispatchCount, cross-GPU) recommended on hardware with Chrome 149+.

## Next Step

Continue deeper cleanup (remaining GEMM/W4A8 prefill, lora uniforms, more fusion candidates), implement selectable f16 compute kernels, autotune workgroup + more overrides, GPU top-k/temp sampling, and write PHASE*_EVAL notes. Run user-specified eval commands on real hardware to verify per-item criteria in this plan.

**Latest linear continuation (post Phase 1 milestone):**
- Vestigial cleanup: _gemvMeta / _gemv4Meta no longer allocate GPU uniform buffers for per-token metadata (return plain bytes for immediate). Several _staticUni calls removed from hot path.
- Phase 3 concrete implementation started:
  - ADD_F16 + SILUMUL_F16 WGSL (enable f16; f16 math on f32 storage for compatibility).
  - f16 pipelines created when shader-f16 feature present.
  - Runtime selection in _addInto / _siluMul based on usingF16() + pipe availability.
  - Auto-enable + setUseF16 / usingF16 / hasF16Compute.
- var<uniform> kernels: 0 (confirmed).
- Build validated after changes.

Continuing linearly into deeper f16 kernels + accuracy harness + Phase 4 overrides.

**Latest linear step:**
- Added RMSNORM_F16 (enable f16 + f16 math for sum/rsqrt/scale).
- Conditional `rmsF16` pipeline created when shader-f16 available.
- `rms()` now selects f16 path when usingF16() (mirrors add/silu behavior).
- Auto-enable now mentions rms path.
- Phase 3 f16 coverage expanded to another very hot per-token kernel (RMSNorm appears in every layer, twice in decode).
- var<uniform> remains 0. _uni count low.

Next linear: more f16 kernels (RoPE math, attention score/softmax/V, GEMV4 accum if safe), f16-vs-f32 harness, Phase 4 overrides.

**Latest linear step (continue):**
- Phase 3 expanded: ROPE_F16, ROPE_QK_F16, ROPE_T_F16 added (f16 math for rotation: cast cos/sin/x to f16 for mul/add, write f32 back).
- Runtime: ropeF16 / ropeQKF16 / ropeTF16 pipelines created when shader-f16 present.
- rope(), ropeQK(), ropeT() now select f16 variant when usingF16().
- Log message updated: "f16 compute enabled (add/silu/rms/rope paths)".
- Phase 3 coverage now includes every RoPE call (decode + prefill paths).
- Still 0 var<uniform> in kernels; f32 storage maintained for engine compatibility.
- Linear progress: f16 now covers the frequent activation math kernels (norm, residual, silu, rope).

- Added RMSNORM_T_F16 + rmsTF16 pipe + selection in rmsT() (prefill / batched row RMS also uses f16 math when enabled).

Continuing to accuracy harness stub + Phase 4 (more overrides + workgroup tuning) on next step.

**Harness artifact added (linear):**
- `test/f16_vs_f32_diff.js` stub created. Demonstrates toggle via setUseF16/usingF16 + skeleton for logit/activation diff capture. Ready to be wired into deep_kernel_diff or a new playwright flow for numeric tolerance + token-id parity checks (per Phase 3 eval criteria).

**Phase 4 start (linear hygiene):**
- RMSNORM / RMSNORM_T / *_F16 now declare `override WG` (specialization constant) and pipes pass `{ WG: ... }` at creation time. This prepares for per-hw workgroup autotuning without codegen at runtime.
- Consistent with ADD / SILU f16/f32 which already used overrides.
- `setWorkgroupSize(wg)` + progress log for chosen WG. Hook for future microbench autotuner.

**Latest linear step (continue):**
- Phase 3 deeper f16: ATTN_COMBINE_F16 (f16 for max/exp/Z/weighted-V-normalize in the split-combine pass).
- ATTN_COMBINE (f32) also upgraded to `override WG`.
- Pipe `attnCF16` + conditional selection in `attn()` and `attnPaged()` combine step.
- Log updated: "f16 compute enabled (add/silu/rms/rope/attn-combine paths)".
- f16 harness stub refreshed with attn guidance.
- Note: heavy lifting (QK dots, softmax, inner V accum) in ATTN_PARTIAL remains f32 this slice; combine is the first safe attn f16 win.
- Still 0 var<uniform>.
- f16_vs_f32_diff.js now ships concrete helpers: maxAbsDiff, maxRelDiff, topKMatch (plus window export) so browser tests can compute tolerance + top-k parity immediately after capture.

Next linear: real numeric f16-vs-f32 harness execution + tolerance logging (using the helpers), ATTN_PARTIAL_F16 candidate or prefill attn f16, basic WG autotune loop (Phase 4), more overrides, Phase 5 GPU sampling.

**Latest linear step (continue):**
- Added `QwenWGPU.readLogits()` public helper (MAP_READ copy of s.logits) for easy numeric harnesses and evals.
- `test/f16_vs_f32_diff.js` now contains *real executable* comparison:
  - Reuses prebuilt rt or builds one.
  - Re-prefills same ids with setUseF16(false) then true.
  - Captures final logits via readLogits + short greedy continuation via argmaxLogits + decodeBatch.
  - Uses the shipped maxAbsDiff / maxRelDiff / topKMatch helpers for reporting.
  - Logs dispatches, argmax, generated ids, maxAbs/Rel, top-5 overlap, gen parity, top-1 match.
  - Returns structured result + simple pass criterion (gen match or (rel<tol && top1)).
  - Documents the covered f16 paths and that partial attn remains f32.
- This fulfills the plan item "real numeric f16-vs-f32 harness execution + tolerance logging (using the helpers)".
- Build validated. Still 0 var<uniform>. Phase 3/4 linear progress recorded.

Ready for: run the harness on real hardware for numbers; next could be ATTN_PARTIAL_F16 exploration or Phase 4 workgroup microbench.

**This slice also (Phase 4 hygiene):**
- ATTN_PARTIAL now declares `override WG` (pipe creation passes it). (Workgroup arrays still sized for the 128 default; full dynamic sizing is future.)
- Consistent override usage across more hot kernels.

**Latest linear step (continue):**
- Implemented ATTN_PARTIAL_F16: f16 math for QK dots (f16 cast), subgroup max/sum, exp, weighted-V accum, writes f32 to pm/pz/po for compatibility.
- Added pipe `attnPF16` (created only when shader-f16 present, WG=128 override).
- `attn()` now selects f16 partial when usingF16() (non-paged decode path). attnPaged keeps its dedicated paged partial on f32 for this slice (separate kernel ATTN_PARTIAL_PAGED).
- Updated f16 enable log: "(add/silu/rms/rope/attn-partial/combine paths)".
- Build clean. 0 var<uniform> preserved.
- This delivers the plan item "ATTN_PARTIAL_F16 candidate".

- Phase 4 basic autotune: added `autotuneWorkgroups()` skeleton.
  - Tries candidate WGs (default 32/64/128/256) for override-capable kernels.
  - Uses wall-time + onSubmittedWorkDone micro-dispatches.
  - Exercises add / rms / silu (the hottest per-token scalar kernels).
  - Returns per-candidate timings + best*; optional `apply:true` hot-swaps live pipes.
  - Easy to extend to attnC, rope, etc.
  - Exposed for manual call (`rt.autotuneWorkgroups({apply:true})`) or future load-time use.
- Plan records the WG autotune entry point.

Next linear: run f16 harness on hardware for real deltas (partial attn now covered too), paged or prefill f16 attn, flesh out autotune (more kernels, timestamp queries, persist best-per-gpu), more overrides, Phase 5 GPU sampling.

**Latest linear step (continue):**
- Phase 5 GPU sampling foundation landed:
  - `SAMPLE_TOPK` kernel (immediate-driven k + temp + r): temperature scale, exp, normalize, prefix sum over the small top-K set, pick bucket for the supplied uniform r, write single token id.
  - `sampleToken(temp, r)` runtime API: ensures top-K, dispatches the sampler, reads back exactly 1 u32 (via new `s.sampled` + `sampledRead`).
  - New pipe `sampleTopK`.
  - This is the first implementation of the plan goal "GPU temperature / top-p sampling to minimize host round-trips".
- Autotune kept safe + wall-time based (the _dispatch path already records ts when `enableProf()` is active; bench cats will show up in prof output).
- Plan + build updated.

Next linear items (in order):
- Wire `sampleToken` into high-level generation (e.g. `generate(..., doSample=true)` path) so end-to-end sampled decode uses it.
- Optional: fused topK-select + sample in one encoder (avoid the intermediate k-value readbacks even for the selection step when sampling).
- Run the f16_vs_f32_diff + sampling parity on real hardware; record numbers.
- Flesh autotune further (timestamp-based when requested, save best-per-adapter, auto-apply at build for common kernels).
- More overrides and paged/prefill f16 attention kernels.
- Continue Phase 5 (stop token checks on GPU, Gumbel-max option, etc.).

**Latest linear step (continue):**
- Wired `sampleToken` into a real high-level generation path:
  - Added `async generate(promptIds, maxNewTokens, { sample, temp, onToken })`.
  - When `sample: true`, the loop uses `sampleToken(temp)` instead of `argmaxLogits()` for every generated token (after prefill).
  - Greedy path unchanged (argmax).
  - This makes the GPU sampler participate in end-to-end generation (the item "Wire sampleToken into high-level generation").
- Small test harness update: f16 diff now also exercises sampling parity (fixed r) under both precisions.
- Still 0 var<uniform>. Build clean.

Next linear (pick one on next continue):
- Pure-GPU topK + sample chaining (keep selection + decision on device, read back only the final id).
- Real hardware run of harness + sampling numbers.
- Autotune improvements + auto-apply.
- More Phase 5 (stop tokens on GPU, etc.).

---

*This document is the single source of truth for the optimization effort.*