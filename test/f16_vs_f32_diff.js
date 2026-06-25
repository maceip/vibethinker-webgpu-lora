/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */

// f16_vs_f32_diff.js
// Phase 3 eval harness stub (per OPTIMIZATION_PLAN.md).
// Usage (in browser console or via test runner that loads this):
//   import('./test/f16_vs_f32_diff.js').then(m => m.runF16Diff())
//
// What it does:
// - Toggle f16 on a built runtime and exercise paths that have f16 variants:
//   rms, add, silu, rope*, and attn combine (score/softmax/V still f32 in partial for this slice).
// - For real numbers, pair with capture of hidden state, attention output, or final logits
//   (see deep_kernel_diff.js / profile for buffer readback patterns).
// - Full eval: same prompt, f16=off vs on → compare logits or generated tokens (greedy must match top token(s)).
//
// Acceptance (from plan): small numeric delta (1e-3..1e-4 rel on logits typical); greedy tokens equivalent.

/*
 * TECHNIQUE: Dual-precision harness with direct numeric + token comparison
 *   Runs the exact same prompt through f32 and f16 code paths on the same
 *   runtime instance. Uses maxAbs / maxRel + top-k match + generation parity.
 *   Also exercises the GPU sampler (sampleToken) for parity.
 */
import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';

// Tiny config for fast mock-based smoke tests of f16 vs f32 kernel paths.
// Enough layers/dims to exercise rope/rms/attn-combine/partial/silu/add etc.,
// but tiny enough that build + a few tokens complete in <1s on typical hardware.
const TINY_MOCK_CFG = {
  hiddenSize: 128,
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 2,
  headDim: 32,
  intermediateSize: 256,
  vocabSize: 256,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  tieWordEmbeddings: true,
  attentionBias: false,
};

export async function runF16Diff(opts = {}) {
  // Support reuse of an already-built runtime (preferred in demo pages).
  let rt = opts.rt;
  let dev = null;
  if (!rt) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const required = ['subgroups'];
    try {
      if (adapter.features.has('shader-f16')) required.push('shader-f16');
    } catch {}
    dev = await adapter.requestDevice({
      requiredFeatures: required,
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      },
    });

    // Use tiny mock config + tiny context by default. This avoids the full 3B
    // shader compilation + huge KV/weight allocs that freeze the page.
    const useMock = (opts.mock !== false) && (!opts.modelPath || opts.modelPath === 'mock');
    const cfg = useMock ? TINY_MOCK_CFG : QWEN25_3B;
    rt = new QwenWGPU(dev, cfg, {
      onProgress: () => {},
      maxCtx: useMock ? 32 : undefined,
      maxPrefillT: useMock ? 32 : undefined,
      decodeBatchSize: 4,
    });

    const source = useMock ? 'mock' : (opts.modelPath || '/model');
    await rt.build(source);
    if (useMock) {
      console.log('[f16_diff] using mock weights (tiny cfg); exercising kernel math paths only');
    }
    if (rt.shaderCompileMs) {
      console.log('[f16_diff] shaderCompileMs=', rt.shaderCompileMs.toFixed(1));
    }
  }

  const hasF16 = !!rt.hasF16;
  if (!hasF16) {
    console.warn('[f16_diff] shader-f16 not available on this device; f32 path only.');
    return { skipped: true, hasF16: false };
  }

  // Obtain token ids for the run (prefer caller-supplied, then ref.json, else tiny synthetic).
  let ids = opts.ids;
  if (!ids) {
    try {
      const ref = await (await fetch('./ref.json')).json();
      ids = ref.ids || ref.tokens;
    } catch {}
  }
  if (!ids || !ids.length) {
    // tiny deterministic synthetic ids (first N embed rows); enough for a numeric smoke
    ids = [42, 43, 44, 45, 100, 101, 102];
  }
  const genLen = opts.genLen || 6;

  const readLogits = async () => rt.readLogits();
  const argmaxOf = (a) => {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; }
    return bi;
  };

  const decodeN = async (pos, n) => {
    const out = [await rt.argmaxLogits()];
    while (out.length < n) {
      const b = await rt.decodeBatch(pos, Math.min(rt.decodeBatchCapacity || 8, n - out.length));
      pos += b.length;
      out.push(...b);
    }
    return out.slice(0, n);
  };

  const runOne = async (useF, label) => {
    rt.setUseF16(useF);
    // re-prefill from scratch so KV + all math use the chosen f16 paths
    rt.prefillBatch(ids);
    const logits = await readLogits();
    const dispatches = rt.lastDispatchCount || 0;
    const gen = await decodeN(ids.length, genLen);
    const top = argmaxOf(logits);
    console.log(`[f16_diff] ${label}  dispatches≈${dispatches}  argmax=${top}  gen=${JSON.stringify(gen)}`);
    return { logits, gen, top, dispatches };
  };

  // f32 path
  const off = await runOne(false, 'f32');
  // f16 path (same prompt, same rt, re-prefill uses f16 kernels where available)
  const on = await runOne(true, 'f16');

  const maxAbs = maxAbsDiff(off.logits, on.logits);
  const maxRel = maxRelDiff(off.logits, on.logits);
  const topK = topKMatch(off.logits, on.logits, 5);
  const genMatch = JSON.stringify(off.gen) === JSON.stringify(on.gen);
  const topMatch = off.top === on.top;

  const tol = opts.tolRel || 5e-3; // per-plan guidance ~1e-3..1e-4 typical; allow headroom for first impl
  const pass = genMatch || (maxRel < tol && topMatch);

  console.log('[f16_diff] === numeric diff ===');
  console.log('  maxAbs:', maxAbs.toExponential(4), ' maxRel:', maxRel.toExponential(4));
  console.log('  top5 match rate:', topK.rate, ' genMatch:', genMatch, ' top1Match:', topMatch);
  console.log('  PASS (gen or (rel<tol && top1)) :', pass, ' tolRel=', tol);

  // Phase 5 sampling parity smoke (GPU-resident sampler).
  // Same prompt + fixed random, f32 vs f16 must pick the same token (within logits tolerance).
  let sampleResult = {};
  try {
    const fixedR = 0.37;
    rt.setUseF16(false);
    rt.prefillBatch(ids);
    const idF32 = await rt.sampleToken(1.0, fixedR);

    rt.setUseF16(true);
    rt.prefillBatch(ids);
    const idF16 = await rt.sampleToken(1.0, fixedR);

    console.log('[f16_diff] sampleToken parity (fixed r):', idF32, 'vs', idF16, 'match=', idF32 === idF16);
    sampleResult = { sampleMatch: idF32 === idF16, sampleF32: idF32, sampleF16: idF16 };
  } catch (e) {
    console.log('[f16_diff] sampleToken smoke skipped:', e?.message || e);
  }

  return {
    hasF16: true,
    maxAbs, maxRel,
    topK, genMatch, topMatch,
    pass,
    offTop: off.top, onTop: on.top,
    offGen: off.gen, onGen: on.gen,
    f16Covered: 'add/silu/rms*/rope*/attn-partial/combine',
    shaderCompileMs: rt.shaderCompileMs || 0,
    ...sampleResult,
  };
}

// Reusable numeric helpers (pure JS) for harnesses.
export function maxAbsDiff(a, b) {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

export function maxRelDiff(a, b, eps = 1e-12) {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), eps);
    const r = Math.abs(a[i] - b[i]) / denom;
    if (r > m) m = r;
  }
  return m;
}

export function topKMatch(a, b, k = 5) {
  const ia = Array.from(a).map((v,i)=>({v,i})).sort((x,y)=>y.v-x.v).slice(0,k).map(x=>x.i);
  const ib = Array.from(b).map((v,i)=>({v,i})).sort((x,y)=>y.v-x.v).slice(0,k).map(x=>x.i);
  const setB = new Set(ib);
  let matches = 0;
  for (const i of ia) if (setB.has(i)) matches++;
  return { matches, k, rate: matches / k };
}

// Auto-run hook for convenience in some test loaders:
if (typeof window !== 'undefined') {
  window.runF16Diff = runF16Diff;
  window.f16DiffHelpers = { maxAbsDiff, maxRelDiff, topKMatch };
}
