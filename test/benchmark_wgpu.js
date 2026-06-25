/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';
import { loadLoraAdapterGPU } from '../src/lora_gpu.js';

/*
 * TECHNIQUE: Structured JSON benchmark output
 *   Every measurement is emitted as a single-line JSON object with a VWG_BENCH
 *   prefix. Makes parsing by external tools trivial and keeps the harness
 *   machine-readable.
 */
const row = (data) => console.log('VWG_BENCH ' + JSON.stringify(data));

async function requestDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  if (!adapter.features.has('subgroups'))
    throw new Error('GPU lacks required "subgroups" feature; no fallback kernels are bundled');
  const hasTimestamp = adapter.features.has('timestamp-query');
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups', ...(hasTimestamp ? ['timestamp-query'] : [])],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
  return { adapter, dev, hasTimestamp };
}

const tile = (base, L) => {
  const out = [];
  while (out.length < L) out.push(base[out.length % base.length]);
  return out;
};

async function tryFetchAdapter(dir) {
  const cfg = await fetch(`/${dir}/adapter_config.json`);
  const st = await fetch(`/${dir}/adapters.safetensors`);
  if (!cfg.ok || !st.ok) return null;
  const wrap = async (resp, name) => {
    const buf = await resp.arrayBuffer();
    return {
      name,
      async text() {
        return new TextDecoder().decode(buf);
      },
      async arrayBuffer() {
        return buf;
      },
    };
  };
  return [await wrap(cfg, 'adapter_config.json'), await wrap(st, 'adapters.safetensors')];
}

async function timeGreedy(rt, ids, startCtx, tokens) {
  rt.prefillBatch(tile(ids, startCtx));
  await rt.argmaxLogits();
  let pos = startCtx,
    emitted = 0;
  await rt.dev.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  while (emitted < tokens && pos < rt.maxCtx) {
    const k = rt.greedyBatchSizeFor({ emitted, remaining: tokens - emitted, pos });
    const b = await rt.decodeGreedyBatch(pos, k);
    emitted += b.length;
    pos += b.length;
  }
  return { tokens: emitted, seconds: (performance.now() - t0) / 1000 };
}

window.run = async () => {
  const { adapter, dev, hasTimestamp } = await requestDevice();
  dev.addEventListener?.('uncapturederror', (e) => row({ type: 'gpu-error', message: e.error.message.slice(0, 200) }));
  row({
    type: 'device',
    timestampQuery: hasTimestamp,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  });
  const ref = await (await fetch('./ref.json')).json();

  const rt = new QwenWGPU(dev, QWEN25_3B, {
    decodeBatchSize: 16,
    maxDecodeBatchSize: 32,
    samplingTopK: 40,
    maxSamplingTopK: 64,
  });
  const tBuild = performance.now();
  await rt.build('mock');
  row({
    type: 'load',
    seconds: (performance.now() - tBuild) / 1000,
    shaderCompileMs: rt.shaderCompileMs || 0,
    kvBytes: rt.estimateKvCacheBytes(),
    pool: rt.poolStats(),
  });
  const workgroupTuning = rt.workgroupAutotunePromise ? await rt.workgroupAutotunePromise : null;
  if (workgroupTuning) row({ type: 'workgroup-autotune', ...workgroupTuning });

  const tuning = await rt.autotuneDecodeBatch();
  row({ type: 'decode-autotune', ...tuning });

  for (const L of [64, 256, 1024, 4096, 8192]) {
    if (L > rt.maxPrefillT) {
      row({ type: 'prefill', L, skipped: 'exceeds maxPrefillT' });
      continue;
    }
    const t0 = performance.now();
    rt.prefillBatch(tile(ref.ids, L));
    await rt.argmaxLogits();
    row({
      type: 'prefill',
      L,
      ms: performance.now() - t0,
      scratchBytes: rt.estimatePrefillScratchBytes(L),
      pool: rt.poolStats(),
    });
  }

  for (const ctx of [128, 1024, 4096, 7800]) {
    if (ctx >= rt.maxCtx) {
      row({ type: 'greedy-decode', ctx, skipped: 'ctx exceeds maxCtx' });
      continue;
    }
    const r = await timeGreedy(rt, ref.ids, ctx, 32);
    row({
      type: 'greedy-decode',
      ctx,
      selectedBatch: rt.MAXBATCH,
      tokens: r.tokens,
      seconds: r.seconds,
      tokPerSec: r.tokens / r.seconds,
      readbackBytesPerToken: 4,
    });
  }

  for (const k of [1, 2, 4, 8, 16, 32]) {
    if (k > rt.decodeBatchCapacity) {
      row({ type: 'batch-candidate', k, skipped: 'exceeds capacity' });
      continue;
    }
    dev.queue.writeBuffer(rt.s.amax, 0, new Uint32Array([0]));
    const t0 = performance.now();
    const got = await rt.decodeGreedyBatch(0, k);
    const seconds = (performance.now() - t0) / 1000;
    row({ type: 'batch-candidate', k, seconds, tokPerSec: got.length / seconds });
  }

  rt.prefillBatch(ref.ids);
  let pos = ref.ids.length;
  const tSample = performance.now();
  for (let i = 0; i < 8; i++) {
    const next = await rt.sampleToken(1.0, 0.5);
    rt.token(next, pos++);
  }
  const sampleSeconds = (performance.now() - tSample) / 1000;
  row({
    type: 'sampling-topk',
    topK: 40,
    tokens: 8,
    seconds: sampleSeconds,
    tokPerSec: 8 / sampleSeconds,
    readbackBytesPerToken: 4,
  });

  if (hasTimestamp) {
    rt.enableProf(700);
    rt.prefillBatch(ref.ids);
    const next = await rt.argmaxLogits();
    const sums = await rt.profToken(next, ref.ids.length);
    row({ type: 'profile-token', ctx: ref.ids.length, categoriesUs: sums });
    rt.prof = null;
  } else row({ type: 'profile-token', skipped: 'timestamp-query unavailable' });

  const adapterFiles = await tryFetchAdapter('adapters_sel');
  if (adapterFiles) {
    const lora = await loadLoraAdapterGPU(dev, adapterFiles, QWEN25_3B);
    rt.setLora(lora);
    const r = await timeGreedy(rt, ref.ids, 256, 16);
    row({
      type: 'lora-greedy-decode',
      modules: Object.keys(lora.modules).length,
      tokens: r.tokens,
      seconds: r.seconds,
      tokPerSec: r.tokens / r.seconds,
    });
  } else row({ type: 'lora-greedy-decode', skipped: 'adapters_sel fixture unavailable' });

  row({ type: 'done' });
  console.log('VWG DONE');
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => {
    row({ type: 'error', message: e.message, stack: (e.stack || '').slice(0, 500) });
    console.log('VWG ERROR ' + e.message);
  }),
);
