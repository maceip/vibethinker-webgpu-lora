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

import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';

const BASE = {
  fuseQKV: false,
  fuseRoPE: false,
  fuseMLP: false,
  fuseResidual: false,
  prefillAttention: 'row',
  prefillChunkSize: 0,
};
const FUSED = {
  fuseQKV: true,
  fuseRoPE: true,
  fuseMLP: true,
  fuseResidual: true,
  prefillAttention: 'block',
  prefillChunkSize: 0,
};

const row = (data) => console.log('VWG_AB ' + JSON.stringify(data));

async function requestDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  return { adapter, dev };
}

async function benchDecode(rt, flags, { batch, latencyCap, ctx, tokens, label }) {
  rt.setFeatureFlags(flags);
  rt.decodeBatchMode = 'fixed';
  rt.MAXBATCH = batch;
  rt.decodeBatchMaxLatencyMs = latencyCap;
  rt.decodeBatchWarmupTokens = 0;
  const ref = await (await fetch('./ref.json')).json();
  const ids = ref.ids;
  const tile = (L) => {
    const out = [];
    while (out.length < L) out.push(ids[out.length % ids.length]);
    return out;
  };
  rt.prefillBatch(tile(ctx));
  await rt.argmaxLogits();
  let pos = ctx,
    emitted = 0;
  await rt.dev.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  while (emitted < tokens && pos < rt.maxCtx) {
    const k = Math.min(batch, tokens - emitted, rt.maxCtx - pos);
    const b = await rt.decodeGreedyBatch(pos, k);
    emitted += b.length;
    pos += b.length;
  }
  const seconds = (performance.now() - t0) / 1000;
  row({
    label,
    flags: flags === FUSED ? 'fused' : 'baseline',
    batch,
    latencyCap,
    ctx,
    tokens: emitted,
    seconds,
    tokPerSec: emitted / seconds,
    dispatchesPerToken: rt.lastDispatchCount,
  });
}

window.run = async () => {
  const { adapter, dev } = await requestDevice();
  row({ type: 'device', adapter: adapter?.info?.description || 'unknown' });
  const rt = new QwenWGPU(dev, QWEN25_3B, { decodeBatchSize: 16, maxDecodeBatchSize: 32 });
  await rt.build('/model');
  row({ type: 'load', seconds: 'done', features: rt.featureFlags() });

  for (const flags of [BASE, FUSED]) {
    for (const batch of [4, 16]) {
      await benchDecode(rt, flags, {
        batch,
        latencyCap: Infinity,
        ctx: 128,
        tokens: 32,
        label: `${flags === FUSED ? 'fused' : 'baseline'}-b${batch}-ctx128`,
      });
    }
  }

  rt.decodeBatchMode = 'auto';
  rt.decodeBatchMaxLatencyMs = 250;
  await rt.autotuneDecodeBatch();
  row({ type: 'autotune-250ms', ...rt.decodeBatchTuning });
  await benchDecode(rt, FUSED, {
    batch: rt.MAXBATCH,
    latencyCap: 250,
    ctx: 128,
    tokens: 32,
    label: `autotuned-b${rt.MAXBATCH}`,
  });

  rt.decodeBatchMaxLatencyMs = Infinity;
  await rt.autotuneDecodeBatch();
  row({ type: 'autotune-nocap', ...rt.decodeBatchTuning });
  await benchDecode(rt, FUSED, {
    batch: rt.MAXBATCH,
    latencyCap: Infinity,
    ctx: 128,
    tokens: 32,
    label: `autotuned-nocap-b${rt.MAXBATCH}`,
  });

  console.log('VWG DONE');
  window.__done = true;
};

window.addEventListener('DOMContentLoaded', () => window.run().catch((e) => console.log('VWG ERROR ' + e.message)));
