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
const CHUNKED = { ...FUSED, prefillChunkSize: 64 };
window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const hasTS = adapter.features.has('timestamp-query');
  console.log('VWG timestamp-query=' + hasTS);
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups', ...(hasTS ? ['timestamp-query'] : [])],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json();
  const ids = ref.ids;
  const rt = new QwenWGPU(dev, QWEN25_3B);
  await rt.build('/model');
  console.log('VWG built; default features=' + JSON.stringify(rt.featureFlags()));
  console.log('VWG memory initial=' + JSON.stringify(rt.memoryFootprintBytes()));
  // prime the KV cache with the prompt
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let nxt = await rt.argmaxLogits();
  let pos = ids.length;
  // warm up to a LONG context to profile the app's long-context decode regime
  const WARM = 3200;
  for (let s = 0; s < WARM; s++) {
    rt.token(nxt, pos);
    pos++;
    nxt = await rt.argmaxLogits();
  }
  console.log('VWG profiling at ctx=' + pos);

  const profileDecode = async (label, flags) => {
    rt.setFeatureFlags(flags);
    rt.enableProf(900);
    const N = 5;
    const agg = {};
    let total = 0;
    let dispatches = 0;
    for (let s = 0; s < N; s++) {
      const sums = await rt.profToken(nxt, pos);
      dispatches += rt.lastDispatchCount;
      pos++;
      nxt = await rt.argmaxLogits();
      for (const k in sums) {
        agg[k] = (agg[k] || 0) + sums[k];
        total += sums[k];
      }
    }
    const rows = Object.entries(agg)
      .map(([k, v]) => [k, v / N])
      .sort((a, b) => b[1] - a[1]);
    console.log('VWG === ' + label + ' per-token GPU breakdown (us), avg of ' + N + ' tokens ===');
    for (const [k, v] of rows)
      console.log(
        'VWG ' +
          label +
          ' ' +
          v.toFixed(1).padStart(8) +
          ' us  ' +
          ((100 * v * N) / total).toFixed(1).padStart(5) +
          '%  ' +
          k,
      );
    console.log(
      'VWG ' +
        label +
        ' dispatches/token=' +
        (dispatches / N).toFixed(1) +
        ' totalGPU=' +
        (total / N).toFixed(1) +
        'us features=' +
        JSON.stringify(rt.featureFlags()),
    );
    rt.prof = null;
    await dev.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    const K = 32;
    const b = await rt.decodeBatch(pos, K);
    pos += b.length;
    nxt = b[b.length - 1];
    const dt = (performance.now() - t0) / 1000;
    console.log(
      'VWG ' + label + ' sampling tok/s=' + (K / dt).toFixed(1) + ' wall=' + ((1000 * dt) / K).toFixed(1) + 'ms/token',
    );
  };

  await profileDecode('baseline', BASE);
  await profileDecode('fused', FUSED);

  const benchPrefill = async (label, flags) => {
    rt.setFeatureFlags(flags);
    await dev.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    rt.prefillBatch(ids);
    const first = await rt.argmaxLogits();
    const ms = performance.now() - t0;
    console.log(
      'VWG ' +
        label +
        ' prefill TTFT=' +
        ms.toFixed(0) +
        'ms dispatches=' +
        rt.lastDispatchCount +
        ' first=' +
        first +
        ' features=' +
        JSON.stringify(rt.featureFlags()) +
        ' memory=' +
        JSON.stringify(rt.memoryFootprintBytes()),
    );
  };
  await benchPrefill('baseline', BASE);
  await benchPrefill('fused-block', FUSED);
  await benchPrefill('fused-chunk64', CHUNKED);
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
