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

/*
 * TECHNIQUE: Explicit device feature guard in tests
 *   All serious harnesses require 'subgroups'. This documents the hard
 *   dependency and fails fast on unsupported browsers/GPUs.
 */
async function requestDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  if (!adapter.features.has('subgroups'))
    throw new Error('GPU lacks required "subgroups" feature; no fallback kernels are bundled');
  return await adapter.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
}

function pick(candidates, temperature = 0.7) {
  const best = candidates[0].logit;
  const weights = candidates.map((c) => Math.exp((c.logit - best) / temperature));
  const sum = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum,
    c = 0;
  for (let i = 0; i < candidates.length; i++) {
    c += weights[i];
    if (r <= c) return candidates[i].id;
  }
  return candidates[candidates.length - 1].id;
}

window.run = async () => {
  const dev = await requestDevice();
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json();
  const rt = new QwenWGPU(dev, QWEN25_3B, { samplingTopK: 16, maxSamplingTopK: 32 });
  await rt.build('/model');
  console.log('VWG built samplingTopK=' + rt.samplingTopK);

  rt.prefillBatch(ref.ids);
  const argmax = await rt.argmaxLogits();
  const top1 = await rt.topKLogits(1);
  console.log('VWG top1=' + top1[0].id + ' argmax=' + argmax + ' ' + (top1[0].id === argmax ? 'PASS' : 'FAIL'));

  const top8 = await rt.topKLogits(8);
  const unique = new Set(top8.map((x) => x.id)).size === top8.length;
  const finite = top8.every((x) => Number.isFinite(x.logit) && x.id < QWEN25_3B.vocabSize);
  const sorted = top8.every((x, i, a) => i === 0 || a[i - 1].logit >= x.logit);
  console.log(
    'VWG top8 unique=' +
      unique +
      ' finite=' +
      finite +
      ' sorted=' +
      sorted +
      ' ' +
      (unique && finite && sorted ? 'PASS' : 'FAIL'),
  );

  let pos = ref.ids.length,
    next = pick(top8);
  const got = [];
  const t0 = performance.now();
  for (let i = 0; i < 8; i++) {
    got.push(next);
    rt.token(next, pos++);
    next = pick(await rt.topKLogits(16));
  }
  const dt = (performance.now() - t0) / 1000;
  console.log('VWG sampled ids=' + JSON.stringify(got));
  console.log('VWG sampled top-k readback bytes/token=' + 16 * 8);
  console.log('VWG SAMPLING SPEED ' + (got.length / dt).toFixed(1) + ' tok/s');
  console.log('VWG ' + (top1[0].id === argmax && unique && finite && sorted ? 'SAMPLING PASS' : 'SAMPLING FAIL'));
  console.log('VWG DONE');
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
