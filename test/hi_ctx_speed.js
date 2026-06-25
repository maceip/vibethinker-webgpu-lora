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
window.run = async () => {
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await a.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: {
      maxBufferSize: a.limits.maxBufferSize,
      maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
    },
  });
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json();
  const rt = new QwenWGPU(dev, QWEN25_3B);
  await rt.build('/model');
  console.log('VWG built maxCtx=' + rt.maxCtx);
  const tile = (L) => {
    const o = [];
    while (o.length < L) o.push(ref.ids[o.length % ref.ids.length]);
    return o;
  };
  for (const startCtx of [2000, 4000, 6000, 7800]) {
    rt.prefillBatch(tile(startCtx)); // jump to ctx fast
    let pos = startCtx,
      nxt = await rt.argmaxLogits();
    await dev.queue.onSubmittedWorkDone();
    const N = 40,
      t0 = performance.now();
    for (let i = 0; i < N; i++) {
      const b = await rt.decodeBatch(pos, Math.min(rt.MAXBATCH, N - (i * rt.MAXBATCH < N ? 0 : 0)));
      pos += b.length;
      i += b.length - 1;
    }
    const dt = (performance.now() - t0) / 1000,
      toks = pos - startCtx,
      tps = toks / dt;
    console.log(
      'VWG ctx~' +
        startCtx +
        ': ' +
        tps.toFixed(1) +
        ' tok/s (' +
        toks +
        ' tok in ' +
        dt.toFixed(2) +
        's) ' +
        (tps >= 20 ? '>=20 OK' : 'BELOW 20'),
    );
  }
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 200))),
);
