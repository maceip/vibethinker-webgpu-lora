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
import { loadLoraAdapterGPU } from '../src/lora_gpu.js';

window.run = async () => {
  const dir =
    window.__ADAPTER_DIR || new URLSearchParams(location.search).get('dir') || 'adapters/highconf-trace-20260623';
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));

  const fetchAdapter = async (base) => {
    const names = ['adapter_config.json', 'adapters.safetensors', 'adapter_model.safetensors'];
    const out = [];
    for (const name of names) {
      const res = await fetch(`/${base}/${name}`);
      if (!res.ok) continue;
      const buf = await res.arrayBuffer();
      out.push({
        name,
        async text() {
          return new TextDecoder().decode(buf);
        },
        async arrayBuffer() {
          return buf;
        },
      });
    }
    if (!out.some((f) => f.name.endsWith('.safetensors'))) throw new Error(`no safetensors under /${base}`);
    if (!out.some((f) => f.name === 'adapter_config.json')) throw new Error(`no adapter_config.json under /${base}`);
    return out;
  };

  console.log('VWG loading adapter dir=/' + dir);
  const rt = new QwenWGPU(dev, QWEN25_3B, { decodeBatchSize: 16 });
  await rt.build('/model');
  console.log('VWG built');

  const ref = await (await fetch('./ref.json')).json();
  const ids = ref.ids;
  const files = await fetchAdapter(dir);
  const lora = await loadLoraAdapterGPU(dev, files, QWEN25_3B);
  const keys = Object.keys(lora.modules);
  const sample = lora.modules[keys[0]];
  console.log(`VWG parsed ${dir}: ${keys.length} modules rank=${sample?.rank} scale=${sample?.scale}`);

  const readLogits = async () => {
    for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    const rb = dev.createBuffer({
      size: QWEN25_3B.vocabSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(rt.s.logits, 0, rb, 0, QWEN25_3B.vocabSize * 4);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice();
    rb.unmap();
    rb.destroy();
    return a;
  };
  const maxAbsDiff = (a, b) => {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  };

  rt.clearLora();
  const base = await readLogits();
  rt.setLora(lora);
  const active = await readLogits();
  rt.clearLora();
  const restored = await readLogits();
  const dActive = maxAbsDiff(active, base);
  const dRestore = maxAbsDiff(restored, base);
  console.log(`VWG logit Δ(active,base)=${dActive.toFixed(3)} Δ(restore,base)=${dRestore.toFixed(6)}`);

  const argmax = (a) => {
    let bi = 0,
      bv = -Infinity;
    for (let i = 0; i < a.length; i++)
      if (a[i] > bv) {
        bv = a[i];
        bi = i;
      }
    return bi;
  };
  rt.setLora(lora);
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  const first = await rt.argmaxLogits();
  console.log('VWG argmax with adapter=' + first + ' (ref.json expects ' + ref.argmax + ' on fine-tuned checkpoint)');

  rt.setLora(lora);
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let pos = ids.length;
  await rt.argmaxLogits();
  const t0 = performance.now();
  let n = 0;
  for (let i = 0; i < 4; i++) {
    const b = await rt.decodeBatch(pos, 8);
    pos += b.length;
    n += b.length;
  }
  console.log('VWG SPEED(LoRA active) ' + (n / ((performance.now() - t0) / 1000)).toFixed(1) + ' tok/s');

  const checks = [
    ['adapter parsed modules', keys.length >= 200],
    ['adapter changes logits', dActive > 0.5],
    ['clearLora restores base bit-exact', dRestore === 0],
    ['logits finite', base.every(Number.isFinite)],
  ];
  let pass = 0;
  for (const [name, ok] of checks) {
    console.log('VWG ' + (ok ? 'PASS' : 'FAIL') + '  ' + name);
    if (ok) pass++;
  }
  console.log(
    'VWG LORA-LOAD-PATH ' +
      (pass === checks.length
        ? 'ALL PASS (' + pass + '/' + checks.length + ')'
        : 'FAILED ' + pass + '/' + checks.length),
  );
  console.log('VWG DONE');
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
