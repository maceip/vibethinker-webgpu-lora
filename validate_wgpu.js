import { QwenWGPU } from './qwgpu/runtime.js';
import { QWEN25_3B } from './config.js';
window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups'], requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  dev.addEventListener?.('uncapturederror', e => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids;
  const rt = new QwenWGPU(dev, QWEN25_3B);
  const t0 = performance.now();
  await rt.build('/model');
  console.log('VWG built in ' + ((performance.now() - t0) / 1000).toFixed(1) + 's');
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let first = await rt.argmaxLogits();
  console.log('VWG first argmax=' + first + ' (ref ' + ref.argmax + ') ' + (first === ref.argmax ? 'OK' : 'MISMATCH'));
  const got = [first]; let pos = ids.length, nxt = first;
  for (let s = 0; s < 15; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); got.push(nxt); }
  console.log('VWG gen=' + JSON.stringify(got));
  console.log('VWG ref=' + JSON.stringify(ref.gen_ids));
  console.log('VWG match=' + (JSON.stringify(got) === JSON.stringify(ref.gen_ids)));
  // batched GPU-resident decode must produce the SAME sequence
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let bg = [await rt.argmaxLogits()], bpos = ids.length;
  while (bg.length < 16) { const b = await rt.decodeBatch(bpos, Math.min(rt.MAXBATCH, 16 - bg.length)); bpos += b.length; bg.push(...b); }
  bg = bg.slice(0, 16);
  console.log('VWG batch gen=' + JSON.stringify(bg));
  console.log('VWG batch match=' + (JSON.stringify(bg) === JSON.stringify(ref.gen_ids)));
  await dev.queue.onSubmittedWorkDone(); const s0 = performance.now();
  for (let s = 0; s < 30; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); }
  const dt = (performance.now() - s0) / 1000; console.log('VWG SPEED ' + (30 / dt).toFixed(1) + ' tok/s');
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack||'').slice(0,200))));
