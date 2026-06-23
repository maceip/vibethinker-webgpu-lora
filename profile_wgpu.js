import { QwenWGPU } from './qwgpu/runtime.js';
import { QWEN25_3B } from './config.js';
window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const hasTS = adapter.features.has('timestamp-query');
  console.log('VWG timestamp-query=' + hasTS);
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups', ...(hasTS ? ['timestamp-query'] : [])], requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  dev.addEventListener?.('uncapturederror', e => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids;
  const rt = new QwenWGPU(dev, QWEN25_3B);
  await rt.build('/model');
  console.log('VWG built');
  // prime the KV cache with the prompt
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let nxt = await rt.argmaxLogits(); let pos = ids.length;
  // warm up to a LONG context to profile the app's long-context decode regime
  const WARM = 3200; for (let s = 0; s < WARM; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); }
  console.log('VWG profiling at ctx=' + pos);

  rt.enableProf(700);
  const N = 10; const agg = {}; let total = 0;
  for (let s = 0; s < N; s++) { const sums = await rt.profToken(nxt, pos); pos++; nxt = await rt.argmaxLogits(); for (const k in sums) { agg[k] = (agg[k] || 0) + sums[k]; total += sums[k]; } }
  const rows = Object.entries(agg).map(([k, v]) => [k, v / N]).sort((a, b) => b[1] - a[1]);
  console.log('VWG === per-token GPU breakdown (us), avg of ' + N + ' tokens ===');
  for (const [k, v] of rows) console.log('VWG ' + v.toFixed(1).padStart(8) + ' us  ' + (100 * v * N / total).toFixed(1).padStart(5) + '%  ' + k);
  console.log('VWG TOTAL GPU ' + (total / N).toFixed(1) + ' us/token (sum of pass durations)');

  // wall-clock incl. submit+argmax sync
  await dev.queue.onSubmittedWorkDone(); const t0 = performance.now();
  rt.prof = null; // disable profiling for clean speed
  for (let s = 0; s < 30; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); }
  const dt = (performance.now() - t0) / 1000;
  console.log('VWG WALLCLOCK ' + (1000 * dt / 30).toFixed(1) + ' ms/token = ' + (30 / dt).toFixed(1) + ' tok/s');
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))));
