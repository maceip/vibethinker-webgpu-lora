// Load trained MLX LoRA adapters (from ~/bbverifier, served via symlink) through
// the same loader the UI uses, and prove runtime hot-swap end-to-end: the adapter
// parses, changes the model's logits, reverts bit-exact on clear, and two different
// checkpoints produce different outputs. Also measures decode speed with it active.
import { QwenWGPU } from './qwgpu/runtime.js';
import { QWEN25_3B } from './config.js';
import { loadLoraAdapterGPU } from './lora_gpu.js';

window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups'], requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  dev.addEventListener?.('uncapturederror', e => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids; const cfg = QWEN25_3B;
  const rt = new QwenWGPU(dev, cfg); await rt.build('/model'); console.log('VWG built');

  // fetch an adapter dir's files and wrap as File-like for loadLoraAdapterGPU
  const fetchAdapter = async (dir) => {
    const mk = async (path, name) => { const buf = await (await fetch(path)).arrayBuffer(); return { name, async text() { return new TextDecoder().decode(buf); }, async arrayBuffer() { return buf; } }; };
    return [await mk(`/${dir}/adapter_config.json`, 'adapter_config.json'), await mk(`/${dir}/adapters.safetensors`, 'adapters.safetensors')];
  };

  const rbuf = dev.createBuffer({ size: cfg.vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const logits = async () => {
    for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(rt.s.logits, 0, rbuf, 0, cfg.vocabSize * 4); dev.queue.submit([enc.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ); const a = new Float32Array(rbuf.getMappedRange()).slice(); rbuf.unmap(); return a;
  };
  const maxAbsDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

  const selA = await loadLoraAdapterGPU(dev, await fetchAdapter('adapters_sel'), cfg);
  const v1A = await loadLoraAdapterGPU(dev, await fetchAdapter('adapters_v1'), cfg);
  const sampleKey = Object.keys(selA.modules)[0];
  console.log(`VWG parsed adapters_sel: ${Object.keys(selA.modules).length} modules, rank=${selA.modules[sampleKey].rank}, scale=${selA.modules[sampleKey].scale}`);
  console.log(`VWG parsed adapters_v1:  ${Object.keys(v1A.modules).length} modules`);

  rt.clearLora(); const Lbase = await logits();
  rt.setLora(selA); const Lsel = await logits();
  rt.clearLora(); const Lbase2 = await logits();
  rt.setLora(v1A); const Lv1 = await logits();
  const dSel = maxAbsDiff(Lsel, Lbase), dRevert = maxAbsDiff(Lbase2, Lbase), dV1Sel = maxAbsDiff(Lv1, Lsel);
  console.log(`VWG logit Δ(sel,base)=${dSel.toFixed(3)}  Δ(revert,base)=${dRevert.toFixed(6)}  Δ(v1,sel)=${dV1Sel.toFixed(3)}`);

  const checks = [
    ['adapters_sel parsed all 252 modules', Object.keys(selA.modules).length === 252],
    ['scale read from MLX config (=20)', selA.modules[sampleKey].scale === 20],
    ['adapter changes logits', dSel > 0.5],
    ['clearLora restores base bit-exact', dRevert === 0],
    ['different checkpoints differ', dV1Sel > 0.5],
  ];
  let pass = 0; for (const [n, ok] of checks) { console.log('VWG ' + (ok ? 'PASS' : 'FAIL') + '  ' + n); if (ok) pass++; }
  console.log('VWG LORA-HOTSWAP ' + (pass === checks.length ? 'ALL PASS (' + pass + '/' + checks.length + ')' : 'FAILED ' + pass + '/' + checks.length));

  // decode speed with the adapter active (greedy batched path)
  rt.setLora(selA); await dev.queue.onSubmittedWorkDone();
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let pos = ids.length; await rt.argmaxLogits();
  const t0 = performance.now(); let got = [];
  for (let i = 0; i < 4; i++) { const b = await rt.decodeBatch(pos, 8); pos += 8; got.push(...b); }
  console.log('VWG SPEED(LoRA active) ' + (32 / ((performance.now() - t0) / 1000)).toFixed(1) + ' tok/s');
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))));
