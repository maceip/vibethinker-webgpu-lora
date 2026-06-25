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

window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 200)));
  const ref = await (await fetch('./ref.json')).json();
  const ids = ref.ids;
  const cfg = QWEN25_3B;
  const rt = new QwenWGPU(dev, cfg, FUSED);
  await rt.build('/model');
  console.log('VWG built; features=' + JSON.stringify(rt.featureFlags()));

  const rbuf = dev.createBuffer({ size: cfg.vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readLogits = async () => {
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(rt.s.logits, 0, rbuf, 0, cfg.vocabSize * 4);
    dev.queue.submit([enc.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rbuf.getMappedRange()).slice();
    rbuf.unmap();
    return a;
  };
  const maxAbsDiff = (a, b) => {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  };
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
  const decodeN = async (pos, n) => {
    let out = [await rt.argmaxLogits()];
    while (out.length < n) {
      const b = await rt.decodeBatch(pos, Math.min(rt.MAXBATCH, n - out.length));
      pos += b.length;
      out.push(...b);
    }
    return out.slice(0, n);
  };
  const runDecode = async (flags, label) => {
    rt.setFeatureFlags(flags);
    for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    const dispatches = rt.lastDispatchCount;
    const logits = await readLogits();
    const gen = await decodeN(ids.length, 8);
    console.log(`VWG ${label} decode dispatches=${dispatches} argmax=${argmax(logits)} gen=${JSON.stringify(gen)}`);
    return { logits, gen };
  };
  const runPrefill = async (flags, label, prompt) => {
    rt.setFeatureFlags(flags);
    rt.prefillBatch(prompt);
    const dispatches = rt.lastDispatchCount;
    const logits = await readLogits();
    const gen = await decodeN(prompt.length, 8);
    console.log(`VWG ${label} prefill dispatches=${dispatches} argmax=${argmax(logits)} gen=${JSON.stringify(gen)}`);
    return { logits, gen };
  };
  const sameGen = (a, b) => JSON.stringify(a.gen) === JSON.stringify(b.gen);

  const oldDec = await runDecode(BASE, 'old');
  const fusedDec = await runDecode(FUSED, 'fused');
  const decodeOk = sameGen(oldDec, fusedDec) && argmax(oldDec.logits) === argmax(fusedDec.logits);
  console.log(
    `VWG ${decodeOk ? 'PASS' : 'FAIL'} old decode vs fused decode logitΔ=${maxAbsDiff(oldDec.logits, fusedDec.logits).toFixed(6)}`,
  );

  const tile = (L) => {
    const out = [];
    while (out.length < L) out.push(ids[out.length % ids.length]);
    return out;
  };
  let prefillOk = true;
  for (const L of [16, 257, 512]) {
    const prompt = tile(L);
    const row = await runPrefill({ ...BASE, prefillAttention: 'row' }, `row L=${L}`, prompt);
    const block = await runPrefill(
      { ...FUSED, prefillAttention: 'block', prefillChunkSize: 0 },
      `block L=${L}`,
      prompt,
    );
    const chunk = await runPrefill(
      { ...FUSED, prefillAttention: 'block', prefillChunkSize: 64 },
      `chunk64 L=${L}`,
      prompt,
    );
    const ok =
      sameGen(row, block) &&
      sameGen(row, chunk) &&
      argmax(row.logits) === argmax(block.logits) &&
      argmax(row.logits) === argmax(chunk.logits);
    prefillOk = prefillOk && ok;
    console.log(
      `VWG ${ok ? 'PASS' : 'FAIL'} prefill row/block/chunk L=${L} blockΔ=${maxAbsDiff(row.logits, block.logits).toFixed(3)} chunkΔ=${maxAbsDiff(row.logits, chunk.logits).toFixed(3)}`,
    );
  }

  try {
    const mk = async (path, name) => {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${path} ${res.status}`);
      const buf = await res.arrayBuffer();
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
    const adapterFiles = [
      await mk('/adapters_sel/adapter_config.json', 'adapter_config.json'),
      await mk('/adapters_sel/adapters.safetensors', 'adapters.safetensors'),
    ];
    const lora = await loadLoraAdapterGPU(dev, adapterFiles, cfg);
    rt.setLora(lora);
    const oldLora = await runDecode(BASE, 'old LoRA');
    const fusedLora = await runDecode(FUSED, 'fused LoRA');
    const loraOk = sameGen(oldLora, fusedLora) && argmax(oldLora.logits) === argmax(fusedLora.logits);
    console.log(
      `VWG ${loraOk ? 'PASS' : 'FAIL'} old LoRA decode vs fused LoRA decode logitΔ=${maxAbsDiff(oldLora.logits, fusedLora.logits).toFixed(6)}`,
    );
    prefillOk = prefillOk && loraOk;
  } catch (e) {
    console.log('VWG SKIP optional LoRA old-vs-fused comparison: ' + e.message);
  } finally {
    rt.clearLora();
  }

  console.log('VWG DEEP-KERNEL-DIFF ' + (decodeOk && prefillOk ? 'ALL PASS' : 'FAILURES'));
  console.log('VWG DONE');
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 400))),
);
