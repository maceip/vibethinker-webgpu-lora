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

// Differential + scale test for the T>1 prefill path.
// (1) batched prefill == sequential (proven path) across boundary lengths — exercises the
//     flash online-softmax multi-block rescale (ctx>256), GEMM tiles, grid-stride kernels.
// (2) smoke at 4096 / 8192: runs end-to-end, logits finite, argmax valid (exercises the
//     65535-dispatch grid-stride paths and flash attention at long ctx).

/*
 * TECHNIQUE: Prefill correctness harness exercising long-context paths
 *   Tests batched prefill against sequential, long context (4096/8192),
 *   and the grid-stride / flash attention logic in one place.
 */
import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';
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
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json();
  const cfg = QWEN25_3B;
  const rt = new QwenWGPU(dev, cfg);
  await rt.build('/model');
  console.log('VWG built; maxPrefillT=' + rt.maxPrefillT + ' maxCtx=' + rt.maxCtx);
  const rbuf = dev.createBuffer({ size: cfg.vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readLogits = async () => {
    const e = dev.createCommandEncoder();
    e.copyBufferToBuffer(rt.s.logits, 0, rbuf, 0, cfg.vocabSize * 4);
    dev.queue.submit([e.finish()]);
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
  const finite = (a) => {
    for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false;
    return true;
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
  const tile = (L) => {
    const o = [];
    while (o.length < L) o.push(ref.ids[o.length % ref.ids.length]);
    return o;
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

  let allOk = true;
  for (const L of [16, 17, 256, 257, 512, 1024]) {
    const ids = tile(L);
    for (let p = 0; p < L; p++) rt.token(ids[p], p);
    const Lseq = await readLogits();
    const seqGen = await decodeN(L, 6);
    rt.prefillBatch(ids);
    const Lbat = await readLogits();
    const batGen = await decodeN(L, 6);
    const ok = seqGen[0] === batGen[0] && JSON.stringify(seqGen) === JSON.stringify(batGen);
    allOk = allOk && ok;
    console.log(
      `VWG L=${String(L).padStart(4)}  argmax ${seqGen[0]}==${batGen[0]}  gen ${JSON.stringify(seqGen) === JSON.stringify(batGen) ? 'match' : 'DIFFER'}  logitΔ=${maxAbsDiff(Lseq, Lbat).toFixed(3)}  ${ok ? 'PASS' : 'FAIL'}`,
    );
  }
  for (const L of [4096, 8192]) {
    const t0 = performance.now();
    rt.prefillBatch(tile(L));
    const lg = await readLogits();
    const ms = performance.now() - t0;
    const ok = finite(lg) && argmax(lg) < cfg.vocabSize;
    allOk = allOk && ok;
    console.log(
      `VWG smoke L=${L}  prefill=${ms.toFixed(0)}ms  finite=${finite(lg)}  argmax=${argmax(lg)}  ${ok ? 'PASS' : 'FAIL'}`,
    );
  }
  console.log('VWG PREFILL-8192 ' + (allOk ? 'ALL PASS' : 'FAILURES'));
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
