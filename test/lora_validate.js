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

/*
  ,;
 \@@#\:          :/.        .:;;:
_@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
.!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
    .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
        .\@@@*;,\@@@@@@@@+,*@@@@@@+.
    :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
    .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
     ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
            ,@@@@#*@@@@@#+__!.  ,*@@@@@/
             \##+_@@@@@@@@,      ,+@@@_:
                  ;;,,..,:         !;.
*/

import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';
import { loadLoraAdapterGPU } from '../src/lora_gpu.js';

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
  const ids = ref.ids;
  const cfg = QWEN25_3B;
  const rt = new QwenWGPU(dev, cfg);
  await rt.build('/model');
  console.log('VWG built');

  // fetch an adapter dir's files and wrap as File-like for loadLoraAdapterGPU
  const fetchAdapter = async (dir) => {
    const mk = async (path, name) => {
      const buf = await (await fetch(path)).arrayBuffer();
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
    return [
      await mk(`/${dir}/adapter_config.json`, 'adapter_config.json'),
      await mk(`/${dir}/adapters.safetensors`, 'adapters.safetensors'),
    ];
  };

  const rbuf = dev.createBuffer({ size: cfg.vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readCurrentLogits = async () => {
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(rt.s.logits, 0, rbuf, 0, cfg.vocabSize * 4);
    dev.queue.submit([enc.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rbuf.getMappedRange()).slice();
    rbuf.unmap();
    return a;
  };
  const logits = async () => {
    for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    return await readCurrentLogits();
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
  const decodeN = async (pos, n) => {
    let out = [await rt.argmaxLogits()];
    while (out.length < n) {
      const b = await rt.decodeBatch(pos, Math.min(rt.MAXBATCH, n - out.length));
      pos += b.length;
      out.push(...b);
    }
    return out.slice(0, n);
  };

  const selA = await loadLoraAdapterGPU(dev, await fetchAdapter('adapters_sel'), cfg);
  const v1A = await loadLoraAdapterGPU(dev, await fetchAdapter('adapters_v1'), cfg);
  const sampleKey = Object.keys(selA.modules)[0];
  console.log(
    `VWG parsed adapters_sel: ${Object.keys(selA.modules).length} modules, rank=${selA.modules[sampleKey].rank}, scale=${selA.modules[sampleKey].scale}`,
  );
  console.log(`VWG parsed adapters_v1:  ${Object.keys(v1A.modules).length} modules`);

  rt.clearLora();
  const Lbase = await logits();
  rt.setLora(selA);
  const Lsel = await logits();
  rt.clearLora();
  const Lbase2 = await logits();
  rt.setLora(v1A);
  const Lv1 = await logits();
  const dSel = maxAbsDiff(Lsel, Lbase),
    dRevert = maxAbsDiff(Lbase2, Lbase),
    dV1Sel = maxAbsDiff(Lv1, Lsel);
  console.log(
    `VWG logit Δ(sel,base)=${dSel.toFixed(3)}  Δ(revert,base)=${dRevert.toFixed(6)}  Δ(v1,sel)=${dV1Sel.toFixed(3)}`,
  );

  const checks = [
    ['adapters_sel parsed all 252 modules', Object.keys(selA.modules).length === 252],
    ['scale read from MLX config (=20)', selA.modules[sampleKey].scale === 20],
    ['adapter changes logits', dSel > 0.5],
    ['clearLora restores base bit-exact', dRevert === 0],
    ['different checkpoints differ', dV1Sel > 0.5],
  ];
  let pass = 0;
  for (const [n, ok] of checks) {
    console.log('VWG ' + (ok ? 'PASS' : 'FAIL') + '  ' + n);
    if (ok) pass++;
  }
  console.log(
    'VWG LORA-HOTSWAP ' +
      (pass === checks.length
        ? 'ALL PASS (' + pass + '/' + checks.length + ')'
        : 'FAILED ' + pass + '/' + checks.length),
  );

  // Active-LoRA batched prefill must match the current proven sequential adapter path.
  rt.setLora(selA);
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  const seqLogits = await readCurrentLogits();
  const seqGen = await decodeN(ids.length, 6);
  rt.setLora(selA);
  rt.prefillBatch(ids);
  const batLogits = await readCurrentLogits();
  const batGen = await decodeN(ids.length, 6);
  const prefillDelta = maxAbsDiff(seqLogits, batLogits);

  // Multi-token diagnostic trace for layer 0 self_attn q_proj (K=2048, rank=16, N=2048)
  const Tdiag = 6;
  const idsDiag = ids.slice(0, Tdiag);

  rt.clearLora();
  rt.setupDebugCapture(Tdiag, 2048, 16, 2048);
  for (let p = 0; p < Tdiag; p++) rt.token(idsDiag[p], p);
  const dataBase = await rt.readDebugCapture();

  rt.clearLora();
  rt.setLora(selA);
  rt.setupDebugCapture(Tdiag, 2048, 16, 2048);
  for (let p = 0; p < Tdiag; p++) rt.token(idsDiag[p], p);
  const dataSeq = await rt.readDebugCapture();

  rt.clearLora();
  rt.setLora(selA);
  rt.setupDebugCapture(Tdiag, 2048, 16, 2048);
  rt.prefillBatch(idsDiag);
  const dataBat = await rt.readDebugCapture();

  const modQ = selA.modules['layers.0.self_attn.q_proj'];
  const B_arr = modQ.rawB;

  const d_x = maxAbsDiff(dataSeq.xSeq, dataBat.xBat);
  const d_d = maxAbsDiff(dataSeq.dSeq, dataBat.dBat);
  const d_y = maxAbsDiff(dataSeq.ySeq, dataBat.yBat);
  console.log(`VWG Diagnostic LoRA: x_diff=${d_x.toFixed(6)} d_diff=${d_d.toFixed(6)} y_diff=${d_y.toFixed(6)}`);

  // Analyze Token 1, column 1601
  const t_chk = 1;
  const col_chk = 1601;
  const N = 2048;
  const rank = 16;
  const scale = modQ.scale;

  const seq_val = dataSeq.ySeq[t_chk * N + col_chk];
  const bat_val = dataBat.yBat[t_chk * N + col_chk];
  const base_val = dataBase.ySeq[t_chk * N + col_chk]; // ySeq or yBat in base mode should be identical

  // Compute CPU delta for token 1
  let cpu_delta = 0;
  for (let r = 0; r < rank; r++) {
    cpu_delta += dataSeq.dSeq[t_chk * rank + r] * B_arr[r * N + col_chk];
  }
  const cpu_scaled_delta = cpu_delta * scale;

  console.log(`VWG CPU check at Token ${t_chk} col ${col_chk}:`);
  console.log(`  base_val:            ${base_val.toFixed(6)}`);
  console.log(`  cpu_scaled_delta:    ${cpu_scaled_delta.toFixed(6)}`);
  console.log(`  expected (base+del): ${(base_val + cpu_scaled_delta).toFixed(6)}`);
  console.log(`  actual seq_val:      ${seq_val.toFixed(6)}`);
  console.log(`  actual bat_val:      ${bat_val.toFixed(6)}`);

  // Find max difference per token
  for (let t = 0; t < Tdiag; t++) {
    let maxDiff = 0;
    let maxI = 0;
    for (let n = 0; n < N; n++) {
      const idx = t * N + n;
      const diff = Math.abs(dataSeq.ySeq[idx] - dataBat.yBat[idx]);
      if (diff > maxDiff) {
        maxDiff = diff;
        maxI = n;
      }
    }
    const idx = t * N + maxI;
    console.log(
      `VWG Token ${t} max_diff=${maxDiff.toFixed(6)} at col ${maxI} (seq=${dataSeq.ySeq[idx].toFixed(4)}, bat=${dataBat.yBat[idx].toFixed(4)})`,
    );
  }

  console.log(`VWG Diagnostic ySeq samples: ${Array.from(dataSeq.ySeq.slice(4, 12)).map((x) => x.toFixed(4))}`);
  console.log(`VWG Diagnostic yBat samples: ${Array.from(dataBat.yBat.slice(4, 12)).map((x) => x.toFixed(4))}`);

  // Let's also log a few sample values if they differ
  if (d_d > 0.01) {
    console.log(`VWG Diagnostic dSeq samples: ${Array.from(dataSeq.dSeq.slice(0, 8)).map((x) => x.toFixed(4))}`);
    console.log(`VWG Diagnostic dBat samples: ${Array.from(dataBat.dBat.slice(0, 8)).map((x) => x.toFixed(4))}`);
  }

  const prefillOk = finite(batLogits) && seqGen[0] === batGen[0] && JSON.stringify(seqGen) === JSON.stringify(batGen);
  console.log(
    `VWG LoRA prefill seq=${JSON.stringify(seqGen)} batched=${JSON.stringify(batGen)} logitΔ=${prefillDelta.toFixed(3)}`,
  );
  console.log('VWG ' + (prefillOk ? 'PASS' : 'FAIL') + '  LoRA batched prefill matches sequential generation');

  // decode speed with the adapter active (greedy batched path)
  rt.setLora(selA);
  await dev.queue.onSubmittedWorkDone();
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let pos = ids.length;
  await rt.argmaxLogits();
  const t0 = performance.now();
  let got = [];
  for (let i = 0; i < 4; i++) {
    const b = await rt.decodeBatch(pos, 8);
    pos += 8;
    got.push(...b);
  }
  console.log('VWG SPEED(LoRA active) ' + (32 / ((performance.now() - t0) / 1000)).toFixed(1) + ' tok/s');
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
