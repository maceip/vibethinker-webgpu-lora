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

import { GEMM4 } from '../src/qwgpu/kernels.js';
import { quantizeInt4Group } from '../src/qwgpu/quantize.js';
window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice();
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const T = 40,
    K = 2048,
    N = 517,
    G = 128; // odd N to test bounds
  // random A, W
  const A = new Float32Array(T * K);
  for (let i = 0; i < A.length; i++) A[i] = (Math.sin(i * 0.3) + Math.cos(i * 0.7)) * 0.5;
  const Wf = new Float32Array(N * K);
  for (let i = 0; i < Wf.length; i++) Wf[i] = Math.sin(i * 0.11) * 0.4;
  const bias = new Float32Array(N);
  for (let i = 0; i < N; i++) bias[i] = Math.cos(i) * 0.1;
  const { packed, scale, groupsPerRow } = quantizeInt4Group(Wf, N, K, G);
  // CPU reference using the SAME dequantized weights
  const i4 = new Int8Array(N * K);
  for (let n = 0; n < N; n++)
    for (let g = 0; g < groupsPerRow; g++) {
      const base = n * K + g * G;
      for (let j = 0; j < G; j++) {
        const word = packed[((base + j) / 8) | 0];
      }
    }
  // dequant from packed: 8 nibbles/word, sign-extended, * group scale
  const deq = new Float32Array(N * K);
  for (let n = 0; n < N; n++)
    for (let c = 0; c < K / 8; c++) {
      const word = packed[n * (K / 8) + c] >>> 0;
      const bk = c * 8;
      const sc = scale[n * groupsPerRow + (bk >> 7)];
      for (let j = 0; j < 8; j++) {
        let nib = (word >> (j * 4)) & 0xf;
        if (nib > 7) nib -= 16;
        deq[n * K + bk + j] = nib * sc;
      }
    }
  const Yref = new Float32Array(T * N);
  for (let t = 0; t < T; t++)
    for (let n = 0; n < N; n++) {
      let s = bias[n];
      for (let k = 0; k < K; k++) s += A[t * K + k] * deq[n * K + k];
      Yref[t * N + n] = s;
    }
  // GPU
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = (arr, u = S) => {
    const b = dev.createBuffer({ size: arr.byteLength, usage: u });
    dev.queue.writeBuffer(b, 0, arr);
    return b;
  };
  const aB = mk(A),
    wB = mk(packed),
    sB = mk(scale),
    bB = mk(bias);
  const yB = dev.createBuffer({ size: T * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const meta = new Uint32Array([K, N, T, groupsPerRow, 1, 0, 0, 0]);
  const uB = mk(meta, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const pipe = dev.createComputePipeline({
    layout: 'auto',
    compute: { module: dev.createShaderModule({ code: GEMM4 }), entryPoint: 'main' },
  });
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [aB, wB, sB, bB, yB, uB].map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe);
  p.setBindGroup(0, bg);
  p.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 16));
  p.end();
  const rb = dev.createBuffer({ size: T * N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyBufferToBuffer(yB, 0, rb, 0, T * N * 4);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const Y = new Float32Array(rb.getMappedRange()).slice();
  rb.unmap();
  let maxAbs = 0,
    maxRel = 0,
    denom = 0;
  for (let i = 0; i < T * N; i++) {
    const e = Math.abs(Y[i] - Yref[i]);
    maxAbs = Math.max(maxAbs, e);
    denom += Yref[i] * Yref[i];
  }
  const rms = Math.sqrt(
    (() => {
      let s = 0;
      for (let i = 0; i < T * N; i++) s += (Y[i] - Yref[i]) ** 2;
      return s;
    })() / denom,
  );
  console.log(
    'VWG GEMM maxAbs=' +
      maxAbs.toExponential(3) +
      ' rms=' +
      rms.toExponential(3) +
      ' (sample Y[0]=' +
      Y[0].toFixed(4) +
      ' ref=' +
      Yref[0].toFixed(4) +
      ')',
  );
  console.log('VWG GEMM ' + (maxAbs < 1e-2 ? 'PASS' : 'FAIL'));
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 200))),
);
