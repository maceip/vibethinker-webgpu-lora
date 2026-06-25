/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

import { ARGMAX, SAMPLE_TOPK, TOPK_SELECT } from '../src/qwgpu/kernels.js';

async function requestDevice() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  return await adapter.requestDevice();
}

async function runTop1TieCase(dev, logits) {
  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const logitsBuf = dev.createBuffer({ size: logits.byteLength, usage: storageUsage });
  dev.queue.writeBuffer(logitsBuf, 0, logits);

  const argmaxOut = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const topkIds = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const topkVals = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const argmaxPipe = dev.createComputePipeline({
    layout: 'auto',
    compute: { module: dev.createShaderModule({ code: ARGMAX }), entryPoint: 'main' },
  });
  const topkPipe = dev.createComputePipeline({
    layout: 'auto',
    compute: { module: dev.createShaderModule({ code: TOPK_SELECT }), entryPoint: 'main' },
  });
  const argmaxBg = dev.createBindGroup({
    layout: argmaxPipe.getBindGroupLayout(0),
    entries: [logitsBuf, argmaxOut].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const topkBg = dev.createBindGroup({
    layout: topkPipe.getBindGroupLayout(0),
    entries: [logitsBuf, topkIds, topkVals].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const readback = dev.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(argmaxPipe);
  pass.setBindGroup(0, argmaxBg);
  pass.setImmediates(0, new Uint32Array([logits.length]));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(topkPipe);
  pass.setBindGroup(0, topkBg);
  pass.setImmediates(0, new Uint32Array([logits.length, 0]));
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(argmaxOut, 0, readback, 0, 4);
  enc.copyBufferToBuffer(topkIds, 0, readback, 4, 4);
  dev.queue.submit([enc.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const ids = new Uint32Array(readback.getMappedRange()).slice();
  readback.unmap();
  return { argmax: ids[0], top1: ids[1] };
}

async function runSampleCase(dev) {
  const ids = new Uint32Array([10, 20, 30, 40]);
  const vals = new Float32Array([4, 3, 2, 1]);
  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const idsBuf = dev.createBuffer({ size: ids.byteLength, usage: storageUsage });
  const valsBuf = dev.createBuffer({ size: vals.byteLength, usage: storageUsage });
  dev.queue.writeBuffer(idsBuf, 0, ids);
  dev.queue.writeBuffer(valsBuf, 0, vals);

  const outBuf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const samplePipe = dev.createComputePipeline({
    layout: 'auto',
    compute: { module: dev.createShaderModule({ code: SAMPLE_TOPK }), entryPoint: 'main' },
  });
  const sampleBg = dev.createBindGroup({
    layout: samplePipe.getBindGroupLayout(0),
    entries: [idsBuf, valsBuf, outBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const imm = new Uint32Array(4);
  imm[0] = ids.length;
  const f32 = new Float32Array(imm.buffer);
  f32[2] = 1.0;
  f32[3] = 0.9;

  const readback = dev.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(samplePipe);
  pass.setBindGroup(0, sampleBg);
  pass.setImmediates(0, imm);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readback, 0, 4);
  dev.queue.submit([enc.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const picked = new Uint32Array(readback.getMappedRange())[0];
  readback.unmap();
  return picked;
}

window.run = async () => {
  const dev = await requestDevice();
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));

  const logits = new Float32Array(300).fill(-5);
  logits[1] = 7;
  logits[256] = 7;
  logits[129] = 6;

  const { argmax, top1 } = await runTop1TieCase(dev, logits);
  const picked = await runSampleCase(dev);
  const pass = argmax === 1 && top1 === 1 && picked === 30;
  console.log('VWG tie argmax=' + argmax + ' top1=' + top1 + ' expected=1 ' + (argmax === 1 && top1 === 1 ? 'PASS' : 'FAIL'));
  console.log('VWG sampleTopK picked=' + picked + ' expected=30 ' + (picked === 30 ? 'PASS' : 'FAIL'));
  console.log('VWG ' + (pass ? 'TOPK_ARGMAX_TIE PASS' : 'TOPK_ARGMAX_TIE FAIL'));
  console.log('VWG DONE');
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
