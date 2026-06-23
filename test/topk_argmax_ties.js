import { ARGMAX, TOPK_SELECT } from '../src/qwgpu/kernels.js';

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
  const argmaxUniform = dev.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(argmaxUniform, 0, new Uint32Array([logits.length]));

  const topkIds = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const topkVals = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const topkUniform = dev.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(topkUniform, 0, new Uint32Array([logits.length, 0]));

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
    entries: [logitsBuf, argmaxOut, argmaxUniform].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const topkBg = dev.createBindGroup({
    layout: topkPipe.getBindGroupLayout(0),
    entries: [logitsBuf, topkIds, topkVals, topkUniform].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const readback = dev.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(argmaxPipe);
  pass.setBindGroup(0, argmaxBg);
  pass.dispatchWorkgroups(1);
  pass.setPipeline(topkPipe);
  pass.setBindGroup(0, topkBg);
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

window.run = async () => {
  const dev = await requestDevice();
  dev.addEventListener?.('uncapturederror', e => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));

  const logits = new Float32Array(300).fill(-5);
  logits[1] = 7;
  logits[256] = 7;
  logits[129] = 6;

  const { argmax, top1 } = await runTop1TieCase(dev, logits);
  const pass = argmax === 1 && top1 === 1;
  console.log('VWG tie argmax=' + argmax + ' top1=' + top1 + ' expected=1 ' + (pass ? 'PASS' : 'FAIL'));
  console.log('VWG ' + (pass ? 'TOPK_ARGMAX_TIE PASS' : 'TOPK_ARGMAX_TIE FAIL'));
  console.log('VWG DONE');
};

window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))));
