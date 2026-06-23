export async function initWebGPUDevice({ log = () => {} } = {}) {
  log('requesting WebGPU device…');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter (use a WebGPU-capable browser)');
  if (!adapter.features.has('subgroups')) throw new Error('GPU lacks the "subgroups" feature (needed by the fast GEMV kernels)');
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  dev.addEventListener?.('uncapturederror', e => console.error('GPUERR', e.error.message));
  log(`WebGPU ready. maxBuffer=${(Number(adapter.limits.maxBufferSize) / 1e9).toFixed(2)}GB`);
  return dev;
}
