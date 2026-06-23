// GPU buffer, uniform, and bind-group helpers for QwenWGPU. Dynamic uniforms
// keep the original 32-byte ring-buffer behavior; immutable uniforms and bind
// groups are cached only when call-sites explicitly opt in with stable metadata.

export class GPUBufferPool {
  constructor(device, { cacheBindGroups = true } = {}) {
    this.dev = device;
    this.cacheBindGroups = cacheBindGroups;
    this.uniformPool = [];
    this.uniformIdx = 0;
    this.staticUniforms = new Map();
    this.bindGroups = new Map();
    this.sensitiveBindGroups = new Set();
    this.bufferIds = new WeakMap();
    this.pipelineIds = new WeakMap();
    this.nextBufferId = 1;
    this.nextPipelineId = 1;
  }

  buffer(size, usage) {
    return this.dev.createBuffer({ size, usage });
  }

  uploadF32(arr, usage) {
    const b = this.buffer(arr.byteLength, usage);
    this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }

  uploadU32(arr, usage) {
    const b = this.buffer(arr.byteLength, usage);
    this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }

  dynamicUniform(arr, usage) {
    let b = this.uniformPool[this.uniformIdx];
    if (!b) { b = this.buffer(32, usage); this.uniformPool[this.uniformIdx] = b; }
    this.uniformIdx++;
    this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  }

  resetUniforms() {
    this.uniformIdx = 0;
  }

  staticUniform(key, arr, usage) {
    let b = this.staticUniforms.get(key);
    if (!b) {
      b = this.buffer(32, usage);
      this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
      this.staticUniforms.set(key, b);
    }
    return b;
  }

  idForBuffer(buffer) {
    let id = this.bufferIds.get(buffer);
    if (!id) { id = this.nextBufferId++; this.bufferIds.set(buffer, id); }
    return id;
  }

  idForPipeline(pipe) {
    let id = this.pipelineIds.get(pipe);
    if (!id) { id = this.nextPipelineId++; this.pipelineIds.set(pipe, id); }
    return id;
  }

  uncachedBindGroup(pipe, buffers) {
    return this.dev.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
  }

  cachedBindGroup(pipe, buffers, key, { sensitive = false } = {}) {
    if (!this.cacheBindGroups || !key) return this.uncachedBindGroup(pipe, buffers);
    const fullKey = `${this.idForPipeline(pipe)}:${key}:${buffers.map(b => this.idForBuffer(b)).join(',')}`;
    let bg = this.bindGroups.get(fullKey);
    if (!bg) {
      bg = this.uncachedBindGroup(pipe, buffers);
      this.bindGroups.set(fullKey, bg);
      if (sensitive) this.sensitiveBindGroups.add(fullKey);
    }
    return bg;
  }

  clearSensitiveBindGroups() {
    for (const key of this.sensitiveBindGroups) this.bindGroups.delete(key);
    this.sensitiveBindGroups.clear();
  }
}
