// Custom pure-WebGPU Qwen2.5 decode runtime. int8 weights (per-channel scale),
// f32 norms/biases, GPU-resident KV cache, runtime-swappable LoRA (A/B f32
// buffers consumed by the GEMV kernel). No tf.js → no per-op dispatch overhead.
//
// Correctness is validated against the tf.js forward (which == HuggingFace).
import {
  GEMV,
  GEMV4,
  GEMV4_ADD,
  QKV_GEMV4,
  GATE_UP_SILU_GEMV4,
  LORA_A,
  LORA_A_BATCH,
  LORA_B_ADD,
  LORA_B_ADD_T,
  RMSNORM,
  RMSNORM_F16,
  ROPE,
  ROPE_F16,
  ROPE_QK,
  ROPE_QK_F16,
  ROPE_T_F16,
  RMSNORM_T_F16,
  ATTN_PARTIAL,
  ATTN_PARTIAL_F16,
  ATTN_COMBINE,
  ATTN_COMBINE_F16,
  ADD,
  ADD_F16,
  SILUMUL,
  SILUMUL_F16,
  EMBED,
  EMBED_BUF,
  ARGMAX,
  TOPK_SELECT,
  SAMPLE_TOPK,
  GEMM4,
  GEMM4_ADD_T,
  RMSNORM_T,
  ROPE_T,
  EMBED_T,
  ATTN_PREFILL,
  ATTN_PREFILL_BLOCK,
  DYN_QUANT_X,
  DYN_QUANT_X_T,
  GEMV4_W4A8,
  GEMV4_ADD_W4A8,
  QKV_GEMV4_W4A8,
  GATE_UP_SILU_GEMV4_W4A8,
  GEMM4_W4A8,
  GEMM4_ADD_T_W4A8,
  RMSNORM_QKV_ROPE,
  GEMV4_QKV_ROPE_RMS,
  WRITE_KV_PAGE,
  WRITE_KV_PAGE_BATCH,
  ATTN_PARTIAL_PAGED,
  ATTN_PREFILL_PAGED,
  ATTN_PREFILL_BLOCK_PAGED,
} from './kernels.js';
import { createQwenSchema } from './model_schema.js';
import { createDispatchPlan } from './dispatch_plan.js';
import { streamSafetensors } from './safetensors_loader.js';
import { ModelUploader } from './model_uploader.js';
import { GPUBufferPool } from './buffer_pool.js';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export class QwenWGPU {
  // opts: { maxCtx, maxPrefillT, decodeBatchSize, samplingTopK } — context
  // window + batched-prefill cap (default 8192 each; KV cache grows linearly).
  constructor(device, cfg, opts = {}) {
    this.dev = device;
    this.cfg = cfg;
    this.lora = null;
    this.bufs = {};
    this.opts = opts;
    this.features = this._normalizeFeatures(opts);
    this.pool = new GPUBufferPool(device, { cacheBindGroups: opts.cacheBindGroups !== false });
    this._loraEpoch = 0;
    this.lastDispatchCount = 0;
    this.packedBytes = 0;
    this._argmaxReadBusy = false;
    this._topKReadBusy = false;
  }

  _normalizeFeatures(opts = {}) {
    const prefillAttention = opts.prefillAttention || 'block';
    if (!['row', 'block'].includes(prefillAttention))
      throw new Error(`unsupported prefillAttention ${prefillAttention}`);
    return {
      fuseQKV: opts.fuseQKV !== false,
      fuseRoPE: opts.fuseRoPE !== false,
      fuseMLP: opts.fuseMLP !== false,
      fuseResidual: opts.fuseResidual !== false,
      prefillAttention,
      prefillChunkSize: Math.max(0, opts.prefillChunkSize || 0),
      actQuant: !!opts.actQuant,
      fuseRMSNormQKVRoPE: opts.fuseRMSNormQKVRoPE !== false,
      pagedAttention: !!opts.pagedAttention,
    };
  }

  setFeatureFlags(flags = {}) {
    this.features = this._normalizeFeatures({ ...this.features, ...flags });
    this.pool.clearSensitiveBindGroups();
  }

  featureFlags() {
    return { ...this.features };
  }

  // Phase 3 (f16): when shader-f16 is available we can switch hot kernels to f16
  // storage/compute for bandwidth wins. Stub for now; real kernel variants + selection
  // will be added. Evaluation: compare f16 vs f32 logits within tolerance + bench speedup.
  hasF16Compute() {
    return !!this.hasF16;
  }

  setUseF16(v) {
    this._useF16 = !!v && this.hasF16Compute();
    // When true, future dispatches can select f16 WGSL variants (to be added).
    // For now just observable.
  }
  usingF16() { return !!this._useF16; }

  // Phase 4: allow caller / autotuner to override workgroup size after build if desired.
  // Note: affects *future* pipes / re-pipes; existing pipes keep their specialization.
  setWorkgroupSize(wg) {
    if (wg && wg > 0) this.workgroupSize = wg | 0;
  }

  // Basic load-time / on-demand workgroup autotuner (Phase 4).
  // Tries a few WG sizes for simple override-supporting kernels (add / rms for now).
  // Uses wall time + onSubmittedWorkDone for broad compatibility.
  // Returns a map of best sizes; optionally hot-swaps the pipe for 'add'.
  async autotuneWorkgroups(opts = {}) {
    const iters = opts.iters || 6;
    const cands = opts.candidates || [32, 64, 128, 256];
    const results = {};

    const timeKernel = async (pipe, n, label) => {
      // tiny synthetic work: n elements
      const a = this._buf(n * 4);
      const y = this._buf(n * 4);

      const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      for (let i = 0; i < iters; i++) {
        const enc = this.dev.createCommandEncoder();
        const bg = this._bg(pipe, [a, y]);
        const imm = new Uint32Array([n]);
        this._dispatch(enc, pipe, bg, Math.ceil(n / (pipe.__wg || 256)), 1, label + ':bench', imm);
        this.dev.queue.submit([enc.finish()]);
        if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
      }
      const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      // cleanup
      a.destroy?.(); y.destroy?.();
      return ms / iters;
    };

    // Autotune a few hot, override-friendly kernels (add, rms, silu).
    const kernels = [
      { name: 'add', src: ADD, n: 8192 },
      { name: 'rms', src: RMSNORM, n: 4096 },   // K=4096 typical
      { name: 'silu', src: SILUMUL, n: 8192 },
    ];

    for (const k of kernels) {
      try {
        let best = { wg: 256, ms: Infinity };
        for (const wg of cands) {
          const p = this._pipe(k.src, `${k.name}:autotune:${wg}`, { WG: wg });
          p.__wg = wg;
          const ms = await timeKernel(p, k.n, `${k.name}${wg}`);
          results[`${k.name}:${wg}`] = ms;
          if (ms < best.ms) best = { wg, ms };
        }
        results[`best${k.name[0].toUpperCase()}${k.name.slice(1)}`] = best;
        if (opts.apply && this.pipes[k.name]) {
          this.pipes[k.name] = this._pipe(k.src, k.name, { WG: best.wg });
          this.pipes[k.name].__wg = best.wg;
        }
      } catch (e) {
        results[`${k.name}Error`] = String(e);
      }
    }

    console.log('[autotune] WG microbench results (ms/iter):', results);
    return results;
  }

  _buf(size, usage = STORAGE) {
    return this.pool.buffer(size, usage);
  }
  _f32(arr, usage = STORAGE) {
    return this.pool.uploadF32(arr, usage);
  }
  _u32(arr) {
    return this.pool.uploadU32(arr, STORAGE);
  }
  _uni(arr) {
    return this.pool.dynamicUniform(arr, UNIFORM);
  }
  _staticUni(key, arr) {
    return this.pool.staticUniform(key, arr, UNIFORM);
  }
  _resetUni() {
    this.pool.resetUniforms();
    this.lastDispatchCount = 0;
  }

  _pipe(code, name, overrides = null) {
    const processedCode = typeof code === 'string' ? code.replaceAll('WG_SIZE', this.workgroupSize || 64) : code;
    const m = this.dev.createShaderModule({
      label: name || undefined,
      code: processedCode,
    });
    const comp = { module: m, entryPoint: 'main' };
    if (overrides && typeof overrides === 'object') comp.constants = overrides;
    return this.dev.createComputePipeline({
      label: name ? `${name}-pipeline` : undefined,
      layout: 'auto',
      compute: comp,
    });
  }

  // `source` is a base URL string OR a reader { range, text } (e.g. hfReader/fileReader).
  async build(source, onProgress = () => {}) {
    const dev = this.dev,
      c = this.cfg;
    this.CHUNK = 128;
    this._initRuntimeOptions();
    this.maxCtx = this.opts.maxCtx || 8192; // context window (KV cache length)
    this.maxPrefillT = Math.min(this.opts.maxPrefillT || 8192, this.maxCtx); // batched-prefill cap (<= ctx)

    const isAppleSilicon = this.dev.limits.minStorageBufferOffsetAlignment === 4;
    const isIntelArc = this.dev.limits.minStorageBufferOffsetAlignment === 256;
    this.workgroupSize = isAppleSilicon || isIntelArc ? 32 : 64;
    // Phase 4: cheap static heuristic. Real autotune (bench a few micro dispatches) can override later.
    onProgress && onProgress(`workgroup size chosen: ${this.workgroupSize} (apple/intel bias toward 32)`, 0);

    let hasDP4a = false;
    if (
      typeof navigator !== 'undefined' &&
      navigator.gpu?.wgslLanguageFeatures?.has?.('packed_4x8_integer_dot_product')
    ) {
      dev.pushErrorScope('validation');
      try {
        dev.createShaderModule({
          code: `enable packed_4x8_integer_dot_product; @compute @workgroup_size(1) fn main() {}`,
        });
        const error = await dev.popErrorScope();
        if (!error) {
          hasDP4a = true;
        }
      } catch (e) {
        await dev.popErrorScope();
      }
    }
    this.hasDP4a = hasDP4a;

    const hasF16 = this.dev.features.has('shader-f16');
    this.hasF16 = hasF16;

    this.pam = new PagedAttentionManager(this.maxCtx);

    this.pipes = {
      gemv: this._pipe(GEMV, 'gemv'),
      loraA: this._pipe(LORA_A, 'loraA'),
      loraABatch: this._pipe(LORA_A_BATCH, 'loraABatch'),
      loraBAdd: this._pipe(LORA_B_ADD, 'loraBAdd'),
      loraBAddT: this._pipe(LORA_B_ADD_T, 'loraBAddT'),
      rms: this._pipe(RMSNORM, 'rms', { WG: this.workgroupSize || 256 }),
      rmsF16: hasF16 ? this._pipe(RMSNORM_F16, 'rmsF16', { WG: this.workgroupSize || 256 }) : null,
      rope: this._pipe(ROPE, 'rope'),
      ropeF16: hasF16 ? this._pipe(ROPE_F16, 'ropeF16') : null,
      ropeQK: this._pipe(ROPE_QK, 'ropeQK'),
      ropeQKF16: hasF16 ? this._pipe(ROPE_QK_F16, 'ropeQKF16') : null,
      ropeT: this._pipe(ROPE_T, 'ropeT'),
      ropeTF16: hasF16 ? this._pipe(ROPE_T_F16, 'ropeTF16') : null,
      attnP: this._pipe(ATTN_PARTIAL, 'attnP', { WG: 128 }),
      attnPF16: hasF16 ? this._pipe(ATTN_PARTIAL_F16, 'attnPF16', { WG: 128 }) : null,
      attnC: this._pipe(ATTN_COMBINE, 'attnC', { WG: 128 }),
      attnCF16: hasF16 ? this._pipe(ATTN_COMBINE_F16, 'attnCF16', { WG: 128 }) : null,
      add: this._pipe(ADD, 'add', { WG: this.workgroupSize || 256 }),
      silu: this._pipe(SILUMUL, 'silu', { WG: this.workgroupSize || 256 }),
      addF16: hasF16 ? this._pipe(ADD_F16, 'addF16', { WG: this.workgroupSize || 256 }) : null,
      siluF16: hasF16 ? this._pipe(SILUMUL_F16, 'siluF16', { WG: this.workgroupSize || 256 }) : null,
      embed: this._pipe(EMBED, 'embed'),
      embedBuf: this._pipe(EMBED_BUF, 'embedBuf'),
      argmax: this._pipe(ARGMAX, 'argmax'),
      gemv4: this._pipe(GEMV4, 'gemv4'),
      gemv4Add: this._pipe(GEMV4_ADD, 'gemv4Add'),
      qkvGemv4: this._pipe(QKV_GEMV4, 'qkvGemv4'),
      gateUpSiluGemv4: this._pipe(GATE_UP_SILU_GEMV4, 'gateUpSiluGemv4'),
      topkSelect: this._pipe(TOPK_SELECT, 'topkSelect'),
      sampleTopK: this._pipe(SAMPLE_TOPK, 'sampleTopK'),
      gemm4: this._pipe(GEMM4, 'gemm4'),
      gemm4AddT: this._pipe(GEMM4_ADD_T, 'gemm4AddT'),
      rmsT: this._pipe(RMSNORM_T, 'rmsT', { WG: this.workgroupSize || 256 }),
      rmsTF16: hasF16 ? this._pipe(RMSNORM_T_F16, 'rmsTF16', { WG: this.workgroupSize || 256 }) : null,
      embedT: this._pipe(EMBED_T, 'embedT'),
      attnPrefill: this._pipe(ATTN_PREFILL, 'attnPrefill'),
      attnPrefillBlock: this._pipe(ATTN_PREFILL_BLOCK, 'attnPrefillBlock'),
      dynQuant: this._pipe(DYN_QUANT_X, 'dynQuant'),
      dynQuantT: this._pipe(DYN_QUANT_X_T, 'dynQuantT'),
      gemv4W4A8: this._pipe(GEMV4_W4A8(hasDP4a, this.workgroupSize), 'gemv4W4A8'),
      gemv4AddW4A8: this._pipe(GEMV4_ADD_W4A8(hasDP4a, this.workgroupSize), 'gemv4AddW4A8'),
      qkvGemv4W4A8: this._pipe(QKV_GEMV4_W4A8(hasDP4a, this.workgroupSize), 'qkvGemv4W4A8'),
      gateUpSiluGemv4W4A8: this._pipe(GATE_UP_SILU_GEMV4_W4A8(hasDP4a, this.workgroupSize), 'gateUpSiluGemv4W4A8'),
      gemm4W4A8: this._pipe(GEMM4_W4A8(hasDP4a), 'gemm4W4A8'),
      gemm4AddTW4A8: this._pipe(GEMM4_ADD_T_W4A8(hasDP4a), 'gemm4AddTW4A8'),
      rmsNormQkvRope: this._pipe(GEMV4_QKV_ROPE_RMS, 'rmsNormQkvRope'),
      writeKvPage: this._pipe(WRITE_KV_PAGE, 'writeKvPage'),
      writeKvPageBatch: this._pipe(WRITE_KV_PAGE_BATCH, 'writeKvPageBatch'),
      attnPartialPaged: this._pipe(ATTN_PARTIAL_PAGED, 'attnPartialPaged'),
      attnPrefillPaged: this._pipe(ATTN_PREFILL_PAGED, 'attnPrefillPaged'),
      attnPrefillBlockPaged: this._pipe(ATTN_PREFILL_BLOCK_PAGED, 'attnPrefillBlockPaged'),
    };

    if (hasF16) {
      this.setUseF16(true);
      onProgress('f16 compute enabled (add/silu/rms/rope/attn-partial/combine paths)', 0);
    }

    onProgress('streaming + quantizing weights', 0);
    this.schema = createQwenSchema(c);
    this.plan = createDispatchPlan(this.schema);
    this.q = {};
    this.q4 = {};
    this.qkv = [];
    this.gateUp = [];
    const uploader = new ModelUploader({
      schema: this.schema,
      q: this.q,
      q4: this.q4,
      bufs: this.bufs,
      uploadF32: (arr) => this._f32(arr),
      uploadU32: (arr) => this._u32(arr),
    });
    if (source === 'mock') {
      for (const name of this.schema.expectedNames) {
        const desc = this.schema.tensors.find((t) => t.name === name);
        const shape = desc.shape;
        const numel = shape.reduce((a, b) => a * b, 1);
        const type = desc.quant === 'int8' ? 'I8' : 'F32';
        uploader.visit({ name, shape, data: new Uint8Array(numel * (type === 'I8' ? 1 : 4)), type });
      }
    } else {
      await streamSafetensors(source, {
        names: this.schema.expectedNames,
        onProgress,
        onTensor: async (tensor) => {
          uploader.visit(tensor);
          if (uploader.seen.size % 48 === 0) await new Promise((r) => setTimeout(r, 0));
        },
      });
    }
    uploader.finalize();
    await this._buildPackedProjectionBuffers();
    // Context window (this.maxCtx) set above from opts; RoPE tables + KV cache sized to it.
    this._buildRope(this.maxCtx);
    // KV cache (f32) per layer
    (this.kc = []), (this.vc = []);
    const kvSize = c.numKVHeads * this.maxCtx * c.headDim * 4;
    for (let i = 0; i < c.numLayers; i++) {
      this.kc.push(this._buf(kvSize));
      this.vc.push(this._buf(kvSize));
    }
    // scratch buffers (reused each token)
    const H = c.hiddenSize,
      qd = c.numHeads * c.headDim,
      kvd = c.numKVHeads * c.headDim,
      I = c.intermediateSize;
    const NSPLITMAX = Math.ceil(this.maxCtx / this.CHUNK);
    this.s = {
      hidden: this._buf(H * 4),
      normed: this._buf(H * 4),
      q: this._buf(qd * 4),
      k: this._buf(kvd * 4),
      v: this._buf(kvd * 4),
      attn: this._buf(qd * 4),
      tmp: this._buf(Math.max(qd, I) * 4),
      tmp2: this._buf(I * 4),
      logits: this._buf(c.vocabSize * 4),
      dummy: this._buf(64),
      loraD: this._buf(256 * 4),
      loraD2: this._buf(256 * 4),
      amax: this._buf(4),
      pm: this._buf(c.numHeads * NSPLITMAX * 4),
      pz: this._buf(c.numHeads * NSPLITMAX * 4),
      po: this._buf(c.numHeads * NSPLITMAX * c.headDim * 4),
      idsBuf: this._buf(this.decodeBatchCapacity * 4),
      sampleIds: this._buf(this.maxSamplingTopK * 4),
      sampleVals: this._buf(this.maxSamplingTopK * 4),
      sampled: this._buf(4),  // single u32 chosen by GPU sampler (Phase 5)
      x_q: this._buf(Math.max(qd, I) * 4),
      scale_x: this._buf(256 * 4),
      blockTableBuf: this._buf(this.pam.maxBlocksPerSeq * 4, STORAGE | GPUBufferUsage.COPY_DST),
    };
    this.idsRead = this._buf(this.decodeBatchCapacity * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.argmaxRead = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampleIdsRead = this._buf(this.maxSamplingTopK * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampleValsRead = this._buf(this.maxSamplingTopK * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampledRead = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    // prefill scratch is allocated lazily (sized to the actual prompt) — see _ensurePrefillScratch.
    this.sT = null;
    this.sTcap = 0;
    this._initStaticUniforms();
    if (this.decodeBatchMode === 'auto') {
      onProgress('autotuning decode batch', 0.98);
      await this.autotuneDecodeBatch();
    }
    onProgress('ready', 1);
    return this;
  }

  _initRuntimeOptions() {
    const opts = this.opts;
    this.decodeBatchMode = opts.decodeBatchSize === 'auto' ? 'auto' : 'fixed';
    this.decodeBatchCandidates = (opts.decodeBatchCandidates || [1, 2, 4, 8, 16, 32])
      .map((x) => Math.max(1, Math.floor(Number(x) || 0)))
      .filter(Boolean);
    const requested =
      opts.decodeBatchSize === undefined || opts.decodeBatchSize === 'auto'
        ? 16
        : Math.max(1, Math.floor(Number(opts.decodeBatchSize)));
    this.maxDecodeBatchSize = Math.max(
      1,
      Math.floor(Number(opts.maxDecodeBatchSize || Math.max(requested, ...this.decodeBatchCandidates, 16))),
    );
    this.decodeBatchCapacity = Math.min(this.maxDecodeBatchSize, Math.max(requested, ...this.decodeBatchCandidates));
    this.MAXBATCH = Math.min(requested, this.decodeBatchCapacity);
    this.decodeBatchWarmupTokens = Math.max(0, Math.floor(Number(opts.decodeBatchWarmupTokens ?? 4)));
    this.decodeBatchWarmupSize = Math.min(
      this.decodeBatchCapacity,
      Math.max(1, Math.floor(Number(opts.decodeBatchWarmupSize ?? 4))),
    );
    this.decodeBatchMaxLatencyMs = Number(opts.decodeBatchMaxLatencyMs ?? 250);
    this.samplingTopK = Math.max(1, Math.floor(Number(opts.samplingTopK ?? 40)));
    this.maxSamplingTopK = Math.max(this.samplingTopK, Math.floor(Number(opts.maxSamplingTopK ?? 64)));
    this.decodeBatchTuning = {
      selected: this.MAXBATCH,
      candidates: [],
      reason: this.decodeBatchMode === 'auto' ? 'pending' : 'fixed',
    };
  }

  _buildRope(maxSeq) {
    const { headDim, ropeTheta } = this.cfg;
    const half = headDim / 2;
    const cos = new Float32Array(maxSeq * headDim),
      sin = new Float32Array(maxSeq * headDim);
    for (let p = 0; p < maxSeq; p++)
      for (let i = 0; i < half; i++) {
        const a = p / Math.pow(ropeTheta, (2 * i) / headDim);
        const cc = Math.cos(a),
          ss = Math.sin(a);
        cos[p * headDim + i] = cc;
        cos[p * headDim + half + i] = cc;
        sin[p * headDim + i] = ss;
        sin[p * headDim + half + i] = ss;
      }
    this.ropeCos = this._f32(cos);
    this.ropeSin = this._f32(sin);
    this._ropeRow = headDim * 4;
  }

  _initStaticUniforms() {
    const c = this.cfg;
    const rms = new ArrayBuffer(8);
    const rmsDv = new DataView(rms);
    rmsDv.setFloat32(0, c.hiddenSize, true);
    rmsDv.setFloat32(4, c.rmsNormEps, true);
    this.u = {
      rmsHidden: this._staticUni(`rms:${c.hiddenSize}:${c.rmsNormEps}`, new Uint8Array(rms)),
      addHidden: this._staticUni(`u32:${c.hiddenSize}`, new Uint32Array([c.hiddenSize])),
      siluIntermediate: this._staticUni(`u32:${c.intermediateSize}`, new Uint32Array([c.intermediateSize])),
      embedBuf: this._staticUni(`embedBuf:${c.hiddenSize}`, new Uint32Array([c.hiddenSize])),
      argmax: this._staticUni(`argmax:${c.vocabSize}`, new Uint32Array([c.vocabSize])),
    };
  }

  async _buildPackedProjectionBuffers() {
    const enc = this.dev.createCommandEncoder();
    const copy = (src, dst, dstOffset, bytes) => enc.copyBufferToBuffer(src, 0, dst, dstOffset, bytes);
    this.packedBytes = 0;
    for (const L of this.plan.layers) {
      const q = this.q4[L.q.weight],
        k = this.q4[L.k.weight],
        v = this.q4[L.v.weight];
      if (q.K !== k.K || q.K !== v.K || q.gpr !== k.gpr || q.gpr !== v.gpr)
        throw new Error(`layer ${L.index} qkv packing requires matching K/gpr`);
      const totalN = q.N + k.N + v.N;
      const wBytes = totalN * (q.K / 8) * 4;
      const scaleBytes = totalN * q.gpr * 4;
      const biasBytes = totalN * 4;
      const w = this._buf(wBytes);
      const scale = this._buf(scaleBytes);
      const bias = this._buf(biasBytes);
      enc.clearBuffer(bias);
      let wOff = 0,
        sOff = 0,
        bOff = 0;
      for (const part of [L.q, L.k, L.v]) {
        const qq = this.q4[part.weight];
        const rowsW = qq.N * (qq.K / 8) * 4;
        const rowsS = qq.N * qq.gpr * 4;
        copy(qq.w, w, wOff, rowsW);
        wOff += rowsW;
        copy(qq.scale, scale, sOff, rowsS);
        sOff += rowsS;
        if (part.bias) copy(this.bufs[part.bias], bias, bOff, qq.N * 4);
        bOff += qq.N * 4;
      }
      this.qkv[L.index] = { w, scale, bias, K: q.K, qN: q.N, kN: k.N, vN: v.N, totalN, gpr: q.gpr };
      this.packedBytes += wBytes + scaleBytes + biasBytes;

      const gate = this.q4[L.gate.weight],
        up = this.q4[L.up.weight];
      if (gate.K !== up.K || gate.N !== up.N || gate.gpr !== up.gpr)
        throw new Error(`layer ${L.index} gate/up packing requires matching shape`);
      const guWBytes = (gate.N + up.N) * (gate.K / 8) * 4;
      const guScaleBytes = (gate.N + up.N) * gate.gpr * 4;
      const guW = this._buf(guWBytes);
      const guScale = this._buf(guScaleBytes);
      copy(gate.w, guW, 0, gate.N * (gate.K / 8) * 4);
      copy(up.w, guW, gate.N * (gate.K / 8) * 4, up.N * (up.K / 8) * 4);
      copy(gate.scale, guScale, 0, gate.N * gate.gpr * 4);
      copy(up.scale, guScale, gate.N * gate.gpr * 4, up.N * up.gpr * 4);
      this.gateUp[L.index] = { w: guW, scale: guScale, K: gate.K, N: gate.N, gpr: gate.gpr };
      this.packedBytes += guWBytes + guScaleBytes;
    }
    this.dev.queue.submit([enc.finish()]);
    await this.dev.queue.onSubmittedWorkDone();
  }

  memoryFootprintBytes() {
    const c = this.cfg;
    const kvBytes = c.numLayers * 2 * c.numKVHeads * this.maxCtx * c.headDim * 4;
    const decodeScratchBytes =
      c.hiddenSize * 2 * 4 +
      (c.numHeads * c.headDim + 2 * c.numKVHeads * c.headDim + c.numHeads * c.headDim) * 4 +
      (Math.max(c.numHeads * c.headDim, c.intermediateSize) + c.intermediateSize + c.vocabSize) * 4;
    const prefillScratchBytes = this.sTcap
      ? this.sTcap *
        (3 * c.hiddenSize +
          c.numHeads * c.headDim +
          2 * c.numKVHeads * c.headDim +
          c.numHeads * c.headDim +
          2 * c.intermediateSize) *
        4
      : 0;
    return { kvBytes, decodeScratchBytes, prefillScratchBytes, packedBytes: this.packedBytes };
  }

  _gemvMeta(q, biasBuf, mod) {
    const gx = Math.min(q.N, 65535);
    const bytes = new Uint8Array(32);
    const dv = new DataView(bytes.buffer);
    dv.setUint32(0, q.K, true);
    dv.setUint32(4, q.N, true);
    dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true);
    dv.setUint32(16, mod ? 1 : 0, true);
    dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true);
    return {
      gx,
      gy: Math.ceil(q.N / gx),
      bytes,
    };
  }

  _gemv4Meta(q, biasBuf, mod) {
    const gx = Math.min(q.N, 65535);
    const bytes = new Uint8Array(32);
    const dv = new DataView(bytes.buffer);
    dv.setUint32(0, q.K, true);
    dv.setUint32(4, q.N, true);
    dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true);
    dv.setUint32(16, mod ? 1 : 0, true);
    dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true);
    dv.setUint32(28, q.gpr, true);
    return {
      gx,
      gy: Math.ceil(q.N / gx),
      bytes,
    };
  }

  setLora(adapter) {
    this.lora = adapter;
    this._loraEpoch++;
    this.pool.clearSensitiveBindGroups();
  } // {modules: {key:{A,B,rank,scale}}}  A:[K][rank], B:[rank][N] f32 GPUBuffers
  clearLora() {
    this.lora = null;
    this._loraEpoch++;
    this.pool.clearSensitiveBindGroups();
  }

  _bg(pipe, buffers) {
    return this.pool.uncachedBindGroup(pipe, buffers);
  }
  _bgCached(pipe, buffers, key, opts) {
    return this.pool.cachedBindGroup(pipe, buffers, key, opts);
  }
  _dispatch(enc, pipe, bg, gx, gy = 1, cat, imm = null) {
    this.lastDispatchCount++;
    let ts;
    if (this.prof && this.prof.idx < this.prof.cap) {
      const i = this.prof.idx++;
      this.prof.cats.push(cat || 'misc');
      ts = { querySet: this.prof.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
    }
    const p = enc.beginComputePass(ts ? { timestampWrites: ts } : undefined);
    p.setPipeline(pipe);
    if (bg) p.setBindGroup(0, bg);
    if (imm) {
      if (Array.isArray(imm)) {
        let off = 0;
        for (const part of imm) {
          p.setImmediates(off, part);
          off += part.byteLength || (part.length * (part.BYTES_PER_ELEMENT || 4));
        }
      } else {
        p.setImmediates(0, imm);
      }
    }
    p.dispatchWorkgroups(gx, gy);
    p.end();
  }
  enableProf(cap = 700) {
    this.prof = {
      qs: this.dev.createQuerySet({ type: 'timestamp', count: cap * 2 }),
      cap,
      idx: 0,
      cats: [],
      resolve: this._buf(cap * 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
      read: this._buf(cap * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    };
  }
  async profToken(id, pos) {
    this._resetUni();
    this.prof.idx = 0;
    this.prof.cats = [];
    const enc = this.dev.createCommandEncoder();
    this.embedRow(enc, id);
    this.step(enc, id, pos);
    const n = this.prof.idx;
    enc.resolveQuerySet(this.prof.qs, 0, n * 2, this.prof.resolve, 0);
    enc.copyBufferToBuffer(this.prof.resolve, 0, this.prof.read, 0, n * 16);
    this.dev.queue.submit([enc.finish()]);
    await this.prof.read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.prof.read.getMappedRange());
    const sums = {};
    for (let i = 0; i < n; i++) {
      const us = Number(t[2 * i + 1] - t[2 * i]) / 1000;
      const c = this.prof.cats[i];
      sums[c] = (sums[c] || 0) + us;
    }
    this.prof.read.unmap();
    return sums;
  }

  poolStats() {
    return this.pool.stats();
  }
  resetPoolStats() {
    this.pool.resetStats();
  }

  estimateKvCacheBytes() {
    const c = this.cfg;
    return c.numLayers * 2 * c.numKVHeads * this.maxCtx * c.headDim * 4;
  }

  estimatePrefillScratchBytes(T, loraRank = this._activeMaxLoraRank()) {
    const c = this.cfg,
      H = c.hiddenSize,
      qd = c.numHeads * c.headDim,
      kvd = c.numKVHeads * c.headDim,
      I = c.intermediateSize;
    return (
      T * H * 4 * 2 +
      T * qd * 4 * 2 +
      T * kvd * 4 * 2 +
      T * I * 4 * 2 +
      T * 4 +
      Math.max(1, T * Math.max(1, loraRank)) * 4
    );
  }

  greedyBatchSizeFor({ emitted = 0, remaining = Infinity, pos = 0 } = {}) {
    const interactive = emitted < this.decodeBatchWarmupTokens ? this.decodeBatchWarmupSize : this.MAXBATCH;
    return Math.max(0, Math.min(interactive, remaining, this.maxCtx - pos, this.decodeBatchCapacity));
  }

  async _resetAutotuneDecodeState(tokens, seedTokenId = 0) {
    const c = this.cfg,
      S = this.s,
      H = c.hiddenSize,
      hd = c.headDim,
      qd = c.numHeads * hd,
      kvd = c.numKVHeads * hd,
      I = c.intermediateSize;
    const nsplitMax = Math.ceil(this.maxCtx / this.CHUNK);
    const touchedTokens = Math.min(Math.max(0, Math.floor(tokens)), this.maxCtx);
    const enc = this.dev.createCommandEncoder();
    const clear = (buf, bytes) => {
      if (bytes > 0) enc.clearBuffer(buf, 0, bytes);
    };

    clear(S.hidden, H * 4);
    clear(S.normed, H * 4);
    clear(S.q, qd * 4);
    clear(S.k, kvd * 4);
    clear(S.v, kvd * 4);
    clear(S.attn, qd * 4);
    clear(S.tmp, Math.max(qd, I) * 4);
    clear(S.tmp2, I * 4);
    clear(S.logits, c.vocabSize * 4);
    clear(S.loraD, 256 * 4);
    clear(S.idsBuf, this.decodeBatchCapacity * 4);
    clear(S.pm, c.numHeads * nsplitMax * 4);
    clear(S.pz, c.numHeads * nsplitMax * 4);
    clear(S.po, c.numHeads * nsplitMax * hd * 4);
    const kvBytes = touchedTokens * kvd * 4;
    for (let i = 0; i < c.numLayers; i++) {
      clear(this.kc[i], kvBytes);
      clear(this.vc[i], kvBytes);
    }

    this.dev.queue.submit([enc.finish()]);
    this.dev.queue.writeBuffer(S.amax, 0, new Uint32Array([seedTokenId]));
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
  }

  async autotuneDecodeBatch() {
    const candidates = [...new Set(this.decodeBatchCandidates)]
      .filter((k) => k >= 1 && k <= this.decodeBatchCapacity && k <= this.maxCtx)
      .sort((a, b) => a - b);
    const rows = [];
    const resetTokens = candidates.length ? Math.max(...candidates) : 0;
    let selected = candidates[0] ?? this.MAXBATCH,
      best = Infinity;
    try {
      for (const k of candidates) {
        await this._resetAutotuneDecodeState(resetTokens);
        const t0 = performance.now();
        await this.decodeGreedyBatch(0, k);
        const ms = performance.now() - t0;
        const msPerToken = ms / k;
        rows.push({ k, ms, msPerToken });
        const latencyOk = !Number.isFinite(this.decodeBatchMaxLatencyMs) || ms <= this.decodeBatchMaxLatencyMs;
        if (latencyOk && msPerToken < best) {
          best = msPerToken;
          selected = k;
        }
      }
      if (!rows.some((r) => r.k === selected) && rows.length)
        selected = rows.reduce((a, b) => (a.msPerToken <= b.msPerToken ? a : b)).k;
      this.MAXBATCH = selected;
      this.decodeBatchTuning = {
        selected,
        candidates: rows,
        reason: 'auto wall-clock decodeGreedyBatch with reset state',
      };
    } catch (e) {
      this.decodeBatchTuning = { selected: this.MAXBATCH, candidates: rows, reason: `auto failed: ${e.message}` };
    } finally {
      if (resetTokens > 0) {
        try {
          await this._resetAutotuneDecodeState(resetTokens);
        } catch {}
      }
    }
    return this.decodeBatchTuning;
  }

  // y = int8-GEMV(x, q) [+bias] [+lora]. q={w,scale,N,K}. moduleKey for LoRA lookup.
  gemv(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) {
      // d = x@A  (rank outputs)
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      const bgA = this._bgCached(
        this.pipes.loraA,
        [xBuf, mod.A, this.s.loraD, uA],
        `loraA:${moduleKey}:${this._loraEpoch}`,
        { sensitive: true },
      );
      this._dispatch(enc, this.pipes.loraA, bgA, mod.rank, 1, 'loraA');
    }
    const meta = this._gemvMeta(q, biasBuf, mod);
    const key = `gemv:${moduleKey || 'base'}:${q.K}:${q.N}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod },
    );
    this._dispatch(enc, this.pipes.gemv, bg, meta.gx, meta.gy, `gemv:${q.N}x${q.K}`, meta.bytes);
  }

  gemv4(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (this.debugCapture) console.log('VWG gemv4: ' + moduleKey + ' mod=' + !!mod);
    if (mod) {
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      this._dispatch(
        enc,
        this.pipes.loraA,
        this._bgCached(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, uA], `loraA:${moduleKey}:${this._loraEpoch}`, {
          sensitive: true,
        }),
        mod.rank,
        1,
        'loraA',
      );
      if (this.debugCapture && moduleKey === 'layers.0.self_attn.q_proj' && this.debugStep < this.debugT) {
        enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xSeq, this.debugStep * q.K * 4, q.K * 4);
        enc.copyBufferToBuffer(this.s.loraD, 0, this.debugBufs.dSeq, this.debugStep * mod.rank * 4, mod.rank * 4);
      }
    }
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4:${moduleKey || 'base'}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod },
    );
    this._dispatch(enc, this.pipes.gemv4, bg, meta.gx, meta.gy, `g4:${q.N}x${q.K}`, meta.bytes);
    if (mod) {
      if (this.debugCapture && moduleKey === 'layers.0.self_attn.q_proj' && this.debugStep < this.debugT) {
        enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.ySeq, this.debugStep * q.N * 4, q.N * 4);
        this.debugStep++;
      }
    }
  }

  _loraA(enc, xBuf, q, mod, dBuf, moduleKey, label = 'loraA') {
    const imm = new Uint32Array([q.K, mod.rank]);
    this._dispatch(
      enc,
      this.pipes.loraA,
      this._bgCached(this.pipes.loraA, [xBuf, mod.A, dBuf], `${label}:${moduleKey}:${this._loraEpoch}`, {
        sensitive: true,
      }),
      mod.rank,
      1,
      label,
      imm,
    );
    if (this.debugCapture && moduleKey === 'layers.0.self_attn.q_proj' && this.debugStep < this.debugT) {
      enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xSeq, this.debugStep * q.K * 4, q.K * 4);
      enc.copyBufferToBuffer(dBuf, 0, this.debugBufs.dSeq, this.debugStep * mod.rank * 4, mod.rank * 4);
    }
  }

  _loraBAdd(enc, yBuf, q, mod, dBuf, moduleKey) {
    const meta = new ArrayBuffer(32);
    const dv = new DataView(meta);
    dv.setUint32(0, q.N, true);
    dv.setUint32(4, mod.rank, true);
    dv.setFloat32(16, mod.scale, true);
    const bg = this._bgCached(
      this.pipes.loraBAdd,
      [dBuf, mod.B, yBuf],
      `loraBAdd:${moduleKey}:${this._loraEpoch}`,
      { sensitive: true },
    );
    this._dispatch(enc, this.pipes.loraBAdd, bg, Math.ceil(q.N / 256), 1, 'loraB', new Uint8Array(meta));
    if (this.debugCapture && moduleKey === 'layers.0.self_attn.q_proj' && this.debugStep < this.debugT) {
      enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.ySeq, this.debugStep * q.N * 4, q.N * 4);
      this.debugStep++;
    }
  }

  gemv4Add(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4add:${moduleKey || 'base'}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4Add,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod },
    );
    this._dispatch(enc, this.pipes.gemv4Add, bg, meta.gx, meta.gy, `g4add:${q.N}x${q.K}`, meta.bytes);
  }

  dynQuant(enc, xBuf, x_qBuf, scale_xBuf, K) {
    const numGroups = Math.ceil(K / 128);
    const imm = new Uint32Array([K]);
    const bg = this._bg(this.pipes.dynQuant, [xBuf, x_qBuf, scale_xBuf]);
    this._dispatch(enc, this.pipes.dynQuant, bg, numGroups, 1, 'dynQuant', imm);
  }

  dynQuantT(enc, xBuf, x_qBuf, scale_xBuf, K, T) {
    const numGroups = Math.ceil(K / 128);
    const imm = new Uint32Array([K, T]);
    const bg = this._bg(this.pipes.dynQuantT, [xBuf, x_qBuf, scale_xBuf]);
    this._dispatch(enc, this.pipes.dynQuantT, bg, numGroups, T, 'dynQuantT', imm);
  }

  gemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) {
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      this._dispatch(
        enc,
        this.pipes.loraA,
        this._bgCached(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, uA], `loraA:${moduleKey}:${this._loraEpoch}`, {
          sensitive: true,
        }),
        mod.rank,
        1,
        'loraA',
      );
    }
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4_w4a8:${moduleKey || 'base'}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4W4A8,
      [
        x_qBuf,
        scale_xBuf,
        q.w,
        q.scale,
        biasBuf || this.s.dummy,
        this.s.loraD,
        mod ? mod.B : this.s.dummy,
        yBuf,
      ],
      key,
      { sensitive: !!mod },
    );
    this._dispatch(enc, this.pipes.gemv4W4A8, bg, meta.gx, meta.gy, `g4w4a8:${q.N}x${q.K}`, meta.bytes);
  }

  gemv4AddW4A8(enc, xBuf, x_qBuf, scale_xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4add_w4a8:${moduleKey || 'base'}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4AddW4A8,
      [
        x_qBuf,
        scale_xBuf,
        q.w,
        q.scale,
        biasBuf || this.s.dummy,
        this.s.loraD,
        mod ? mod.B : this.s.dummy,
        yBuf,
      ],
      key,
      { sensitive: !!mod },
    );
    this._dispatch(enc, this.pipes.gemv4AddW4A8, bg, meta.gx, meta.gy, `g4addw4a8:${q.N}x${q.K}`, meta.bytes);
  }

  qkvGemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, packed, qBuf, kBuf, vBuf, L) {
    const gx = Math.min(packed.totalN, 65535);
    const imm = new Uint32Array([packed.K, packed.totalN, packed.qN, packed.kN, packed.vN, packed.gpr, gx, 0]);
    const bg = this._bgCached(
      this.pipes.qkvGemv4W4A8,
      [x_qBuf, scale_xBuf, packed.w, packed.scale, packed.bias, qBuf, kBuf, vBuf],
      `qkv_w4a8:${L.index}`,
      { sensitive: false },
    );
    this._dispatch(
      enc,
      this.pipes.qkvGemv4W4A8,
      bg,
      gx,
      Math.ceil(packed.totalN / gx),
      `qkvw4a8:${packed.totalN}x${packed.K}`,
      imm,
    );
    for (const [part, out] of [
      [L.q, qBuf],
      [L.k, kBuf],
      [L.v, vBuf],
    ]) {
      const mod = this.lora?.modules?.[part.loraKey];
      if (!mod) continue;
      const q = this.q4[part.weight];
      this._loraA(enc, xBuf, q, mod, this.s.loraD, part.loraKey);
      this._loraBAdd(enc, out, q, mod, this.s.loraD, part.loraKey);
    }
  }

  gateUpSiluGemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, packed, yBuf, L) {
    const gate = this.q4[L.gate.weight],
      up = this.q4[L.up.weight];
    const gateMod = this.lora?.modules?.[L.gate.loraKey];
    const upMod = this.lora?.modules?.[L.up.loraKey];
    if (gateMod) this._loraA(enc, xBuf, gate, gateMod, this.s.loraD, L.gate.loraKey, 'loraA:gate');
    if (upMod) this._loraA(enc, xBuf, up, upMod, this.s.loraD2, L.up.loraKey, 'loraA:up');
    const gx = Math.min(packed.N, 65535);
    const m0 = new Uint32Array([
      packed.K,
      packed.N,
      packed.gpr,
      gx,
      gateMod ? gateMod.rank : 0,
      upMod ? upMod.rank : 0,
      gateMod ? 1 : 0,
      upMod ? 1 : 0,
    ]);
    const m1 = new Float32Array([gateMod ? gateMod.scale : 0, upMod ? upMod.scale : 0, 0, 0]);
    const bg = this._bgCached(
      this.pipes.gateUpSiluGemv4W4A8,
      [
        x_qBuf,
        scale_xBuf,
        packed.w,
        packed.scale,
        yBuf,
        this.s.loraD,
        gateMod ? gateMod.B : this.s.dummy,
        this.s.loraD2,
        upMod ? upMod.B : this.s.dummy,
      ],
      `gu_w4a8:${L.index}:${this._loraEpoch}:${gateMod ? 1 : 0}:${upMod ? 1 : 0}`,
      { sensitive: !!(gateMod || upMod) },
    );
    this._dispatch(
      enc,
      this.pipes.gateUpSiluGemv4W4A8,
      bg,
      gx,
      Math.ceil(packed.N / gx),
      `guw4a8:${packed.N}x${packed.K}`,
      [m0, m1],
    );
  }

  gemm4W4A8(enc, aBuf, a_qBuf, scale_xBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4W4A8, [a_qBuf, scale_xBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4W4A8, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), 'gemm4W4A8', imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }

  gemm4AddTW4A8(enc, aBuf, a_qBuf, scale_xBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4AddTW4A8, [
      a_qBuf,
      scale_xBuf,
      q.w,
      q.scale,
      biasBuf || this.s.dummy,
      yBuf,
    ]);
    this._dispatch(enc, this.pipes.gemm4AddTW4A8, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), 'gemm4AddTW4A8', imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }

  rmsNormQkvRope(enc, xBuf, layerIndex, pos) {
    const c = this.cfg,
      L = this.plan.layers[layerIndex];
    const packed = this.qkv[L.index];
    const meta = new Uint32Array([
      packed.K, packed.totalN, packed.qN, packed.kN, packed.vN, packed.gpr, 20 /*gx placeholder*/, pos, c.headDim,
      ...new Uint32Array(new Float32Array([c.rmsNormEps, packed.qN, packed.kN]).buffer)
    ]);
    const bg = this._bg(
      this.pipes.rmsNormQkvRope,
      [
        xBuf,
        this.bufs[L.inputNorm],
        packed.w,
        packed.scale,
        packed.bias,
        this.ropeCos,
        this.ropeSin,
        this.s.q,
        this.s.k,
        this.s.v,
      ]
    );
    this._dispatch(enc, this.pipes.rmsNormQkvRope, bg, 20, 1, 'rmsNormQkvRope', meta);
    for (const [part, out] of [
      [L.q, this.s.q],
      [L.k, this.s.k],
      [L.v, this.s.v],
    ]) {
      const mod = this.lora?.modules?.[part.loraKey];
      if (!mod) continue;
      const q = this.q4[part.weight];
      this._loraA(enc, this.s.normed, q, mod, this.s.loraD, part.loraKey);
      this._loraBAdd(enc, out, q, mod, this.s.loraD, part.loraKey);
    }
  }

  writeKvPage(enc, kBuf, vBuf, kcBuf, vcBuf, pos, layerIndex) {
    const c = this.cfg;
    const kvd = c.numKVHeads * c.headDim;
    this.pam.ensureBlocks(0, pos + 1);
    const btArr = this.pam.getBlockTableArray(0);
    this.dev.queue.writeBuffer(this.s.blockTableBuf, 0, btArr);
    const meta = new Uint32Array([pos, 0, this.pam.maxBlocksPerSeq, kvd]);
    const bg = this._bg(this.pipes.writeKvPage, [kBuf, vBuf, kcBuf, vcBuf, this.s.blockTableBuf]);
    this._dispatch(enc, this.pipes.writeKvPage, bg, Math.ceil(kvd / 256), 1, 'writeKvPage', meta);
  }

  writeKvPageBatch(enc, kBuf, vBuf, kcBuf, vcBuf, T, off, layerIndex) {
    const c = this.cfg;
    const kvd = c.numKVHeads * c.headDim;
    this.pam.ensureBlocks(0, off + T);
    const btArr = this.pam.getBlockTableArray(0);
    this.dev.queue.writeBuffer(this.s.blockTableBuf, 0, btArr);
    const meta = new Uint32Array([T, 0, this.pam.maxBlocksPerSeq, kvd, off]);
    const bg = this._bg(this.pipes.writeKvPageBatch, [kBuf, vBuf, kcBuf, vcBuf, this.s.blockTableBuf]);
    this._dispatch(enc, this.pipes.writeKvPageBatch, bg, Math.ceil((T * kvd) / 256), 1, 'writeKvPageBatch', meta);
  }

  attnPaged(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg,
      S = this.s;
    const nsplit = Math.ceil(ctx / this.CHUNK);
    const bgP = this._bg(this.pipes.attnPartialPaged, [
      qBuf,
      kc,
      vc,
      S.pm,
      S.pz,
      S.po,
      S.blockTableBuf,
    ]);
    const immP = new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim]);
    const immP2 = new Uint32Array([nsplit, this.CHUNK, 0, this.pam.maxBlocksPerSeq]);
    this._dispatch(enc, this.pipes.attnPartialPaged, bgP, c.numHeads, nsplit, 'attnP_paged', [immP, immP2]);
    const useF16C = this.usingF16() && this.pipes.attnCF16;
    const pipeC = useF16C ? this.pipes.attnCF16 : this.pipes.attnC;
    const bgC = this._bg(pipeC, [
      S.pm,
      S.pz,
      S.po,
      oBuf,
    ]);
    const immC = new Uint32Array([c.numHeads, c.headDim, nsplit, 0]);
    this._dispatch(enc, pipeC, bgC, c.numHeads, 1, useF16C ? 'attnCF16' : 'attnC', immC);
  }

  attnPrefillPaged(enc, qBuf, kc, vc, oBuf, T, qStart = 0, ctx = T) {
    const c = this.cfg;
    if (this.features.prefillAttention === 'block' || qStart !== 0 || ctx !== T) {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, qStart, ctx, 0, this.pam.maxBlocksPerSeq]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillBlockPaged,
        this._bg(this.pipes.attnPrefillBlockPaged, [qBuf, kc, vc, oBuf, this.s.blockTableBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        'attnPrefillBlockPaged',
        imm,
      );
    } else {
      const imm1 = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T]);
      const imm2 = new Uint32Array([0, this.pam.maxBlocksPerSeq]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillPaged,
        this._bg(this.pipes.attnPrefillPaged, [
          qBuf,
          kc,
          vc,
          oBuf,
          this.s.blockTableBuf,
        ]),
        c.numHeads,
        T,
        'attnPrefillPaged',
        [imm1, imm2],
      );
    }
  }

  qkvGemv4(enc, xBuf, packed, qBuf, kBuf, vBuf, L) {
    const gx = Math.min(packed.totalN, 65535);
    const imm = new Uint32Array([packed.K, packed.totalN, packed.qN, packed.kN, packed.vN, packed.gpr, gx, 0]);
    const bg = this._bgCached(
      this.pipes.qkvGemv4,
      [xBuf, packed.w, packed.scale, packed.bias, qBuf, kBuf, vBuf],
      `qkv:${L.index}`,
      { sensitive: false },
    );
    this._dispatch(enc, this.pipes.qkvGemv4, bg, gx, Math.ceil(packed.totalN / gx), `qkv:${packed.totalN}x${packed.K}`, imm);
    for (const [part, out] of [
      [L.q, qBuf],
      [L.k, kBuf],
      [L.v, vBuf],
    ]) {
      const mod = this.lora?.modules?.[part.loraKey];
      if (!mod) continue;
      const q = this.q4[part.weight];
      this._loraA(enc, xBuf, q, mod, this.s.loraD, part.loraKey);
      this._loraBAdd(enc, out, q, mod, this.s.loraD, part.loraKey);
    }
  }

  fusedRmsQkvRope(enc, hiddenBuf, inputNormBuf, packed, qBuf, kBuf, vBuf, pos, L) {
    const qPairs = packed.qN / 2;
    const kPairs = packed.kN / 2;
    const vPairs = packed.vN / 2;
    const totalPairs = qPairs + kPairs + vPairs;
    const gx = Math.min(totalPairs, 65535);
    const meta = new Uint32Array([
      packed.K, totalPairs, qPairs, kPairs, vPairs, packed.gpr, gx, pos, this.cfg.headDim,
      ...new Uint32Array(new Float32Array([this.cfg.rmsNormEps, packed.qN, packed.kN]).buffer)
    ]);
    const bg = this._bg(
      this.pipes.rmsNormQkvRope,
      [
        hiddenBuf,
        inputNormBuf,
        packed.w,
        packed.scale,
        packed.bias,
        this.ropeCos,
        this.ropeSin,
        qBuf,
        kBuf,
        vBuf,
      ]
    );
    this._dispatch(
      enc,
      this.pipes.rmsNormQkvRope,
      bg,
      gx,
      Math.ceil(totalPairs / gx),
      `fusedQkvRope:${totalPairs}x${packed.K}`,
      meta
    );
  }

  gateUpSiluGemv4(enc, xBuf, packed, yBuf, L) {
    const gate = this.q4[L.gate.weight],
      up = this.q4[L.up.weight];
    const gateMod = this.lora?.modules?.[L.gate.loraKey];
    const upMod = this.lora?.modules?.[L.up.loraKey];
    if (gateMod) this._loraA(enc, xBuf, gate, gateMod, this.s.loraD, L.gate.loraKey, 'loraA:gate');
    if (upMod) this._loraA(enc, xBuf, up, upMod, this.s.loraD2, L.up.loraKey, 'loraA:up');
    const gx = Math.min(packed.N, 65535);
    const m0 = new Uint32Array([
      packed.K,
      packed.N,
      packed.gpr,
      gx,
      gateMod ? gateMod.rank : 0,
      upMod ? upMod.rank : 0,
      gateMod ? 1 : 0,
      upMod ? 1 : 0,
    ]);
    const m1 = new Float32Array([gateMod ? gateMod.scale : 0, upMod ? upMod.scale : 0, 0, 0]);
    const bg = this._bgCached(
      this.pipes.gateUpSiluGemv4,
      [
        xBuf,
        packed.w,
        packed.scale,
        yBuf,
        this.s.loraD,
        gateMod ? gateMod.B : this.s.dummy,
        this.s.loraD2,
        upMod ? upMod.B : this.s.dummy,
      ],
      `gu:${L.index}:${this._loraEpoch}:${gateMod ? 1 : 0}:${upMod ? 1 : 0}`,
      { sensitive: !!(gateMod || upMod) },
    );
    this._dispatch(enc, this.pipes.gateUpSiluGemv4, bg, gx, Math.ceil(packed.N / gx), `gu:${packed.N}x${packed.K}`, [m0, m1]);
  }
  rms(enc, xBuf, gBuf, yBuf, K) {
    const imm = new Float32Array([K, this.cfg.rmsNormEps]);
    const useF16 = this.usingF16() && this.pipes.rmsF16;
    const pipe = useF16 ? this.pipes.rmsF16 : this.pipes.rms;
    const key = `rms:${K}${useF16 ? ':f16' : ''}`;
    this._dispatch(enc, pipe, this._bgCached(pipe, [xBuf, gBuf, yBuf], key), 1, 1, useF16 ? 'rmsF16' : 'rms', imm);
  }
  rope(enc, xBuf, pos, nHeads) {
    const useF16 = this.usingF16() && this.pipes.ropeF16;
    const pipe = useF16 ? this.pipes.ropeF16 : this.pipes.rope;
    this._dispatch(
      enc,
      pipe,
      this._bg(pipe, [
        xBuf,
        this.ropeCos,
        this.ropeSin,
      ]),
      Math.ceil((nHeads * (this.cfg.headDim / 2)) / 256),
      1,
      useF16 ? 'ropeF16' : 'rope',
      new Uint32Array([nHeads, this.cfg.headDim, pos])
    );
  }
  ropeQK(enc, qBuf, kBuf, pos) {
    const c = this.cfg;
    const pairs = (c.numHeads + c.numKVHeads) * (c.headDim / 2);
    const useF16 = this.usingF16() && this.pipes.ropeQKF16;
    const pipe = useF16 ? this.pipes.ropeQKF16 : this.pipes.ropeQK;
    this._dispatch(
      enc,
      pipe,
      this._bg(pipe, [
        qBuf,
        kBuf,
        this.ropeCos,
        this.ropeSin,
      ]),
      Math.ceil(pairs / 256),
      1,
      useF16 ? 'ropeQKF16' : 'ropeQK',
      new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, pos])
    );
  }
  attn(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg,
      S = this.s;
    const nsplit = Math.ceil(ctx / this.CHUNK);
    // pass 1: per (head, ctx-chunk) partial softmax → pm/pz/po (nHeads*nsplit workgroups)
    const useF16P = this.usingF16() && this.pipes.attnPF16;
    const pipeP = useF16P ? this.pipes.attnPF16 : this.pipes.attnP;
    const bgP = this._bg(pipeP, [
      qBuf,
      kc,
      vc,
      S.pm,
      S.pz,
      S.po,
    ]);
    const immP = new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim, nsplit, this.CHUNK]);
    this._dispatch(enc, pipeP, bgP, c.numHeads, nsplit, useF16P ? 'attnPF16' : 'attnP', immP);
    // pass 2: combine splits per head → o
    const useF16C = this.usingF16() && this.pipes.attnCF16;
    const pipeC = useF16C ? this.pipes.attnCF16 : this.pipes.attnC;
    const bgC = this._bg(pipeC, [
      S.pm,
      S.pz,
      S.po,
      oBuf,
    ]);
    const immC = new Uint32Array([c.numHeads, c.headDim, nsplit, 0]);
    this._dispatch(enc, pipeC, bgC, c.numHeads, 1, useF16C ? 'attnCF16' : 'attnC', immC);
  }

  // Decode one token at absolute position `pos`. Writes logits to s.logits. Returns nothing.
  step(enc, tokenId, pos) {
    const c = this.cfg,
      S = this.s,
      hd = c.headDim,
      kvd = c.numKVHeads * hd;
    for (let i = 0; i < c.numLayers; i++) {
      const L = this.plan.layers[i];
      if (this.features.fuseRMSNormQKVRoPE) {
        this.rmsNormQkvRope(enc, S.hidden, i, pos);
      } else {
        this.rms(enc, S.hidden, this.bufs[L.inputNorm], S.normed, c.hiddenSize);
        if (this.features.actQuant) {
          this.dynQuant(enc, S.normed, S.x_q, S.scale_x, c.hiddenSize);
          this.qkvGemv4W4A8(enc, S.normed, S.x_q, S.scale_x, this.qkv[L.index], S.q, S.k, S.v, L);
        } else {
          const hasQkvLora =
            this.lora &&
            (this.lora.modules[L.q.loraKey] || this.lora.modules[L.k.loraKey] || this.lora.modules[L.v.loraKey]);
          if (!hasQkvLora && this.features.fuseQKV) {
            this.fusedRmsQkvRope(enc, S.hidden, this.bufs[L.inputNorm], this.qkv[L.index], S.q, S.k, S.v, pos, L);
          } else if (this.features.fuseQKV) {
            this.qkvGemv4(enc, S.normed, this.qkv[L.index], S.q, S.k, S.v, L);
            if (this.features.fuseRoPE) this.ropeQK(enc, S.q, S.k, pos);
            else {
              this.rope(enc, S.q, pos, c.numHeads);
              this.rope(enc, S.k, pos, c.numKVHeads);
            }
          } else {
            this.gemv4(enc, S.normed, this.q4[L.q.weight], S.q, this.bufs[L.q.bias], L.q.loraKey);
            this.gemv4(enc, S.normed, this.q4[L.k.weight], S.k, this.bufs[L.k.bias], L.k.loraKey);
            this.gemv4(enc, S.normed, this.q4[L.v.weight], S.v, this.bufs[L.v.bias], L.v.loraKey);
            if (this.features.fuseRoPE) this.ropeQK(enc, S.q, S.k, pos);
            else {
              this.rope(enc, S.q, pos, c.numHeads);
              this.rope(enc, S.k, pos, c.numKVHeads);
            }
          }
        }
      }

      if (this.features.pagedAttention) {
        this.writeKvPage(enc, S.k, S.v, this.kc[i], this.vc[i], pos, i);
      } else {
        enc.copyBufferToBuffer(S.k, 0, this.kc[i], pos * kvd * 4, kvd * 4);
        enc.copyBufferToBuffer(S.v, 0, this.vc[i], pos * kvd * 4, kvd * 4);
      }

      if (this.features.pagedAttention) {
        this.attnPaged(enc, S.q, this.kc[i], this.vc[i], S.attn, pos + 1);
      } else {
        this.attn(enc, S.q, this.kc[i], this.vc[i], S.attn, pos + 1);
      }

      if (this.features.actQuant) {
        this.dynQuant(enc, S.attn, S.x_q, S.scale_x, c.hiddenSize);
        if (this.features.fuseResidual) {
          this.gemv4AddW4A8(enc, S.attn, S.x_q, S.scale_x, this.q4[L.o.weight], S.hidden, null, L.o.loraKey);
        } else {
          this.gemv4W4A8(enc, S.attn, S.x_q, S.scale_x, this.q4[L.o.weight], S.tmp, null, L.o.loraKey);
          this._addInto(enc, S.hidden, S.tmp, c.hiddenSize);
        }
      } else {
        if (this.features.fuseResidual) this.gemv4Add(enc, S.attn, this.q4[L.o.weight], S.hidden, null, L.o.loraKey);
        else {
          this.gemv4(enc, S.attn, this.q4[L.o.weight], S.tmp, null, L.o.loraKey);
          this._addInto(enc, S.hidden, S.tmp, c.hiddenSize);
        }
      }

      this.rms(enc, S.hidden, this.bufs[L.postAttentionNorm], S.normed, c.hiddenSize);
      if (this.features.actQuant) {
        this.dynQuant(enc, S.normed, S.x_q, S.scale_x, c.hiddenSize);
        this.gateUpSiluGemv4W4A8(enc, S.normed, S.x_q, S.scale_x, this.gateUp[L.index], S.tmp, L);
      } else {
        if (this.features.fuseMLP) {
          this.gateUpSiluGemv4(enc, S.normed, this.gateUp[L.index], S.tmp, L);
        } else {
          this.gemv4(enc, S.normed, this.q4[L.gate.weight], S.tmp, null, L.gate.loraKey);
          this.gemv4(enc, S.normed, this.q4[L.up.weight], S.tmp2, null, L.up.loraKey);
          this._siluMul(enc, S.tmp, S.tmp2, c.intermediateSize);
        }
      }

      if (this.features.actQuant) {
        this.dynQuant(enc, S.tmp, S.x_q, S.scale_x, c.intermediateSize);
        if (this.features.fuseResidual) {
          this.gemv4AddW4A8(enc, S.tmp, S.x_q, S.scale_x, this.q4[L.down.weight], S.hidden, null, L.down.loraKey);
        } else {
          this.gemv4W4A8(enc, S.tmp, S.x_q, S.scale_x, this.q4[L.down.weight], S.normed, null, L.down.loraKey);
          this._addInto(enc, S.hidden, S.normed, c.hiddenSize);
        }
      } else {
        if (this.features.fuseResidual)
          this.gemv4Add(enc, S.tmp, this.q4[L.down.weight], S.hidden, null, L.down.loraKey);
        else {
          this.gemv4(enc, S.tmp, this.q4[L.down.weight], S.normed, null, L.down.loraKey);
          this._addInto(enc, S.hidden, S.normed, c.hiddenSize);
        }
      }
    }
    this.rms(enc, S.hidden, this.bufs[this.plan.finalNorm.name], S.normed, c.hiddenSize);
    this.gemv(enc, S.normed, this.q[this.plan.embed.name], S.logits, null, null);
  }

  _addInto(enc, yBuf, aBuf, n) {
    const imm = new Uint32Array([n]);
    const useF16 = this.usingF16() && this.pipes.addF16;
    const pipe = useF16 ? this.pipes.addF16 : this.pipes.add;
    const bg = this._bgCached(pipe, [aBuf, yBuf], `add:${n}${useF16 ? ':f16' : ''}`);
    this._dispatch(enc, pipe, bg, Math.min(Math.ceil(n / 256), 65535), 1, useF16 ? 'addF16' : 'add', imm);
  }
  _siluMul(enc, gateBuf, upBuf, n) {
    const imm = new Uint32Array([n]);
    const useF16 = this.usingF16() && this.pipes.siluF16;
    const pipe = useF16 ? this.pipes.siluF16 : this.pipes.silu;
    const bg = this._bgCached(pipe, [gateBuf, upBuf], `silu:${n}${useF16 ? ':f16' : ''}`);
    this._dispatch(enc, pipe, bg, Math.min(Math.ceil(n / 256), 65535), 1, useF16 ? 'siluF16' : 'silu', imm);
  }
  embedRow(enc, id) {
    const e = this.q[this.plan.embed.name];
    const imm = new Uint32Array([id, this.cfg.hiddenSize]);
    this._dispatch(
      enc,
      this.pipes.embed,
      this._bg(this.pipes.embed, [e.w, e.scale, this.s.hidden]),
      Math.ceil(this.cfg.hiddenSize / 256),
      1,
      'embed',
      imm,
    );
  }
  async argmaxLogits() {
    if (this._argmaxReadBusy)
      throw new Error('argmaxLogits() is already in flight; concurrent generation is not supported');
    this._argmaxReadBusy = true;
    const enc = this.dev.createCommandEncoder();
    // argmax n is passed via immediate now; the cached u.argmax can be dropped over time
    const n = this.cfg.vocabSize || 0;
    this._dispatch(
      enc,
      this.pipes.argmax,
      this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax], 'argmax'),
      1,
      1,
      'argmax',
      new Uint32Array([n]),
    );
    enc.copyBufferToBuffer(this.s.amax, 0, this.argmaxRead, 0, 4);
    this.dev.queue.submit([enc.finish()]);
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
    try {
      await this.argmaxRead.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(this.argmaxRead.getMappedRange())[0];
      this.argmaxRead.unmap();
      return id;
    } finally {
      this._argmaxReadBusy = false;
    }
  }

  // Convenience for numeric comparison harnesses (Phase 3 f16 eval etc.).
  // Returns a fresh Float32Array copy of the current final logits buffer.
  async readLogits() {
    const n = this.cfg.vocabSize;
    if (!this._logitsRead) {
      this._logitsRead = this._buf(n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    }
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.s.logits, 0, this._logitsRead, 0, n * 4);
    this.dev.queue.submit([enc.finish()]);
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
    await this._logitsRead.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this._logitsRead.getMappedRange()).slice();
    this._logitsRead.unmap();
    return out;
  }

  async topKLogits(k = this.samplingTopK) {
    if (this._topKReadBusy) throw new Error('topKLogits() is already in flight; concurrent sampling is not supported');
    this._topKReadBusy = true;
    try {
      k = Math.min(Math.max(1, Math.floor(k)), this.maxSamplingTopK, this.cfg.vocabSize);
      const enc = this.dev.createCommandEncoder();
      for (let i = 0; i < k; i++) {
        const imm = new Uint32Array([this.cfg.vocabSize, i]);
        this._dispatch(
          enc,
          this.pipes.topkSelect,
          this._bgCached(this.pipes.topkSelect, [this.s.logits, this.s.sampleIds, this.s.sampleVals], `topk:${i}`),
          1,
          1,
          'topk',
          imm,
        );
      }
      enc.copyBufferToBuffer(this.s.sampleIds, 0, this.sampleIdsRead, 0, k * 4);
      enc.copyBufferToBuffer(this.s.sampleVals, 0, this.sampleValsRead, 0, k * 4);
      this.dev.queue.submit([enc.finish()]);
      await Promise.all([this.sampleIdsRead.mapAsync(GPUMapMode.READ), this.sampleValsRead.mapAsync(GPUMapMode.READ)]);
      const ids = Array.from(new Uint32Array(this.sampleIdsRead.getMappedRange(), 0, k));
      const vals = Array.from(new Float32Array(this.sampleValsRead.getMappedRange(), 0, k));
      return ids.map((id, i) => ({ id, logit: vals[i] }));
    } finally {
      if (this.sampleIdsRead.mapState !== 'unmapped') this.sampleIdsRead.unmap();
      if (this.sampleValsRead.mapState !== 'unmapped') this.sampleValsRead.unmap();
      this._topKReadBusy = false;
    }
  }

  // Phase 5: GPU-resident sampling.
  // Populates top-K on GPU (via existing machinery), then runs a tiny kernel that
  // applies temperature, softmax + nucleus over the k candidates using a host-supplied
  // uniform r, and writes exactly **one** token id. Only one u32 is read back.
  // This is the first step toward eliminating large per-token host round-trips for sampling.
  async sampleToken(temp = 1.0, r = (typeof Math !== 'undefined' ? Math.random() : 0.5)) {
    const k = Math.min(this.samplingTopK, this.maxSamplingTopK, this.cfg.vocabSize);
    // Populate the top-K buffers (current impl reads k values; future work can keep
    // everything resident and chain directly into the sample kernel).
    await this.topKLogits(k);

    const enc = this.dev.createCommandEncoder();
    const bg = this._bg(this.pipes.sampleTopK, [
      this.s.sampleIds,
      this.s.sampleVals,
      this.s.sampled,
    ]);
    const immK = new Uint32Array([k]);
    const immP = new Float32Array([temp > 0 ? temp : 1.0, Math.max(0, Math.min(1, r))]);
    this._dispatch(enc, this.pipes.sampleTopK, bg, 1, 1, 'sampleTopK', [immK, immP]);

    enc.copyBufferToBuffer(this.s.sampled, 0, this.sampledRead, 0, 4);
    this.dev.queue.submit([enc.finish()]);
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
    await this.sampledRead.mapAsync(GPUMapMode.READ);
    const id = new Uint32Array(this.sampledRead.getMappedRange())[0];
    this.sampledRead.unmap();
    return id;
  }

  // Run one token end-to-end (embed + step) and submit.
  token(id, pos) {
    this._resetUni();
    const enc = this.dev.createCommandEncoder();
    this.embedRow(enc, id);
    this.step(enc, id, pos);
    this.dev.queue.submit([enc.finish()]);
  }

  // embed the token id held in s.amax (GPU-resident, from a prior argmax)
  embedFromBuf(enc) {
    const e = this.q[this.plan.embed.name];
    const imm = new Uint32Array([this.cfg.hiddenSize]);
    this._dispatch(
      enc,
      this.pipes.embedBuf,
      this._bgCached(this.pipes.embedBuf, [e.w, e.scale, this.s.hidden, this.s.amax], 'embedBuf'),
      Math.ceil(this.cfg.hiddenSize / 256),
      1,
      'embed',
      imm,
    );
  }
  // argmax(logits) -> s.amax, within the given encoder (no submit/readback)
  argmaxInto(enc) {
    this._dispatch(
      enc,
      this.pipes.argmax,
      this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax, this.u.argmax], 'argmax'),
      1,
      1,
      'argmax',
    );
  }

  // GPU-resident batched GREEDY decode only: chains embed->step->argmax for K
  // tokens in ONE submit, reads back K ids once, and checks stop tokens only
  // after readback. It assumes s.amax already holds the current token id to
  // embed. Do not use for sampled decoding; sampled tokens must be written by
  // the CPU/GPU sampler one step at a time.
  async decodeBatch(startPos, K) {
    K = Math.min(K, this.decodeBatchCapacity, this.maxCtx - startPos); // never write/read past cache or ids buffers
    if (K <= 0) return [];
    this._resetUni();
    const enc = this.dev.createCommandEncoder();
    for (let k = 0; k < K; k++) {
      this.embedFromBuf(enc);
      this.step(enc, 0, startPos + k);
      this.argmaxInto(enc);
      enc.copyBufferToBuffer(this.s.amax, 0, this.s.idsBuf, k * 4, 4);
    }
    enc.copyBufferToBuffer(this.s.idsBuf, 0, this.idsRead, 0, K * 4);
    this.dev.queue.submit([enc.finish()]);
    await this.idsRead.mapAsync(GPUMapMode.READ);
    const ids = Array.from(new Uint32Array(this.idsRead.getMappedRange(), 0, K));
    this.idsRead.unmap();
    return ids;
  }
  async decodeGreedyBatch(startPos, K) {
    return this.decodeBatch(startPos, K);
  }

  // ---- PREFILL (T>1): process the whole prompt at once via tiled GEMM. If a LoRA
  // adapter has the projection module, add its batched delta immediately after base GEMM.
  gemm4(enc, aBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4, [aBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), 'gemm4', imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  gemm4AddT(enc, aBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4AddT, [aBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4AddT, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), 'gemm4AddT', imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  loraBatchDelta(enc, xBuf, yBuf, q, T, mod, moduleKey) {
    if (this.debugCapture) console.log('VWG loraBatchDelta: ' + moduleKey + ' mod=' + !!mod);
    const imm = new Uint32Array([q.K, mod.rank, T, 0]);
    const bgA = this._bg(this.pipes.loraABatch, [xBuf, mod.A, this.sT.loraD]);
    this._dispatch(enc, this.pipes.loraABatch, bgA, mod.rank, T, 'loraA:T', imm);
    if (this.debugCapture && moduleKey === 'layers.0.self_attn.q_proj') {
      enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xBat, 0, T * q.K * 4);
      enc.copyBufferToBuffer(this.sT.loraD, 0, this.debugBufs.dBat, 0, T * mod.rank * 4);
    }
    const totalGroups = Math.ceil((T * q.N) / 256);
    let gx = totalGroups;
    let gy = 1;
    if (gx > 65535) {
      gx = 256;
      gy = Math.ceil(totalGroups / 256);
    }
    const meta = new ArrayBuffer(32);
    const dv = new DataView(meta);
    dv.setUint32(0, T, true);
    dv.setUint32(4, q.N, true);
    dv.setUint32(8, mod.rank, true);
    dv.setUint32(12, gx, true);
    dv.setFloat32(16, mod.scale, true);
    const bgB = this._bg(this.pipes.loraBAddT, [this.sT.loraD, mod.B, yBuf]);
    this._dispatch(enc, this.pipes.loraBAddT, bgB, gx, gy, 'loraB:T', new Uint8Array(meta));
    if (this.debugCapture && moduleKey === 'layers.0.self_attn.q_proj') {
      enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.yBat, 0, T * q.N * 4);
      this.debugCaptured = true;
    }
  }
  rmsT(enc, xBuf, gBuf, yBuf, T, K) {
    const imm = new Float32Array([K, this.cfg.rmsNormEps]);
    const useF16 = this.usingF16() && this.pipes.rmsTF16;
    const pipe = useF16 ? this.pipes.rmsTF16 : this.pipes.rmsT;
    this._dispatch(enc, pipe, this._bg(pipe, [xBuf, gBuf, yBuf]), T, 1, useF16 ? 'rmsTF16' : 'rmsT', imm);
  }
  ropeT(enc, xBuf, T, nHeads, pos0 = 0) {
    const hd = this.cfg.headDim;
    const imm = new Uint32Array([nHeads, hd, T, pos0]);
    const useF16 = this.usingF16() && this.pipes.ropeTF16;
    const pipe = useF16 ? this.pipes.ropeTF16 : this.pipes.ropeT;
    this._dispatch(
      enc,
      pipe,
      this._bg(pipe, [xBuf, this.ropeCos, this.ropeSin]),
      Math.ceil((T * nHeads * (hd / 2)) / 256),
      1,
      useF16 ? 'ropeTF16' : 'ropeT',
      imm,
    );
  }
  attnPrefill(enc, qBuf, kc, vc, oBuf, T, qStart = 0, ctx = T) {
    const c = this.cfg;
    if (this.features.prefillAttention === 'block' || qStart !== 0 || ctx !== T) {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, qStart, ctx, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillBlock,
        this._bg(this.pipes.attnPrefillBlock, [qBuf, kc, vc, oBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        'attnPrefillBlock',
        imm,
      );
    } else {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T]);
      this._dispatch(
        enc,
        this.pipes.attnPrefill,
        this._bg(this.pipes.attnPrefill, [qBuf, kc, vc, oBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        'attnPrefill',
        imm
      );
    }
  }

  // (re)allocate prefill scratch sized to T (grows as needed; only paid when prefilling).
  _ensurePrefillScratch(T, loraRank = 0, idsCap = T) {
    if (this.sTcap >= T && (this.sTLoraRank || 0) >= loraRank && (this.sTidsCap || 0) >= idsCap) return;
    const need = this.estimatePrefillScratchBytes(T, loraRank);
    if (this.opts.maxPrefillScratchBytes && need > this.opts.maxPrefillScratchBytes) {
      throw new Error(
        `prefill scratch ${Math.ceil(need / 1048576)}MiB exceeds maxPrefillScratchBytes; lower maxPrefillT or use shorter prompt chunks`,
      );
    }
    if (this.sT) for (const k in this.sT) this.sT[k].destroy();
    const c = this.cfg,
      H = c.hiddenSize,
      qd = c.numHeads * c.headDim,
      kvd = c.numKVHeads * c.headDim,
      I = c.intermediateSize;
    this.sT = {
      hidden: this._buf(T * H * 4),
      normed: this._buf(T * H * 4),
      q: this._buf(T * qd * 4),
      k: this._buf(T * kvd * 4),
      v: this._buf(T * kvd * 4),
      attn: this._buf(T * qd * 4),
      tmp: this._buf(T * I * 4),
      tmp2: this._buf(T * I * 4),
      ids: this._buf(idsCap * 4),
      loraD: this._buf(Math.max(1, T * Math.max(1, loraRank)) * 4),
      x_q: this._buf(T * Math.max(H, I) * 4),
      scale_x: this._buf(((T * Math.max(H, I)) / 128) * 4),
    };
    this.sTcap = T;
    this.sTLoraRank = loraRank;
    this.sTidsCap = idsCap;
  }

  _activeMaxLoraRank() {
    let rank = 0;
    const mods = this.lora?.modules;
    if (!mods) return 0;
    for (const key of Object.keys(mods)) rank = Math.max(rank, mods[key].rank || 0);
    return rank;
  }

  // Prefill the prompt (positions 0..T-1). Leaves last-row logits in s.logits and the
  // KV cache populated, so decode continues from pos=T. T must be <= maxPrefillT.
  prefillBatch(ids) {
    const T = ids.length;
    if (T > this.maxPrefillT) throw new Error(`prompt ${T} > maxPrefillT ${this.maxPrefillT}`);
    if (T > this.maxCtx) throw new Error(`prompt ${T} > maxCtx ${this.maxCtx}`);
    const chunk = this.features.prefillChunkSize;
    if (chunk > 0 && T > chunk) return this._prefillChunked(ids, chunk);
    return this._prefillFull(ids);
  }

  _prefillFull(ids) {
    const c = this.cfg,
      S = this.s,
      T = ids.length,
      hd = c.headDim,
      kvd = c.numKVHeads * hd,
      H = c.hiddenSize;
    this._ensurePrefillScratch(T, this._activeMaxLoraRank());
    const ST = this.sT;
    this._resetUni();
    this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(ids));
    const enc = this.dev.createCommandEncoder();
    const e = this.q[this.plan.embed.name];
    const imm = new Uint32Array([T, H, 0, 0]);
    this._dispatch(
      enc,
      this.pipes.embedT,
      this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
      Math.min(Math.ceil((T * H) / 256), 65535),
      1,
      'embedT',
      imm,
    );
    for (let i = 0; i < c.numLayers; i++) {
      const L = this.plan.layers[i];
      this.rmsT(enc, ST.hidden, this.bufs[L.inputNorm], ST.normed, T, H);

      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
        this.gemm4W4A8(
          enc,
          ST.normed,
          ST.x_q,
          ST.scale_x,
          this.q4[L.q.weight],
          ST.q,
          T,
          this.bufs[L.q.bias],
          L.q.loraKey,
        );
        this.gemm4W4A8(
          enc,
          ST.normed,
          ST.x_q,
          ST.scale_x,
          this.q4[L.k.weight],
          ST.k,
          T,
          this.bufs[L.k.bias],
          L.k.loraKey,
        );
        this.gemm4W4A8(
          enc,
          ST.normed,
          ST.x_q,
          ST.scale_x,
          this.q4[L.v.weight],
          ST.v,
          T,
          this.bufs[L.v.bias],
          L.v.loraKey,
        );
      } else {
        this.gemm4(enc, ST.normed, this.q4[L.q.weight], ST.q, T, this.bufs[L.q.bias], L.q.loraKey);
        this.gemm4(enc, ST.normed, this.q4[L.k.weight], ST.k, T, this.bufs[L.k.bias], L.k.loraKey);
        this.gemm4(enc, ST.normed, this.q4[L.v.weight], ST.v, T, this.bufs[L.v.bias], L.v.loraKey);
      }
      this.ropeT(enc, ST.q, T, c.numHeads);
      this.ropeT(enc, ST.k, T, c.numKVHeads);

      if (this.features.pagedAttention) {
        this.writeKvPageBatch(enc, ST.k, ST.v, this.kc[i], this.vc[i], T, 0, i);
      } else {
        enc.copyBufferToBuffer(ST.k, 0, this.kc[i], 0, T * kvd * 4);
        enc.copyBufferToBuffer(ST.v, 0, this.vc[i], 0, T * kvd * 4);
      }

      if (this.features.pagedAttention) {
        this.attnPrefillPaged(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, 0, T);
      } else {
        this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, 0, T);
      }

      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.attn, ST.x_q, ST.scale_x, H, T);
        if (this.features.fuseResidual) {
          this.gemm4AddTW4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
        } else {
          this.gemm4W4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
          this._addInto(enc, ST.hidden, ST.tmp, T * H);
        }
      } else {
        if (this.features.fuseResidual)
          this.gemm4AddT(enc, ST.attn, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
        else {
          this.gemm4(enc, ST.attn, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
          this._addInto(enc, ST.hidden, ST.tmp, T * H);
        }
      }

      this.rmsT(enc, ST.hidden, this.bufs[L.postAttentionNorm], ST.normed, T, H);
      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
        this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
        this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
      } else {
        this.gemm4(enc, ST.normed, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
        this.gemm4(enc, ST.normed, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
      }
      this._siluMul(enc, ST.tmp, ST.tmp2, T * c.intermediateSize);

      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.tmp, ST.x_q, ST.scale_x, c.intermediateSize, T);
        if (this.features.fuseResidual) {
          this.gemm4AddTW4A8(
            enc,
            ST.tmp,
            ST.x_q,
            ST.scale_x,
            this.q4[L.down.weight],
            ST.hidden,
            T,
            null,
            L.down.loraKey,
          );
        } else {
          this.gemm4W4A8(enc, ST.tmp, ST.x_q, ST.scale_x, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
          this._addInto(enc, ST.hidden, ST.normed, T * H);
        }
      } else {
        if (this.features.fuseResidual)
          this.gemm4AddT(enc, ST.tmp, this.q4[L.down.weight], ST.hidden, T, null, L.down.loraKey);
        else {
          this.gemm4(enc, ST.tmp, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
          this._addInto(enc, ST.hidden, ST.normed, T * H);
        }
      }
    }
    enc.copyBufferToBuffer(ST.hidden, (T - 1) * H * 4, S.hidden, 0, H * 4);
    this.rms(enc, S.hidden, this.bufs[this.plan.finalNorm.name], S.normed, H);
    this.gemv(enc, S.normed, this.q[this.plan.embed.name], S.logits, null, null);
    this.dev.queue.submit([enc.finish()]);
  }

  _prefillChunked(ids, chunkSize) {
    const c = this.cfg,
      S = this.s,
      H = c.hiddenSize,
      hd = c.headDim,
      kvd = c.numKVHeads * hd;
    const T = ids.length;
    this._ensurePrefillScratch(Math.min(chunkSize, T), this._activeMaxLoraRank(), T);
    const ST = this.sT;
    this._resetUni();
    this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(ids));
    const enc = this.dev.createCommandEncoder();
    const e = this.q[this.plan.embed.name];
    for (let off = 0; off < T; off += chunkSize) {
      const end = Math.min(T, off + chunkSize);
      const CT = end - off;
      this._dispatch(
        enc,
        this.pipes.embedT,
        this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
        Math.min(Math.ceil((CT * H) / 256), 65535),
        1,
        'embedT',
        new Uint32Array([CT, H, off, 0]),
      );
      for (let i = 0; i < c.numLayers; i++) {
        const L = this.plan.layers[i];
        this.rmsT(enc, ST.hidden, this.bufs[L.inputNorm], ST.normed, CT, H);

        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, CT);
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.q.weight],
            ST.q,
            CT,
            this.bufs[L.q.bias],
            L.q.loraKey,
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.k.weight],
            ST.k,
            CT,
            this.bufs[L.k.bias],
            L.k.loraKey,
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.v.weight],
            ST.v,
            CT,
            this.bufs[L.v.bias],
            L.v.loraKey,
          );
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.q.weight], ST.q, CT, this.bufs[L.q.bias], L.q.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.k.weight], ST.k, CT, this.bufs[L.k.bias], L.k.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.v.weight], ST.v, CT, this.bufs[L.v.bias], L.v.loraKey);
        }
        this.ropeT(enc, ST.q, CT, c.numHeads, off);
        this.ropeT(enc, ST.k, CT, c.numKVHeads, off);

        if (this.features.pagedAttention) {
          this.writeKvPageBatch(enc, ST.k, ST.v, this.kc[i], this.vc[i], CT, off, i);
        } else {
          enc.copyBufferToBuffer(ST.k, 0, this.kc[i], off * kvd * 4, CT * kvd * 4);
          enc.copyBufferToBuffer(ST.v, 0, this.vc[i], off * kvd * 4, CT * kvd * 4);
        }

        if (this.features.pagedAttention) {
          this.attnPrefillPaged(enc, ST.q, this.kc[i], this.vc[i], ST.attn, CT, off, end);
        } else {
          this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, CT, off, end);
        }

        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.attn, ST.x_q, ST.scale_x, H, CT);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.hidden, CT, null, L.o.loraKey);
          } else {
            this.gemm4W4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.tmp, CT, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, CT * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.attn, this.q4[L.o.weight], ST.hidden, CT, null, L.o.loraKey);
          else {
            this.gemm4(enc, ST.attn, this.q4[L.o.weight], ST.tmp, CT, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, CT * H);
          }
        }

        this.rmsT(enc, ST.hidden, this.bufs[L.postAttentionNorm], ST.normed, CT, H);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, CT);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.gate.weight], ST.tmp, CT, null, L.gate.loraKey);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.up.weight], ST.tmp2, CT, null, L.up.loraKey);
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.gate.weight], ST.tmp, CT, null, L.gate.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.up.weight], ST.tmp2, CT, null, L.up.loraKey);
        }
        this._siluMul(enc, ST.tmp, ST.tmp2, CT * c.intermediateSize);

        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.tmp, ST.x_q, ST.scale_x, c.intermediateSize, CT);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(
              enc,
              ST.tmp,
              ST.x_q,
              ST.scale_x,
              this.q4[L.down.weight],
              ST.hidden,
              CT,
              null,
              L.down.loraKey,
            );
          } else {
            this.gemm4W4A8(
              enc,
              ST.tmp,
              ST.x_q,
              ST.scale_x,
              this.q4[L.down.weight],
              ST.normed,
              CT,
              null,
              L.down.loraKey,
            );
            this._addInto(enc, ST.hidden, ST.normed, CT * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.tmp, this.q4[L.down.weight], ST.hidden, CT, null, L.down.loraKey);
          else {
            this.gemm4(enc, ST.tmp, this.q4[L.down.weight], ST.normed, CT, null, L.down.loraKey);
            this._addInto(enc, ST.hidden, ST.normed, CT * H);
          }
        }
      }
      if (end === T) {
        enc.copyBufferToBuffer(ST.hidden, (CT - 1) * H * 4, S.hidden, 0, H * 4);
      }
    }
    this.rms(enc, S.hidden, this.bufs[this.plan.finalNorm.name], S.normed, H);
    this.gemv(enc, S.normed, this.q[this.plan.embed.name], S.logits, null, null);
    this.dev.queue.submit([enc.finish()]);
  }

  async speculativeDecode(draftModel, promptIds, maxNewTokens, onToken) {
    await this.prefillBatch(promptIds);
    await draftModel.prefillBatch(promptIds);

    let currentPos = promptIds.length;
    const generatedIds = [];

    let nextToken = await this.argmaxLogits();
    generatedIds.push(nextToken);
    if (onToken) onToken(nextToken);

    draftModel.dev.queue.writeBuffer(draftModel.s.amax, 0, new Uint32Array([nextToken]));
    this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([nextToken]));

    const gamma = 4;

    while (generatedIds.length < maxNewTokens) {
      const draftCandidates = await draftModel.decodeBatch(currentPos, gamma);
      if (draftCandidates.length === 0) break;

      const T = draftCandidates.length;
      this._resetUni();
      this._ensurePrefillScratch(T, this._activeMaxLoraRank());
      const ST = this.sT;
      const c = this.cfg,
        H = c.hiddenSize,
        kvd = c.numKVHeads * c.headDim;

      this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(draftCandidates));

      const enc = this.dev.createCommandEncoder();
      const e = this.q[this.plan.embed.name];
      const embedUni = new Uint32Array([T, H, 0, 0]);

      this._dispatch(
        enc,
        this.pipes.embedT,
        this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
        Math.min(Math.ceil((T * H) / 256), 65535),
        1,
        'embedT',
        embedUni,
      );
      for (let i = 0; i < c.numLayers; i++) {
        const L = this.plan.layers[i];
        this.rmsT(enc, ST.hidden, this.bufs[L.inputNorm], ST.normed, T, H);

        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.q.weight],
            ST.q,
            T,
            this.bufs[L.q.bias],
            L.q.loraKey,
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.k.weight],
            ST.k,
            T,
            this.bufs[L.k.bias],
            L.k.loraKey,
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.v.weight],
            ST.v,
            T,
            this.bufs[L.v.bias],
            L.v.loraKey,
          );
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.q.weight], ST.q, T, this.bufs[L.q.bias], L.q.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.k.weight], ST.k, T, this.bufs[L.k.bias], L.k.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.v.weight], ST.v, T, this.bufs[L.v.bias], L.v.loraKey);
        }
        this.ropeT(enc, ST.q, T, c.numHeads, currentPos);
        this.ropeT(enc, ST.k, T, c.numKVHeads, currentPos);

        if (this.features.pagedAttention) {
          this.writeKvPageBatch(enc, ST.k, ST.v, this.kc[i], this.vc[i], T, currentPos, i);
        } else {
          enc.copyBufferToBuffer(ST.k, 0, this.kc[i], currentPos * kvd * 4, T * kvd * 4);
          enc.copyBufferToBuffer(ST.v, 0, this.vc[i], currentPos * kvd * 4, T * kvd * 4);
        }

        if (this.features.pagedAttention) {
          this.attnPrefillPaged(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, currentPos, currentPos + T);
        } else {
          this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, currentPos, currentPos + T);
        }

        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.attn, ST.x_q, ST.scale_x, H, T);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
          } else {
            this.gemm4W4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, T * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.attn, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
          else {
            this.gemm4(enc, ST.attn, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, T * H);
          }
        }

        this.rmsT(enc, ST.hidden, this.bufs[L.postAttentionNorm], ST.normed, T, H);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
        }
        this._siluMul(enc, ST.tmp, ST.tmp2, T * c.intermediateSize);

        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.tmp, ST.x_q, ST.scale_x, c.intermediateSize, T);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(
              enc,
              ST.tmp,
              ST.x_q,
              ST.scale_x,
              this.q4[L.down.weight],
              ST.hidden,
              T,
              null,
              L.down.loraKey,
            );
          } else {
            this.gemm4W4A8(enc, ST.tmp, ST.x_q, ST.scale_x, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
            this._addInto(enc, ST.hidden, ST.normed, T * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.tmp, this.q4[L.down.weight], ST.hidden, T, null, L.down.loraKey);
          else {
            this.gemm4(enc, ST.tmp, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
            this._addInto(enc, ST.hidden, ST.normed, T * H);
          }
        }
      }

      if (!this.s.logitsT || this.sTcap < T) {
        if (this.s.logitsT) this.s.logitsT.destroy();
        this.s.logitsT = this._buf(T * c.vocabSize * 4);
        if (this.logitsTRead) this.logitsTRead.destroy();
        this.logitsTRead = this._buf(T * c.vocabSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      }

      for (let t = 0; t < T; t++) {
        enc.copyBufferToBuffer(ST.hidden, t * H * 4, this.s.hidden, 0, H * 4);
        this.rms(enc, this.s.hidden, this.bufs[this.plan.finalNorm.name], this.s.normed, H);
        this.gemv(enc, this.s.normed, this.q[this.plan.embed.name], this.s.logits, null, null);
        enc.copyBufferToBuffer(this.s.logits, 0, this.s.logitsT, t * c.vocabSize * 4, c.vocabSize * 4);
      }

      enc.copyBufferToBuffer(this.s.logitsT, 0, this.logitsTRead, 0, T * c.vocabSize * 4);
      this.dev.queue.submit([enc.finish()]);

      await this.logitsTRead.mapAsync(GPUMapMode.READ);
      const logitsArray = new Float32Array(this.logitsTRead.getMappedRange());

      let acceptedCount = 0;
      let targetToken = 0;

      for (let t = 0; t < T; t++) {
        let maxVal = -1e30;
        let argmaxId = 0;
        const offset = t * c.vocabSize;
        for (let v = 0; v < c.vocabSize; v++) {
          const l = logitsArray[offset + v];
          if (l > maxVal) {
            maxVal = l;
            argmaxId = v;
          }
        }

        targetToken = argmaxId;

        if (t < T) {
          if (draftCandidates[t] === targetToken) {
            acceptedCount++;
          } else {
            break;
          }
        }
      }

      this.logitsTRead.unmap();

      for (let a = 0; a < acceptedCount; a++) {
        generatedIds.push(draftCandidates[a]);
        if (onToken) onToken(draftCandidates[a]);
      }

      generatedIds.push(targetToken);
      if (onToken) onToken(targetToken);

      const nextPos = currentPos + acceptedCount + 1;
      this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([targetToken]));
      draftModel.dev.queue.writeBuffer(draftModel.s.amax, 0, new Uint32Array([targetToken]));

      if (this.features.pagedAttention) {
        this.pam.ensureBlocks(0, nextPos);
      }

      currentPos = nextPos;
    }

    return generatedIds;
  }

  // Simple high-level generation helper (Phase 5 wiring).
  // If opts.sample === true, uses the GPU sampler (sampleToken) with given temp;
  // otherwise falls back to argmax (greedy).
  // This makes sampleToken part of the real generation path.
  async generate(promptIds, maxNewTokens = 32, opts = {}) {
    const doSample = !!opts.sample;
    const temp = (opts.temp != null && opts.temp > 0) ? opts.temp : 1.0;
    await this.prefillBatch(promptIds);

    const generatedIds = [];
    let pos = promptIds.length;

    // first token after prefill
    let next = doSample ? await this.sampleToken(temp) : await this.argmaxLogits();
    generatedIds.push(next);
    if (opts.onToken) opts.onToken(next);
    this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([next]));

    while (generatedIds.length < maxNewTokens) {
      this._resetUni();
      const enc = this.dev.createCommandEncoder();
      this.embedFromBuf(enc);
      this.step(enc, 0, pos);
      this.dev.queue.submit([enc.finish()]);

      next = doSample ? await this.sampleToken(temp) : await this.argmaxLogits();
      generatedIds.push(next);
      if (opts.onToken) opts.onToken(next);
      this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([next]));
      pos += 1;
    }
    return generatedIds;
  }

  setupDebugCapture(T, K, rank, N) {
    this.debugCapture = true;
    this.debugT = T;
    this.debugK = K;
    this.debugRank = rank;
    this.debugN = N;
    this.debugStep = 0;
    this.debugCaptured = false;
    this.debugBufs = {
      xSeq: this._buf(T * K * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      dSeq: this._buf(T * rank * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      ySeq: this._buf(T * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      xBat: this._buf(T * K * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      dBat: this._buf(T * rank * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      yBat: this._buf(T * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    };
  }

  async readDebugCapture() {
    this.debugCapture = false;
    const bufs = this.debugBufs;
    if (!bufs) return null;
    await Promise.all([
      bufs.xSeq.mapAsync(GPUMapMode.READ),
      bufs.dSeq.mapAsync(GPUMapMode.READ),
      bufs.ySeq.mapAsync(GPUMapMode.READ),
      bufs.xBat.mapAsync(GPUMapMode.READ),
      bufs.dBat.mapAsync(GPUMapMode.READ),
      bufs.yBat.mapAsync(GPUMapMode.READ),
    ]);
    const res = {
      xSeq: new Float32Array(bufs.xSeq.getMappedRange()).slice(),
      dSeq: new Float32Array(bufs.dSeq.getMappedRange()).slice(),
      ySeq: new Float32Array(bufs.ySeq.getMappedRange()).slice(),
      xBat: new Float32Array(bufs.xBat.getMappedRange()).slice(),
      dBat: new Float32Array(bufs.dBat.getMappedRange()).slice(),
      yBat: new Float32Array(bufs.yBat.getMappedRange()).slice(),
    };
    bufs.xSeq.unmap();
    bufs.xSeq.destroy();
    bufs.dSeq.unmap();
    bufs.dSeq.destroy();
    bufs.ySeq.unmap();
    bufs.ySeq.destroy();
    bufs.xBat.unmap();
    bufs.xBat.destroy();
    bufs.dBat.unmap();
    bufs.dBat.destroy();
    bufs.yBat.unmap();
    bufs.yBat.destroy();
    this.debugBufs = null;
    return res;
  }
}

export class PagedAttentionManager {
  constructor(maxCtx, pageSize = 16) {
    this.pageSize = pageSize;
    this.maxCtx = maxCtx;
    this.maxBlocksPerSeq = Math.ceil(maxCtx / pageSize);
    this.freeBlocks = [];
    this.seqBlocks = new Map();
    const totalBlocks = this.maxBlocksPerSeq * 4;
    for (let i = 0; i < totalBlocks; i++) {
      this.freeBlocks.push(i);
    }
  }

  allocateSeq(seqId) {
    this.seqBlocks.set(seqId, []);
  }

  freeSeq(seqId) {
    const blocks = this.seqBlocks.get(seqId) || [];
    this.freeBlocks.push(...blocks);
    this.seqBlocks.delete(seqId);
  }

  ensureBlocks(seqId, numTokens) {
    const neededBlocks = Math.ceil(numTokens / this.pageSize);
    const blocks = this.seqBlocks.get(seqId);
    if (!blocks) throw new Error(`Sequence ${seqId} not allocated`);
    while (blocks.length < neededBlocks) {
      if (this.freeBlocks.length === 0) {
        const newBlock = blocks.length + 1000;
        this.freeBlocks.push(newBlock);
      }
      blocks.push(this.freeBlocks.pop());
    }
    return blocks;
  }

  getBlockTableArray(seqId) {
    const blocks = this.seqBlocks.get(seqId) || [];
    const arr = new Uint32Array(this.maxBlocksPerSeq);
    arr.set(blocks);
    return arr;
  }
}
