// Custom pure-WebGPU Qwen2.5 decode runtime. int8 weights (per-channel scale),
// f32 norms/biases, GPU-resident KV cache, runtime-swappable LoRA (A/B f32
// buffers consumed by the GEMV kernel). No tf.js → no per-op dispatch overhead.
//
// Correctness is validated against the tf.js forward (which == HuggingFace).
import { GEMV, GEMV4, LORA_A, RMSNORM, ROPE, ATTN_PARTIAL, ATTN_COMBINE, ADD, SILUMUL, EMBED, EMBED_BUF, ARGMAX,
  GEMM4, RMSNORM_T, ROPE_T, EMBED_T, ATTN_PREFILL } from './kernels.js';
import { createQwenSchema } from './model_schema.js';
import { streamSafetensors } from './safetensors_loader.js';
import { ModelUploader } from './model_uploader.js';
import { GPUBufferPool } from './buffer_pool.js';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export class QwenWGPU {
  // opts: { maxCtx, maxPrefillT } — context window + batched-prefill cap (default 8192 each;
  // raise toward the base model's limit, e.g. 32768, memory permitting — KV cache grows linearly).
  constructor(device, cfg, opts = {}) {
    this.dev = device; this.cfg = cfg; this.lora = null; this.bufs = {}; this.opts = opts;
    this.pool = new GPUBufferPool(device, { cacheBindGroups: opts.cacheBindGroups !== false });
    this._loraEpoch = 0;
  }

  _buf(size, usage = STORAGE) { return this.pool.buffer(size, usage); }
  _f32(arr, usage = STORAGE) { return this.pool.uploadF32(arr, usage); }
  _u32(arr) { return this.pool.uploadU32(arr, STORAGE); }
  _uni(arr) { return this.pool.dynamicUniform(arr, UNIFORM); }
  _staticUni(key, arr) { return this.pool.staticUniform(key, arr, UNIFORM); }
  _resetUni() { this.pool.resetUniforms(); }

  _pipe(code) {
    const m = this.dev.createShaderModule({ code });
    return this.dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
  }

  // `source` is a base URL string OR a reader { range, text } (e.g. hfReader/fileReader).
  async build(source, onProgress = () => {}) {
    const dev = this.dev, c = this.cfg;
    this.CHUNK = 128; this.MAXBATCH = 16;
    this.maxCtx = this.opts.maxCtx || 8192;                       // context window (KV cache length)
    this.maxPrefillT = Math.min(this.opts.maxPrefillT || 8192, this.maxCtx); // batched-prefill cap (<= ctx)
    this.pipes = { gemv: this._pipe(GEMV), loraA: this._pipe(LORA_A), rms: this._pipe(RMSNORM), rope: this._pipe(ROPE), attnP: this._pipe(ATTN_PARTIAL), attnC: this._pipe(ATTN_COMBINE), add: this._pipe(ADD), silu: this._pipe(SILUMUL), embed: this._pipe(EMBED), embedBuf: this._pipe(EMBED_BUF), argmax: this._pipe(ARGMAX), gemv4: this._pipe(GEMV4),
      gemm4: this._pipe(GEMM4), rmsT: this._pipe(RMSNORM_T), ropeT: this._pipe(ROPE_T), embedT: this._pipe(EMBED_T), attnPrefill: this._pipe(ATTN_PREFILL) };
    onProgress('streaming + quantizing weights', 0);
    this.schema = createQwenSchema(c);
    this.layers = this.schema.layers;
    this.q = {}; this.q4 = {};
    const uploader = new ModelUploader({
      schema: this.schema,
      q: this.q,
      q4: this.q4,
      bufs: this.bufs,
      uploadF32: (arr) => this._f32(arr),
      uploadU32: (arr) => this._u32(arr),
    });
    await streamSafetensors(source, {
      names: this.schema.expectedNames,
      onProgress,
      onTensor: async (tensor) => {
        uploader.visit(tensor);
        if (uploader.seen.size % 48 === 0) await new Promise(r => setTimeout(r, 0));
      },
    });
    uploader.finalize();
    // Context window (this.maxCtx) set above from opts; RoPE tables + KV cache sized to it.
    this._buildRope(this.maxCtx);
    // KV cache (f32) per layer
    this.kc = [], this.vc = [];
    const kvSize = c.numKVHeads * this.maxCtx * c.headDim * 4;
    for (let i = 0; i < c.numLayers; i++) { this.kc.push(this._buf(kvSize)); this.vc.push(this._buf(kvSize)); }
    // scratch buffers (reused each token)
    const H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    const NSPLITMAX = Math.ceil(this.maxCtx / this.CHUNK);
    this.s = {
      hidden: this._buf(H * 4), normed: this._buf(H * 4), q: this._buf(qd * 4), k: this._buf(kvd * 4), v: this._buf(kvd * 4),
      attn: this._buf(qd * 4), tmp: this._buf(Math.max(qd, I) * 4), tmp2: this._buf(I * 4), logits: this._buf(c.vocabSize * 4),
      dummy: this._buf(64), loraD: this._buf(256 * 4), amax: this._buf(4),
      pm: this._buf(c.numHeads * NSPLITMAX * 4), pz: this._buf(c.numHeads * NSPLITMAX * 4), po: this._buf(c.numHeads * NSPLITMAX * c.headDim * 4),
      idsBuf: this._buf(this.MAXBATCH * 4),
    };
    this.idsRead = this._buf(this.MAXBATCH * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    // prefill scratch is allocated lazily (sized to the actual prompt) — see _ensurePrefillScratch.
    this.sT = null; this.sTcap = 0;
    this._initStaticUniforms();
    onProgress('ready', 1);
    return this;
  }

  _buildRope(maxSeq) {
    const { headDim, ropeTheta } = this.cfg; const half = headDim / 2;
    const cos = new Float32Array(maxSeq * headDim), sin = new Float32Array(maxSeq * headDim);
    for (let p = 0; p < maxSeq; p++) for (let i = 0; i < half; i++) {
      const a = p / Math.pow(ropeTheta, (2 * i) / headDim); const cc = Math.cos(a), ss = Math.sin(a);
      cos[p * headDim + i] = cc; cos[p * headDim + half + i] = cc; sin[p * headDim + i] = ss; sin[p * headDim + half + i] = ss;
    }
    this.ropeCos = this._f32(cos); this.ropeSin = this._f32(sin); this._ropeRow = headDim * 4;
  }

  _initStaticUniforms() {
    const c = this.cfg;
    const rms = new ArrayBuffer(8); const rmsDv = new DataView(rms);
    rmsDv.setFloat32(0, c.hiddenSize, true); rmsDv.setFloat32(4, c.rmsNormEps, true);
    this.u = {
      rmsHidden: this._staticUni(`rms:${c.hiddenSize}:${c.rmsNormEps}`, new Uint8Array(rms)),
      addHidden: this._staticUni(`u32:${c.hiddenSize}`, new Uint32Array([c.hiddenSize])),
      siluIntermediate: this._staticUni(`u32:${c.intermediateSize}`, new Uint32Array([c.intermediateSize])),
      embedBuf: this._staticUni(`embedBuf:${c.hiddenSize}`, new Uint32Array([c.hiddenSize])),
      argmax: this._staticUni(`argmax:${c.vocabSize}`, new Uint32Array([c.vocabSize])),
    };
  }

  _gemvMeta(q, biasBuf, mod) {
    const gx = Math.min(q.N, 65535);
    const meta = new ArrayBuffer(32); const dv = new DataView(meta);
    dv.setUint32(0, q.K, true); dv.setUint32(4, q.N, true); dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true); dv.setUint32(16, mod ? 1 : 0, true); dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true);
    return { gx, gy: Math.ceil(q.N / gx), buf: this._staticUni(`gemv:${q.K}:${q.N}:${biasBuf ? 1 : 0}:${mod ? `${this._loraEpoch}:${mod.rank}:${mod.scale}` : 'base'}`, new Uint8Array(meta)) };
  }

  _gemv4Meta(q, biasBuf, mod) {
    const gx = Math.min(q.N, 65535);
    const meta = new ArrayBuffer(32); const dv = new DataView(meta);
    dv.setUint32(0, q.K, true); dv.setUint32(4, q.N, true); dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true); dv.setUint32(16, mod ? 1 : 0, true); dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true); dv.setUint32(28, q.gpr, true);
    return { gx, gy: Math.ceil(q.N / gx), buf: this._staticUni(`gemv4:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? `${this._loraEpoch}:${mod.rank}:${mod.scale}` : 'base'}`, new Uint8Array(meta)) };
  }

  setLora(adapter) { this.lora = adapter; this._loraEpoch++; this.pool.clearSensitiveBindGroups(); }   // {modules: {key:{A,B,rank,scale}}}  A:[K][rank], B:[rank][N] f32 GPUBuffers
  clearLora() { this.lora = null; this._loraEpoch++; this.pool.clearSensitiveBindGroups(); }

  _bg(pipe, buffers) {
    return this.pool.uncachedBindGroup(pipe, buffers);
  }
  _bgCached(pipe, buffers, key, opts) {
    return this.pool.cachedBindGroup(pipe, buffers, key, opts);
  }
  _dispatch(enc, pipe, bg, gx, gy=1, cat) {
    let ts;
    if (this.prof && this.prof.idx < this.prof.cap) { const i = this.prof.idx++; this.prof.cats.push(cat || 'misc'); ts = { querySet: this.prof.qs, beginningOfPassWriteIndex: 2*i, endOfPassWriteIndex: 2*i+1 }; }
    const p = enc.beginComputePass(ts ? { timestampWrites: ts } : undefined); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(gx, gy); p.end();
  }
  enableProf(cap = 700) { this.prof = { qs: this.dev.createQuerySet({ type: 'timestamp', count: cap * 2 }), cap, idx: 0, cats: [], resolve: this._buf(cap * 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC), read: this._buf(cap * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) }; }
  async profToken(id, pos) {
    this._resetUni(); this.prof.idx = 0; this.prof.cats = [];
    const enc = this.dev.createCommandEncoder(); this.embedRow(enc, id); this.step(enc, id, pos);
    const n = this.prof.idx; enc.resolveQuerySet(this.prof.qs, 0, n * 2, this.prof.resolve, 0);
    enc.copyBufferToBuffer(this.prof.resolve, 0, this.prof.read, 0, n * 16);
    this.dev.queue.submit([enc.finish()]); await this.prof.read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.prof.read.getMappedRange()); const sums = {};
    for (let i = 0; i < n; i++) { const us = Number(t[2*i+1] - t[2*i]) / 1000; const c = this.prof.cats[i]; sums[c] = (sums[c] || 0) + us; }
    this.prof.read.unmap(); return sums;
  }

  // y = int8-GEMV(x, q) [+bias] [+lora]. q={w,scale,N,K}. moduleKey for LoRA lookup.
  gemv(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) { // d = x@A  (rank outputs)
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      const bgA = this._bgCached(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, uA], `loraA:${moduleKey}:${this._loraEpoch}`, { sensitive: true });
      this._dispatch(enc, this.pipes.loraA, bgA, mod.rank, 1, 'loraA');
    }
    const meta = this._gemvMeta(q, biasBuf, mod);
    const key = `gemv:${moduleKey || 'base'}:${q.K}:${q.N}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(this.pipes.gemv, [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf, meta.buf], key, { sensitive: !!mod });
    this._dispatch(enc, this.pipes.gemv, bg, meta.gx, meta.gy, `gemv:${q.N}x${q.K}`);
  }

  gemv4(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) {
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      this._dispatch(enc, this.pipes.loraA, this._bgCached(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, uA], `loraA:${moduleKey}:${this._loraEpoch}`, { sensitive: true }), mod.rank, 1, 'loraA');
    }
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4:${moduleKey || 'base'}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(this.pipes.gemv4, [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf, meta.buf], key, { sensitive: !!mod });
    this._dispatch(enc, this.pipes.gemv4, bg, meta.gx, meta.gy, `g4:${q.N}x${q.K}`);
  }
  rms(enc, xBuf, gBuf, yBuf, K) {
    let u = this.u?.rmsHidden;
    if (!u || K !== this.cfg.hiddenSize) {
      const raw = new ArrayBuffer(8); const dv = new DataView(raw); dv.setFloat32(0, K, true); dv.setFloat32(4, this.cfg.rmsNormEps, true);
      u = this._staticUni(`rms:${K}:${this.cfg.rmsNormEps}`, new Uint8Array(raw));
    }
    this._dispatch(enc, this.pipes.rms, this._bgCached(this.pipes.rms, [xBuf, gBuf, yBuf, u], `rms:${K}`), 1, 1, 'rms');
  }
  rope(enc, xBuf, pos, nHeads) {
    this._dispatch(enc, this.pipes.rope, this._bg(this.pipes.rope, [xBuf, this.ropeCos, this.ropeSin, this._uni(new Uint32Array([nHeads, this.cfg.headDim, pos]))]), Math.ceil(nHeads*(this.cfg.headDim/2)/256), 1, 'rope');
  }
  attn(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg, S = this.s; const nsplit = Math.ceil(ctx / this.CHUNK);
    // pass 1: per (head, ctx-chunk) partial softmax → pm/pz/po (nHeads*nsplit workgroups)
    const bgP = this._bg(this.pipes.attnP, [qBuf, kc, vc, S.pm, S.pz, S.po,
      this._uni(new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim])), this._uni(new Uint32Array([nsplit, this.CHUNK]))]);
    this._dispatch(enc, this.pipes.attnP, bgP, c.numHeads, nsplit, 'attnP');
    // pass 2: combine splits per head → o
    const bgC = this._bg(this.pipes.attnC, [S.pm, S.pz, S.po, oBuf, this._uni(new Uint32Array([c.numHeads, c.headDim, nsplit, 0]))]);
    this._dispatch(enc, this.pipes.attnC, bgC, c.numHeads, 1, 'attnC');
  }

  // Decode one token at absolute position `pos`. Writes logits to s.logits. Returns nothing.
  step(enc, tokenId, pos) {
    const c = this.cfg, S = this.s, hd = c.headDim, kvd = c.numKVHeads * hd;
    // embed: dequant row tokenId of embed_tokens int8 -> hidden (use gemv? no; copy+scale). Use a tiny loraA-style? Simplest: a gemv with a one-hot is overkill.
    // We do embed lookup on CPU-uploaded row: handled by caller via this.embedRow(tokenId) into S.hidden.
    for (let i = 0; i < c.numLayers; i++) {
      const p = `model.layers.${i}`;
      this.rms(enc, S.hidden, this.bufs[`${p}.input_layernorm.weight`], S.normed, c.hiddenSize);
      this.gemv4(enc, S.normed, this.q4[`${p}.self_attn.q_proj.weight`], S.q, this.bufs[`${p}.self_attn.q_proj.bias`], `layers.${i}.self_attn.q_proj`);
      this.gemv4(enc, S.normed, this.q4[`${p}.self_attn.k_proj.weight`], S.k, this.bufs[`${p}.self_attn.k_proj.bias`], `layers.${i}.self_attn.k_proj`);
      this.gemv4(enc, S.normed, this.q4[`${p}.self_attn.v_proj.weight`], S.v, this.bufs[`${p}.self_attn.v_proj.bias`], `layers.${i}.self_attn.v_proj`);
      this.rope(enc, S.q, pos, c.numHeads); this.rope(enc, S.k, pos, c.numKVHeads);
      // append k,v to cache at position pos
      enc.copyBufferToBuffer(S.k, 0, this.kc[i], pos * kvd * 4, kvd * 4);
      enc.copyBufferToBuffer(S.v, 0, this.vc[i], pos * kvd * 4, kvd * 4);
      this.attn(enc, S.q, this.kc[i], this.vc[i], S.attn, pos + 1);
      this.gemv4(enc, S.attn, this.q4[`${p}.self_attn.o_proj.weight`], S.tmp, null, `layers.${i}.self_attn.o_proj`);
      this._addInto(enc, S.hidden, S.tmp, c.hiddenSize);             // residual
      this.rms(enc, S.hidden, this.bufs[`${p}.post_attention_layernorm.weight`], S.normed, c.hiddenSize);
      this.gemv4(enc, S.normed, this.q4[`${p}.mlp.gate_proj.weight`], S.tmp, null, `layers.${i}.mlp.gate_proj`);
      this.gemv4(enc, S.normed, this.q4[`${p}.mlp.up_proj.weight`], S.tmp2, null, `layers.${i}.mlp.up_proj`);
      this._siluMul(enc, S.tmp, S.tmp2, c.intermediateSize);          // tmp = silu(gate)*up
      this.gemv4(enc, S.tmp, this.q4[`${p}.mlp.down_proj.weight`], S.normed, null, `layers.${i}.mlp.down_proj`);
      this._addInto(enc, S.hidden, S.normed, c.hiddenSize);
    }
    this.rms(enc, S.hidden, this.bufs['model.norm.weight'], S.normed, c.hiddenSize);
    this.gemv(enc, S.normed, this.q['model.embed_tokens.weight'], S.logits, null, null); // lm_head (tied)
  }

  _addInto(enc, yBuf, aBuf, n) {
    const u = n === this.cfg.hiddenSize ? this.u.addHidden : this._uni(new Uint32Array([n]));
    const bg = n === this.cfg.hiddenSize ? this._bgCached(this.pipes.add, [aBuf, yBuf, u], `add:${n}`) : this._bg(this.pipes.add, [aBuf, yBuf, u]);
    this._dispatch(enc, this.pipes.add, bg, Math.min(Math.ceil(n/256), 65535), 1, 'add');
  }
  _siluMul(enc, gateBuf, upBuf, n) {
    const u = n === this.cfg.intermediateSize ? this.u.siluIntermediate : this._uni(new Uint32Array([n]));
    const bg = n === this.cfg.intermediateSize ? this._bgCached(this.pipes.silu, [gateBuf, upBuf, u], `silu:${n}`) : this._bg(this.pipes.silu, [gateBuf, upBuf, u]);
    this._dispatch(enc, this.pipes.silu, bg, Math.min(Math.ceil(n/256), 65535), 1, 'silu');
  }
  embedRow(enc, id) { const e = this.q['model.embed_tokens.weight']; this._dispatch(enc, this.pipes.embed, this._bg(this.pipes.embed, [e.w, e.scale, this.s.hidden, this._uni(new Uint32Array([id, this.cfg.hiddenSize]))]), Math.ceil(this.cfg.hiddenSize/256), 1, 'embed'); }
  async argmaxLogits() {
    const enc = this.dev.createCommandEncoder();
    this._dispatch(enc, this.pipes.argmax, this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax, this.u.argmax], 'argmax'), 1);
    const rb = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(this.s.amax, 0, rb, 0, 4); this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const id = new Uint32Array(rb.getMappedRange())[0]; rb.unmap(); rb.destroy(); return id;
  }
  // Run one token end-to-end (embed + step) and submit.
  token(id, pos) { this._resetUni(); const enc = this.dev.createCommandEncoder(); this.embedRow(enc, id); this.step(enc, id, pos); this.dev.queue.submit([enc.finish()]); }

  // embed the token id held in s.amax (GPU-resident, from a prior argmax)
  embedFromBuf(enc) { const e = this.q['model.embed_tokens.weight']; this._dispatch(enc, this.pipes.embedBuf, this._bgCached(this.pipes.embedBuf, [e.w, e.scale, this.s.hidden, this.s.amax, this.u.embedBuf], 'embedBuf'), Math.ceil(this.cfg.hiddenSize/256), 1, 'embed'); }
  // argmax(logits) -> s.amax, within the given encoder (no submit/readback)
  argmaxInto(enc) { this._dispatch(enc, this.pipes.argmax, this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax, this.u.argmax], 'argmax'), 1, 1, 'argmax'); }

  // GPU-resident batched greedy decode: chains embed->step->argmax->embed for K
  // tokens in ONE submit (no per-token CPU sync), reads back K ids once. Assumes
  // s.amax holds the current token to embed. Returns the K generated ids.
  async decodeBatch(startPos, K) {
    K = Math.min(K, this.maxCtx - startPos);   // never write KV past the cache
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
    const ids = Array.from(new Uint32Array(this.idsRead.getMappedRange(), 0, K)); this.idsRead.unmap();
    return ids;
  }

  // ---- PREFILL (T>1): process the whole prompt at once via tiled GEMM. Base model
  // only (no LoRA — caller falls back to sequential token() when an adapter is active).
  gemm4(enc, aBuf, q, yBuf, T, biasBuf) {
    const meta = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4, [aBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf, this._uni(meta)]);
    this._dispatch(enc, this.pipes.gemm4, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), 'gemm4');
  }
  rmsT(enc, xBuf, gBuf, yBuf, T, K) {
    const u = new ArrayBuffer(8); const dv = new DataView(u); dv.setFloat32(0, K, true); dv.setFloat32(4, this.cfg.rmsNormEps, true);
    this._dispatch(enc, this.pipes.rmsT, this._bg(this.pipes.rmsT, [xBuf, gBuf, yBuf, this._uni(new Uint8Array(u))]), T, 1, 'rmsT');
  }
  ropeT(enc, xBuf, T, nHeads) {
    const hd = this.cfg.headDim;
    this._dispatch(enc, this.pipes.ropeT, this._bg(this.pipes.ropeT, [xBuf, this.ropeCos, this.ropeSin, this._uni(new Uint32Array([nHeads, hd, T, 0]))]), Math.ceil(T * nHeads * (hd / 2) / 256), 1, 'ropeT');
  }
  attnPrefill(enc, qBuf, kc, vc, oBuf, T) {
    const c = this.cfg;
    this._dispatch(enc, this.pipes.attnPrefill, this._bg(this.pipes.attnPrefill, [qBuf, kc, vc, oBuf, this._uni(new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T]))]), c.numHeads, T, 'attnPrefill');
  }

  // (re)allocate prefill scratch sized to T (grows as needed; only paid when prefilling).
  _ensurePrefillScratch(T) {
    if (this.sTcap >= T) return;
    if (this.sT) for (const k in this.sT) this.sT[k].destroy();
    const c = this.cfg, H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    this.sT = {
      hidden: this._buf(T * H * 4), normed: this._buf(T * H * 4), q: this._buf(T * qd * 4), k: this._buf(T * kvd * 4), v: this._buf(T * kvd * 4),
      attn: this._buf(T * qd * 4), tmp: this._buf(T * I * 4), tmp2: this._buf(T * I * 4), ids: this._buf(T * 4),
    };
    this.sTcap = T;
  }

  // Prefill the prompt (positions 0..T-1). Leaves last-row logits in s.logits and the
  // KV cache populated, so decode continues from pos=T. T must be <= maxPrefillT and no LoRA.
  prefillBatch(ids) {
    const c = this.cfg, S = this.s, T = ids.length, hd = c.headDim, kvd = c.numKVHeads * hd, H = c.hiddenSize;
    if (T > this.maxPrefillT) throw new Error(`prompt ${T} > maxPrefillT ${this.maxPrefillT}`);
    if (T > this.maxCtx) throw new Error(`prompt ${T} > maxCtx ${this.maxCtx}`);
    this._ensurePrefillScratch(T); const ST = this.sT;
    this._resetUni();
    this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(ids));
    const enc = this.dev.createCommandEncoder();
    const e = this.q['model.embed_tokens.weight'];
    this._dispatch(enc, this.pipes.embedT, this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids, this._uni(new Uint32Array([T, H]))]), Math.min(Math.ceil(T * H / 256), 65535), 1, 'embedT');
    for (let i = 0; i < c.numLayers; i++) {
      const p = `model.layers.${i}`;
      this.rmsT(enc, ST.hidden, this.bufs[`${p}.input_layernorm.weight`], ST.normed, T, H);
      this.gemm4(enc, ST.normed, this.q4[`${p}.self_attn.q_proj.weight`], ST.q, T, this.bufs[`${p}.self_attn.q_proj.bias`]);
      this.gemm4(enc, ST.normed, this.q4[`${p}.self_attn.k_proj.weight`], ST.k, T, this.bufs[`${p}.self_attn.k_proj.bias`]);
      this.gemm4(enc, ST.normed, this.q4[`${p}.self_attn.v_proj.weight`], ST.v, T, this.bufs[`${p}.self_attn.v_proj.bias`]);
      this.ropeT(enc, ST.q, T, c.numHeads); this.ropeT(enc, ST.k, T, c.numKVHeads);
      enc.copyBufferToBuffer(ST.k, 0, this.kc[i], 0, T * kvd * 4);
      enc.copyBufferToBuffer(ST.v, 0, this.vc[i], 0, T * kvd * 4);
      this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T);
      this.gemm4(enc, ST.attn, this.q4[`${p}.self_attn.o_proj.weight`], ST.tmp, T, null);
      this._addInto(enc, ST.hidden, ST.tmp, T * H);
      this.rmsT(enc, ST.hidden, this.bufs[`${p}.post_attention_layernorm.weight`], ST.normed, T, H);
      this.gemm4(enc, ST.normed, this.q4[`${p}.mlp.gate_proj.weight`], ST.tmp, T, null);
      this.gemm4(enc, ST.normed, this.q4[`${p}.mlp.up_proj.weight`], ST.tmp2, T, null);
      this._siluMul(enc, ST.tmp, ST.tmp2, T * c.intermediateSize);
      this.gemm4(enc, ST.tmp, this.q4[`${p}.mlp.down_proj.weight`], ST.normed, T, null);
      this._addInto(enc, ST.hidden, ST.normed, T * H);
    }
    // last row -> final norm -> lm_head (reuse decode single-row kernels)
    enc.copyBufferToBuffer(ST.hidden, (T - 1) * H * 4, S.hidden, 0, H * 4);
    this.rms(enc, S.hidden, this.bufs['model.norm.weight'], S.normed, H);
    this.gemv(enc, S.normed, this.q['model.embed_tokens.weight'], S.logits, null, null);
    this.dev.queue.submit([enc.finish()]);
  }

}
