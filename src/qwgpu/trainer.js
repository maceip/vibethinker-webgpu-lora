/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

// QwenLoraTrainer: in-browser LoRA fine-tuning on top of the frozen-int4 inference
// engine (QwenWGPU). Implements gradient-checkpointed full-network backprop where
// only the LoRA A/B matrices receive gradients; the base int4 projections and int8
// tied embeddings stay frozen and are dequantized on the fly in the backward kernels.
//
// Design:
//   - Forward stores only per-layer input hidden states (checkpoints). The backward
//     pass recomputes each layer's internals from its checkpoint, then backprops.
//   - All math is f32 (master A/B, grads, Adam moments). The base stays int4/int8.
//   - The adapter object is the SAME one used for inference hot-swap; AdamW mutates
//     mod.A / mod.B in place and calls rt.invalidateLora() so generation stays valid.

import {
  GEMM_DX_INT4,
  LORA_DD,
  LORA_GRAD_A,
  LORA_GRAD_B,
  LORA_DX_ADD,
  RMSNORM_BWD_T,
  SWIGLU_BWD,
  ROPE_BWD_T,
  ATTN_BWD_STATS,
  ATTN_BWD_DQ,
  ATTN_BWD_DKV,
  LOGITS_GEMM_I8,
  CE_SOFTMAX_GRAD,
  DHIDDEN_FROM_DLOGITS_I8,
  ADAMW_STEP,
  SUMSQ,
} from './backward_kernels.js';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const READBACK = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;

// Default training modules (PEFT/MLX-style loraKeys are matched against these suffixes).
const ALL_PROJ = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];

// Build a fresh, trainable LoRA adapter sized to the model (A ~ N(0,stddev), B = 0,
// matching PEFT init so the initial delta is zero). Buffers get COPY_SRC so the
// trainer can read them back for export. Returns { name, modules } for setLora/attach.
//   A layout [rank][K] (transposed, matches loraABatch); B layout [rank][N].
export function createTrainableAdapter(rt, opts = {}) {
  const rank = Math.max(1, Math.floor(opts.rank ?? 16));
  const alpha = opts.alpha ?? rank * 2;
  const scale = opts.scale ?? alpha / rank;
  const targets = opts.targetModules ?? ALL_PROJ;
  const stddev = opts.stddev ?? 1 / Math.sqrt(rank);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const gauss = () => {
    // Box–Muller; deterministic enough for init.
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const modules = {};
  for (const L of rt.plan.layers) {
    for (const name of ALL_PROJ) {
      if (!targets.includes(name)) continue;
      const part = L[name];
      const q4 = rt.q4[part.weight];
      const K = q4.K,
        N = q4.N;
      const Aarr = new Float32Array(rank * K);
      for (let i = 0; i < Aarr.length; i++) Aarr[i] = gauss() * stddev;
      const Barr = new Float32Array(rank * N); // zeros
      const A = rt.dev.createBuffer({ size: Aarr.byteLength, usage });
      const B = rt.dev.createBuffer({ size: Barr.byteLength, usage });
      rt.dev.queue.writeBuffer(A, 0, Aarr);
      rt.dev.queue.writeBuffer(B, 0, Barr);
      modules[part.loraKey] = { A, B, rank, scale, inDim: K, outDim: N };
    }
  }
  return { name: opts.name || 'trainable', modules };
}

export class QwenLoraTrainer {
  // rt: a built QwenWGPU. opts: see _normalizeOpts.
  constructor(rt, opts = {}) {
    this.rt = rt;
    this.dev = rt.dev;
    this.cfg = rt.cfg;
    this.opts = this._normalizeOpts(opts);
    this.step = 0; // optimizer steps taken
    this._microInWindow = 0; // micro-batches accumulated since last optimizer step
    this.scratchT = 0; // current allocated sequence capacity
    this._buildPipes();
  }

  _normalizeOpts(o) {
    return {
      lr: o.lr ?? 1e-4,
      beta1: o.beta1 ?? 0.9,
      beta2: o.beta2 ?? 0.999,
      eps: o.eps ?? 1e-8,
      weightDecay: o.weightDecay ?? 0.0,
      maxGradNorm: o.maxGradNorm ?? 1.0,
      gradAccumSteps: Math.max(1, Math.floor(o.gradAccumSteps ?? 1)),
      lmHeadBlock: Math.max(1, Math.floor(o.lmHeadBlock ?? 128)),
      maxTrainSeq: Math.max(1, Math.floor(o.maxTrainSeq ?? 512)),
      warmupSteps: Math.max(0, Math.floor(o.warmupSteps ?? 0)),
      totalSteps: o.totalSteps ?? 0, // for cosine decay; 0 disables decay
      minLrRatio: o.minLrRatio ?? 0.1,
      targetModules: o.targetModules ?? ALL_PROJ,
    };
  }

  _buildPipes() {
    const rt = this.rt;
    this.p = {
      dx4: rt._pipe(GEMM_DX_INT4, 'bwd_dx4'),
      dd: rt._pipe(LORA_DD, 'bwd_lora_dd'),
      gradA: rt._pipe(LORA_GRAD_A, 'bwd_lora_dA'),
      gradB: rt._pipe(LORA_GRAD_B, 'bwd_lora_dB'),
      dxAdd: rt._pipe(LORA_DX_ADD, 'bwd_lora_dx'),
      rmsBwd: rt._pipe(RMSNORM_BWD_T, 'bwd_rms'),
      swiglu: rt._pipe(SWIGLU_BWD, 'bwd_swiglu'),
      ropeBwd: rt._pipe(ROPE_BWD_T, 'bwd_rope'),
      attnStats: rt._pipe(ATTN_BWD_STATS, 'bwd_attn_stats'),
      attnDq: rt._pipe(ATTN_BWD_DQ, 'bwd_attn_dq'),
      attnDkv: rt._pipe(ATTN_BWD_DKV, 'bwd_attn_dkv'),
      logits: rt._pipe(LOGITS_GEMM_I8, 'bwd_logits'),
      ceGrad: rt._pipe(CE_SOFTMAX_GRAD, 'bwd_ce'),
      dHidden: rt._pipe(DHIDDEN_FROM_DLOGITS_I8, 'bwd_dhidden'),
      adamw: rt._pipe(ADAMW_STEP, 'adamw'),
      sumsq: rt._pipe(SUMSQ, 'sumsq'),
    };
  }

  // ---- adapter attach: build per-module grad + Adam moment state ----
  // The adapter must already be uploaded (loadLoraAdapterGPU) and set on rt.
  attach(adapter) {
    if (!adapter || !adapter.modules) throw new Error('trainer.attach: adapter with modules required');
    this.adapter = adapter;
    this.rt.setLora(adapter);
    const rt = this.rt;
    // Map every projection loraKey -> its int4 weight descriptor (for backward dX dims).
    const byKey = new Map();
    for (const L of rt.plan.layers) {
      for (const name of ALL_PROJ) {
        const part = L[name];
        byKey.set(part.loraKey, { part, kind: name, q4: rt.q4[part.weight] });
      }
    }
    this.state = {};
    let maxRank = 1;
    for (const key of Object.keys(adapter.modules)) {
      const mod = adapter.modules[key];
      const info = byKey.get(key);
      if (!info) continue; // adapter module that doesn't map to a known projection
      const kind = info.kind.replace(/_proj$/, '');
      if (!this.opts.targetModules.includes(kind)) continue;
      const K = info.q4.K, N = info.q4.N, rank = mod.rank;
      maxRank = Math.max(maxRank, rank);
      this.state[key] = {
        mod,
        q4: info.q4,
        K,
        N,
        rank,
        scale: mod.scale,
        dA: rt._buf(rank * K * 4),
        dB: rt._buf(rank * N * 4),
        mA: rt._buf(rank * K * 4),
        vA: rt._buf(rank * K * 4),
        mB: rt._buf(rank * N * 4),
        vB: rt._buf(rank * N * 4),
      };
    }
    this.maxRank = maxRank;
    this.trainedKeys = Object.keys(this.state);
    if (!this.trainedKeys.length) throw new Error('trainer.attach: no trainable modules matched targetModules');
    this._zeroAdamMoments();
    this.zeroGrads();
    return this;
  }

  _zeroAdamMoments() {
    const enc = this.dev.createCommandEncoder();
    for (const k of this.trainedKeys) {
      const st = this.state[k];
      enc.clearBuffer(st.mA);
      enc.clearBuffer(st.vA);
      enc.clearBuffer(st.mB);
      enc.clearBuffer(st.vB);
    }
    this.dev.queue.submit([enc.finish()]);
  }

  zeroGrads() {
    const enc = this.dev.createCommandEncoder();
    for (const k of this.trainedKeys) {
      enc.clearBuffer(this.state[k].dA);
      enc.clearBuffer(this.state[k].dB);
    }
    this.dev.queue.submit([enc.finish()]);
    this._microInWindow = 0;
  }

  // ---- activation/gradient scratch sized to the sequence ----
  _ensureScratch(T) {
    if (this.scratchT >= T && this.s) return;
    if (this.s) for (const k in this.s) this.s[k].destroy?.();
    if (this.ckpt) for (const c of this.ckpt) c.destroy?.();
    this.lossRead?.destroy?.();
    this.normRead?.destroy?.();
    const c = this.cfg;
    const H = c.hiddenSize,
      qd = c.numHeads * c.headDim,
      kvd = c.numKVHeads * c.headDim,
      I = c.intermediateSize,
      nH = c.numHeads,
      R = this.maxRank,
      lmB = this.opts.lmHeadBlock,
      V = c.vocabSize;
    const b = (n) => this.rt._buf(n * 4);
    this.ckpt = [];
    for (let i = 0; i <= c.numLayers; i++) this.ckpt.push(b(T * H));
    this.s = {
      hid: b(T * H),
      normed1: b(T * H),
      normed2: b(T * H),
      normedF: b(T * H),
      q: b(T * qd),
      k: b(T * kvd),
      v: b(T * kvd),
      attn: b(T * qd),
      hmid: b(T * H),
      gate: b(T * I),
      up: b(T * I),
      swig: b(T * I),
      dHidden: b(T * H),
      dnorm: b(T * H),
      dtmp: b(T * H),
      dhmid: b(T * H),
      dq: b(T * qd),
      dk: b(T * kvd),
      dv: b(T * kvd),
      dob: b(T * qd),
      dgate: b(T * I),
      dup: b(T * I),
      dswig: b(T * I),
      dD: b(T * R),
      Dmat: b(T * R),
      lse: b(nH * T),
      delta: b(nH * T),
      logits: b(lmB * V),
      loss: b(T),
      targets: this.rt._buf(T * 4),
      mask: b(T),
      normBuf: b(1),
    };
    this.lossRead = this.rt._buf(T * 4, READBACK);
    this.normRead = this.rt._buf(4, READBACK);
    this.scratchT = T;
  }

  // ---- small dispatch helpers ----
  _grid1d(n) {
    return Math.min(Math.ceil(n / 256), 65535);
  }
  _disp(enc, pipe, buffers, gx, gy, imm, cat) {
    const bg = this.rt._bg(pipe, buffers);
    this.rt._dispatch(enc, pipe, bg, gx, gy, cat || 'train', imm);
  }
  _u32(arr) {
    return new Uint32Array(arr);
  }
  _meta(u32parts, f32parts = {}) {
    // build a 48-byte immediate with u32 at given word indices and f32 at others
    const buf = new ArrayBuffer(48);
    const dv = new DataView(buf);
    for (const [i, v] of u32parts) dv.setUint32(i * 4, v >>> 0, true);
    for (const [i, v] of Object.entries(f32parts)) dv.setFloat32(Number(i) * 4, v, true);
    return new Uint8Array(buf);
  }

  // ---- forward with checkpoints (LoRA-modified, f32) ----
  _layerForward(enc, L, hid, T) {
    const rt = this.rt,
      c = this.cfg,
      s = this.s;
    const H = c.hiddenSize;
    // input norm -> q,k,v -> rope -> attn -> +o_proj
    rt.rmsT(enc, hid, rt.bufs[L.inputNorm], s.normed1, T, H);
    rt.gemm4(enc, s.normed1, rt.q4[L.q.weight], s.q, T, rt.bufs[L.q.bias], L.q.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.k.weight], s.k, T, rt.bufs[L.k.bias], L.k.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.v.weight], s.v, T, rt.bufs[L.v.bias], L.v.loraKey);
    rt.ropeT(enc, s.q, T, c.numHeads);
    rt.ropeT(enc, s.k, T, c.numKVHeads);
    rt.attnPrefill(enc, s.q, s.k, s.v, s.attn, T, 0, T);
    rt.gemm4AddT(enc, s.attn, rt.q4[L.o.weight], hid, T, null, L.o.loraKey);
    // post-attn norm -> gate/up -> swiglu -> +down_proj
    rt.rmsT(enc, hid, rt.bufs[L.postAttentionNorm], s.normed2, T, H);
    rt.gemm4(enc, s.normed2, rt.q4[L.gate.weight], s.gate, T, null, L.gate.loraKey);
    rt.gemm4(enc, s.normed2, rt.q4[L.up.weight], s.up, T, null, L.up.loraKey);
    enc.copyBufferToBuffer(s.gate, 0, s.swig, 0, T * c.intermediateSize * 4);
    rt._siluMul(enc, s.swig, s.up, T * c.intermediateSize);
    rt.gemm4AddT(enc, s.swig, rt.q4[L.down.weight], hid, T, null, L.down.loraKey);
  }

  _forward(enc, ids, T) {
    const rt = this.rt,
      c = this.cfg,
      s = this.s,
      H = c.hiddenSize;
    rt._ensurePrefillScratch(T, this.maxRank);
    rt._resetUni();
    // embed into ckpt[0] and hid; ids go to the runtime's prefill ids buffer.
    const e = rt.q[rt.plan.embed.name];
    this.dev.queue.writeBuffer(rt.sT.ids, 0, new Uint32Array(ids));
    rt._dispatch(
      enc,
      rt.pipes.embedT,
      rt._bg(rt.pipes.embedT, [e.w, e.scale, this.ckpt[0], rt.sT.ids]),
      Math.min(Math.ceil((T * H) / 256), 65535),
      1,
      'embedT',
      this._u32([T, H, 0, 0]),
    );
    enc.copyBufferToBuffer(this.ckpt[0], 0, s.hid, 0, T * H * 4);
    for (let i = 0; i < c.numLayers; i++) {
      this._layerForward(enc, rt.plan.layers[i], s.hid, T);
      enc.copyBufferToBuffer(s.hid, 0, this.ckpt[i + 1], 0, T * H * 4);
    }
  }

  // recompute one layer's forward internals (from its checkpoint) into scratch, also
  // producing hmid (= ckpt + attnProj) which the backward needs as the post-attn input.
  _recomputeLayer(enc, L, T) {
    const rt = this.rt,
      c = this.cfg,
      s = this.s,
      H = c.hiddenSize,
      idx = L.index;
    rt.rmsT(enc, this.ckpt[idx], rt.bufs[L.inputNorm], s.normed1, T, H);
    rt.gemm4(enc, s.normed1, rt.q4[L.q.weight], s.q, T, rt.bufs[L.q.bias], L.q.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.k.weight], s.k, T, rt.bufs[L.k.bias], L.k.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.v.weight], s.v, T, rt.bufs[L.v.bias], L.v.loraKey);
    rt.ropeT(enc, s.q, T, c.numHeads);
    rt.ropeT(enc, s.k, T, c.numKVHeads);
    rt.attnPrefill(enc, s.q, s.k, s.v, s.attn, T, 0, T);
    // hmid = ckpt[idx] + o_proj(attn)
    enc.copyBufferToBuffer(this.ckpt[idx], 0, s.hmid, 0, T * H * 4);
    rt.gemm4AddT(enc, s.attn, rt.q4[L.o.weight], s.hmid, T, null, L.o.loraKey);
    rt.rmsT(enc, s.hmid, rt.bufs[L.postAttentionNorm], s.normed2, T, H);
    rt.gemm4(enc, s.normed2, rt.q4[L.gate.weight], s.gate, T, null, L.gate.loraKey);
    rt.gemm4(enc, s.normed2, rt.q4[L.up.weight], s.up, T, null, L.up.loraKey);
    enc.copyBufferToBuffer(s.gate, 0, s.swig, 0, T * c.intermediateSize * 4);
    rt._siluMul(enc, s.swig, s.up, T * c.intermediateSize);
  }

  // ---- LoRA + base projection backward ----
  // dY [T][N] -> accumulate into dXbuf [T][K] (base + LoRA), plus dA/dB grads.
  _projBackward(enc, key, Xbuf, dYbuf, dXbuf, T) {
    const st = this.state[key];
    if (!st) {
      // module not trained: still need base dX so upstream gradient flows
      this._dispatch_dx4(enc, dYbuf, st, dXbuf, T, key);
      return;
    }
    const { K, N, rank, scale, q4, dA, dB } = st;
    const s = this.s;
    // base dX += dY @ deq(W)
    this._disp(
      enc,
      this.p.dx4,
      [dYbuf, q4.w, q4.scale, dXbuf],
      this._grid1d(T * K),
      1,
      this._meta([[0, T], [1, N], [2, K], [3, q4.gpr]]),
      'dx4',
    );
    // dD = scale * dY @ B^T   (one workgroup per (t,r))
    this._disp(
      enc,
      this.p.dd,
      [dYbuf, st.mod.B, s.dD],
      T * rank,
      1,
      this._meta([[0, T], [1, N], [2, rank]], { 4: scale }),
      'dd',
    );
    // dA += dD^T @ X
    this._disp(
      enc,
      this.p.gradA,
      [s.dD, Xbuf, dA],
      this._grid1d(rank * K),
      1,
      this._meta([[0, T], [1, K], [2, rank]]),
      'gradA',
    );
    // D = X @ A (recompute, no scale) for dB
    this._disp(
      enc,
      this.rt.pipes.loraABatch,
      [Xbuf, st.mod.A, s.Dmat],
      rank,
      T,
      this._u32([K, rank, T, 0]),
      'loraABatch',
    );
    // dB += scale * D^T @ dY
    this._disp(
      enc,
      this.p.gradB,
      [s.Dmat, dYbuf, dB],
      this._grid1d(rank * N),
      1,
      this._meta([[0, T], [1, N], [2, rank]], { 4: scale }),
      'gradB',
    );
    // dX += dD @ A
    this._disp(
      enc,
      this.p.dxAdd,
      [s.dD, st.mod.A, dXbuf],
      this._grid1d(T * K),
      1,
      this._meta([[0, T], [1, K], [2, rank]]),
      'dxAdd',
    );
  }

  _dispatch_dx4(enc, dYbuf, st, dXbuf, T, key) {
    // base-only dX for an untrained module (rare). Look up dims from plan.
    const info = this._infoForKey(key);
    const q4 = info.q4;
    this._disp(
      enc,
      this.p.dx4,
      [dYbuf, q4.w, q4.scale, dXbuf],
      this._grid1d(T * q4.K),
      1,
      this._meta([[0, T], [1, q4.N], [2, q4.K], [3, q4.gpr]]),
      'dx4',
    );
  }

  _infoForKey(key) {
    for (const L of this.rt.plan.layers)
      for (const name of ALL_PROJ) if (L[name].loraKey === key) return { q4: this.rt.q4[L[name].weight] };
    throw new Error(`unknown loraKey ${key}`);
  }

  _rmsBwd(enc, xBuf, gBuf, dyBuf, dxBuf, T) {
    const c = this.cfg;
    this._disp(
      enc,
      this.p.rmsBwd,
      [xBuf, gBuf, dyBuf, dxBuf],
      T,
      1,
      new Float32Array([c.hiddenSize, c.rmsNormEps]),
      'rmsBwd',
    );
  }

  // ---- full backward for one micro-batch; accumulates grads, returns nothing ----
  _backward(enc, T, numActive) {
    const rt = this.rt,
      c = this.cfg,
      s = this.s,
      H = c.hiddenSize,
      qd = c.numHeads * c.headDim,
      kvd = c.numKVHeads * c.headDim,
      I = c.intermediateSize,
      V = c.vocabSize;

    // final norm + LM head backward (streamed over token blocks)
    rt.rmsT(enc, this.ckpt[c.numLayers], rt.bufs[rt.plan.finalNorm.name], s.normedF, T, H);
    enc.clearBuffer(s.dnorm); // dNormedF accumulator
    const e = rt.q[rt.plan.embed.name];
    const lossScale = 1.0 / Math.max(1, numActive);
    const lmB = this.opts.lmHeadBlock;
    for (let off = 0; off < T; off += lmB) {
      const bt = Math.min(lmB, T - off);
      this._disp(
        enc,
        this.p.logits,
        [s.normedF, e.w, e.scale, s.logits],
        this._grid1d(bt * V),
        1,
        this._meta([[0, bt], [1, V], [2, H], [3, off]]),
        'logits',
      );
      // CE meta: { vocab, tOff, lossScale, p } — logits is block-local, target/mask/loss global.
      this._disp(
        enc,
        this.p.ceGrad,
        [s.logits, s.targets, s.mask, s.loss],
        bt,
        1,
        this._meta([[0, V], [1, off]], { 2: lossScale }),
        'ce',
      );
      this._disp(
        enc,
        this.p.dHidden,
        [s.logits, e.w, e.scale, s.dnorm],
        this._grid1d(bt * H),
        1,
        this._meta([[0, bt], [1, V], [2, H], [3, off]]),
        'dHidden',
      );
    }
    // finalNorm backward: dNormedF -> dHidden (grad of ckpt[numLayers])
    this._rmsBwd(enc, this.ckpt[c.numLayers], rt.bufs[rt.plan.finalNorm.name], s.dnorm, s.dHidden, T);

    for (let i = c.numLayers - 1; i >= 0; i--) {
      const L = rt.plan.layers[i];
      this._recomputeLayer(enc, L, T);

      // --- MLP block ---
      // h_out = hmid + down(swig). dhmid += dHidden ; ddown = dHidden.
      enc.clearBuffer(s.dswig);
      this._projBackward(enc, L.down.loraKey, s.swig, s.dHidden, s.dswig, T);
      // swiglu backward -> dgate, dup
      this._disp(
        enc,
        this.p.swiglu,
        [s.gate, s.up, s.dswig, s.dgate, s.dup],
        this._grid1d(T * I),
        1,
        this._u32([T * I]),
        'swiglu',
      );
      // gate/up backward -> dnorm (shared input normed2)
      enc.clearBuffer(s.dnorm);
      this._projBackward(enc, L.gate.loraKey, s.normed2, s.dgate, s.dnorm, T);
      this._projBackward(enc, L.up.loraKey, s.normed2, s.dup, s.dnorm, T);
      // post-attn rmsnorm backward: dnorm -> dtmp ; dhmid = dHidden(residual) + dtmp
      this._rmsBwd(enc, s.hmid, rt.bufs[L.postAttentionNorm], s.dnorm, s.dtmp, T);
      enc.copyBufferToBuffer(s.dHidden, 0, s.dhmid, 0, T * H * 4);
      rt._addInto(enc, s.dhmid, s.dtmp, T * H);

      // --- attention block ---
      // h_mid = ckpt[i] + o_proj(attn). dckpt += dhmid (residual) ; dattnProj = dhmid.
      enc.clearBuffer(s.dob);
      this._projBackward(enc, L.o.loraKey, s.attn, s.dhmid, s.dob, T);
      // attention backward (recompute) -> dq,dk,dv
      const am = this._u32([c.numHeads, c.numKVHeads, c.headDim, T]);
      this._disp(enc, this.p.attnStats, [s.q, s.k, s.attn, s.dob, s.lse, s.delta], c.numHeads, T, am, 'attnStats');
      enc.clearBuffer(s.dq);
      enc.clearBuffer(s.dk);
      enc.clearBuffer(s.dv);
      this._disp(enc, this.p.attnDq, [s.q, s.k, s.v, s.dob, s.lse, s.delta, s.dq], c.numHeads, T, am, 'attnDq');
      this._disp(
        enc,
        this.p.attnDkv,
        [s.q, s.k, s.v, s.dob, s.lse, s.delta, s.dk, s.dv],
        c.numKVHeads,
        T,
        am,
        'attnDkv',
      );
      // rope backward on dq, dk
      this._disp(
        enc,
        this.p.ropeBwd,
        [s.dq, rt.ropeCos, rt.ropeSin],
        Math.ceil((T * c.numHeads * (c.headDim / 2)) / 256),
        1,
        this._u32([c.numHeads, c.headDim, T, 0]),
        'ropeBwd',
      );
      this._disp(
        enc,
        this.p.ropeBwd,
        [s.dk, rt.ropeCos, rt.ropeSin],
        Math.ceil((T * c.numKVHeads * (c.headDim / 2)) / 256),
        1,
        this._u32([c.numKVHeads, c.headDim, T, 0]),
        'ropeBwd',
      );
      // q/k/v backward -> dnorm (shared input normed1)
      enc.clearBuffer(s.dnorm);
      this._projBackward(enc, L.q.loraKey, s.normed1, s.dq, s.dnorm, T);
      this._projBackward(enc, L.k.loraKey, s.normed1, s.dk, s.dnorm, T);
      this._projBackward(enc, L.v.loraKey, s.normed1, s.dv, s.dnorm, T);
      // input rmsnorm backward: dnorm -> dtmp ; dHidden_next = dhmid(residual into ckpt) + dtmp
      this._rmsBwd(enc, this.ckpt[i], rt.bufs[L.inputNorm], s.dnorm, s.dtmp, T);
      enc.copyBufferToBuffer(s.dhmid, 0, s.dHidden, 0, T * H * 4);
      rt._addInto(enc, s.dHidden, s.dtmp, T * H);
    }
  }

  // shifted-label targets + mask into the scratch buffers; returns numActive.
  _writeTargets(tokens, lossMask, T) {
    const targets = new Uint32Array(T);
    const mask = new Float32Array(T);
    let numActive = 0;
    for (let t = 0; t < T - 1; t++) {
      targets[t] = tokens[t + 1] >>> 0;
      const mk = lossMask ? (lossMask[t] ? 1 : 0) : 1;
      mask[t] = mk;
      numActive += mk;
    }
    targets[T - 1] = 0;
    mask[T - 1] = 0;
    this.dev.queue.writeBuffer(this.s.targets, 0, targets);
    this.dev.queue.writeBuffer(this.s.mask, 0, mask);
    return numActive;
  }

  // loss head only (final norm + streamed logits + CE), no backward sweep. Used by
  // evalLoss(). CE overwrites s.logits with dLogits but we ignore that here.
  _lossOnly(enc, T, numActive) {
    const rt = this.rt,
      c = this.cfg,
      s = this.s,
      H = c.hiddenSize,
      V = c.vocabSize;
    rt.rmsT(enc, this.ckpt[c.numLayers], rt.bufs[rt.plan.finalNorm.name], s.normedF, T, H);
    const e = rt.q[rt.plan.embed.name];
    const lossScale = 1.0 / Math.max(1, numActive);
    const lmB = this.opts.lmHeadBlock;
    for (let off = 0; off < T; off += lmB) {
      const bt = Math.min(lmB, T - off);
      this._disp(enc, this.p.logits, [s.normedF, e.w, e.scale, s.logits], this._grid1d(bt * V), 1, this._meta([[0, bt], [1, V], [2, H], [3, off]]), 'logits');
      this._disp(enc, this.p.ceGrad, [s.logits, s.targets, s.mask, s.loss], bt, 1, this._meta([[0, V], [1, off]], { 2: lossScale }), 'ce');
    }
  }

  // ---- public: forward-only mean cross-entropy (no grads). For held-out eval. ----
  async evalLoss(tokens, lossMask) {
    const T = tokens.length;
    if (T > this.opts.maxTrainSeq) throw new Error(`seq ${T} > maxTrainSeq ${this.opts.maxTrainSeq}`);
    this._ensureScratch(T);
    const wasF16 = this.rt.usingF16?.();
    this.rt.setUseF16?.(false);
    try {
      const numActive = this._writeTargets(tokens, lossMask, T);
      const enc = this.dev.createCommandEncoder();
      this._forward(enc, tokens, T);
      this._lossOnly(enc, T, numActive);
      enc.copyBufferToBuffer(this.s.loss, 0, this.lossRead, 0, T * 4);
      this.dev.queue.submit([enc.finish()]);
      await this.lossRead.mapAsync(GPUMapMode.READ);
      const arr = new Float32Array(this.lossRead.getMappedRange().slice(0));
      this.lossRead.unmap();
      let sum = 0;
      for (let t = 0; t < T; t++) sum += arr[t];
      return { loss: sum / Math.max(1, numActive), numActive };
    } finally {
      if (wasF16) this.rt.setUseF16?.(true);
    }
  }

  // ---- public: accumulate one micro-batch. tokens: Int array, lossMask: 0/1 array. ----
  // lossMask[t]==1 means "train the prediction of tokens[t+1] from position t".
  async microStep(tokens, lossMask) {
    const c = this.cfg;
    const T = tokens.length;
    if (T > this.opts.maxTrainSeq) throw new Error(`seq ${T} > maxTrainSeq ${this.opts.maxTrainSeq}`);
    this._ensureScratch(T);
    const wasF16 = this.rt.usingF16?.();
    this.rt.setUseF16?.(false); // f32 recompute for gradient stability
    try {
      const numActive = this._writeTargets(tokens, lossMask, T);
      const enc = this.dev.createCommandEncoder();
      this._forward(enc, tokens, T);
      this._backward(enc, T, numActive);
      enc.copyBufferToBuffer(this.s.loss, 0, this.lossRead, 0, T * 4);
      this.dev.queue.submit([enc.finish()]);

      await this.lossRead.mapAsync(GPUMapMode.READ);
      const lossArr = new Float32Array(this.lossRead.getMappedRange().slice(0));
      this.lossRead.unmap();
      let lossSum = 0;
      for (let t = 0; t < T; t++) lossSum += lossArr[t];
      this._microInWindow++;
      return { loss: lossSum / Math.max(1, numActive), numActive };
    } finally {
      // Always restore the runtime's precision mode, even if a dispatch/readback
      // failed — otherwise inference would silently keep running in f32.
      if (wasF16) this.rt.setUseF16?.(true);
    }
  }

  // ---- public: apply accumulated grads with AdamW + global-norm clip ----
  async optimizerStep() {
    const o = this.opts;
    const accum = this._microInWindow || 1;
    // global grad norm
    const encN = this.dev.createCommandEncoder();
    encN.clearBuffer(this.s.normBuf);
    for (const k of this.trainedKeys) {
      const st = this.state[k];
      this._disp(encN, this.p.sumsq, [st.dA, this.s.normBuf], 1, 1, this._u32([st.rank * st.K]), 'sumsq');
      this._disp(encN, this.p.sumsq, [st.dB, this.s.normBuf], 1, 1, this._u32([st.rank * st.N]), 'sumsq');
    }
    encN.copyBufferToBuffer(this.s.normBuf, 0, this.normRead, 0, 4);
    this.dev.queue.submit([encN.finish()]);
    await this.normRead.mapAsync(GPUMapMode.READ);
    const sumsq = new Float32Array(this.normRead.getMappedRange().slice(0))[0];
    this.normRead.unmap();
    // grads were summed over micro-batches; effective grad = sum/accum.
    const gradScale = 1.0 / accum;
    const gnorm = Math.sqrt(sumsq) * gradScale;
    const clip = o.maxGradNorm > 0 && gnorm > o.maxGradNorm ? o.maxGradNorm / (gnorm + 1e-6) : 1.0;
    const gScale = gradScale * clip;

    this.step++;
    const lr = this._lrAt(this.step);
    const b1c = 1 - Math.pow(o.beta1, this.step);
    const b2c = 1 - Math.pow(o.beta2, this.step);

    const enc = this.dev.createCommandEncoder();
    for (const k of this.trainedKeys) {
      const st = this.state[k];
      const metaA = this._adamMeta(st.rank * st.K, lr, gScale, b1c, b2c);
      this._disp(enc, this.p.adamw, [st.mod.A, st.dA, st.mA, st.vA], this._grid1d(st.rank * st.K), 1, metaA, 'adamw');
      const metaB = this._adamMeta(st.rank * st.N, lr, gScale, b1c, b2c);
      this._disp(enc, this.p.adamw, [st.mod.B, st.dB, st.mB, st.vB], this._grid1d(st.rank * st.N), 1, metaB, 'adamw');
    }
    this.dev.queue.submit([enc.finish()]);
    // mutated A/B in place: refresh inference bind groups against new contents.
    this.rt.invalidateLora();
    this.zeroGrads();
    return { lr, gradNorm: gnorm, clip };
  }

  _lrAt(step) {
    const o = this.opts;
    if (o.warmupSteps > 0 && step <= o.warmupSteps) return o.lr * (step / o.warmupSteps);
    if (o.totalSteps > 0 && step > o.warmupSteps) {
      const prog = (step - o.warmupSteps) / Math.max(1, o.totalSteps - o.warmupSteps);
      const cos = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, prog)));
      return o.lr * (o.minLrRatio + (1 - o.minLrRatio) * cos);
    }
    return o.lr;
  }

  _adamMeta(n, lr, gScale, b1c, b2c) {
    const o = this.opts;
    const buf = new ArrayBuffer(48);
    const dv = new DataView(buf);
    dv.setUint32(0, n >>> 0, true);
    dv.setFloat32(8, lr, true);
    dv.setFloat32(12, o.beta1, true);
    dv.setFloat32(16, o.beta2, true);
    dv.setFloat32(20, o.eps, true);
    dv.setFloat32(24, o.weightDecay, true);
    dv.setFloat32(28, gScale, true);
    dv.setFloat32(32, b1c, true);
    dv.setFloat32(36, b2c, true);
    return new Uint8Array(buf);
  }

  // ---- convenience: one full optimization step over a list of micro-batches ----
  async trainStep(batches) {
    const list = Array.isArray(batches) ? batches : [batches];
    let lossSum = 0,
      n = 0;
    for (const b of list) {
      const r = await this.microStep(b.tokens, b.lossMask);
      lossSum += r.loss;
      n++;
    }
    const opt = await this.optimizerStep();
    return { loss: lossSum / Math.max(1, n), ...opt };
  }
}
