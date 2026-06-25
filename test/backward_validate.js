/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

// GPU-level validation of the backward WGSL kernels (src/qwgpu/backward_kernels.js).
// Runs each kernel on a real device with tiny inputs and compares the read-back
// result to a CPU reference. No base model needed — self-contained, so it can run
// in any WebGPU webshell. Prints "BWD-GPU ..." lines; "BWD-GPU DONE" when finished.

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
} from '../src/qwgpu/backward_kernels.js';

// Every kernel the trainer compiles, so the compile-gate below catches missing
// builtins / WGSL errors (e.g. unpack4xI8 in the LM-head kernels) that a partial
// import would miss.
const ALL_KERNELS = {
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
};

const TOL = 3e-3;
let PASS = 0,
  FAIL = 0;

class MiniGPU {
  constructor(dev) {
    this.dev = dev;
  }
  pipe(code, name) {
    const m = this.dev.createShaderModule({ code, label: name });
    return this.dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' }, label: name });
  }
  // Async pipeline creation rejects deterministically on a WGSL compile/validation
  // error (GPUPipelineError) instead of surfacing it as an async uncaptured error.
  async pipeAsync(code, name) {
    const m = this.dev.createShaderModule({ code, label: name });
    return await this.dev.createComputePipelineAsync({ layout: 'auto', compute: { module: m, entryPoint: 'main' }, label: name });
  }
  buf(data, extraUsage = 0) {
    const arr = data instanceof Float32Array || data instanceof Uint32Array ? data : new Float32Array(data);
    const b = this.dev.createBuffer({
      size: Math.max(4, arr.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
    });
    this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }
  empty(bytes) {
    const b = this.dev.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    return b;
  }
  run(pipe, buffers, gx, gy, imm) {
    const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    const bg = this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const enc = this.dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe);
    p.setBindGroup(0, bg);
    if (imm) p.setImmediates(0, imm);
    p.dispatchWorkgroups(gx, gy || 1);
    p.end();
    this.dev.queue.submit([enc.finish()]);
  }
  async read(b, floats) {
    const rb = this.dev.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(b, 0, rb, 0, floats * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return out;
  }
}

function relerr(a, b) {
  let num = 0,
    den = 0;
  for (let i = 0; i < a.length; i++) {
    num = Math.max(num, Math.abs(a[i] - b[i]));
    den = Math.max(den, Math.abs(b[i]));
  }
  return num / (den + 1e-9);
}
function check(name, got, ref) {
  const e = relerr(got, ref);
  const ok = e < TOL && Number.isFinite(e);
  console.log(`BWD-GPU ${ok ? 'PASS' : 'FAIL'}  ${name}  relerr=${e.toExponential(2)}`);
  ok ? PASS++ : FAIL++;
}
function u8meta(u32parts, f32parts = {}, bytes = 32) {
  const buf = new ArrayBuffer(bytes);
  const dv = new DataView(buf);
  for (const [i, v] of u32parts) dv.setUint32(i * 4, v >>> 0, true);
  for (const [i, v] of Object.entries(f32parts)) dv.setFloat32(Number(i) * 4, v, true);
  return new Uint8Array(buf);
}
function rnd(n, s = 1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * s;
  return a;
}

window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    console.log('BWD-GPU FAIL  no WebGPU adapter');
    console.log('BWD-GPU DONE');
    return;
  }
  // Gate on the exact WGSL language feature the backward kernels rely on
  // (`requires immediate_address_space;`) — same hard requirement as the app's
  // device_service.js. Chrome 149+ exposes it via wgslLanguageFeatures.
  if (!navigator.gpu.wgslLanguageFeatures?.has('immediate_address_space')) {
    console.log('BWD-GPU FAIL  WGSL immediate_address_space unavailable (need Chrome 149+)');
    console.log('BWD-GPU DONE');
    return;
  }
  if (!adapter.features.has('subgroups')) {
    console.log('BWD-GPU FAIL  adapter lacks "subgroups" feature');
    console.log('BWD-GPU DONE');
    return;
  }
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups'] });
  dev.addEventListener?.('uncapturederror', (e) => console.log('BWD-GPU GPUERR ' + e.error.message.slice(0, 200)));
  const g = new MiniGPU(dev);
  console.log('BWD-GPU device ready');

  // ---- compile-gate: every trainer kernel must compile (catches missing builtins) ----
  // Uses async pipeline creation + an error scope so a WGSL failure is deterministic
  // (rejects here) rather than escaping as an async uncaptured device error.
  for (const [name, src] of Object.entries(ALL_KERNELS)) {
    dev.pushErrorScope('validation');
    let err = null;
    try {
      await g.pipeAsync(src, name);
    } catch (e) {
      err = e.message;
    }
    const scoped = await dev.popErrorScope();
    const ok = !err && !scoped;
    console.log(`BWD-GPU ${ok ? 'PASS' : 'FAIL'}  compile ${name}${ok ? '' : '  ' + (err || scoped.message).slice(0, 120)}`);
    ok ? PASS++ : FAIL++;
  }

  // ---- GEMM_DX_INT4: dX = dY @ deq(W) with int4 group weights ----
  {
    const T = 3,
      N = 4,
      K = 8,
      gpr = 1; // one 128-group covers K=8
    const dY = rnd(T * N);
    // build int4 weights: nibble in [-8,7], one scale per row
    const nib = new Int8Array(N * K);
    for (let i = 0; i < nib.length; i++) nib[i] = ((Math.random() * 16) | 0) - 8;
    const scaleW = rnd(N, 0.5).map((x) => Math.abs(x) + 0.05);
    const words = new Uint32Array(N * (K / 8));
    for (let n = 0; n < N; n++) {
      let w = 0;
      for (let k = 0; k < 8; k++) w |= (nib[n * K + k] & 0xf) << (k * 4);
      words[n] = w >>> 0;
    }
    const dX = g.empty(T * K * 4);
    const pipe = g.pipe(GEMM_DX_INT4, 'dx4');
    g.run(pipe, [g.buf(dY), g.buf(words), g.buf(scaleW), dX], Math.ceil((T * K) / 256), 1, u8meta([[0, T], [1, N], [2, K], [3, gpr]], {}, 16));
    const got = await g.read(dX, T * K);
    const ref = new Float32Array(T * K);
    for (let t = 0; t < T; t++)
      for (let k = 0; k < K; k++) {
        let acc = 0;
        for (let n = 0; n < N; n++) acc += dY[t * N + n] * nib[n * K + k] * scaleW[n];
        ref[t * K + k] = acc;
      }
    check('GEMM_DX_INT4', got, ref);
  }

  // ---- LoRA grads: full dA/dB/dX path ----
  {
    const T = 4,
      K = 6,
      N = 5,
      rank = 2,
      scale = 1.3;
    const X = rnd(T * K),
      dY = rnd(T * N),
      A = rnd(rank * K),
      B = rnd(rank * N);
    // dD = scale*dY@B^T  (LORA_DD): one wg per (t,r), 256 threads
    const dD = g.empty(T * rank * 4);
    g.run(g.pipe(LORA_DD, 'dd'), [g.buf(dY), g.buf(B), dD], T * rank, 1, u8meta([[0, T], [1, N], [2, rank]], { 4: scale }));
    const dDg = await g.read(dD, T * rank);
    const dDref = new Float32Array(T * rank);
    for (let t = 0; t < T; t++) for (let r = 0; r < rank; r++) { let a = 0; for (let n = 0; n < N; n++) a += dY[t * N + n] * B[r * N + n]; dDref[t * rank + r] = scale * a; }
    check('LORA_DD', dDg, dDref);
    // dA += dD^T@X
    const dA = g.empty(rank * K * 4);
    g.run(g.pipe(LORA_GRAD_A, 'gA'), [dD, g.buf(X), dA], Math.ceil((rank * K) / 256), 1, u8meta([[0, T], [1, K], [2, rank]], {}, 16));
    const dAg = await g.read(dA, rank * K);
    const dAref = new Float32Array(rank * K);
    for (let r = 0; r < rank; r++) for (let k = 0; k < K; k++) { let a = 0; for (let t = 0; t < T; t++) a += dDref[t * rank + r] * X[t * K + k]; dAref[r * K + k] = a; }
    check('LORA_GRAD_A', dAg, dAref);
    // dB += scale*D^T@dY where D=X@A
    const Dmat = new Float32Array(T * rank);
    for (let t = 0; t < T; t++) for (let r = 0; r < rank; r++) { let a = 0; for (let k = 0; k < K; k++) a += X[t * K + k] * A[r * K + k]; Dmat[t * rank + r] = a; }
    const dB = g.empty(rank * N * 4);
    g.run(g.pipe(LORA_GRAD_B, 'gB'), [g.buf(Dmat), g.buf(dY), dB], Math.ceil((rank * N) / 256), 1, u8meta([[0, T], [1, N], [2, rank]], { 4: scale }));
    const dBg = await g.read(dB, rank * N);
    const dBref = new Float32Array(rank * N);
    for (let r = 0; r < rank; r++) for (let n = 0; n < N; n++) { let a = 0; for (let t = 0; t < T; t++) a += Dmat[t * rank + r] * dY[t * N + n]; dBref[r * N + n] = scale * a; }
    check('LORA_GRAD_B', dBg, dBref);
    // dX += dD@A
    const dX = g.empty(T * K * 4);
    g.run(g.pipe(LORA_DX_ADD, 'dxAdd'), [dD, g.buf(A), dX], Math.ceil((T * K) / 256), 1, u8meta([[0, T], [1, K], [2, rank]], {}, 16));
    const dXg = await g.read(dX, T * K);
    const dXref = new Float32Array(T * K);
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) { let a = 0; for (let r = 0; r < rank; r++) a += dDref[t * rank + r] * A[r * K + k]; dXref[t * K + k] = a; }
    check('LORA_DX_ADD', dXg, dXref);
  }

  // ---- RMSNORM_BWD_T ----
  {
    const T = 3,
      K = 8,
      eps = 1e-6;
    const X = rnd(T * K),
      gg = rnd(K),
      dy = rnd(T * K);
    const dx = g.empty(T * K * 4);
    g.run(g.pipe(RMSNORM_BWD_T, 'rms'), [g.buf(X), g.buf(gg), g.buf(dy), dx], T, 1, new Float32Array([K, eps]));
    const got = await g.read(dx, T * K);
    const ref = new Float32Array(T * K);
    for (let t = 0; t < T; t++) {
      let ss = 0;
      for (let k = 0; k < K; k++) ss += X[t * K + k] ** 2;
      const inv = 1 / Math.sqrt(ss / K + eps);
      let c = 0;
      for (let k = 0; k < K; k++) c += dy[t * K + k] * gg[k] * X[t * K + k];
      const i3 = (inv * inv * inv) / K;
      for (let k = 0; k < K; k++) ref[t * K + k] = inv * gg[k] * dy[t * K + k] - i3 * X[t * K + k] * c;
    }
    check('RMSNORM_BWD_T', got, ref);
  }

  // ---- SWIGLU_BWD ----
  {
    const n = 16;
    const gate = rnd(n),
      up = rnd(n),
      dOut = rnd(n);
    const dGate = g.empty(n * 4),
      dUp = g.empty(n * 4);
    g.run(g.pipe(SWIGLU_BWD, 'sw'), [g.buf(gate), g.buf(up), g.buf(dOut), dGate, dUp], Math.ceil(n / 256), 1, new Uint32Array([n]));
    const dGg = await g.read(dGate, n),
      dUg = await g.read(dUp, n);
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const dGr = new Float32Array(n),
      dUr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const z = gate[i],
        sg = sig(z);
      dUr[i] = dOut[i] * z * sg;
      dGr[i] = dOut[i] * up[i] * (sg * (1 + z * (1 - sg)));
    }
    check('SWIGLU_BWD dGate', dGg, dGr);
    check('SWIGLU_BWD dUp', dUg, dUr);
  }

  // ---- CE_SOFTMAX_GRAD ----
  {
    const T = 2,
      V = 7;
    const logits = rnd(T * V);
    const targets = new Uint32Array([3, 5]);
    const mask = new Float32Array([1, 1]);
    const lossScale = 1.0;
    const logitsBuf = g.buf(logits);
    const lossBuf = g.empty(T * 4);
    g.run(g.pipe(CE_SOFTMAX_GRAD, 'ce'), [logitsBuf, g.buf(targets), g.buf(mask), lossBuf], T, 1, u8meta([[0, V], [1, 0]], { 2: lossScale }));
    const dL = await g.read(logitsBuf, T * V);
    const ref = new Float32Array(T * V);
    for (let t = 0; t < T; t++) {
      let mx = -Infinity;
      for (let v = 0; v < V; v++) mx = Math.max(mx, logits[t * V + v]);
      let Z = 0;
      for (let v = 0; v < V; v++) Z += Math.exp(logits[t * V + v] - mx);
      for (let v = 0; v < V; v++) {
        let p = Math.exp(logits[t * V + v] - mx) / Z;
        if (v === targets[t]) p -= 1;
        ref[t * V + v] = p;
      }
    }
    check('CE_SOFTMAX_GRAD', dL, ref);
  }

  // ---- ATTN_BWD_* (single + GQA) ----
  {
    const nHeads = 4,
      nKV = 2,
      hd = 4,
      T = 5,
      group = nHeads / nKV;
    const scl = 1 / Math.sqrt(hd);
    const Q = rnd(T * nHeads * hd, 0.5),
      Kc = rnd(T * nKV * hd, 0.5),
      Vc = rnd(T * nKV * hd, 0.5),
      dO = rnd(T * nHeads * hd);
    const qoff = (t, h) => t * nHeads * hd + h * hd;
    const kvoff = (t, kvh) => t * nKV * hd + kvh * hd;
    // CPU forward O
    const O = new Float32Array(T * nHeads * hd);
    for (let h = 0; h < nHeads; h++) {
      const kvh = (h / group) | 0;
      for (let t = 0; t < T; t++) {
        const sc = [];
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) { let d = 0; for (let x = 0; x < hd; x++) d += Q[qoff(t, h) + x] * Kc[kvoff(j, kvh) + x]; sc[j] = d * scl; mx = Math.max(mx, sc[j]); }
        let Z = 0;
        for (let j = 0; j <= t; j++) { sc[j] = Math.exp(sc[j] - mx); Z += sc[j]; }
        for (let x = 0; x < hd; x++) { let a = 0; for (let j = 0; j <= t; j++) a += (sc[j] / Z) * Vc[kvoff(j, kvh) + x]; O[qoff(t, h) + x] = a; }
      }
    }
    const meta = new Uint32Array([nHeads, nKV, hd, T]);
    const lse = g.empty(nHeads * T * 4),
      delta = g.empty(nHeads * T * 4);
    const qB = g.buf(Q), kB = g.buf(Kc), vB = g.buf(Vc), oB = g.buf(O), doB = g.buf(dO);
    g.run(g.pipe(ATTN_BWD_STATS, 'st'), [qB, kB, oB, doB, lse, delta], nHeads, T, meta);
    const dq = g.empty(T * nHeads * hd * 4);
    g.run(g.pipe(ATTN_BWD_DQ, 'dq'), [qB, kB, vB, doB, lse, delta, dq], nHeads, T, meta);
    const dk = g.empty(T * nKV * hd * 4),
      dv = g.empty(T * nKV * hd * 4);
    g.run(g.pipe(ATTN_BWD_DKV, 'dkv'), [qB, kB, vB, doB, lse, delta, dk, dv], nKV, T, meta);
    const dqg = await g.read(dq, T * nHeads * hd),
      dkg = await g.read(dk, T * nKV * hd),
      dvg = await g.read(dv, T * nKV * hd);
    // CPU reference grads
    const dQr = new Float32Array(T * nHeads * hd),
      dKr = new Float32Array(T * nKV * hd),
      dVr = new Float32Array(T * nKV * hd);
    for (let h = 0; h < nHeads; h++) {
      const kvh = (h / group) | 0;
      for (let t = 0; t < T; t++) {
        const sc = [];
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) { let d = 0; for (let x = 0; x < hd; x++) d += Q[qoff(t, h) + x] * Kc[kvoff(j, kvh) + x]; sc[j] = d * scl; mx = Math.max(mx, sc[j]); }
        let Z = 0;
        for (let j = 0; j <= t; j++) Z += Math.exp(sc[j] - mx);
        const lseV = mx + Math.log(Z);
        let del = 0;
        for (let x = 0; x < hd; x++) del += dO[qoff(t, h) + x] * O[qoff(t, h) + x];
        for (let j = 0; j <= t; j++) {
          const p = Math.exp(sc[j] - lseV);
          let dp = 0;
          for (let x = 0; x < hd; x++) dp += dO[qoff(t, h) + x] * Vc[kvoff(j, kvh) + x];
          const ds = p * (dp - del);
          for (let x = 0; x < hd; x++) {
            dQr[qoff(t, h) + x] += scl * ds * Kc[kvoff(j, kvh) + x];
            dKr[kvoff(j, kvh) + x] += scl * ds * Q[qoff(t, h) + x];
            dVr[kvoff(j, kvh) + x] += p * dO[qoff(t, h) + x];
          }
        }
      }
    }
    check('ATTN_BWD_DQ', dqg, dQr);
    check('ATTN_BWD_DKV dK', dkg, dKr);
    check('ATTN_BWD_DKV dV', dvg, dVr);
  }

  // ---- SUMSQ + ADAMW_STEP ----
  {
    const n = 5;
    const x = rnd(n);
    const out = g.empty(4);
    g.run(g.pipe(SUMSQ, 'ss'), [g.buf(x), out], 1, 1, new Uint32Array([n]));
    const ssg = (await g.read(out, 1))[0];
    let ssr = 0;
    for (let i = 0; i < n; i++) ssr += x[i] * x[i];
    check('SUMSQ', new Float32Array([ssg]), new Float32Array([ssr]));

    const param = rnd(n),
      grad = rnd(n),
      m = rnd(n, 0.1),
      v = rnd(n, 0.1).map(Math.abs);
    const lr = 1e-3,
      b1 = 0.9,
      b2 = 0.999,
      eps = 1e-8,
      wd = 0.01,
      step = 7,
      gScale = 1.0;
    const b1c = 1 - b1 ** step,
      b2c = 1 - b2 ** step;
    const pBuf = g.buf(param);
    const meta = new ArrayBuffer(48);
    const dv = new DataView(meta);
    dv.setUint32(0, n, true);
    dv.setFloat32(8, lr, true);
    dv.setFloat32(12, b1, true);
    dv.setFloat32(16, b2, true);
    dv.setFloat32(20, eps, true);
    dv.setFloat32(24, wd, true);
    dv.setFloat32(28, gScale, true);
    dv.setFloat32(32, b1c, true);
    dv.setFloat32(36, b2c, true);
    g.run(g.pipe(ADAMW_STEP, 'adam'), [pBuf, g.buf(grad), g.buf(m), g.buf(v)], Math.ceil(n / 256), 1, new Uint8Array(meta));
    const got = await g.read(pBuf, n);
    const ref = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const mm = b1 * m[i] + (1 - b1) * grad[i];
      const vv = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
      ref[i] = param[i] - lr * (mm / b1c / (Math.sqrt(vv / b2c) + eps) + wd * param[i]);
    }
    check('ADAMW_STEP', got, ref);
  }

  // ---- LM head: LOGITS_GEMM_I8 + DHIDDEN_FROM_DLOGITS_I8 (tied int8 embeddings) ----
  {
    const T = 2,
      V = 5,
      K = 8,
      K4 = K / 4,
      tOff = 0;
    const normed = rnd(T * K);
    const Ei8 = new Int8Array(V * K);
    for (let i = 0; i < Ei8.length; i++) Ei8[i] = ((Math.random() * 256) | 0) - 128;
    const scaleE = rnd(V, 0.3).map((x) => Math.abs(x) + 0.05);
    const Ewords = new Uint32Array(V * K4);
    for (let v = 0; v < V; v++)
      for (let c = 0; c < K4; c++) {
        let w = 0;
        for (let l = 0; l < 4; l++) w |= (Ei8[v * K + c * 4 + l] & 0xff) << (l * 8);
        Ewords[v * K4 + c] = w >>> 0;
      }
    const EwB = g.buf(Ewords),
      scB = g.buf(scaleE),
      nB = g.buf(normed);
    // forward logits
    const logits = g.empty(T * V * 4);
    g.run(g.pipe(LOGITS_GEMM_I8, 'logits'), [nB, EwB, scB, logits], Math.ceil((T * V) / 256), 1, u8meta([[0, T], [1, V], [2, K], [3, tOff]], {}, 16));
    const lg = await g.read(logits, T * V);
    const lref = new Float32Array(T * V);
    for (let t = 0; t < T; t++)
      for (let v = 0; v < V; v++) {
        let acc = 0;
        for (let k = 0; k < K; k++) acc += normed[t * K + k] * Ei8[v * K + k];
        lref[t * V + v] = acc * scaleE[v];
      }
    check('LOGITS_GEMM_I8', lg, lref);
    // dHidden from dLogits
    const dLogits = rnd(T * V);
    const dHidden = g.empty(T * K * 4); // zero-init
    g.run(g.pipe(DHIDDEN_FROM_DLOGITS_I8, 'dh'), [g.buf(dLogits), EwB, scB, dHidden], Math.ceil((T * K) / 256), 1, u8meta([[0, T], [1, V], [2, K], [3, tOff]], {}, 16));
    const dhg = await g.read(dHidden, T * K);
    const dhref = new Float32Array(T * K);
    for (let t = 0; t < T; t++)
      for (let k = 0; k < K; k++) {
        let acc = 0;
        for (let v = 0; v < V; v++) acc += dLogits[t * V + v] * scaleE[v] * Ei8[v * K + k];
        dhref[t * K + k] = acc;
      }
    check('DHIDDEN_FROM_DLOGITS_I8', dhg, dhref);
  }

  // ---- ROPE_BWD_T: transpose of forward rotation ----
  {
    const nHeads = 2,
      hd = 4,
      T = 3,
      half = hd / 2;
    // cos/sin tables [T][hd] (only first half used)
    const cosT = new Float32Array(T * hd),
      sinT = new Float32Array(T * hd);
    for (let t = 0; t < T; t++)
      for (let j = 0; j < half; j++) {
        const ang = 0.3 * (t + 1) * (j + 1);
        cosT[t * hd + j] = Math.cos(ang);
        sinT[t * hd + j] = Math.sin(ang);
      }
    const dx = rnd(T * nHeads * hd);
    const dxB = g.buf(dx);
    g.run(g.pipe(ROPE_BWD_T, 'rope'), [dxB, g.buf(cosT), g.buf(sinT)], Math.ceil((T * nHeads * half) / 256), 1, new Uint32Array([nHeads, hd, T, 0]));
    const got = await g.read(dxB, T * nHeads * hd);
    const ref = new Float32Array(dx);
    for (let t = 0; t < T; t++)
      for (let h = 0; h < nHeads; h++)
        for (let j = 0; j < half; j++) {
          const lo = t * nHeads * hd + h * hd + j,
            hi = lo + half;
          const c = cosT[t * hd + j],
            s = sinT[t * hd + j];
          const dl = dx[lo],
            dh = dx[hi];
          ref[lo] = c * dl + s * dh;
          ref[hi] = -s * dl + c * dh;
        }
    check('ROPE_BWD_T', got, ref);
  }

  console.log(`BWD-GPU ${FAIL === 0 ? 'ALL PASS' : 'FAILED'} (${PASS}/${PASS + FAIL})`);
  console.log('BWD-GPU DONE');
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('BWD-GPU ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))),
);
