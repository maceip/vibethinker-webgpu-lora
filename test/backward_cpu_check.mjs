/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

// Pure-Node finite-difference validation of the gradient ALGEBRA implemented by the
// WGSL backward kernels in src/qwgpu/backward_kernels.js. Each reference function
// below mirrors exactly one kernel's math; we verify it against numerical gradients
// of a scalar loss L = sum(G ⊙ Y) for random upstream G. No GPU / no model needed,
// so it runs anywhere and prints PASS/FAIL per kernel.

const TOL = 2e-3; // relative tolerance for central finite differences
let PASS = 0,
  FAIL = 0;

function rnd(n, s = 1) {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * s;
  return a;
}
function relerr(a, b) {
  let num = 0,
    den = 0;
  for (let i = 0; i < a.length; i++) {
    num = Math.max(num, Math.abs(a[i] - b[i]));
    den = Math.max(den, Math.abs(b[i]));
  }
  return num / (den + 1e-12);
}
function fdGrad(f, x, eps = 1e-4) {
  const g = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const o = x[i];
    x[i] = o + eps;
    const fp = f();
    x[i] = o - eps;
    const fm = f();
    x[i] = o;
    g[i] = (fp - fm) / (2 * eps);
  }
  return g;
}
function check(name, analytic, numeric) {
  const e = relerr(analytic, numeric);
  const ok = e < TOL && Number.isFinite(e);
  console.log(`BWD-CPU ${ok ? 'PASS' : 'FAIL'}  ${name}  relerr=${e.toExponential(2)}`);
  ok ? PASS++ : FAIL++;
}

// ---------- LoRA + frozen projection: Y = X@W + scale*(X@A)@B ----------
// kernels: GEMM_DX_INT4 (dX base), LORA_DD/GRAD_A/GRAD_B/DX_ADD
function loraProj() {
  const T = 4,
    K = 6,
    N = 5,
    rank = 2,
    scale = 1.7;
  const X = rnd(T * K),
    W = rnd(N * K),
    A = rnd(rank * K),
    B = rnd(rank * N),
    G = rnd(T * N);
  const fwdY = () => {
    const Y = new Float64Array(T * N);
    for (let t = 0; t < T; t++)
      for (let n = 0; n < N; n++) {
        let acc = 0;
        for (let k = 0; k < K; k++) acc += X[t * K + k] * W[n * K + k];
        Y[t * N + n] = acc;
      }
    // D = X@A  (A is [rank][K]); delta = scale*D@B
    const D = new Float64Array(T * rank);
    for (let t = 0; t < T; t++)
      for (let r = 0; r < rank; r++) {
        let acc = 0;
        for (let k = 0; k < K; k++) acc += X[t * K + k] * A[r * K + k];
        D[t * rank + r] = acc;
      }
    for (let t = 0; t < T; t++)
      for (let n = 0; n < N; n++) {
        let acc = 0;
        for (let r = 0; r < rank; r++) acc += D[t * rank + r] * B[r * N + n];
        Y[t * N + n] += scale * acc;
      }
    return Y;
  };
  const loss = () => {
    const Y = fwdY();
    let s = 0;
    for (let i = 0; i < Y.length; i++) s += G[i] * Y[i];
    return s;
  };
  // analytic (mirrors kernels): dY = G
  // dD = scale * G @ B^T ; dA = dD^T @ X ; dB = scale * D^T @ G ; dX = G@W + dD@A
  const D = new Float64Array(T * rank);
  for (let t = 0; t < T; t++)
    for (let r = 0; r < rank; r++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += X[t * K + k] * A[r * K + k];
      D[t * rank + r] = acc;
    }
  const dD = new Float64Array(T * rank);
  for (let t = 0; t < T; t++)
    for (let r = 0; r < rank; r++) {
      let acc = 0;
      for (let n = 0; n < N; n++) acc += G[t * N + n] * B[r * N + n];
      dD[t * rank + r] = scale * acc;
    }
  const dA = new Float64Array(rank * K);
  for (let r = 0; r < rank; r++)
    for (let k = 0; k < K; k++) {
      let acc = 0;
      for (let t = 0; t < T; t++) acc += dD[t * rank + r] * X[t * K + k];
      dA[r * K + k] = acc;
    }
  const dB = new Float64Array(rank * N);
  for (let r = 0; r < rank; r++)
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let t = 0; t < T; t++) acc += D[t * rank + r] * G[t * N + n];
      dB[r * N + n] = scale * acc;
    }
  const dX = new Float64Array(T * K);
  for (let t = 0; t < T; t++)
    for (let k = 0; k < K; k++) {
      let acc = 0;
      for (let n = 0; n < N; n++) acc += G[t * N + n] * W[n * K + k];
      for (let r = 0; r < rank; r++) acc += dD[t * rank + r] * A[r * K + k];
      dX[t * K + k] = acc;
    }
  check('LoRA dX (GEMM_DX_INT4 + LORA_DX_ADD)', dX, fdGrad(loss, X));
  check('LoRA dA (LORA_GRAD_A)', dA, fdGrad(loss, A));
  check('LoRA dB (LORA_GRAD_B)', dB, fdGrad(loss, B));
}

// ---------- RMSNorm backward (RMSNORM_BWD_T) ----------
function rmsnorm() {
  const T = 3,
    K = 8,
    eps = 1e-6;
  const X = rnd(T * K),
    g = rnd(K, 1),
    G = rnd(T * K);
  const fwd = () => {
    const Y = new Float64Array(T * K);
    for (let t = 0; t < T; t++) {
      let ss = 0;
      for (let k = 0; k < K; k++) ss += X[t * K + k] ** 2;
      const inv = 1 / Math.sqrt(ss / K + eps);
      for (let k = 0; k < K; k++) Y[t * K + k] = X[t * K + k] * inv * g[k];
    }
    return Y;
  };
  const loss = () => {
    const Y = fwd();
    let s = 0;
    for (let i = 0; i < Y.length; i++) s += G[i] * Y[i];
    return s;
  };
  const dX = new Float64Array(T * K);
  for (let t = 0; t < T; t++) {
    let ss = 0;
    for (let k = 0; k < K; k++) ss += X[t * K + k] ** 2;
    const inv = 1 / Math.sqrt(ss / K + eps);
    let c = 0;
    for (let k = 0; k < K; k++) c += G[t * K + k] * g[k] * X[t * K + k];
    const inv3overK = (inv * inv * inv) / K;
    for (let k = 0; k < K; k++) dX[t * K + k] = inv * g[k] * G[t * K + k] - inv3overK * X[t * K + k] * c;
  }
  check('RMSNorm dX (RMSNORM_BWD_T)', dX, fdGrad(loss, X));
}

// ---------- SwiGLU backward (SWIGLU_BWD) ----------
function swiglu() {
  const n = 16;
  const gate = rnd(n),
    up = rnd(n),
    G = rnd(n);
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const fwd = () => {
    const o = new Float64Array(n);
    for (let i = 0; i < n; i++) o[i] = gate[i] * sig(gate[i]) * up[i];
    return o;
  };
  const loss = () => {
    const o = fwd();
    let s = 0;
    for (let i = 0; i < n; i++) s += G[i] * o[i];
    return s;
  };
  const dGate = new Float64Array(n),
    dUp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const z = gate[i],
      sg = sig(z),
      sl = z * sg;
    dUp[i] = G[i] * sl;
    dGate[i] = G[i] * up[i] * (sg * (1 + z * (1 - sg)));
  }
  check('SwiGLU dGate (SWIGLU_BWD)', dGate, fdGrad(loss, gate));
  check('SwiGLU dUp (SWIGLU_BWD)', dUp, fdGrad(loss, up));
}

// ---------- RoPE backward = J^T (ROPE_BWD_T) ----------
function rope() {
  // single pair (lo,hi) under rotation by angle θ; backward must be J^T.
  const c = Math.cos(0.7),
    s = Math.sin(0.7);
  const x = rnd(2),
    G = rnd(2);
  const fwd = () => [x[0] * c - x[1] * s, x[1] * c + x[0] * s];
  const loss = () => {
    const y = fwd();
    return G[0] * y[0] + G[1] * y[1];
  };
  const dlo = c * G[0] + s * G[1];
  const dhi = -s * G[0] + c * G[1];
  check('RoPE dx (ROPE_BWD_T)', new Float64Array([dlo, dhi]), fdGrad(loss, x));
}

// ---------- Causal GQA attention backward (ATTN_BWD_*) ----------
function attention() {
  const nHeads = 4,
    nKV = 2,
    hd = 4,
    T = 5;
  const group = nHeads / nKV;
  const scl = 1 / Math.sqrt(hd);
  const Q = rnd(T * nHeads * hd, 0.5),
    Kc = rnd(T * nKV * hd, 0.5),
    Vc = rnd(T * nKV * hd, 0.5),
    G = rnd(T * nHeads * hd);
  const qoff = (t, h) => t * nHeads * hd + h * hd;
  const kvoff = (t, kvh) => t * nKV * hd + kvh * hd;
  const fwdO = () => {
    const O = new Float64Array(T * nHeads * hd);
    for (let h = 0; h < nHeads; h++) {
      const kvh = Math.floor(h / group);
      for (let t = 0; t < T; t++) {
        // softmax over j<=t
        const sc = [];
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) {
          let dot = 0;
          for (let d = 0; d < hd; d++) dot += Q[qoff(t, h) + d] * Kc[kvoff(j, kvh) + d];
          dot *= scl;
          sc[j] = dot;
          mx = Math.max(mx, dot);
        }
        let Z = 0;
        for (let j = 0; j <= t; j++) {
          sc[j] = Math.exp(sc[j] - mx);
          Z += sc[j];
        }
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let j = 0; j <= t; j++) acc += (sc[j] / Z) * Vc[kvoff(j, kvh) + d];
          O[qoff(t, h) + d] = acc;
        }
      }
    }
    return O;
  };
  const loss = () => {
    const O = fwdO();
    let s = 0;
    for (let i = 0; i < O.length; i++) s += G[i] * O[i];
    return s;
  };
  // analytic via flash formulas
  const O = fwdO();
  const dQ = new Float64Array(T * nHeads * hd),
    dK = new Float64Array(T * nKV * hd),
    dV = new Float64Array(T * nKV * hd);
  for (let h = 0; h < nHeads; h++) {
    const kvh = Math.floor(h / group);
    for (let t = 0; t < T; t++) {
      // recompute p, lse, delta
      const sc = [];
      let mx = -Infinity;
      for (let j = 0; j <= t; j++) {
        let dot = 0;
        for (let d = 0; d < hd; d++) dot += Q[qoff(t, h) + d] * Kc[kvoff(j, kvh) + d];
        dot *= scl;
        sc[j] = dot;
        mx = Math.max(mx, dot);
      }
      let Z = 0;
      for (let j = 0; j <= t; j++) Z += Math.exp(sc[j] - mx);
      const lse = mx + Math.log(Z);
      let delta = 0;
      for (let d = 0; d < hd; d++) delta += G[qoff(t, h) + d] * O[qoff(t, h) + d];
      for (let j = 0; j <= t; j++) {
        const p = Math.exp(sc[j] - lse);
        let dp = 0;
        for (let d = 0; d < hd; d++) dp += G[qoff(t, h) + d] * Vc[kvoff(j, kvh) + d];
        const ds = p * (dp - delta);
        for (let d = 0; d < hd; d++) {
          dQ[qoff(t, h) + d] += scl * ds * Kc[kvoff(j, kvh) + d];
          dK[kvoff(j, kvh) + d] += scl * ds * Q[qoff(t, h) + d];
          dV[kvoff(j, kvh) + d] += p * G[qoff(t, h) + d];
        }
      }
    }
  }
  check('Attn dQ (ATTN_BWD_DQ)', dQ, fdGrad(loss, Q));
  check('Attn dK (ATTN_BWD_DKV)', dK, fdGrad(loss, Kc));
  check('Attn dV (ATTN_BWD_DKV)', dV, fdGrad(loss, Vc));
}

// ---------- Cross-entropy + softmax grad (CE_SOFTMAX_GRAD) ----------
function ce() {
  const V = 7;
  const logits = rnd(V),
    tgt = 3;
  const loss = () => {
    let mx = -Infinity;
    for (let v = 0; v < V; v++) mx = Math.max(mx, logits[v]);
    let Z = 0;
    for (let v = 0; v < V; v++) Z += Math.exp(logits[v] - mx);
    return Math.log(Z) - (logits[tgt] - mx);
  };
  let mx = -Infinity;
  for (let v = 0; v < V; v++) mx = Math.max(mx, logits[v]);
  let Z = 0;
  for (let v = 0; v < V; v++) Z += Math.exp(logits[v] - mx);
  const dL = new Float64Array(V);
  for (let v = 0; v < V; v++) {
    let p = Math.exp(logits[v] - mx) / Z;
    if (v === tgt) p -= 1;
    dL[v] = p;
  }
  check('CE dLogits (CE_SOFTMAX_GRAD)', dL, fdGrad(loss, logits));
}

// ---------- LM head dHidden (DHIDDEN_FROM_DLOGITS_I8) ----------
// logits[v] = scaleE[v] * sum_k hidden[k]*E[v][k]  -> dHidden[k] = sum_v dLogits[v]*scaleE[v]*E[v][k]
function lmHead() {
  const K = 6,
    V = 5;
  const hidden = rnd(K),
    E = rnd(V * K),
    scaleE = rnd(V, 0.5).map((x) => Math.abs(x) + 0.1),
    dLogits = rnd(V);
  // loss = sum_v dLogits[v]*logits[v]  => grad wrt hidden is the dHidden formula
  const loss = () => {
    let s = 0;
    for (let v = 0; v < V; v++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += hidden[k] * E[v * K + k];
      s += dLogits[v] * scaleE[v] * acc;
    }
    return s;
  };
  const dHidden = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let acc = 0;
    for (let v = 0; v < V; v++) acc += dLogits[v] * scaleE[v] * E[v * K + k];
    dHidden[k] = acc;
  }
  check('LM head dHidden (DHIDDEN_FROM_DLOGITS_I8)', dHidden, fdGrad(loss, hidden));
}

// ---------- AdamW step (ADAMW_STEP) ----------
function adamw() {
  const n = 5;
  const param = rnd(n),
    grad = rnd(n),
    m = rnd(n, 0.1),
    v = rnd(n, 0.1).map(Math.abs);
  const lr = 1e-3,
    b1 = 0.9,
    b2 = 0.999,
    eps = 1e-8,
    wd = 0.01,
    step = 7;
  const b1c = 1 - b1 ** step,
    b2c = 1 - b2 ** step;
  // reference
  const ref = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const mm = b1 * m[i] + (1 - b1) * grad[i];
    const vv = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
    const mhat = mm / b1c,
      vhat = vv / b2c;
    ref[i] = param[i] - lr * (mhat / (Math.sqrt(vhat) + eps) + wd * param[i]);
  }
  // kernel formula (gScale=1 here)
  const got = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const gr = grad[i] * 1.0;
    const mm = b1 * m[i] + (1 - b1) * gr;
    const vv = b2 * v[i] + (1 - b2) * gr * gr;
    const mhat = mm / b1c,
      vhat = vv / b2c;
    got[i] = param[i] - lr * (mhat / (Math.sqrt(vhat) + eps) + wd * param[i]);
  }
  check('AdamW update (ADAMW_STEP)', got, ref);
}

console.log('--- backward kernel gradient algebra (finite-difference) ---');
loraProj();
rmsnorm();
swiglu();
rope();
attention();
ce();
lmHead();
adamw();
console.log(`BWD-CPU ${FAIL === 0 ? 'ALL PASS' : 'FAILED'} (${PASS}/${PASS + FAIL})`);
process.exit(FAIL === 0 ? 0 : 1);
