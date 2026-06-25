/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

// WGSL backward / training kernels for the custom Qwen2.5 WebGPU runtime.
//
// These mirror the forward kernels in kernels.js but compute gradients. The base
// model (int4 group-128 projections, int8 tied embeddings, f32 norms/biases) stays
// FROZEN: backward dequantizes the same int4/int8 buffers on the fly and never
// produces a weight gradient for them. Only the LoRA A/B matrices receive grads.
//
// Conventions:
//   - Forward proj: Y[t,n] = sum_k X[t,k] * deq(W[n,k]) (+ bias[n]) (+ scale*(X@A)@B)
//     W int4 row-major [N][K/8] (8 signed nibbles/word); scale[N][gpr], gpr=K/128.
//   - LoRA: A stored [rank][K] (transposed), B stored [rank][N]. D = X@A -> [T][rank].
//   - All "*_ADD" / grad kernels ACCUMULATE into their output (caller zeroes first),
//     so gradient accumulation across micro-batches is a no-op (just don't zero).
//   - Metadata is pushed via the immediate address space (var<immediate>), matching
//     the forward kernels and runtime _dispatch().

// ---- dX through a frozen int4 projection ----
// dX[t,k] += sum_n dY[t,n] * deq(W[n,k])   (contraction over the forward output dim N)
// Grid-strided over T*K. Correctness-first (re-reads W column-strided); the LoRA
// path dominates trainable FLOPs so this stays simple.
export const GEMM_DX_INT4 = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, K:u32, gpr:u32 };
@group(0) @binding(0) var<storage,read> dY: array<f32>;       // [T][N]
@group(0) @binding(1) var<storage,read> W: array<u32>;        // [N][K/8] int4
@group(0) @binding(2) var<storage,read> scaleW: array<f32>;   // [N][gpr]
@group(0) @binding(3) var<storage,read_write> dX: array<f32>; // [T][K]
var<immediate> m: Meta;
fn deq4(n: u32, k: u32, K8: u32) -> f32 {
  let word = W[n*K8 + (k >> 3u)];
  let shift = (k & 7u) * 4u;
  let nib = i32(word << (28u - shift)) >> 28u;
  return f32(nib) * scaleW[n*m.gpr + (k >> 7u)];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.K; let stride = nwg.x * 256u; let K8 = m.K / 8u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.K; let k = i % m.K;
    var acc = 0.0;
    let yb = t * m.N;
    for (var n = 0u; n < m.N; n = n + 1u) { acc = acc + dY[yb + n] * deq4(n, k, K8); }
    dX[i] = dX[i] + acc;
  }
}`;

// ---- LoRA gradients ----
// dD[t,r] = scale * sum_n dY[t,n] * B[r,n]   (gradient w.r.t. the rank-projection D=X@A)
export const LORA_DD = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, rank:u32, p:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> dY: array<f32>;       // [T][N]
@group(0) @binding(1) var<storage,read> B: array<f32>;        // [rank][N]
@group(0) @binding(2) var<storage,read_write> dD: array<f32>; // [T][rank]
var<immediate> m: Meta;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wid.x; let t = idx / m.rank; let r = idx % m.rank; let tid = lid.x;
  if (t >= m.T) { return; }
  var s = 0.0; let yb = t*m.N; let bb = r*m.N;
  for (var n = tid; n < m.N; n = n + 256u) { s = s + dY[yb + n] * B[bb + n]; }
  part[tid] = s; workgroupBarrier();
  for (var st = 128u; st > 0u; st = st/2u) { if (tid < st) { part[tid] = part[tid] + part[tid+st]; } workgroupBarrier(); }
  if (tid == 0u) { dD[t*m.rank + r] = m.scale * part[0]; }
}`;

// dA[r,k] += sum_t dD[t,r] * X[t,k]   (scale already folded into dD)
export const LORA_GRAD_A = `
requires immediate_address_space;
struct Meta { T:u32, K:u32, rank:u32, p:u32 };
@group(0) @binding(0) var<storage,read> dD: array<f32>;       // [T][rank]
@group(0) @binding(1) var<storage,read> X: array<f32>;        // [T][K]
@group(0) @binding(2) var<storage,read_write> dA: array<f32>; // [rank][K]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.rank * m.K; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let r = i / m.K; let k = i % m.K;
    var acc = 0.0;
    for (var t = 0u; t < m.T; t = t + 1u) { acc = acc + dD[t*m.rank + r] * X[t*m.K + k]; }
    dA[i] = dA[i] + acc;
  }
}`;

// dB[r,n] += scale * sum_t D[t,r] * dY[t,n]   (D = X@A, recomputed without scale)
export const LORA_GRAD_B = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, rank:u32, p:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> D: array<f32>;        // [T][rank]
@group(0) @binding(1) var<storage,read> dY: array<f32>;       // [T][N]
@group(0) @binding(2) var<storage,read_write> dB: array<f32>; // [rank][N]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.rank * m.N; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let r = i / m.N; let n = i % m.N;
    var acc = 0.0;
    for (var t = 0u; t < m.T; t = t + 1u) { acc = acc + D[t*m.rank + r] * dY[t*m.N + n]; }
    dB[i] = dB[i] + m.scale * acc;
  }
}`;

// dX[t,k] += sum_r dD[t,r] * A[r,k]   (LoRA contribution to the input gradient)
export const LORA_DX_ADD = `
requires immediate_address_space;
struct Meta { T:u32, K:u32, rank:u32, p:u32 };
@group(0) @binding(0) var<storage,read> dD: array<f32>;       // [T][rank]
@group(0) @binding(1) var<storage,read> A: array<f32>;        // [rank][K]
@group(0) @binding(2) var<storage,read_write> dX: array<f32>; // [T][K]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.K; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.K; let k = i % m.K;
    var acc = 0.0;
    for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + dD[t*m.rank + r] * A[r*m.K + k]; }
    dX[i] = dX[i] + acc;
  }
}`;

// ---- RMSNorm backward (g frozen, so only dx) ----
// Forward: y[k] = x[k]*inv*g[k], inv = 1/sqrt(mean(x^2)+eps).
// dx_j = inv*g_j*dy_j - inv^3 * x_j * c / K,  c = sum_k dy_k*g_k*x_k.
export const RMSNORM_BWD_T = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;        // [T][K]
@group(0) @binding(1) var<storage,read> g: array<f32>;        // [K]
@group(0) @binding(2) var<storage,read> dy: array<f32>;       // [T][K]
@group(0) @binding(3) var<storage,read_write> dx: array<f32>; // [T][K]
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  // sum of squares for inv
  var ss = 0.0;
  for (var k = tid; k < K; k = k + WG) { let v = x[base+k]; ss = ss + v*v; }
  red[tid] = ss; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  let ms = red[0] / m.x;
  let inv = inverseSqrt(ms + m.y);
  workgroupBarrier();
  // c = sum dy*g*x
  var cc = 0.0;
  for (var k = tid; k < K; k = k + WG) { cc = cc + dy[base+k]*g[k]*x[base+k]; }
  red[tid] = cc; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  let c = red[0];
  let inv3overK = inv*inv*inv / m.x;
  for (var k = tid; k < K; k = k + WG) {
    dx[base+k] = inv*g[k]*dy[base+k] - inv3overK * x[base+k] * c;
  }
}`;

// ---- SwiGLU backward ----
// Forward: out = silu(gate)*up, silu(z)=z*sig(z). Given dOut:
//   dUp   = dOut * silu(gate)
//   dGate = dOut * up * silu'(gate),  silu'(z)=sig(z)*(1 + z*(1-sig(z)))
export const SWIGLU_BWD = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
@group(0) @binding(2) var<storage,read> dOut: array<f32>;
@group(0) @binding(3) var<storage,read_write> dGate: array<f32>;
@group(0) @binding(4) var<storage,read_write> dUp: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = gid.x; i < n; i = i + stride) {
    let z = gate[i]; let sig = 1.0/(1.0+exp(-z)); let sl = z*sig;
    let d = dOut[i];
    dUp[i] = d * sl;
    dGate[i] = d * up[i] * (sig * (1.0 + z*(1.0 - sig)));
  }
}`;

// ---- RoPE backward (transpose of the forward rotation = rotate by -angle) ----
// Forward pair: lo' = lo*c - hi*s ; hi' = hi*c + lo*s.
// Backward:     dlo = c*dlo' + s*dhi' ; dhi = -s*dlo' + c*dhi'  (in place on the grad).
export const ROPE_BWD_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> dx: array<f32>;   // [T][nHeads*headDim] gradient
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = cosT[off]; let s = sinT[off];
  let dl = dx[lo]; let dh = dx[hi];
  dx[lo] = c*dl + s*dh;
  dx[hi] = -s*dl + c*dh;
}`;

// ---- Attention backward (flash-style, recompute) ----
// Stage 1: per (head,t) logsumexp lse[h,t] and delta[h,t]=sum_d do[t,h,d]*o[t,h,d].
export const ATTN_BWD_STATS = `
requires immediate_address_space;
override WG: u32 = 128u;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;     // [T][nHeads*hd]
@group(0) @binding(1) var<storage,read> kc: array<f32>;    // [T][nKV*hd]
@group(0) @binding(2) var<storage,read> o: array<f32>;     // [T][nHeads*hd] attn output
@group(0) @binding(3) var<storage,read> doo: array<f32>;   // [T][nHeads*hd] grad of attn output
@group(0) @binding(4) var<storage,read_write> lse: array<f32>;   // [nHeads*T]
@group(0) @binding(5) var<storage,read_write> delta: array<f32>; // [nHeads*T]
var<immediate> m: Meta;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let t = wid.y; let tid = lid.x;
  let hd = m.hd; let nKV = m.nKV; let kvh = h / (m.nHeads / nKV);
  let qb = t*m.nHeads*hd + h*hd; let kvstride = nKV*hd; let hoff = kvh*hd;
  let scl = 1.0 / sqrt(f32(hd));
  // running max
  var lmax = -1e30;
  for (var j = tid; j <= t; j = j + WG) {
    var dot = 0.0; let kb = j*kvstride + hoff;
    for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qb+d]*kc[kb+d]; }
    lmax = max(lmax, dot*scl);
  }
  red[tid] = lmax; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = max(red[tid], red[tid+s]); } workgroupBarrier(); }
  let M = red[0];
  workgroupBarrier();
  var lsum = 0.0;
  for (var j = tid; j <= t; j = j + WG) {
    var dot = 0.0; let kb = j*kvstride + hoff;
    for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qb+d]*kc[kb+d]; }
    lsum = lsum + exp(dot*scl - M);
  }
  red[tid] = lsum; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  // delta
  var dl = 0.0;
  for (var d = tid; d < hd; d = d + WG) { dl = dl + doo[qb+d]*o[qb+d]; }
  // reuse red after sum captured
  let Z = red[0];
  workgroupBarrier();
  red[tid] = dl; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  if (tid == 0u) { lse[h*m.T + t] = M + log(Z); delta[h*m.T + t] = red[0]; }
}`;

// Stage 2: dq[t,h,:] = scl * sum_{j<=t} ds_{t,j} k_j, ds = p*(dp - delta), p = exp(s - lse).
// One workgroup per (head,t); one thread per head-dim channel (hd <= 128).
export const ATTN_BWD_DQ = `
requires immediate_address_space;
override WG: u32 = 128u;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read> doo: array<f32>;
@group(0) @binding(4) var<storage,read> lse: array<f32>;
@group(0) @binding(5) var<storage,read> delta: array<f32>;
@group(0) @binding(6) var<storage,read_write> dq: array<f32>;
var<immediate> m: Meta;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let t = wid.y; let d = lid.x;
  let hd = m.hd; let nKV = m.nKV; let kvh = h / (m.nHeads / nKV);
  let qb = t*m.nHeads*hd + h*hd; let kvstride = nKV*hd; let hoff = kvh*hd;
  let scl = 1.0 / sqrt(f32(hd));
  let lse_t = lse[h*m.T + t]; let delta_t = delta[h*m.T + t];
  // Guard every storage read behind (d < hd): WGSL select() is eager and would
  // still evaluate the buffer load for inactive lanes (OOB when hd < WG). Barriers
  // stay at uniform control flow so the reductions remain valid.
  let inHd = d < hd;
  var acc = 0.0;
  for (var j = 0u; j <= t; j = j + 1u) {
    let kb = j*kvstride + hoff;
    // s = scl * dot(q, k_j)
    var sv = 0.0; if (inHd) { sv = q[qb+d] * kc[kb+d]; }
    red[d] = sv; workgroupBarrier();
    for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
    let sval = red[0] * scl;
    workgroupBarrier();
    // dp = dot(do, v_j)
    var dpv = 0.0; if (inHd) { dpv = doo[qb+d] * vc[kb+d]; }
    red[d] = dpv; workgroupBarrier();
    for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
    let dp = red[0];
    workgroupBarrier();
    let p = exp(sval - lse_t);
    let ds = p * (dp - delta_t);
    if (inHd) { acc = acc + ds * kc[kb+d]; }
  }
  if (inHd) { dq[qb+d] = dq[qb+d] + scl * acc; }
}`;

// Stage 3: per (kvHead, j) accumulate dk_j and dv_j over all heads in the GQA group
// and all queries t >= j. dk_j += scl*ds*q_t ; dv_j += p*do_t.
export const ATTN_BWD_DKV = `
requires immediate_address_space;
override WG: u32 = 128u;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read> doo: array<f32>;
@group(0) @binding(4) var<storage,read> lse: array<f32>;
@group(0) @binding(5) var<storage,read> delta: array<f32>;
@group(0) @binding(6) var<storage,read_write> dk: array<f32>;
@group(0) @binding(7) var<storage,read_write> dv: array<f32>;
var<immediate> m: Meta;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let kvh = wid.x; let j = wid.y; let d = lid.x;
  let hd = m.hd; let nKV = m.nKV; let group = m.nHeads / nKV;
  let kvstride = nKV*hd; let hoff = kvh*hd; let kb = j*kvstride + hoff;
  let scl = 1.0 / sqrt(f32(hd));
  // Guard storage reads behind (d < hd) — see ATTN_BWD_DQ note on eager select().
  let inHd = d < hd;
  var dkacc = 0.0; var dvacc = 0.0;
  for (var hi = 0u; hi < group; hi = hi + 1u) {
    let h = kvh*group + hi;
    for (var t = j; t < m.T; t = t + 1u) {
      let qb = t*m.nHeads*hd + h*hd;
      var sv = 0.0; if (inHd) { sv = q[qb+d] * kc[kb+d]; }
      red[d] = sv; workgroupBarrier();
      for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
      let sval = red[0] * scl;
      workgroupBarrier();
      var dpv = 0.0; if (inHd) { dpv = doo[qb+d] * vc[kb+d]; }
      red[d] = dpv; workgroupBarrier();
      for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
      let dp = red[0];
      workgroupBarrier();
      let p = exp(sval - lse[h*m.T + t]);
      let ds = p * (dp - delta[h*m.T + t]);
      if (inHd) {
        dkacc = dkacc + scl * ds * q[qb+d];
        dvacc = dvacc + p * doo[qb+d];
      }
    }
  }
  if (inHd) { dk[kb+d] = dk[kb+d] + dkacc; dv[kb+d] = dv[kb+d] + dvacc; }
}`;

// ---- LM head (tied int8 embeddings) ----
// Forward logits block: Y[t,v] = scaleE[v] * sum_k normed[t,k]*i8(E[v,k]).
// Grid-strided over Tblock*vocab; each invocation does the full K reduction itself.
export const LOGITS_GEMM_I8 = `
requires immediate_address_space;
struct Meta { T:u32, vocab:u32, K:u32, tOff:u32 };
@group(0) @binding(0) var<storage,read> normed: array<f32>;   // [T][K] (full-seq buffer, offset by tOff)
@group(0) @binding(1) var<storage,read> E: array<u32>;        // [vocab][K/4] int8
@group(0) @binding(2) var<storage,read> scaleE: array<f32>;   // [vocab]
@group(0) @binding(3) var<storage,read_write> logits: array<f32>; // [Tblock][vocab]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.vocab; let stride = nwg.x * 256u; let K4 = m.K / 4u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.vocab; let v = i % m.vocab;
    let nb = (m.tOff + t) * m.K; let eb = v * K4;
    var acc = 0.0;
    for (var c = 0u; c < K4; c = c + 1u) {
      let p = unpack4xI8(E[eb + c]); let kk = c*4u;
      acc = acc + normed[nb+kk]*f32(p.x) + normed[nb+kk+1u]*f32(p.y)
                + normed[nb+kk+2u]*f32(p.z) + normed[nb+kk+3u]*f32(p.w);
    }
    logits[i] = acc * scaleE[v];
  }
}`;

// CE + softmax gradient. Per token (one workgroup): softmax over vocab, write
// dLogits = (softmax - onehot(target)) * lossScale in place, and per-token loss.
// Masked tokens (mask==0) produce zero grad and zero loss.
// logits is a token-BLOCK buffer [bt][vocab] (local index), while target/mask/loss are
// full-sequence buffers indexed by the global token id (tOff + localT).
export const CE_SOFTMAX_GRAD = `
requires immediate_address_space;
override WG: u32 = 256u;
struct Meta { vocab:u32, tOff:u32, lossScale:f32, p:u32 };
@group(0) @binding(0) var<storage,read_write> logits: array<f32>; // [bt][vocab] -> dLogits
@group(0) @binding(1) var<storage,read> labels: array<u32>;       // [T] token id (global)
@group(0) @binding(2) var<storage,read> mask: array<f32>;         // [T] 1 train / 0 skip (global)
@group(0) @binding(3) var<storage,read_write> lossOut: array<f32>;// [T] (global)
var<immediate> m: Meta;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lt = wid.x; let tid = lid.x; let base = lt*m.vocab;
  let gt = m.tOff + lt;            // global token index for target/mask/loss
  let mk = mask[gt];
  // max
  var mx = -1e30;
  for (var v = tid; v < m.vocab; v = v + WG) { mx = max(mx, logits[base+v]); }
  red[tid] = mx; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = max(red[tid], red[tid+s]); } workgroupBarrier(); }
  let M = red[0]; workgroupBarrier();
  // sum exp
  var sm = 0.0;
  for (var v = tid; v < m.vocab; v = v + WG) { sm = sm + exp(logits[base+v] - M); }
  red[tid] = sm; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  let Z = red[0];
  let tgt = labels[gt];
  if (tid == 0u) {
    let ltgt = logits[base + tgt];
    lossOut[gt] = mk * (log(Z) - (ltgt - M));
  }
  // dLogits = mask*lossScale*(p - onehot)
  let invZ = 1.0 / Z; let g = mk * m.lossScale;
  for (var v = tid; v < m.vocab; v = v + WG) {
    var p = exp(logits[base+v] - M) * invZ;
    if (v == tgt) { p = p - 1.0; }
    logits[base+v] = g * p;
  }
}`;

// dHidden[t,k] += sum_v dLogits[t,v] * scaleE[v] * i8(E[v,k])   (tied embeddings, frozen)
export const DHIDDEN_FROM_DLOGITS_I8 = `
requires immediate_address_space;
struct Meta { T:u32, vocab:u32, K:u32, tOff:u32 };
@group(0) @binding(0) var<storage,read> dLogits: array<f32>;  // [Tblock][vocab]
@group(0) @binding(1) var<storage,read> E: array<u32>;        // [vocab][K/4] int8
@group(0) @binding(2) var<storage,read> scaleE: array<f32>;   // [vocab]
@group(0) @binding(3) var<storage,read_write> dHidden: array<f32>; // [T][K] (offset tOff)
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.K; let stride = nwg.x * 256u; let K4 = m.K / 4u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.K; let k = i % m.K;
    let lb = t * m.vocab;
    var acc = 0.0;
    let word_idx = k >> 2u; let lane = k & 3u;
    for (var v = 0u; v < m.vocab; v = v + 1u) {
      let p = unpack4xI8(E[v*K4 + word_idx]);
      var b: i32; if (lane==0u){b=p.x;} else if (lane==1u){b=p.y;} else if (lane==2u){b=p.z;} else {b=p.w;}
      acc = acc + dLogits[lb + v] * scaleE[v] * f32(b);
    }
    dHidden[(m.tOff + t)*m.K + k] = dHidden[(m.tOff + t)*m.K + k] + acc;
  }
}`;

// ---- AdamW (decoupled weight decay), in place on a LoRA param + its moments ----
// Caller passes bias-corrected step. gScale folds 1/accumSteps + grad-clip factor.
export const ADAMW_STEP = `
requires immediate_address_space;
struct Meta { n:u32, p:u32, lr:f32, beta1:f32, beta2:f32, eps:f32, wd:f32, gScale:f32, b1c:f32, b2c:f32, f0:f32, f1:f32 };
@group(0) @binding(0) var<storage,read_write> param: array<f32>;
@group(0) @binding(1) var<storage,read> grad: array<f32>;
@group(0) @binding(2) var<storage,read_write> mBuf: array<f32>;
@group(0) @binding(3) var<storage,read_write> vBuf: array<f32>;
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  for (var i = gid.x; i < m.n; i = i + stride) {
    let gr = grad[i] * m.gScale;
    let mm = m.beta1 * mBuf[i] + (1.0 - m.beta1) * gr;
    let vv = m.beta2 * vBuf[i] + (1.0 - m.beta2) * gr * gr;
    mBuf[i] = mm; vBuf[i] = vv;
    let mhat = mm / m.b1c; let vhat = vv / m.b2c;
    param[i] = param[i] - m.lr * (mhat / (sqrt(vhat) + m.eps) + m.wd * param[i]);
  }
}`;

// Sum of squares of a buffer into a single-element accumulator (for global grad-norm).
// out[0] += sum_i x[i]^2  (caller zeroes out[0]; one workgroup, grid-strided load).
export const SUMSQ = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;  // [1]
var<immediate> n: u32;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; var s = 0.0;
  for (var i = tid; i < n; i = i + WG) { let v = x[i]; s = s + v*v; }
  red[tid] = s; workgroupBarrier();
  for (var st = WG/2u; st > 0u; st = st/2u) { if (tid < st) { red[tid] = red[tid] + red[tid+st]; } workgroupBarrier(); }
  if (tid == 0u) { out[0] = out[0] + red[0]; }
}`;
