// WGSL kernels for the custom Qwen2.5 WebGPU runtime. All decode-path (T=1)
// kernels: a token is a vector, so these are GEMV / vector ops. Weights are
// int8 (per-output-channel scale). LoRA A/B are f32 and applied in-kernel, so
// adapters hot-swap by swapping the A/B buffers (no base reload, no requant).

// y[n] = scale[n] * sum_k x[k]*W[n,k]  (+ optional bias)  (+ optional LoRA delta)
// W packed int8 row-major [N][K] (4 per u32). Workgroup per output, 256-thread
// K-reduction (coalesced). LoRA: d[r] = sum_k x[k]*A[k,r]; y += sum_r d[r]*B[r,n].
export const GEMV = `
enable subgroups;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;       // [N][K/4] int8
@group(0) @binding(2) var<storage,read> scale: array<f32>;   // [N]
@group(0) @binding(3) var<storage,read> bias: array<f32>;    // [N] or dummy
@group(0) @binding(4) var<storage,read> loraD: array<f32>;   // [rank] precomputed x@A (or dummy)
@group(0) @binding(5) var<storage,read> loraB: array<f32>;   // [rank][N] (or dummy)
@group(0) @binding(6) var<storage,read_write> y: array<f32>; // [N]
@group(0) @binding(7) var<uniform> m: Meta;
var<workgroup> part: array<f32,64>;       // one slot per subgroup
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }               // workgroup-uniform: whole group exits together
  let K4 = m.K/4u; let rb = n*K4;
  var acc = 0.0;
  for (var k = tid; k < K4; k = k + 64u) {
    let p = w[rb+k];
    let v = unpack4xI8(p);                 // vec4<i32>
    let kk = k*4u;
    acc = acc + x[kk]*f32(v.x) + x[kk+1u]*f32(v.y) + x[kk+2u]*f32(v.z) + x[kk+3u]*f32(v.w);
  }
  let ssum = subgroupAdd(acc);            // reduce within subgroup (no barrier)
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var red = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { red = red + part[i]; }
    var o = red * scale[n];
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = o;
  }
}`;

// LoRA's first matmul d = x @ A. A stored TRANSPOSED [rank][K] so each row r is
// contiguous → coalesced reads. One workgroup per rank output, subgroup reduce.
export const LORA_A = `
enable subgroups;
@group(0) @binding(0) var<storage,read> x: array<f32>;     // [K]
@group(0) @binding(1) var<storage,read> A: array<f32>;     // [rank][K] (transposed)
@group(0) @binding(2) var<storage,read_write> d: array<f32>; // [rank]
@group(0) @binding(3) var<uniform> m: vec2<u32>;           // K, rank
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let r = wid.x; let K = m.x; if (r >= m.y) { return; }
  let rb = r*K; var acc = 0.0;
  for (var k = lid.x; k < K; k = k + 64u) { acc = acc + x[k]*A[rb + k]; }
  let s = subgroupAdd(acc);
  if (sgid == 0u) { part[lid.x / sgsz] = s; }
  workgroupBarrier();
  if (lid.x == 0u) { let nsg=(64u+sgsz-1u)/sgsz; var o=0.0; for(var i=0u;i<nsg;i=i+1u){o=o+part[i];} d[r]=o; }
}`;

// Batched LoRA first matmul for prefill:
// D[t,r] = X[t,:] @ A[r,:]. A layout matches decode LORA_A: [rank][K].
export const LORA_A_BATCH = `
enable subgroups;
@group(0) @binding(0) var<storage,read> x: array<f32>;       // [T][K]
@group(0) @binding(1) var<storage,read> A: array<f32>;       // [rank][K]
@group(0) @binding(2) var<storage,read_write> d: array<f32>; // [T][rank]
@group(0) @binding(3) var<uniform> m: vec4<u32>;             // K, rank, T, _
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let r = wid.x; let t = wid.y; let K = m.x; let rank = m.y; if (r >= rank || t >= m.z) { return; }
  let xb = t*K; let ab = r*K; var acc = 0.0;
  for (var k = lid.x; k < K; k = k + 64u) { acc = acc + x[xb + k]*A[ab + k]; }
  let s = subgroupAdd(acc);
  if (sgid == 0u) { part[lid.x / sgsz] = s; }
  workgroupBarrier();
  if (lid.x == 0u) { let nsg=(64u+sgsz-1u)/sgsz; var o=0.0; for(var i=0u;i<nsg;i=i+1u){o=o+part[i];} d[t*rank + r]=o; }
}`;

// Add batched LoRA second matmul into an existing GEMM output:
// Y[t,n] += scale * sum_r D[t,r] * B[r,n].
export const LORA_B_ADD_T = `
struct Meta { T:u32, N:u32, rank:u32, pad:u32, scale:f32, p1:f32, p2:f32, p3:f32 };
@group(0) @binding(0) var<storage,read> d: array<f32>;        // [T][rank]
@group(0) @binding(1) var<storage,read> B: array<f32>;        // [rank][N]
@group(0) @binding(2) var<storage,read_write> Y: array<f32>;  // [T][N]
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.N; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.N; let n = i % m.N; var acc = 0.0;
    for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + d[t*m.rank + r] * B[r*m.N + n]; }
    Y[i] = Y[i] + m.scale * acc;
  }
}`;

// Add decode LoRA second matmul into an existing vector:
// y[n] += scale * sum_r d[r] * B[r,n].
export const LORA_B_ADD = `
struct Meta { N:u32, rank:u32, p0:u32, p1:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> d: array<f32>;       // [rank]
@group(0) @binding(1) var<storage,read> B: array<f32>;       // [rank][N]
@group(0) @binding(2) var<storage,read_write> y: array<f32>; // [N]
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= m.N) { return; }
  var acc = 0.0;
  for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + d[r] * B[r*m.N + n]; }
  y[n] = y[n] + m.scale * acc;
}`;

// RMSNorm: y = x * rsqrt(mean(x^2)+eps) * g   (single row, K elements)
export const RMSNORM = `
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x);
  var s = 0.0; for (var k = tid; k < K; k = k + 256u) { let v = x[k]; s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = 128u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/m.x + m.y);
  for (var k = tid; k < K; k = k + 256u) { y[k] = x[k]*inv*g[k]; }
}`;

// RoPE on a single row reshaped [nHeads, headDim] for position pos. Each thread
// owns one (lo, hi) PAIR — reads both, then writes both. No cross-thread r/w race
// (in-place rotation needs the original of both halves).
export const ROPE = `
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
@group(0) @binding(3) var<uniform> m: vec3<u32>;             // nHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let pos = m.z; let half = D/2u;
  if (g >= H*half) { return; }
  let h = g / half; let j = g % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = cosT[off]; let s = sinT[off];
  let xl = x[lo]; let xh = x[hi];
  x[lo] = xl*c - xh*s;
  x[hi] = xh*c + xl*s;
}`;

// Combined decode RoPE for Q and K in one dispatch. Qwen layout is
// [head][headDim] with the low/high half rotation at the absolute token position.
export const ROPE_QK = `
@group(0) @binding(0) var<storage,read_write> q: array<f32>;
@group(0) @binding(1) var<storage,read_write> k: array<f32>;
@group(0) @binding(2) var<storage,read> cosT: array<f32>;
@group(0) @binding(3) var<storage,read> sinT: array<f32>;
@group(0) @binding(4) var<uniform> m: vec4<u32>;             // qHeads, kvHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let qH = m.x; let kH = m.y; let D = m.z; let pos = m.w; let half = D/2u;
  let qPairs = qH * half; let kPairs = kH * half; let total = qPairs + kPairs;
  if (g >= total) { return; }
  let isK = g >= qPairs;
  var r = g;
  if (isK) { r = g - qPairs; }
  let h = r / half; let j = r % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = cosT[off]; let s = sinT[off];
  if (isK) {
    let xl = k[lo]; let xh = k[hi];
    k[lo] = xl*c - xh*s; k[hi] = xh*c + xl*s;
  } else {
    let xl = q[lo]; let xh = q[hi];
    q[lo] = xl*c - xh*s; q[hi] = xh*c + xl*s;
  }
}`;

// Split-K (flash-style) decode attention. One workgroup per (head, ctx-chunk) →
// nHeads*nsplit workgroups for high GPU occupancy (vs 16 when one wg/head idles
// most of the GPU at long context). Each writes a partial softmax (max, sum,
// unnormalized weighted-V); ATTN_COMBINE merges the splits per head.
// CHUNK = 128 positions/split (must match runtime). q [nHeads,hd], KV [ctx][nKV][hd].
export const ATTN_PARTIAL = `
enable subgroups;
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;  // [nHeads*nsplit] per-split max
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;  // [nHeads*nsplit] per-split sum
@group(0) @binding(5) var<storage,read_write> po: array<f32>;  // [nHeads*nsplit*hd] unnorm weighted V
@group(0) @binding(6) var<uniform> m: vec4<u32>;               // nHeads, nKV, ctx, hd
@group(0) @binding(7) var<uniform> m2: vec2<u32>;              // nsplit, chunk
var<workgroup> sc: array<f32,128>;
var<workgroup> red: array<f32,32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.x; let nKV = m.y; let ctx = m.z; let hd = m.w; let nsplit = m2.x; let chunk = m2.y;
  let kvh = h / (nHeads / nKV);
  let qbase = h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scale = 1.0/sqrt(f32(hd));
  let nsg = (128u + sgsz - 1u) / sgsz;
  let t0 = s*chunk; var t1 = t0 + chunk; if (t1 > ctx) { t1 = ctx; }
  let t = t0 + tid; var sv = -1e30;
  if (t < t1) { var dot = 0.0; let kb = t*stride + hoff; for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; } sv = dot*scale; }
  let sgm = subgroupMax(sv); if (sgid == 0u) { red[tid/sgsz] = sgm; }
  workgroupBarrier();
  var M = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { M = max(M, red[i]); }
  workgroupBarrier();
  var ev = 0.0; if (t < t1) { ev = exp(sv - M); } sc[tid] = ev;
  let sgs = subgroupAdd(ev); if (sgid == 0u) { red[tid/sgsz] = sgs; }
  workgroupBarrier();
  var Z = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { Z = Z + red[i]; }
  workgroupBarrier();
  let len = t1 - t0; let pbase = (h*nsplit + s)*hd;
  for (var d = tid; d < hd; d = d + 128u) {
    var acc = 0.0; for (var tt = 0u; tt < len; tt = tt + 1u) { acc = acc + sc[tt]*vc[(t0+tt)*stride + hoff + d]; }
    po[pbase + d] = acc;
  }
  if (tid == 0u) { pm[h*nsplit + s] = M; pz[h*nsplit + s] = Z; }
}`;

// Combine split partials per head via online softmax → final o[nHeads*hd].
export const ATTN_COMBINE = `
@group(0) @binding(0) var<storage,read> pm: array<f32>;
@group(0) @binding(1) var<storage,read> pz: array<f32>;
@group(0) @binding(2) var<storage,read> po: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> m: vec4<u32>;   // nHeads, hd, nsplit, _
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let tid = lid.x; let hd = m.y; let nsplit = m.z; let base = h*nsplit;
  var M = -1e30; for (var s = 0u; s < nsplit; s = s + 1u) { M = max(M, pm[base+s]); }
  var Z = 0.0; for (var s = 0u; s < nsplit; s = s + 1u) { Z = Z + pz[base+s]*exp(pm[base+s]-M); }
  let invZ = 1.0 / Z;
  for (var d = tid; d < hd; d = d + 128u) {
    var acc = 0.0;
    for (var s = 0u; s < nsplit; s = s + 1u) { acc = acc + exp(pm[base+s]-M)*po[(base+s)*hd + d]; }
    o[h*hd + d] = acc * invZ;
  }
}`;

// Tiled int4 GEMM for PREFILL (T>1): Y[T][N] = A[T][K] @ dequant(W[N][K])^T (+bias).
// Each workgroup computes a BM(tokens)×BN(cols) output tile; the A K-slice is staged
// in shared memory so activations are reused across the BN columns (the naive
// per-column kernel re-reads activations N times and is slower than sequential decode).
export const GEMM4 = `
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;       // [T][K]
@group(0) @binding(1) var<storage,read> W: array<u32>;       // [N][K/8] int4
@group(0) @binding(2) var<storage,read> scale: array<f32>;   // [N][gpr]
@group(0) @binding(3) var<storage,read> bias: array<f32>;    // [N] or dummy
@group(0) @binding(4) var<storage,read_write> Y: array<f32>; // [T][N]
@group(0) @binding(5) var<uniform> m: Meta;
const BM = 16u; const BN = 64u;
var<workgroup> As: array<f32, 128>;   // BM*8 — A staged for one 8-wide K chunk
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  for (var c = 0u; c < K8; c = c + 1u) {
    for (var l = lid.x; l < BM*8u; l = l + 64u) {
      let tt = l / 8u; let trow = tTile + tt;
      As[l] = select(0.0, A[trow*m.K + c*8u + (l % 8u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0=f32(i32(word<<28u)>>28u)*sc; let w1=f32(i32(word<<24u)>>28u)*sc;
      let w2=f32(i32(word<<20u)>>28u)*sc; let w3=f32(i32(word<<16u)>>28u)*sc;
      let w4=f32(i32(word<<12u)>>28u)*sc; let w5=f32(i32(word<<8u)>>28u)*sc;
      let w6=f32(i32(word<<4u)>>28u)*sc;  let w7=f32(i32(word)>>28u)*sc;
      for (var t = 0u; t < BM; t = t + 1u) {
        let b = t*8u;
        acc[t] = acc[t] + As[b]*w0+As[b+1u]*w1+As[b+2u]*w2+As[b+3u]*w3+As[b+4u]*w4+As[b+5u]*w5+As[b+6u]*w6+As[b+7u]*w7;
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) { let trow = tTile + t; if (trow < m.T) { Y[trow*m.N + col] = acc[t] + bv; } }
  }
}`;

// Batched int4 GEMM that adds directly into an existing residual buffer:
// Y[T][N] += A[T][K] @ dequant(W[N][K])^T (+bias).
export const GEMM4_ADD_T = `
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;
@group(0) @binding(1) var<storage,read> W: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
const BM = 16u; const BN = 64u;
var<workgroup> As: array<f32, 128>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  for (var c = 0u; c < K8; c = c + 1u) {
    for (var l = lid.x; l < BM*8u; l = l + 64u) {
      let tt = l / 8u; let trow = tTile + tt;
      As[l] = select(0.0, A[trow*m.K + c*8u + (l % 8u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0=f32(i32(word<<28u)>>28u)*sc; let w1=f32(i32(word<<24u)>>28u)*sc;
      let w2=f32(i32(word<<20u)>>28u)*sc; let w3=f32(i32(word<<16u)>>28u)*sc;
      let w4=f32(i32(word<<12u)>>28u)*sc; let w5=f32(i32(word<<8u)>>28u)*sc;
      let w6=f32(i32(word<<4u)>>28u)*sc;  let w7=f32(i32(word)>>28u)*sc;
      for (var t = 0u; t < BM; t = t + 1u) {
        let b = t*8u;
        acc[t] = acc[t] + As[b]*w0+As[b+1u]*w1+As[b+2u]*w2+As[b+3u]*w3+As[b+4u]*w4+As[b+5u]*w5+As[b+6u]*w6+As[b+7u]*w7;
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) {
      let trow = tTile + t;
      if (trow < m.T) { Y[trow*m.N + col] = Y[trow*m.N + col] + acc[t] + bv; }
    }
  }
}`;

// y += a (elementwise). Grid-stride so it covers any n with a dispatch capped at the
// 65535-workgroup limit (n reaches T*H ~ 16.7M during prefill).
export const ADD = `
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  for (var i = g.x; i < n; i = i + stride) { y[i] = y[i] + a[i]; }
}`;

// gate = silu(gate) * up  (in place). Grid-stride (n reaches T*I ~ 90M during prefill).
export const SILUMUL = `
@group(0) @binding(0) var<storage,read_write> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  for (var i = g.x; i < n; i = i + stride) { let v = gate[i]; gate[i] = (v/(1.0+exp(-v)))*up[i]; }
}`;

// embed lookup: dequant row `id` of int8 embed [vocab][hidden] -> out[hidden]
export const EMBED = `
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> m: vec2<u32>;   // id, hidden
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; let id = m.x; let H = m.y; if (k >= H) { return; }
  let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
  var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
  out[k] = f32(b) * scale[id];
}`;


// embed lookup but token id comes from a GPU buffer (the argmax output) — lets the
// decode loop chain argmax -> embed on the GPU with no per-token CPU readback.
export const EMBED_BUF = `
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> idbuf: array<u32>;   // idbuf[0] = token id
@group(0) @binding(4) var<uniform> H: u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; let id = idbuf[0]; if (k >= H) { return; }
  let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
  var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
  out[k] = f32(b) * scale[id];
}`;

// ---- PREFILL (T>1) batched ops ----

// RMSNorm over T rows (one workgroup per row).
export const RMSNORM_T = `
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  var s = 0.0; for (var k = tid; k < K; k = k + 256u) { let v = x[base+k]; s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = 128u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/m.x + m.y);
  for (var k = tid; k < K; k = k + 256u) { y[base+k] = x[base+k]*inv*g[k]; }
}`;

// RoPE over T rows [T][nHeads*headDim]; row r is at absolute position pos0+r. Pair-wise (no race).
export const ROPE_T = `
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
@group(0) @binding(3) var<uniform> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = cosT[off]; let s = sinT[off]; let xl = x[lo]; let xh = x[hi];
  x[lo] = xl*c - xh*s; x[hi] = xh*c + xl*s;
}`;

// Embed T tokens: out[t][k] = dequant(embed[ids[idOffset+t]])[k].
export const EMBED_T = `
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> ids: array<u32>;
@group(0) @binding(4) var<uniform> m: vec4<u32>;   // T, H, idOffset, _
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let T = m.x; let H = m.y; let N = T*H; let stride = nwg.x * 256u;
  for (var i = gid.x; i < N; i = i + stride) {
    let t = i / H; let k = i % H; let id = ids[m.z + t];
    let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
    var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
    out[i] = f32(b) * scale[id];
  }
}`;

// Causal attention for prefill: query row t attends keys 0..t. One workgroup per (head, t).
// FLASH / online-softmax: keys are streamed in 256-wide blocks, maintaining a running
// max (mrun), denominator (lrun) and weighted-V accumulator (acc[hd]) — so workgroup
// memory is O(block), NOT O(ctx). This is what lets prefill scale to ctx=8192+ (a full
// sc[ctx] array would be 32KB at 8192 = the entire workgroup-storage budget).
export const ATTN_PREFILL = `
enable subgroups;
@group(0) @binding(0) var<storage,read> q: array<f32>;       // [T][nHeads*hd]
@group(0) @binding(1) var<storage,read> kc: array<f32>;      // [ctx][nKV*hd]
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>; // [T][nHeads*hd]
@group(0) @binding(4) var<uniform> m: vec4<u32>;             // nHeads, nKV, hd, T
var<workgroup> ps: array<f32,256>;   // exp-scores for the current key block
var<workgroup> acc: array<f32,128>;  // running weighted-V accumulator (hd<=128)
var<workgroup> red: array<f32,64>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let t = wid.y; let tid = lid.x; let nHeads = m.x; let nKV = m.y; let hd = m.z;
  let ctx = t + 1u; let kvh = h / (nHeads / nKV);
  let qbase = t*nHeads*hd + h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scl = 1.0/sqrt(f32(hd));
  let nsg = (256u + sgsz - 1u) / sgsz;
  for (var d = tid; d < hd; d = d + 256u) { acc[d] = 0.0; }
  var mrun = -1e30; var lrun = 0.0;
  let nblk = (ctx + 255u) / 256u;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk*256u; let kk = kbase + tid;
    var s = -1e30;
    if (kk < ctx) { var dot = 0.0; let kb = kk*stride + hoff; for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; } s = dot*scl; }
    let sgm = subgroupMax(s); if (sgid == 0u) { red[tid/sgsz] = sgm; }
    workgroupBarrier();                                   // A: block-max partials visible
    var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[i]); }
    let mnew = max(mrun, bm); let corr = exp(mrun - mnew);
    var p = 0.0; if (kk < ctx) { p = exp(s - mnew); }
    ps[tid] = p;
    workgroupBarrier();                                   // B: bm reads done + ps visible
    let sgs = subgroupAdd(p); if (sgid == 0u) { red[tid/sgsz] = sgs; }
    workgroupBarrier();                                   // C: block-sum partials visible
    var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[i]; }
    lrun = lrun*corr + bs;
    let bcount = min(256u, ctx - kbase);
    for (var d = tid; d < hd; d = d + 256u) {
      var aa = acc[d]*corr;
      for (var j = 0u; j < bcount; j = j + 1u) { aa = aa + ps[j]*vc[(kbase+j)*stride + hoff + d]; }
      acc[d] = aa;
    }
    mrun = mnew;
    workgroupBarrier();                                   // D: acc's ps reads done before next block
  }
  let invL = 1.0/lrun;
  for (var d = tid; d < hd; d = d + 256u) { o[qbase + d] = acc[d]*invL; }
}`;

// Block-tiled causal prefill attention. One workgroup handles a small block of
// query rows for a head and streams key blocks from the global KV cache, reusing
// the loaded K/V block across BQ query rows while maintaining independent online
// softmax state per query row. q/o are chunk-local [T][nHeads*hd]; kc/vc are
// global caches indexed by absolute position.
export const ATTN_PREFILL_BLOCK = `
enable subgroups;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, qStart:u32, ctx:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BQ = 4u; const BK = 128u;
var<workgroup> ps: array<f32, 512>;    // BQ*BK
var<workgroup> acc: array<f32, 512>;   // BQ*hd (hd<=128)
var<workgroup> red: array<f32, 128>;   // BQ*subgroup-count
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let qBlock = wid.y; let tid = lid.x; let hd = m.hd;
  let kvh = h / (m.nHeads / m.nKV); let stride = m.nKV * hd; let hoff = kvh * hd;
  let nsg = (128u + sgsz - 1u) / sgsz; let scl = 1.0 / sqrt(f32(hd));
  var mrun: array<f32, 4>; var lrun: array<f32, 4>;
  for (var r = 0u; r < BQ; r = r + 1u) { mrun[r] = -1e30; lrun[r] = 0.0; }
  for (var i = tid; i < BQ*hd; i = i + 128u) { acc[i] = 0.0; }
  workgroupBarrier();
  let nblk = (m.ctx + BK - 1u) / BK;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk * BK; let kk = kbase + tid;
    var score: array<f32, 4>;
    var validQ: array<bool, 4>;
    var dot: array<f32, 4>;
    var corrRun: array<f32, 4>;
    for (var r = 0u; r < BQ; r = r + 1u) {
      let qt = qBlock * BQ + r; let absQ = m.qStart + qt;
      validQ[r] = qt < m.T && kk < m.ctx && kk <= absQ;
      dot[r] = 0.0; score[r] = -1e30;
    }
    if (kk < m.ctx) {
      let kb = kk*stride + hoff;
      for (var d = 0u; d < hd; d = d + 1u) {
        let kval = kc[kb+d];
        for (var r = 0u; r < BQ; r = r + 1u) {
          let qt = qBlock * BQ + r;
          if (validQ[r]) { dot[r] = dot[r] + q[qt*m.nHeads*hd + h*hd + d] * kval; }
        }
      }
      for (var r = 0u; r < BQ; r = r + 1u) {
        if (validQ[r]) { score[r] = dot[r] * scl; }
      }
    }
    for (var r = 0u; r < BQ; r = r + 1u) {
      let s = score[r];
      let sgm = subgroupMax(s);
      if (sgid == 0u) { red[r*32u + tid/sgsz] = sgm; }
      workgroupBarrier();
      var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[r*32u+i]); }
      let mnew = max(mrun[r], bm); let corr = exp(mrun[r] - mnew);
      corrRun[r] = corr;
      var p = 0.0; if (validQ[r]) { p = exp(s - mnew); }
      ps[r*BK + tid] = p;
      workgroupBarrier();
      let sgs = subgroupAdd(p);
      if (sgid == 0u) { red[r*32u + tid/sgsz] = sgs; }
      workgroupBarrier();
      var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[r*32u+i]; }
      lrun[r] = lrun[r] * corr + bs;
      mrun[r] = mnew;
      workgroupBarrier();
    }
    let bcount = min(BK, m.ctx - kbase);
    for (var d = tid; d < hd; d = d + 128u) {
      var aa: array<f32, 4>;
      for (var r = 0u; r < BQ; r = r + 1u) { aa[r] = acc[r*hd+d] * corrRun[r]; }
      for (var j = 0u; j < bcount; j = j + 1u) {
        let vv = vc[(kbase+j)*stride + hoff + d];
        for (var r = 0u; r < BQ; r = r + 1u) { aa[r] = aa[r] + ps[r*BK+j] * vv; }
      }
      for (var r = 0u; r < BQ; r = r + 1u) { acc[r*hd+d] = aa[r]; }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < BQ; r = r + 1u) {
    let qt = qBlock * BQ + r;
    if (qt < m.T) {
      let invL = 1.0 / lrun[r]; let ob = qt*m.nHeads*hd + h*hd;
      for (var d = tid; d < hd; d = d + 128u) { o[ob+d] = acc[r*hd+d] * invL; }
    }
  }
}`;

// GPU argmax over logits -> out[0] = best index (one 4-byte readback instead of 608KB)
export const ARGMAX = `
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> n: u32;
var<workgroup> bv: array<f32,256>; var<workgroup> bi: array<u32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; var v = -1e30; var idx = 0xffffffffu;
  for (var i = tid; i < n; i = i + 256u) { let x = logits[i]; if (x > v || (x == v && i < idx)) { v = x; idx = i; } }
  bv[tid] = v; bi[tid] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s/2u) { if (tid < s) { let ov = bv[tid+s]; let oi = bi[tid+s]; if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; } } workgroupBarrier(); }
  if (tid == 0u) { out[0] = bi[0]; }
}`;

// Repeated exact top-k selection over logits. Each dispatch selects one rank:
// ids[selectedCount] / vals[selectedCount] = best logit whose id is not already
// present in ids[0..selectedCount). This keeps sampling readback to O(k) tokens
// instead of copying the full vocab-sized logits buffer every generated token.
export const TOPK_SELECT = `
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> ids: array<u32>;
@group(0) @binding(2) var<storage,read_write> vals: array<f32>;
@group(0) @binding(3) var<uniform> m: vec2<u32>; // vocabSize, selectedCount
var<workgroup> bv: array<f32,256>; var<workgroup> bi: array<u32,256>;
fn alreadySelected(id: u32, n: u32) -> bool {
  for (var j = 0u; j < n; j = j + 1u) { if (ids[j] == id) { return true; } }
  return false;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let n = m.x; let selected = m.y;
  var v = -1e30; var idx = 0xffffffffu;
  for (var i = tid; i < n; i = i + 256u) {
    let x = logits[i];
    if (!alreadySelected(i, selected) && (x > v || (x == v && i < idx))) { v = x; idx = i; }
  }
  bv[tid] = v; bi[tid] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s/2u) {
    if (tid < s) {
      let ov = bv[tid+s]; let oi = bi[tid+s];
      if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { ids[selected] = bi[0]; vals[selected] = bv[0]; }
}`;

// int4 group-128 GEMV. w: [N][K/8] (8 signed nibbles/word). scale: [N][gpr].
export const GEMV4 = `
enable subgroups;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read> loraD: array<f32>;
@group(0) @binding(5) var<storage,read> loraB: array<f32>;
@group(0) @binding(6) var<storage,read_write> y: array<f32>;
@group(0) @binding(7) var<uniform> m: Meta;
var<workgroup> part: array<f32,64>;       // one slot per subgroup
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }               // workgroup-uniform: whole group exits together
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]    * f32(i32(word << 28u) >> 28u);
    p = p + x[bk+1u] * f32(i32(word << 24u) >> 28u);
    p = p + x[bk+2u] * f32(i32(word << 20u) >> 28u);
    p = p + x[bk+3u] * f32(i32(word << 16u) >> 28u);
    p = p + x[bk+4u] * f32(i32(word << 12u) >> 28u);
    p = p + x[bk+5u] * f32(i32(word << 8u)  >> 28u);
    p = p + x[bk+6u] * f32(i32(word << 4u)  >> 28u);
    p = p + x[bk+7u] * f32(i32(word)        >> 28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);            // reduce within subgroup (no barrier)
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = o;
  }
}`;

// int4 group-128 GEMV that adds directly into y (residual fusion).
export const GEMV4_ADD = `
enable subgroups;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read> loraD: array<f32>;
@group(0) @binding(5) var<storage,read> loraB: array<f32>;
@group(0) @binding(6) var<storage,read_write> y: array<f32>;
@group(0) @binding(7) var<uniform> m: Meta;
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]    * f32(i32(word << 28u) >> 28u);
    p = p + x[bk+1u] * f32(i32(word << 24u) >> 28u);
    p = p + x[bk+2u] * f32(i32(word << 20u) >> 28u);
    p = p + x[bk+3u] * f32(i32(word << 16u) >> 28u);
    p = p + x[bk+4u] * f32(i32(word << 12u) >> 28u);
    p = p + x[bk+5u] * f32(i32(word << 8u)  >> 28u);
    p = p + x[bk+6u] * f32(i32(word << 4u)  >> 28u);
    p = p + x[bk+7u] * f32(i32(word)        >> 28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = y[n] + o;
  }
}`;

// Fused packed QKV decode projection. The packed W/scale/bias layout is
// [Q rows][K rows][V rows], each row still int4 group-128 with the same input K.
export const QKV_GEMV4 = `
enable subgroups;
struct Meta { K:u32, totalN:u32, qN:u32, kN:u32, vN:u32, gpr:u32, gridX:u32, p0:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read_write> qOut: array<f32>;
@group(0) @binding(5) var<storage,read_write> kOut: array<f32>;
@group(0) @binding(6) var<storage,read_write> vOut: array<f32>;
@group(0) @binding(7) var<uniform> m: Meta;
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.totalN) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]    * f32(i32(word << 28u) >> 28u);
    p = p + x[bk+1u] * f32(i32(word << 24u) >> 28u);
    p = p + x[bk+2u] * f32(i32(word << 20u) >> 28u);
    p = p + x[bk+3u] * f32(i32(word << 16u) >> 28u);
    p = p + x[bk+4u] * f32(i32(word << 12u) >> 28u);
    p = p + x[bk+5u] * f32(i32(word << 8u)  >> 28u);
    p = p + x[bk+6u] * f32(i32(word << 4u)  >> 28u);
    p = p + x[bk+7u] * f32(i32(word)        >> 28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    o = o + bias[n];
    if (n < m.qN) {
      qOut[n] = o;
    } else if (n < m.qN + m.kN) {
      kOut[n - m.qN] = o;
    } else {
      vOut[n - m.qN - m.kN] = o;
    }
  }
}`;

// Fused gate/up projections plus SwiGLU for decode over packed [gate][up] rows.
// LoRA deltas for gate and up are optionally added before applying SiLU(gate)*up.
export const GATE_UP_SILU_GEMV4 = `
enable subgroups;
struct Meta0 { K:u32, N:u32, gpr:u32, gridX:u32, gateRank:u32, upRank:u32, hasGateLora:u32, hasUpLora:u32 };
struct Meta1 { gateScaleLo:f32, upScaleLo:f32, p0:f32, p1:f32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
@group(0) @binding(4) var<storage,read> gateD: array<f32>;
@group(0) @binding(5) var<storage,read> gateB: array<f32>;
@group(0) @binding(6) var<storage,read> upD: array<f32>;
@group(0) @binding(7) var<storage,read> upB: array<f32>;
@group(0) @binding(8) var<uniform> m0: Meta0;
@group(0) @binding(9) var<uniform> m1: Meta1;
var<workgroup> partG: array<f32,64>;
var<workgroup> partU: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m0.gridX; let tid = lid.x;
  if (n >= m0.N) { return; }
  let K8 = m0.K/8u; let rbG = n*K8; let rbU = (m0.N + n)*K8;
  let sbG = n*m0.gpr; let sbU = (m0.N + n)*m0.gpr;
  var accG = 0.0; var accU = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let bk = c*8u; let wg = w[rbG+c]; let wu = w[rbU+c];
    let scG = scale[sbG + (bk >> 7u)]; let scU = scale[sbU + (bk >> 7u)];
    let x0=x[bk]; let x1=x[bk+1u]; let x2=x[bk+2u]; let x3=x[bk+3u];
    let x4=x[bk+4u]; let x5=x[bk+5u]; let x6=x[bk+6u]; let x7=x[bk+7u];
    var pg = 0.0; var pu = 0.0;
    pg = pg + x0*f32(i32(wg<<28u)>>28u) + x1*f32(i32(wg<<24u)>>28u) + x2*f32(i32(wg<<20u)>>28u) + x3*f32(i32(wg<<16u)>>28u);
    pg = pg + x4*f32(i32(wg<<12u)>>28u) + x5*f32(i32(wg<<8u)>>28u)  + x6*f32(i32(wg<<4u)>>28u)  + x7*f32(i32(wg)>>28u);
    pu = pu + x0*f32(i32(wu<<28u)>>28u) + x1*f32(i32(wu<<24u)>>28u) + x2*f32(i32(wu<<20u)>>28u) + x3*f32(i32(wu<<16u)>>28u);
    pu = pu + x4*f32(i32(wu<<12u)>>28u) + x5*f32(i32(wu<<8u)>>28u)  + x6*f32(i32(wu<<4u)>>28u)  + x7*f32(i32(wu)>>28u);
    accG = accG + pg * scG; accU = accU + pu * scU;
  }
  let sg = subgroupAdd(accG); let su = subgroupAdd(accU);
  if (sgid == 0u) { partG[tid / sgsz] = sg; partU[tid / sgsz] = su; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var gate = 0.0; var up = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { gate = gate + partG[i]; up = up + partU[i]; }
    if (m0.hasGateLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m0.gateRank; r = r + 1u) { dl = dl + gateD[r] * gateB[r*m0.N + n]; }
      gate = gate + m1.gateScaleLo * dl;
    }
    if (m0.hasUpLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m0.upRank; r = r + 1u) { dl = dl + upD[r] * upB[r*m0.N + n]; }
      up = up + m1.upScaleLo * dl;
    }
    y[n] = (gate / (1.0 + exp(-gate))) * up;
  }
}`;