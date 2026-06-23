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

// Embed T tokens: out[t][k] = dequant(embed[ids[t]])[k].
export const EMBED_T = `
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> ids: array<u32>;
@group(0) @binding(4) var<uniform> m: vec2<u32>;   // T, H
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let T = m.x; let H = m.y; let N = T*H; let stride = nwg.x * 256u;
  for (var i = gid.x; i < N; i = i + stride) {
    let t = i / H; let k = i % H; let id = ids[t];
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

// GPU argmax over logits -> out[0] = best index (one 4-byte readback instead of 608KB)
export const ARGMAX = `
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> n: u32;
var<workgroup> bv: array<f32,256>; var<workgroup> bi: array<u32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; var v = -1e30; var idx = 0u;
  for (var i = tid; i < n; i = i + 256u) { let x = logits[i]; if (x > v) { v = x; idx = i; } }
  bv[tid] = v; bi[tid] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s/2u) { if (tid < s) { if (bv[tid+s] > bv[tid]) { bv[tid] = bv[tid+s]; bi[tid] = bi[tid+s]; } } workgroupBarrier(); }
  if (tid == 0u) { out[0] = bi[0]; }
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