/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */

const N = 11008,
  K = 2048,
  G = 128; // gate_proj shape
const K8 = K / 8,
  GPR = K / G;

// --- kernel variants (parametrized) ---
// V0: baseline — wg=256, 1 row/wg, subgroup reduce (current runtime kernel)
const V0 = (WG) => `
enable subgroups;
struct Meta { K:u32, N:u32, gridX:u32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${WG}u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]*f32(i32(word<<28u)>>28u) + x[bk+1u]*f32(i32(word<<24u)>>28u) + x[bk+2u]*f32(i32(word<<20u)>>28u) + x[bk+3u]*f32(i32(word<<16u)>>28u);
    p = p + x[bk+4u]*f32(i32(word<<12u)>>28u) + x[bk+5u]*f32(i32(word<<8u)>>28u) + x[bk+6u]*f32(i32(word<<4u)>>28u) + x[bk+7u]*f32(i32(word)>>28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) { let nsg=(${WG}u+sgsz-1u)/sgsz; var o=0.0; for(var i=0u;i<nsg;i=i+1u){o=o+part[i];} y[n]=o; }
}`;

// V1: ROWS rows per workgroup, full WG cooperates per row, subgroup reduce, x cached in shared mem.
const V1 = (WG, ROWS) => `
enable subgroups;
struct Meta { K:u32, N:u32, gridX:u32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
var<workgroup> xs: array<f32, ${K}>;
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let tid = lid.x; let K8 = m.K/8u;
  for (var k = tid; k < m.K; k = k + ${WG}u) { xs[k] = x[k]; }
  workgroupBarrier();
  let row0 = (wid.x + wid.y * m.gridX) * ${ROWS}u;
  for (var rr = 0u; rr < ${ROWS}u; rr = rr + 1u) {
    let n = row0 + rr; if (n >= m.N) { break; }
    let rb = n*K8; let sbase = n*m.gpr; var acc = 0.0;
    for (var c = tid; c < K8; c = c + ${WG}u) {
      let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
      var p = 0.0;
      p = p + xs[bk]*f32(i32(word<<28u)>>28u) + xs[bk+1u]*f32(i32(word<<24u)>>28u) + xs[bk+2u]*f32(i32(word<<20u)>>28u) + xs[bk+3u]*f32(i32(word<<16u)>>28u);
      p = p + xs[bk+4u]*f32(i32(word<<12u)>>28u) + xs[bk+5u]*f32(i32(word<<8u)>>28u) + xs[bk+6u]*f32(i32(word<<4u)>>28u) + xs[bk+7u]*f32(i32(word)>>28u);
      acc = acc + p * sc;
    }
    let ssum = subgroupAdd(acc);
    if (sgid == 0u) { part[tid / sgsz] = ssum; }
    workgroupBarrier();
    if (tid == 0u) { let nsg=(${WG}u+sgsz-1u)/sgsz; var o=0.0; for(var i=0u;i<nsg;i=i+1u){o=o+part[i];} y[n]=o; }
    workgroupBarrier();
  }
}`;

// V2: one subgroup per row (32 lanes), ROWS = WG/sgsz rows per workgroup. No workgroupBarrier in inner loop.
const V2 = (WG) => `
enable subgroups;
struct Meta { K:u32, N:u32, gridX:u32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let sgPerWg = ${WG}u / sgsz; let sgIdx = lid.x / sgsz;
  let n = (wid.x + wid.y * m.gridX) * sgPerWg + sgIdx; if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr; var acc = 0.0;
  for (var c = sgid; c < K8; c = c + sgsz) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]*f32(i32(word<<28u)>>28u) + x[bk+1u]*f32(i32(word<<24u)>>28u) + x[bk+2u]*f32(i32(word<<20u)>>28u) + x[bk+3u]*f32(i32(word<<16u)>>28u);
    p = p + x[bk+4u]*f32(i32(word<<12u)>>28u) + x[bk+5u]*f32(i32(word<<8u)>>28u) + x[bk+6u]*f32(i32(word<<4u)>>28u) + x[bk+7u]*f32(i32(word)>>28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { y[n] = ssum; }
}`;

window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups', 'timestamp-query'] });
  dev.addEventListener?.('uncapturederror', (e) => console.log('VWG GPUERR ' + e.error.message.slice(0, 200)));
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const wbuf = dev.createBuffer({ size: N * K8 * 4, usage: S });
  const sbuf = dev.createBuffer({ size: N * GPR * 4, usage: S });
  const xbuf = dev.createBuffer({ size: K * 4, usage: S });
  const ybuf = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const ubuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // random-ish fill
  const wd = new Uint32Array(N * K8);
  for (let i = 0; i < wd.length; i++) wd[i] = (i * 2654435761) >>> 0;
  dev.queue.writeBuffer(wbuf, 0, wd);
  const sd = new Float32Array(N * GPR).fill(0.01);
  dev.queue.writeBuffer(sbuf, 0, sd);
  const xd = new Float32Array(K);
  for (let i = 0; i < K; i++) xd[i] = Math.sin(i) * 0.1;
  dev.queue.writeBuffer(xbuf, 0, xd);

  const qs = dev.createQuerySet({ type: 'timestamp', count: 2 });
  const qres = dev.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qread = dev.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const bench = async (label, code, dispatchFn) => {
    let pipe;
    try {
      pipe = dev.createComputePipeline({
        layout: 'auto',
        compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' },
      });
    } catch (e) {
      console.log('VWG ' + label + ' COMPILE-ERR ' + e.message.slice(0, 120));
      return;
    }
    const meta = new Uint32Array([K, N, 65535, GPR]);
    dev.queue.writeBuffer(ubuf, 0, meta);
    const bg = dev.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xbuf } },
        { binding: 1, resource: { buffer: wbuf } },
        { binding: 2, resource: { buffer: sbuf } },
        { binding: 3, resource: { buffer: ybuf } },
        { binding: 4, resource: { buffer: ubuf } },
      ],
    });
    const REP = 200;
    // warmup
    {
      const e = dev.createCommandEncoder();
      for (let i = 0; i < 5; i++) dispatchFn(e, pipe, bg);
      dev.queue.submit([e.finish()]);
      await dev.queue.onSubmittedWorkDone();
    }
    const enc = dev.createCommandEncoder();
    const p = enc.beginComputePass({
      timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    });
    for (let i = 0; i < REP; i++) {
      p.setPipeline(pipe);
      dispatchFn(null, pipe, bg, p);
    }
    p.end();
    enc.resolveQuerySet(qs, 0, 2, qres, 0);
    enc.copyBufferToBuffer(qres, 0, qread, 0, 16);
    dev.queue.submit([enc.finish()]);
    await qread.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(qread.getMappedRange());
    const us = Number(t[1] - t[0]) / 1000 / REP;
    qread.unmap();
    const gbW = (N * K8 * 4) / (us / 1e6) / 1e9; // weight bytes/s
    console.log('VWG ' + us.toFixed(1).padStart(7) + ' us  ' + gbW.toFixed(0).padStart(4) + ' GB/s  ' + label);
  };
  const wgDispatch = (rowsPerWg) => (e, pipe, bg, pass) => {
    const groups = Math.ceil(N / rowsPerWg);
    const gx = Math.min(groups, 65535),
      gy = Math.ceil(groups / gx);
    if (pass) {
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(gx, gy);
    }
  };
  // sgPerWg assumed 8 (Apple sgsz=32, WG=256) for V2 row count
  await bench('V0 wg=256 1row', V0(256), wgDispatch(1));
  await bench('V0 wg=64  1row', V0(64), wgDispatch(1));
  await bench('V1 wg=256 ROWS=4', V1(256, 4), wgDispatch(4));
  await bench('V1 wg=256 ROWS=8', V1(256, 8), wgDispatch(8));
  await bench('V1 wg=128 ROWS=8', V1(128, 8), wgDispatch(8));
  await bench('V2 wg=256 sg/row', V2(256), wgDispatch(8)); // 8 subgroups/wg → 8 rows
  await bench('V2 wg=128 sg/row', V2(128), wgDispatch(4)); // 4 subgroups/wg → 4 rows
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 200))),
);
