var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// src/config.js
var QWEN25_3B = {
  hiddenSize: 2048,
  numLayers: 36,
  numHeads: 16,
  numKVHeads: 2,
  headDim: 128,
  intermediateSize: 11008,
  vocabSize: 151936,
  rmsNormEps: 1e-6,
  ropeTheta: 1e6,
  /*
   * TECHNIQUE: Tie word embeddings
   *   input embedding == output head.
   *   Simplifies loading (one tensor), schema, and final projection math.
   *   Required by the current model_uploader + schema.
   */
  tieWordEmbeddings: true,
  // QKV projections carry a bias in Qwen2.5; o_proj and the MLP do not.
  attentionBias: true
};

// src/qwgpu/model_schema.js
var arrEq = /* @__PURE__ */ __name((a, b) => a.length === b.length && a.every((v, i) => v === b[i]), "arrEq");
function projDesc(layer, subpath, outDim, inDim, { bias = false } = {}) {
  const name = `model.layers.${layer}.${subpath}.weight`;
  const m = subpath.match(/^(self_attn|mlp)\.(.+)$/);
  const loraKey = `layers.${layer}.${m[1]}.${m[2]}`;
  return {
    name,
    role: "projection",
    quant: "int4",
    shape: [outDim, inDim],
    loraKey,
    biasName: bias ? name.replace(/\.weight$/, ".bias") : null
  };
}
__name(projDesc, "projDesc");
function f32Desc(name, shape, role = "f32") {
  return { name, role, quant: "f32", shape };
}
__name(f32Desc, "f32Desc");
function createQwenSchema(cfg) {
  if (!cfg.tieWordEmbeddings && cfg.tieWordEmbeddings !== void 0) {
    throw new Error("QwenWGPU currently requires tied input/output embeddings");
  }
  const H = cfg.hiddenSize;
  const QD = cfg.numHeads * cfg.headDim;
  const KVD = cfg.numKVHeads * cfg.headDim;
  const I = cfg.intermediateSize;
  const tensors = [];
  const layers = [];
  const add = /* @__PURE__ */ __name((d) => {
    tensors.push(d);
    return d;
  }, "add");
  const embed = add({ name: "model.embed_tokens.weight", role: "embedding", quant: "int8", shape: [cfg.vocabSize, H] });
  const finalNorm = add(f32Desc("model.norm.weight", [H], "final_norm"));
  for (let i = 0; i < cfg.numLayers; i++) {
    const p = `model.layers.${i}`;
    const layer = {
      index: i,
      inputNorm: add(f32Desc(`${p}.input_layernorm.weight`, [H], "input_norm")),
      postAttentionNorm: add(f32Desc(`${p}.post_attention_layernorm.weight`, [H], "post_attention_norm")),
      projections: {},
      biases: {}
    };
    layer.projections.q = add(projDesc(i, "self_attn.q_proj", QD, H, { bias: !!cfg.attentionBias }));
    layer.projections.k = add(projDesc(i, "self_attn.k_proj", KVD, H, { bias: !!cfg.attentionBias }));
    layer.projections.v = add(projDesc(i, "self_attn.v_proj", KVD, H, { bias: !!cfg.attentionBias }));
    layer.projections.o = add(projDesc(i, "self_attn.o_proj", H, QD));
    layer.projections.gate = add(projDesc(i, "mlp.gate_proj", I, H));
    layer.projections.up = add(projDesc(i, "mlp.up_proj", I, H));
    layer.projections.down = add(projDesc(i, "mlp.down_proj", H, I));
    for (const key of ["q", "k", "v"]) {
      const proj = layer.projections[key];
      if (proj.biasName) {
        const bias = add(f32Desc(proj.biasName, [proj.shape[0]], `${key}_bias`));
        layer.biases[key] = bias;
      }
    }
    layers.push(layer);
  }
  const byName = new Map(tensors.map((t) => [t.name, t]));
  const expectedNames = new Set(byName.keys());
  return {
    cfg,
    tensors,
    byName,
    expectedNames,
    layers,
    embed,
    finalNorm,
    projectionDescs: tensors.filter((t) => t.role === "projection"),
    validateTensor(name, shape) {
      const desc = byName.get(name);
      if (!desc) return null;
      if (!arrEq(shape, desc.shape)) {
        throw new Error(`shape mismatch for ${name}: got [${shape.join(",")}], expected [${desc.shape.join(",")}]`);
      }
      return desc;
    },
    assertComplete(seen) {
      const missing = [];
      for (const name of expectedNames) if (!seen.has(name)) missing.push(name);
      if (missing.length) {
        const sample = missing.slice(0, 12).join(", ");
        throw new Error(`missing ${missing.length} required tensor(s): ${sample}${missing.length > 12 ? ", \u2026" : ""}`);
      }
    }
  };
}
__name(createQwenSchema, "createQwenSchema");
function moduleKeyFromTensorName(name) {
  const m = name.match(/layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+?)(_proj)?\.(lora_[ABab])/i);
  if (!m) return null;
  return `layers.${m[1]}.${m[2]}.${m[3].replace(/_proj$/, "")}_proj`;
}
__name(moduleKeyFromTensorName, "moduleKeyFromTensorName");

// src/lora_gpu.js
function parseSt(buf) {
  const dv = new DataView(buf);
  const hl = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  return { header, dataStart: 8 + hl, u8: new Uint8Array(buf) };
}
__name(parseSt, "parseSt");
function bf16f32(u8, off, n) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset + off, n);
  const o = new Float32Array(n);
  const o32 = new Uint32Array(o.buffer);
  for (let i = 0; i < n; i++) o32[i] = u16[i] << 16;
  return o;
}
__name(bf16f32, "bf16f32");
function f32(u8, off, n) {
  return new Float32Array(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + n * 4));
}
__name(f32, "f32");
function readTensor(st, name) {
  const t = st.header[name];
  const n = t.shape.reduce((a, b) => a * b, 1);
  const dt = t.dtype.toUpperCase();
  const arr = dt === "BF16" ? bf16f32(st.u8, st.dataStart + t.data_offsets[0], n) : f32(st.u8, st.dataStart + t.data_offsets[0], n);
  return { arr, shape: t.shape };
}
__name(readTensor, "readTensor");
var isA = /* @__PURE__ */ __name((name) => /lora_a/i.test(name), "isA");
function transpose2d(arr, rows, cols) {
  const o = new Float32Array(arr.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = arr[r * cols + c];
  return o;
}
__name(transpose2d, "transpose2d");
async function loadLoraAdapterGPU(dev, files, cfg) {
  const stFile = files.find((f) => f.name.endsWith(".safetensors"));
  if (!stFile) throw new Error("no .safetensors in adapter files");
  const cfgFile = files.find((f) => /adapter_config\.json|config\.json/.test(f.name));
  let rankCfg = 16, scaleCfg = null;
  if (cfgFile) {
    const c = JSON.parse(await cfgFile.text());
    const lp = c.lora_parameters || {};
    rankCfg = c.r ?? c.rank ?? c.lora_rank ?? lp.rank ?? rankCfg;
    if (lp.scale != null)
      scaleCfg = lp.scale;
    else if (c.lora_alpha != null)
      scaleCfg = c.lora_alpha / rankCfg;
    else if (c.alpha != null) scaleCfg = c.alpha / rankCfg;
  }
  const st = parseSt(await stFile.arrayBuffer());
  const names = Object.keys(st.header).filter((k) => k !== "__metadata__" && /lora_[abAB]/.test(k));
  const groups = {};
  for (const nm of names) {
    const key = moduleKeyFromTensorName(nm);
    if (!key) continue;
    (groups[key] ||= {})[isA(nm) ? "A" : "B"] = readTensor(st, nm);
  }
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = /* @__PURE__ */ __name((arr) => {
    const b = dev.createBuffer({ size: arr.byteLength, usage: S });
    dev.queue.writeBuffer(b, 0, arr);
    return b;
  }, "mk");
  const modules = {};
  for (const key of Object.keys(groups)) {
    const g = groups[key];
    if (!g.A || !g.B) continue;
    const r = Math.min(...g.A.shape, ...g.B.shape);
    let Aarr = g.A.arr;
    if (g.A.shape[0] !== r) Aarr = transpose2d(g.A.arr, g.A.shape[0], g.A.shape[1]);
    let Barr = g.B.arr;
    if (g.B.shape[0] !== r) Barr = transpose2d(g.B.arr, g.B.shape[0], g.B.shape[1]);
    const scale = scaleCfg != null ? scaleCfg : 2;
    modules[key] = { A: mk(Aarr), B: mk(Barr), rawA: Aarr, rawB: Barr, rank: r, scale };
  }
  if (!Object.keys(modules).length) throw new Error("no LoRA modules matched layers.*.{self_attn,mlp}.*_proj");
  const name = stFile.name.replace(/\.safetensors$/, "");
  return { name, modules };
}
__name(loadLoraAdapterGPU, "loadLoraAdapterGPU");

// src/readers.js
function urlReader(baseUrl, headers = {}) {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  return {
    async range(path, start, end) {
      const r = await fetch(base + path, {
        headers: { ...headers, Range: `bytes=${start}-${end - 1}` }
      });
      if (!r.ok && r.status !== 206) {
        throw new Error(`range ${path} ${start}-${end}: ${r.status}`);
      }
      return await r.arrayBuffer();
    },
    async text(path) {
      const r = await fetch(base + path, { headers });
      if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
      return await r.text();
    }
  };
}
__name(urlReader, "urlReader");
function hfReader(repo, token = "", rev = "main") {
  return urlReader(
    `https://huggingface.co/${repo}/resolve/${rev}`,
    token ? { Authorization: `Bearer ${token}` } : {}
  );
}
__name(hfReader, "hfReader");
function fileReader(fileMap) {
  const pick = /* @__PURE__ */ __name((path) => fileMap[path] || fileMap[path.split("/").pop()], "pick");
  return {
    async range(path, start, end) {
      const f = pick(path);
      if (!f) throw new Error(`file not provided: ${path}`);
      return await f.slice(start, end).arrayBuffer();
    },
    async text(path) {
      const f = pick(path);
      if (!f) throw new Error(`file not provided: ${path}`);
      return await f.text();
    }
  };
}
__name(fileReader, "fileReader");

// src/services/adapter_registry.js
var AdapterRegistry = class {
  static {
    __name(this, "AdapterRegistry");
  }
  constructor() {
    this.adapters = { none: null };
  }
  add(name, modules) {
    this.adapters[name] = { modules };
    return this.adapters[name];
  }
  get(name) {
    return this.adapters[name] || null;
  }
  /*
   * TECHNIQUE: Runtime adapter swapping via setLora
   *   Registry holds pre-uploaded A/B buffers. applyToRuntime calls
   *   rt.setLora which just swaps references — no weight reload.
   */
  applyToRuntime(name, rt) {
    const adapter = this.get(name);
    if (adapter) rt.setLora(adapter);
    else rt.clearLora();
    return adapter;
  }
};

// src/services/generation_controller.js
var GenerationController = class {
  static {
    __name(this, "GenerationController");
  }
  constructor({ session: session2, adapters: adapters2, systemPrompt, log: log2 = /* @__PURE__ */ __name(() => {
  }, "log") }) {
    this.session = session2;
    this.adapters = adapters2;
    this.systemPrompt = systemPrompt;
    this.log = log2;
  }
  /*
   * TECHNIQUE: Streaming text output with TextNode append (O(n) not O(n^2))
   *   Uses a single Text node and appends characters instead of setting
   *   .textContent repeatedly. Avoids quadratic cost during long generations.
   */
  async runTriage({ adapterName, report, outputNode, maxTemperature = 0 }) {
    const rt = this.session.rt;
    if (!rt) return;
    outputNode.textContent = "";
    const node = document.createTextNode("");
    outputNode.appendChild(node);
    this.adapters.applyToRuntime(adapterName, rt);
    this.log(`generating (adapter=${adapterName})\u2026`);
    const messages = [
      { role: "system", content: this.systemPrompt },
      { role: "user", content: report }
    ];
    const t0 = performance.now();
    let n = 0;
    for await (const delta of this.session.generate(messages, { maxTokens: rt.maxCtx, temperature: maxTemperature })) {
      node.appendData(delta);
      n++;
    }
    const dt = (performance.now() - t0) / 1e3;
    this.log(`done: ${n} tokens in ${dt.toFixed(1)}s (${(n / dt).toFixed(1)} tok/s) adapter=${adapterName}`);
  }
};

// src/qwgpu/kernels.js
var GEMV = `
enable subgroups;
requires immediate_address_space;
requires subgroup_id;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;       // [N][K/4] int8
@group(0) @binding(2) var<storage,read> scale: array<f32>;   // [N]
@group(0) @binding(3) var<storage,read> bias: array<f32>;    // [N] or dummy
@group(0) @binding(4) var<storage,read> loraD: array<f32>;   // [rank] precomputed x@A (or dummy)
@group(0) @binding(5) var<storage,read> loraB: array<f32>;   // [rank][N] (or dummy)
@group(0) @binding(6) var<storage,read_write> y: array<f32>; // [N]
var<immediate> m: Meta;
var<workgroup> part: array<f32,64>;       // one slot per subgroup
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32,
        @builtin(subgroup_id) sgroup: u32) {
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
var LORA_A = `
enable subgroups;
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> x: array<f32>;     // [K]
@group(0) @binding(1) var<storage,read> A: array<f32>;     // [rank][K] (transposed)
@group(0) @binding(2) var<storage,read_write> d: array<f32>; // [rank]
var<immediate> m: vec2<u32>;           // K, rank
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
var LORA_A_BATCH = `
enable subgroups;
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> x: array<f32>;       // [T][K]
@group(0) @binding(1) var<storage,read> A: array<f32>;       // [rank][K]
@group(0) @binding(2) var<storage,read_write> d: array<f32>; // [T][rank]
var<immediate> m: vec4<u32>;             // K, rank, T, _
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
var LORA_B_ADD_T = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, rank:u32, gx:u32, scale:f32, p1:f32, p2:f32, p3:f32 };
@group(0) @binding(0) var<storage,read> d: array<f32>;        // [T][rank]
@group(0) @binding(1) var<storage,read> B: array<f32>;        // [rank][N]
@group(0) @binding(2) var<storage,read_write> Y: array<f32>;  // [T][N]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * (m.gx * 256u) + gid.x;
  if (i >= m.T * m.N) { return; }
  let t = i / m.N; let n = i % m.N; var acc = 0.0;
  for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + d[t*m.rank + r] * B[r*m.N + n]; }
  Y[i] = Y[i] + m.scale * acc;
}`;
var LORA_B_ADD = `
requires immediate_address_space;
struct Meta { N:u32, rank:u32, p0:u32, p1:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> d: array<f32>;       // [rank]
@group(0) @binding(1) var<storage,read> B: array<f32>;       // [rank][N]
@group(0) @binding(2) var<storage,read_write> y: array<f32>; // [N]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= m.N) { return; }
  var acc = 0.0;
  for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + d[r] * B[r*m.N + n]; }
  y[n] = y[n] + m.scale * acc;
}`;
var RMSNORM = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x);
  var s = 0.0; for (var k = tid; k < K; k = k + WG) { let v = x[k]; s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/m.x + m.y);
  for (var k = tid; k < K; k = k + WG) { y[k] = x[k]*inv*g[k]; }
}`;
var RMSNORM_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f16,256>;
@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x);
  var s = 0.0h;
  for (var k = tid; k < K; k = k + WG) { let v = f16(x[k]); s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/f16(m.x) + f16(m.y));
  for (var k = tid; k < K; k = k + WG) { y[k] = f32( f16(x[k]) * inv * f16(g[k]) ); }
}`;
var ROPE = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec3<u32>;             // nHeads, headDim, pos
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
var ROPE_F16 = `
requires immediate_address_space;
enable f16;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec3<u32>;             // nHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let pos = m.z; let half = D/2u;
  if (g >= H*half) { return; }
  let h = g / half; let j = g % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = f16(cosT[off]); let s = f16(sinT[off]);
  let xl = f16(x[lo]); let xh = f16(x[hi]);
  x[lo] = f32( xl*c - xh*s );
  x[hi] = f32( xh*c + xl*s );
}`;
var ROPE_QK = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> q: array<f32>;
@group(0) @binding(1) var<storage,read_write> k: array<f32>;
@group(0) @binding(2) var<storage,read> cosT: array<f32>;
@group(0) @binding(3) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;             // qHeads, kvHeads, headDim, pos
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
var ROPE_QK_F16 = `
requires immediate_address_space;
enable f16;
@group(0) @binding(0) var<storage,read_write> q: array<f32>;
@group(0) @binding(1) var<storage,read_write> k: array<f32>;
@group(0) @binding(2) var<storage,read> cosT: array<f32>;
@group(0) @binding(3) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;             // qHeads, kvHeads, headDim, pos
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
  let c = f16(cosT[off]); let s = f16(sinT[off]);
  if (isK) {
    let xl = f16(k[lo]); let xh = f16(k[hi]);
    k[lo] = f32( xl*c - xh*s ); k[hi] = f32( xh*c + xl*s );
  } else {
    let xl = f16(q[lo]); let xh = f16(q[hi]);
    q[lo] = f32( xl*c - xh*s ); q[hi] = f32( xh*c + xl*s );
  }
}`;
var ATTN_PARTIAL = `
requires immediate_address_space;
enable subgroups;
override WG: u32 = 128u;
struct AttnP { nHeads: u32, nKV: u32, ctx: u32, hd: u32, nsplit: u32, chunk: u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;  // [nHeads*nsplit] per-split max
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;  // [nHeads*nsplit] per-split sum
@group(0) @binding(5) var<storage,read_write> po: array<f32>;  // [nHeads*nsplit*hd] unnorm weighted V
var<immediate> m: AttnP;
var<workgroup> sc: array<f32,128>;
var<workgroup> red: array<f32,32>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.nHeads; let nKV = m.nKV; let ctx = m.ctx; let hd = m.hd; let nsplit = m.nsplit; let chunk = m.chunk;
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
var ATTN_PARTIAL_F16 = `
requires immediate_address_space;
enable subgroups;
enable f16;
override WG: u32 = 128u;
struct AttnP { nHeads: u32, nKV: u32, ctx: u32, hd: u32, nsplit: u32, chunk: u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;  // [nHeads*nsplit] per-split max
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;  // [nHeads*nsplit] per-split sum
@group(0) @binding(5) var<storage,read_write> po: array<f32>;  // [nHeads*nsplit*hd] unnorm weighted V
var<immediate> m: AttnP;
var<workgroup> sc: array<f16,128>;
var<workgroup> red: array<f16,32>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.nHeads; let nKV = m.nKV; let ctx = m.ctx; let hd = m.hd; let nsplit = m.nsplit; let chunk = m.chunk;
  let kvh = h / (nHeads / nKV);
  let qbase = h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scale = 1.0h / sqrt(f16(hd));
  let nsg = (WG + sgsz - 1u) / sgsz;
  let t0 = s*chunk; var t1 = t0 + chunk; if (t1 > ctx) { t1 = ctx; }
  let t = t0 + tid; var sv = -1e4h;
  if (t < t1) { var dot = 0.0h; let kb = t*stride + hoff; for (var d = 0u; d < hd; d = d + 1u) { dot = dot + f16(q[qbase+d])*f16(kc[kb+d]); } sv = dot*scale; }
  let sgm = subgroupMax(sv); if (sgid == 0u) { red[tid/sgsz] = sgm; }
  workgroupBarrier();
  var M = -1e4h; for (var i = 0u; i < nsg; i = i + 1u) { M = max(M, red[i]); }
  workgroupBarrier();
  var ev = 0.0h; if (t < t1) { ev = exp(sv - M); } sc[tid] = ev;
  let sgs = subgroupAdd(ev); if (sgid == 0u) { red[tid/sgsz] = sgs; }
  workgroupBarrier();
  var Z = 0.0h; for (var i = 0u; i < nsg; i = i + 1u) { Z = Z + red[i]; }
  workgroupBarrier();
  let len = t1 - t0; let pbase = (h*nsplit + s)*hd;
  for (var d = tid; d < hd; d = d + WG) {
    var acc = 0.0h; for (var tt = 0u; tt < len; tt = tt + 1u) { acc = acc + sc[tt] * f16(vc[(t0+tt)*stride + hoff + d]); }
    po[pbase + d] = f32(acc);
  }
  if (tid == 0u) { pm[h*nsplit + s] = f32(M); pz[h*nsplit + s] = f32(Z); }
}`;
var ATTN_COMBINE = `
requires immediate_address_space;
override WG: u32 = 128u;
@group(0) @binding(0) var<storage,read> pm: array<f32>;
@group(0) @binding(1) var<storage,read> pz: array<f32>;
@group(0) @binding(2) var<storage,read> po: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, hd, nsplit, _
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let tid = lid.x; let hd = m.y; let nsplit = m.z; let base = h*nsplit;
  var M = -1e30; for (var s = 0u; s < nsplit; s = s + 1u) { M = max(M, pm[base+s]); }
  var Z = 0.0; for (var s = 0u; s < nsplit; s = s + 1u) { Z = Z + pz[base+s]*exp(pm[base+s]-M); }
  let invZ = 1.0 / Z;
  for (var d = tid; d < hd; d = d + WG) {
    var acc = 0.0;
    for (var s = 0u; s < nsplit; s = s + 1u) { acc = acc + exp(pm[base+s]-M)*po[(base+s)*hd + d]; }
    o[h*hd + d] = acc * invZ;
  }
}`;
var ATTN_COMBINE_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 128u;
@group(0) @binding(0) var<storage,read> pm: array<f32>;
@group(0) @binding(1) var<storage,read> pz: array<f32>;
@group(0) @binding(2) var<storage,read> po: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, hd, nsplit, _
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let tid = lid.x; let hd = m.y; let nsplit = m.z; let base = h*nsplit;
  var M = -1e4h; for (var s = 0u; s < nsplit; s = s + 1u) { M = max(M, f16(pm[base+s])); }
  var Z = 0.0h; for (var s = 0u; s < nsplit; s = s + 1u) { Z = Z + f16(pz[base+s]) * exp(f16(pm[base+s]) - M); }
  let invZ = 1.0h / Z;
  for (var d = tid; d < hd; d = d + WG) {
    var acc = 0.0h;
    for (var s = 0u; s < nsplit; s = s + 1u) { acc = acc + exp(f16(pm[base+s]) - M) * f16(po[(base+s)*hd + d]); }
    o[h*hd + d] = f32(acc * invZ);
  }
}`;
var GEMM4 = `
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;       // [T][K]
@group(0) @binding(1) var<storage,read> W: array<u32>;       // [N][K/8] int4
@group(0) @binding(2) var<storage,read> scale: array<f32>;   // [N][gpr]
@group(0) @binding(3) var<storage,read> bias: array<f32>;    // [N] or dummy
@group(0) @binding(4) var<storage,read_write> Y: array<f32>; // [T][N]
var<immediate> m: Meta;
const BM = 16u; const BN = 64u;
var<workgroup> As: array<f32, 128>;   // BM*8 \u2014 A staged for one 8-wide K chunk
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
var GEMM4_ADD_T = `
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;
@group(0) @binding(1) var<storage,read> W: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read_write> Y: array<f32>;
var<immediate> m: Meta;
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
var ADD = `
requires immediate_address_space;
requires linear_indexing;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_index) gid: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = gid; i < n; i = i + stride) { y[i] = y[i] + a[i]; }
}`;
var ADD_F16 = `
requires immediate_address_space;
requires linear_indexing;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_index) gid: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = gid; i < n; i = i + stride) { y[i] = f32(f16(y[i]) + f16(a[i])); }
}`;
var SILUMUL_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read_write> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = g.x; i < n; i = i + stride) { let v = f16(gate[i]); gate[i] = f32( (v/(1.0h+exp(-v))) * f16(up[i]) ); }
}`;
var SILUMUL = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read_write> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = g.x; i < n; i = i + stride) { let v = gate[i]; gate[i] = (v/(1.0+exp(-v)))*up[i]; }
}`;
var EMBED = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
var<immediate> m: vec2<u32>;   // id, hidden
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; let id = m.x; let H = m.y; if (k >= H) { return; }
  let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
  var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
  out[k] = f32(b) * scale[id];
}`;
var EMBED_BUF = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> idbuf: array<u32>;   // idbuf[0] = token id
var<immediate> H: u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; let id = idbuf[0]; if (k >= H) { return; }
  let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
  var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
  out[k] = f32(b) * scale[id];
}`;
var RMSNORM_T = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  var s = 0.0; for (var k = tid; k < K; k = k + WG) { let v = x[base+k]; s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/m.x + m.y);
  for (var k = tid; k < K; k = k + WG) { y[base+k] = x[base+k]*inv*g[k]; }
}`;
var RMSNORM_T_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f16,256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  var s = 0.0h;
  for (var k = tid; k < K; k = k + WG) { let v = f16(x[base+k]); s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/f16(m.x) + f16(m.y));
  for (var k = tid; k < K; k = k + WG) { y[base+k] = f32( f16(x[base+k]) * inv * f16(g[k]) ); }
}`;
var ROPE_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = cosT[off]; let s = sinT[off]; let xl = x[lo]; let xh = x[hi];
  x[lo] = xl*c - xh*s; x[hi] = xh*c + xl*s;
}`;
var ROPE_T_F16 = `
requires immediate_address_space;
enable f16;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = f16(cosT[off]); let s = f16(sinT[off]); let xl = f16(x[lo]); let xh = f16(x[hi]);
  x[lo] = f32( xl*c - xh*s ); x[hi] = f32( xh*c + xl*s );
}`;
var EMBED_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> ids: array<u32>;
var<immediate> m: vec4<u32>;   // T, H, idOffset, _
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
var ATTN_PREFILL = `
enable subgroups;
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> q: array<f32>;       // [T][nHeads*hd]
@group(0) @binding(1) var<storage,read> kc: array<f32>;      // [ctx][nKV*hd]
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>; // [T][nHeads*hd]
var<immediate> m: vec4<u32>;             // nHeads, nKV, hd, T
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
var ATTN_PREFILL_BLOCK = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, qStart:u32, ctx:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
var<immediate> m: Meta;
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
var ARGMAX = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
var<immediate> n: u32;
var<workgroup> bv: array<f32,256>; var<workgroup> bi: array<u32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; var v = -1e30; var idx = 0xffffffffu;
  for (var i = tid; i < n; i = i + 256u) { let x = logits[i]; if (x > v || (x == v && i < idx)) { v = x; idx = i; } }
  bv[tid] = v; bi[tid] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s/2u) { if (tid < s) { let ov = bv[tid+s]; let oi = bi[tid+s]; if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; } } workgroupBarrier(); }
  if (tid == 0u) { out[0] = bi[0]; }
}`;
var TOPK_SELECT = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> ids: array<u32>;
@group(0) @binding(2) var<storage,read_write> vals: array<f32>;
var<immediate> m: vec2<u32>; // vocabSize, selectedCount
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
var SAMPLE_TOPK = `
requires immediate_address_space;
struct Meta { k:u32, pad:u32, temp:f32, r:f32 };
@group(0) @binding(0) var<storage,read> ids: array<u32>;
@group(0) @binding(1) var<storage,read> vals: array<f32>;
@group(0) @binding(2) var<storage,read_write> outId: array<u32>;  // [1] the chosen token
var<immediate> m: Meta;
var<workgroup> s: array<f32, 64>;    // working softmax probs / prefix sums (small k)
var<workgroup> red: array<f32, 64>;  // reduction scratch for the softmax denominator
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let k = m.k;
  let temp = m.temp;
  let r = m.r;
  let t = select(temp, 1.0, temp <= 0.0);

  // Load + temperature scale into shared (one thread per slot)
  var v = -1e30;
  if (tid < k) {
    let lv = vals[tid];
    v = lv;
    if (t != 1.0) { v = lv / t; }
  }
  let ev = select(0.0, exp(v), tid < k);
  s[tid] = ev;
  red[tid] = ev;
  workgroupBarrier();

  // sum
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (tid < stride && (tid + stride) < 64u) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
  }
  let sum = red[0];
  let invSum = select(0.0, 1.0 / sum, sum > 0.0);

  // normalize + prefix sum for nucleus / categorical pick
  if (tid < k) {
    s[tid] = s[tid] * invSum;
  } else {
    s[tid] = 0.0;
  }
  workgroupBarrier();

  // prefix sum (small k, simple scan)
  for (var stride = 1u; stride < 64u; stride = stride * 2u) {
    var add = 0.0;
    if (tid >= stride && tid < 64u) {
      add = s[tid - stride];
    }
    workgroupBarrier();
    if (tid >= stride && tid < 64u) {
      s[tid] = s[tid] + add;
    }
    workgroupBarrier();
  }

  // find the smallest j such that prefix[j] >= r  (or last if r>=1)
  if (tid == 0u) {
    var chosen = select(0u, k - 1u, k > 0u);
    if (sum > 0.0) {
      for (var j = 0u; j < k; j = j + 1u) {
        let pj = s[j];
        if (r <= pj) { chosen = j; break; }
      }
    }
    outId[0] = select(0u, ids[chosen], k > 0u);
  }
}`;
var GEMV4 = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read> loraD: array<f32>;
@group(0) @binding(5) var<storage,read> loraB: array<f32>;
@group(0) @binding(6) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;
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
var GEMV4_ADD = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read> loraD: array<f32>;
@group(0) @binding(5) var<storage,read> loraB: array<f32>;
@group(0) @binding(6) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;
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
var QKV_GEMV4 = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, totalN:u32, qN:u32, kN:u32, vN:u32, gpr:u32, gridX:u32, p0:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read_write> qOut: array<f32>;
@group(0) @binding(5) var<storage,read_write> kOut: array<f32>;
@group(0) @binding(6) var<storage,read_write> vOut: array<f32>;
var<immediate> m: Meta;
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
var GATE_UP_SILU_GEMV4 = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, N:u32, gpr:u32, gridX:u32, gateRank:u32, upRank:u32, hasGateLora:u32, hasUpLora:u32, gateScaleLo:f32, upScaleLo:f32, p0:f32, p1:f32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
@group(0) @binding(4) var<storage,read> gateD: array<f32>;
@group(0) @binding(5) var<storage,read> gateB: array<f32>;
@group(0) @binding(6) var<storage,read> upD: array<f32>;
@group(0) @binding(7) var<storage,read> upB: array<f32>;
var<immediate> m: Meta;
var<workgroup> partG: array<f32,64>;
var<workgroup> partU: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rbG = n*K8; let rbU = (m.N + n)*K8;
  let sbG = n*m.gpr; let sbU = (m.N + n)*m.gpr;
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
    if (m.hasGateLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.gateRank; r = r + 1u) { dl = dl + gateD[r] * gateB[r*m.N + n]; }
      gate = gate + m.gateScaleLo * dl;
    }
    if (m.hasUpLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.upRank; r = r + 1u) { dl = dl + upD[r] * upB[r*m.N + n]; }
      up = up + m.upScaleLo * dl;
    }
    y[n] = (gate / (1.0 + exp(-gate))) * up;
  }
}`;
var DYN_QUANT_X = `
requires immediate_address_space;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> scale_x: array<f32>;
var<immediate> K: u32;
var<workgroup> sh_max: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wid.x; let tid = lid.x; let base = g * 128u;
  var local_max = 0.0;
  let idx0 = base + tid; let idx1 = base + tid + 64u;
  if (idx0 < K) { local_max = max(local_max, abs(x[idx0])); }
  if (idx1 < K) { local_max = max(local_max, abs(x[idx1])); }
  sh_max[tid] = local_max;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (tid < s) { sh_max[tid] = max(sh_max[tid], sh_max[tid + s]); }
    workgroupBarrier();
  }
  let gmax = sh_max[0]; let scale = select(gmax / 127.0, 1.0, gmax == 0.0);
  if (tid == 0u) { scale_x[g] = scale; }
  let pidx = base + tid * 4u;
  if (pidx < K) {
    let q0 = clamp(i32(round(x[pidx] / scale)), -128, 127) & 0xff;
    let q1 = clamp(i32(round(x[pidx + 1u] / scale)), -128, 127) & 0xff;
    let q2 = clamp(i32(round(x[pidx + 2u] / scale)), -128, 127) & 0xff;
    let q3 = clamp(i32(round(x[pidx + 3u] / scale)), -128, 127) & 0xff;
    x_q[g * 32u + tid] = u32(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
  }
}
`;
var DYN_QUANT_X_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> scale_x: array<f32>;
var<immediate> m: vec2<u32>; // K, T
var<workgroup> sh_max: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wid.x; let t = wid.y; let tid = lid.x; let K = m.x; let T = m.y;
  if (t >= T) { return; }
  let row_base = t * K; let base = row_base + g * 128u;
  var local_max = 0.0;
  let idx0 = base + tid; let idx1 = base + tid + 64u;
  if (g * 128u + tid < K) { local_max = max(local_max, abs(x[idx0])); }
  if (g * 128u + tid + 64u < K) { local_max = max(local_max, abs(x[idx1])); }
  sh_max[tid] = local_max;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (tid < s) { sh_max[tid] = max(sh_max[tid], sh_max[tid + s]); }
    workgroupBarrier();
  }
  let gmax = sh_max[0]; let scale = select(gmax / 127.0, 1.0, gmax == 0.0);
  let groupsPerRow = K / 128u;
  if (tid == 0u) { scale_x[t * groupsPerRow + g] = scale; }
  let pidx = base + tid * 4u;
  if (g * 128u + tid * 4u < K) {
    let q0 = clamp(i32(round(x[pidx] / scale)), -128, 127) & 0xff;
    let q1 = clamp(i32(round(x[pidx + 1u] / scale)), -128, 127) & 0xff;
    let q2 = clamp(i32(round(x[pidx + 2u] / scale)), -128, 127) & 0xff;
    let q3 = clamp(i32(round(x[pidx + 3u] / scale)), -128, 127) & 0xff;
    x_q[t * (K / 4u) + g * 32u + tid] = u32(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
  }
}
`;
var GEMV4_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? "enable packed_4x8_integer_dot_product;" : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read> loraD: array<f32>;
@group(0) @binding(6) var<storage,read> loraB: array<f32>;
@group(0) @binding(7) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> part: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let word = w[rb+c]; let bk = c*8u;
    let sc_w = scale[sbase + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let w0 = (i32(word << 28u) >> 28u) & 0xff;
    let w1 = (i32(word << 24u) >> 28u) & 0xff;
    let w2 = (i32(word << 20u) >> 28u) & 0xff;
    let w3 = (i32(word << 16u) >> 28u) & 0xff;
    let w4 = (i32(word << 12u) >> 28u) & 0xff;
    let w5 = (i32(word << 8u)  >> 28u) & 0xff;
    let w6 = (i32(word << 4u)  >> 28u) & 0xff;
    let w7 = (i32(word)        >> 28u) & 0xff;
    let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
    let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
    acc = acc + f32(sum) * sc_w * sc_x;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = o;
  }
}
`, "GEMV4_W4A8");
var GEMV4_ADD_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? "enable packed_4x8_integer_dot_product;" : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read> loraD: array<f32>;
@group(0) @binding(6) var<storage,read> loraB: array<f32>;
@group(0) @binding(7) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> part: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let word = w[rb+c]; let bk = c*8u;
    let sc_w = scale[sbase + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let w0 = (i32(word << 28u) >> 28u) & 0xff;
    let w1 = (i32(word << 24u) >> 28u) & 0xff;
    let w2 = (i32(word << 20u) >> 28u) & 0xff;
    let w3 = (i32(word << 16u) >> 28u) & 0xff;
    let w4 = (i32(word << 12u) >> 28u) & 0xff;
    let w5 = (i32(word << 8u)  >> 28u) & 0xff;
    let w6 = (i32(word << 4u)  >> 28u) & 0xff;
    let w7 = (i32(word)        >> 28u) & 0xff;
    let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
    let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
    acc = acc + f32(sum) * sc_w * sc_x;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = y[n] + o;
  }
}
`, "GEMV4_ADD_W4A8");
var QKV_GEMV4_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? "enable packed_4x8_integer_dot_product;" : ""}
requires immediate_address_space;
struct Meta { K:u32, totalN:u32, qN:u32, kN:u32, vN:u32, gpr:u32, gridX:u32, p0:u32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read_write> qOut: array<f32>;
@group(0) @binding(6) var<storage,read_write> kOut: array<f32>;
@group(0) @binding(7) var<storage,read_write> vOut: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> part: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.totalN) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let word = w[rb+c]; let bk = c*8u;
    let sc_w = scale[sbase + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let w0 = (i32(word << 28u) >> 28u) & 0xff;
    let w1 = (i32(word << 24u) >> 28u) & 0xff;
    let w2 = (i32(word << 20u) >> 28u) & 0xff;
    let w3 = (i32(word << 16u) >> 28u) & 0xff;
    let w4 = (i32(word << 12u) >> 28u) & 0xff;
    let w5 = (i32(word << 8u)  >> 28u) & 0xff;
    let w6 = (i32(word << 4u)  >> 28u) & 0xff;
    let w7 = (i32(word)        >> 28u) & 0xff;
    let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
    let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
    acc = acc + f32(sum) * sc_w * sc_x;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var o = 0.0;
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
}
`, "QKV_GEMV4_W4A8");
var GATE_UP_SILU_GEMV4_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? "enable packed_4x8_integer_dot_product;" : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, gpr:u32, gridX:u32, gateRank:u32, upRank:u32, hasGateLora:u32, hasUpLora:u32, gateScaleLo:f32, upScaleLo:f32, p0:f32, p1:f32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read_write> y: array<f32>;
@group(0) @binding(5) var<storage,read> gateD: array<f32>;
@group(0) @binding(6) var<storage,read> gateB: array<f32>;
@group(0) @binding(7) var<storage,read> upD: array<f32>;
@group(0) @binding(8) var<storage,read> upB: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> partG: array<f32, ${wgSize}>;
var<workgroup> partU: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rbG = n*K8; let rbU = (m.N + n)*K8;
  let sbG = n*m.gpr; let sbU = (m.N + n)*m.gpr;
  var accG = 0.0; var accU = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let wg = w[rbG+c]; let wu = w[rbU+c];
    let bk = c*8u;
    let scG = scale[sbG + (bk >> 7u)]; let scU = scale[sbU + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let wg0 = (i32(wg << 28u) >> 28u) & 0xff;
    let wg1 = (i32(wg << 24u) >> 28u) & 0xff;
    let wg2 = (i32(wg << 20u) >> 28u) & 0xff;
    let wg3 = (i32(wg << 16u) >> 28u) & 0xff;
    let wg4 = (i32(wg << 12u) >> 28u) & 0xff;
    let wg5 = (i32(wg << 8u)  >> 28u) & 0xff;
    let wg6 = (i32(wg << 4u)  >> 28u) & 0xff;
    let wg7 = (i32(wg)        >> 28u) & 0xff;
    let pwg0 = u32(wg0 | (wg1 << 8u) | (wg2 << 16u) | (wg3 << 24u));
    let pwg1 = u32(wg4 | (wg5 << 8u) | (wg6 << 16u) | (wg7 << 24u));
    let wu0 = (i32(wu << 28u) >> 28u) & 0xff;
    let wu1 = (i32(wu << 24u) >> 28u) & 0xff;
    let wu2 = (i32(wu << 20u) >> 28u) & 0xff;
    let wu3 = (i32(wu << 16u) >> 28u) & 0xff;
    let wu4 = (i32(wu << 12u) >> 28u) & 0xff;
    let wu5 = (i32(wu << 8u)  >> 28u) & 0xff;
    let wu6 = (i32(wu << 4u)  >> 28u) & 0xff;
    let wu7 = (i32(wu)        >> 28u) & 0xff;
    let pwu0 = u32(wu0 | (wu1 << 8u) | (wu2 << 16u) | (wu3 << 24u));
    let pwu1 = u32(wu4 | (wu5 << 8u) | (wu6 << 16u) | (wu7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sumG = dot4I8Packed(pwg0, px0) + dot4I8Packed(pwg1, px1);
    let sumU = dot4I8Packed(pwu0, px0) + dot4I8Packed(pwu1, px1);
    accG = accG + f32(sumG) * scG * sc_x;
    accU = accU + f32(sumU) * scU * sc_x;
  }
  let sg = subgroupAdd(accG); let su = subgroupAdd(accU);
  if (sgid == 0u) { partG[tid / sgsz] = sg; partU[tid / sgsz] = su; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var gate = 0.0; var up = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { gate = gate + partG[i]; up = up + partU[i]; }
    if (m.hasGateLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.gateRank; r = r + 1u) { dl = dl + gateD[r] * gateB[r*m.N + n]; }
      gate = gate + m.gateScaleLo * dl;
    }
    if (m.hasUpLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.upRank; r = r + 1u) { dl = dl + upD[r] * upB[r*m.N + n]; }
      up = up + m.upScaleLo * dl;
    }
    y[n] = (gate / (1.0 + exp(-gate))) * up;
  }
}
`, "GATE_UP_SILU_GEMV4_W4A8");
var GEMM4_W4A8 = /* @__PURE__ */ __name((hasDP4a) => `
enable subgroups;
${hasDP4a ? "enable packed_4x8_integer_dot_product;" : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> W: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read_write> Y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

const BM = 16u; const BN = 64u;
var<workgroup> As_q: array<u32, 32>;
var<workgroup> As_scale: array<f32, 16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  let groupsPerRow = m.K / 128u;
  for (var c = 0u; c < K8; c = c + 1u) {
    if (lid.x < BM * 2u) {
      let tt = lid.x / 2u; let trow = tTile + tt; let wordIdx = lid.x % 2u;
      As_q[lid.x] = select(0u, A_q[trow * (m.K / 4u) + c * 2u + wordIdx], trow < m.T);
    }
    if (lid.x < BM) {
      let trow = tTile + lid.x;
      As_scale[lid.x] = select(0.0, scale_x[trow * groupsPerRow + ((c * 8u) >> 7u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc_w = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0 = (i32(word << 28u) >> 28u) & 0xff;
      let w1 = (i32(word << 24u) >> 28u) & 0xff;
      let w2 = (i32(word << 20u) >> 28u) & 0xff;
      let w3 = (i32(word << 16u) >> 28u) & 0xff;
      let w4 = (i32(word << 12u) >> 28u) & 0xff;
      let w5 = (i32(word << 8u)  >> 28u) & 0xff;
      let w6 = (i32(word << 4u)  >> 28u) & 0xff;
      let w7 = (i32(word)        >> 28u) & 0xff;
      let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
      let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
      for (var t = 0u; t < BM; t = t + 1u) {
        let px0 = As_q[t * 2u]; let px1 = As_q[t * 2u + 1u];
        let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
        acc[t] = acc[t] + f32(sum) * sc_w * As_scale[t];
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) { let trow = tTile + t; if (trow < m.T) { Y[trow*m.N + col] = acc[t] + bv; } }
  }
}
`, "GEMM4_W4A8");
var GEMM4_ADD_T_W4A8 = /* @__PURE__ */ __name((hasDP4a) => `
enable subgroups;
${hasDP4a ? "enable packed_4x8_integer_dot_product;" : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> W: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read_write> Y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

const BM = 16u; const BN = 64u;
var<workgroup> As_q: array<u32, 32>;
var<workgroup> As_scale: array<f32, 16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  let groupsPerRow = m.K / 128u;
  for (var c = 0u; c < K8; c = c + 1u) {
    if (lid.x < BM * 2u) {
      let tt = lid.x / 2u; let trow = tTile + tt; let wordIdx = lid.x % 2u;
      As_q[lid.x] = select(0u, A_q[trow * (m.K / 4u) + c * 2u + wordIdx], trow < m.T);
    }
    if (lid.x < BM) {
      let trow = tTile + lid.x;
      As_scale[lid.x] = select(0.0, scale_x[trow * groupsPerRow + ((c * 8u) >> 7u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc_w = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0 = (i32(word << 28u) >> 28u) & 0xff;
      let w1 = (i32(word << 24u) >> 28u) & 0xff;
      let w2 = (i32(word << 20u) >> 28u) & 0xff;
      let w3 = (i32(word << 16u) >> 28u) & 0xff;
      let w4 = (i32(word << 12u) >> 28u) & 0xff;
      let w5 = (i32(word << 8u)  >> 28u) & 0xff;
      let w6 = (i32(word << 4u)  >> 28u) & 0xff;
      let w7 = (i32(word)        >> 28u) & 0xff;
      let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
      let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
      for (var t = 0u; t < BM; t = t + 1u) {
        let px0 = As_q[t * 2u]; let px1 = As_q[t * 2u + 1u];
        let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
        acc[t] = acc[t] + f32(sum) * sc_w * As_scale[t];
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
}
`, "GEMM4_ADD_T_W4A8");
var WRITE_KV_PAGE = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> k_src: array<f32>;
@group(0) @binding(1) var<storage,read> v_src: array<f32>;
@group(0) @binding(2) var<storage,read_write> kc: array<f32>;
@group(0) @binding(3) var<storage,read_write> vc: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: vec4<u32>; // pos, seq_id, max_blocks, kvd
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; let pos = m.x; let seq_id = m.y; let max_blocks = m.z; let kvd = m.w;
  if (idx >= kvd) { return; }
  let page_idx = block_table[seq_id * max_blocks + (pos / 16u)];
  let page_offset = pos % 16u;
  let physical_pos = page_idx * 16u + page_offset;
  let dst_offset = physical_pos * kvd + idx;
  kc[dst_offset] = k_src[idx];
  vc[dst_offset] = v_src[idx];
}
`;
var WRITE_KV_PAGE_BATCH = `
requires immediate_address_space;
struct KVBatchMeta { T:u32, seq_id:u32, max_blocks:u32, kvd:u32, off:u32 };
@group(0) @binding(0) var<storage,read> k_src: array<f32>;
@group(0) @binding(1) var<storage,read> v_src: array<f32>;
@group(0) @binding(2) var<storage,read_write> kc: array<f32>;
@group(0) @binding(3) var<storage,read_write> vc: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: KVBatchMeta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; let T = m.T; let seq_id = m.seq_id; let max_blocks = m.max_blocks; let kvd = m.kvd; let off = m.off;
  let total = T * kvd; if (idx >= total) { return; }
  let t = idx / kvd; let d = idx % kvd;
  let page_idx = block_table[seq_id * max_blocks + ((off + t) / 16u)];
  let page_offset = (off + t) % 16u;
  let physical_pos = page_idx * 16u + page_offset;
  let dst_offset = physical_pos * kvd + d;
  kc[dst_offset] = k_src[idx];
  vc[dst_offset] = v_src[idx];
}
`;
var ATTN_PARTIAL_PAGED = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, ctx:u32, hd:u32, nsplit:u32, chunk:u32, seq_id:u32, max_blocks:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;
@group(0) @binding(5) var<storage,read_write> po: array<f32>;
@group(0) @binding(6) var<storage,read> block_table: array<u32>;
var<immediate> m: Meta;
var<workgroup> sc: array<f32,128>;
var<workgroup> red: array<f32,32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.nHeads; let nKV = m.nKV; let ctx = m.ctx; let hd = m.hd;
  let nsplit = m.nsplit; let chunk = m.chunk; let seq_id = m.seq_id; let max_blocks = m.max_blocks;
  let kvh = h / (nHeads / nKV);
  let qbase = h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scale = 1.0/sqrt(f32(hd));
  let nsg = (128u + sgsz - 1u) / sgsz;
  let t0 = s*chunk; var t1 = t0 + chunk; if (t1 > ctx) { t1 = ctx; }
  let t = t0 + tid; var sv = -1e30;
  if (t < t1) {
    var dot = 0.0;
    let page_idx = block_table[seq_id * max_blocks + (t / 16u)];
    let page_offset = t % 16u;
    let kb = (page_idx * 16u + page_offset) * stride + hoff;
    for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; }
    sv = dot*scale;
  }
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
    var acc = 0.0;
    for (var tt = 0u; tt < len; tt = tt + 1u) {
      let t_curr = t0 + tt;
      let page_idx = block_table[seq_id * max_blocks + (t_curr / 16u)];
      let page_offset = t_curr % 16u;
      let physical_t = page_idx * 16u + page_offset;
      acc = acc + sc[tt]*vc[physical_t*stride + hoff + d];
    }
    po[pbase + d] = acc;
  }
  if (tid == 0u) { pm[h*nsplit + s] = M; pz[h*nsplit + s] = Z; }
}
`;
var ATTN_PREFILL_PAGED = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, seq_id:u32, max_blocks:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: Meta;
var<workgroup> ps: array<f32,256>;
var<workgroup> acc: array<f32,128>;
var<workgroup> red: array<f32,64>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let t = wid.y; let tid = lid.x; let nHeads = m.nHeads; let nKV = m.nKV; let hd = m.hd;
  let ctx = t + 1u; let kvh = h / (nHeads / nKV);
  let qbase = t*nHeads*hd + h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scl = 1.0/sqrt(f32(hd));
  let nsg = (256u + sgsz - 1u) / sgsz;
  let seq_id = m.seq_id; let max_blocks = m.max_blocks;
  for (var d = tid; d < hd; d = d + 256u) { acc[d] = 0.0; }
  var mrun = -1e30; var lrun = 0.0;
  let nblk = (ctx + 255u) / 256u;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk*256u; let kk = kbase + tid;
    var s = -1e30;
    if (kk < ctx) {
      var dot = 0.0;
      let page_idx = block_table[seq_id * max_blocks + (kk / 16u)];
      let page_offset = kk % 16u;
      let kb = (page_idx * 16u + page_offset)*stride + hoff;
      for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; }
      s = dot*scl;
    }
    let sgm = subgroupMax(s); if (sgid == 0u) { red[tid/sgsz] = sgm; }
    workgroupBarrier();
    var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[i]); }
    let mnew = max(mrun, bm); let corr = exp(mrun - mnew);
    var p = 0.0; if (kk < ctx) { p = exp(s - mnew); }
    ps[tid] = p;
    workgroupBarrier();
    let sgs = subgroupAdd(p); if (sgid == 0u) { red[tid/sgsz] = sgs; }
    workgroupBarrier();
    var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[i]; }
    lrun = lrun*corr + bs;
    let bcount = min(256u, ctx - kbase);
    for (var d = tid; d < hd; d = d + 256u) {
      var aa = acc[d]*corr;
      for (var j = 0u; j < bcount; j = j + 1u) {
        let t_curr = kbase + j;
        let page_idx = block_table[seq_id * max_blocks + (t_curr / 16u)];
        let page_offset = t_curr % 16u;
        let physical_t = page_idx * 16u + page_offset;
        aa = aa + ps[j]*vc[physical_t*stride + hoff + d];
      }
      acc[d] = aa;
    }
    mrun = mnew;
    workgroupBarrier();
  }
  let invL = 1.0/lrun;
  for (var d = tid; d < hd; d = d + 256u) { o[qbase + d] = acc[d]*invL; }
}
`;
var ATTN_PREFILL_BLOCK_PAGED = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, qStart:u32, ctx:u32, seq_id:u32, max_blocks:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: Meta;
const BQ = 4u; const BK = 128u;
var<workgroup> ps: array<f32, 512>;
var<workgroup> acc: array<f32, 512>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let qBlock = wid.y; let tid = lid.x; let hd = m.hd;
  let kvh = h / (m.nHeads / m.nKV); let stride = m.nKV * hd; let hoff = kvh * hd;
  let nsg = (128u + sgsz - 1u) / sgsz; let scl = 1.0 / sqrt(f32(hd));
  let seq_id = m.seq_id; let max_blocks = m.max_blocks;
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
      let page_idx = block_table[seq_id * max_blocks + (kk / 16u)];
      let page_offset = kk % 16u;
      let kb = (page_idx * 16u + page_offset)*stride + hoff;
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
        let t_curr = kbase + j;
        let page_idx = block_table[seq_id * max_blocks + (t_curr / 16u)];
        let page_offset = t_curr % 16u;
        let physical_t = page_idx * 16u + page_offset;
        let vv = vc[physical_t*stride + hoff + d];
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
}
`;
var GEMV4_QKV_ROPE_RMS = `
enable subgroups;
requires immediate_address_space;
struct Meta { 
  K: u32, totalPairs: u32, qPairs: u32, kPairs: u32, vPairs: u32, gpr: u32, gridX: u32, 
  pos: u32, headDim: u32, eps: f32,
  qN: u32, kN: u32
};

@group(0) @binding(0) var<storage,read> hidden: array<f32>;      
@group(0) @binding(1) var<storage,read> rms_g: array<f32>;       
@group(0) @binding(2) var<storage,read> w: array<u32>;           
@group(0) @binding(3) var<storage,read> scale: array<f32>;       
@group(0) @binding(4) var<storage,read> bias: array<f32>;        
@group(0) @binding(5) var<storage,read> cosT: array<f32>;
@group(0) @binding(6) var<storage,read> sinT: array<f32>;
@group(0) @binding(7) var<storage,read_write> qOut: array<f32>;  
@group(0) @binding(8) var<storage,read_write> kOut: array<f32>;  
@group(0) @binding(9) var<storage,read_write> vOut: array<f32>;  
var<immediate> m: Meta;

var<workgroup> partSum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  
  let pair_idx = wid.x + wid.y * m.gridX;
  if (pair_idx >= m.totalPairs) { return; }
  let tid = lid.x;
  
  var s = 0.0;
  for (var k = tid; k < m.K; k = k + 64u) { let v = hidden[k]; s = s + v*v; }
  let ssum = subgroupAdd(s);
  if (sgid == 0u) { partSum[tid / sgsz] = ssum; }
  workgroupBarrier();
  
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var red = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { red = red + partSum[i]; }
    partSum[0] = inverseSqrt(red / f32(m.K) + m.eps);
  }
  workgroupBarrier();
  let inv = partSum[0];

  let half = m.headDim / 2u;
  var n0: u32; var n1: u32;
  var isQ = false; var isK = false; var isV = false;
  var out_idx0: u32; var out_idx1: u32;
  var rope_j: u32 = 0u;

  if (pair_idx < m.qPairs) {
    isQ = true;
    let h = pair_idx / half; let j = pair_idx % half;
    n0 = h * m.headDim + j;
    n1 = n0 + half;
    out_idx0 = n0; out_idx1 = n1;
    rope_j = j;
  } else if (pair_idx < m.qPairs + m.kPairs) {
    isK = true;
    let p = pair_idx - m.qPairs;
    let h = p / half; let j = p % half;
    n0 = m.qN + h * m.headDim + j;
    n1 = n0 + half;
    out_idx0 = h * m.headDim + j; out_idx1 = out_idx0 + half;
    rope_j = j;
  } else {
    isV = true;
    let p = pair_idx - m.qPairs - m.kPairs;
    n0 = m.qN + m.kN + p * 2u;
    n1 = n0 + 1u;
    out_idx0 = p * 2u; out_idx1 = out_idx0 + 1u;
  }

  let K8 = m.K / 8u;
  let rb0 = n0 * K8; let rb1 = n1 * K8;
  let sbase0 = n0 * m.gpr; let sbase1 = n1 * m.gpr;

  var acc0 = 0.0; var acc1 = 0.0;
  
  for (var c = tid; c < K8; c = c + 64u) {
    let w0 = w[rb0 + c]; let w1 = w[rb1 + c];
    let bk = c * 8u;
    let sc0 = scale[sbase0 + (bk >> 7u)]; let sc1 = scale[sbase1 + (bk >> 7u)];
    
    // We compute normalized X on the fly
    let x0 = hidden[bk] * inv * rms_g[bk];
    let x1 = hidden[bk+1u] * inv * rms_g[bk+1u];
    let x2 = hidden[bk+2u] * inv * rms_g[bk+2u];
    let x3 = hidden[bk+3u] * inv * rms_g[bk+3u];
    let x4 = hidden[bk+4u] * inv * rms_g[bk+4u];
    let x5 = hidden[bk+5u] * inv * rms_g[bk+5u];
    let x6 = hidden[bk+6u] * inv * rms_g[bk+6u];
    let x7 = hidden[bk+7u] * inv * rms_g[bk+7u];

    var p0 = 0.0; var p1 = 0.0;
    p0 = p0 + x0 * f32(i32(w0 << 28u) >> 28u); p1 = p1 + x0 * f32(i32(w1 << 28u) >> 28u);
    p0 = p0 + x1 * f32(i32(w0 << 24u) >> 28u); p1 = p1 + x1 * f32(i32(w1 << 24u) >> 28u);
    p0 = p0 + x2 * f32(i32(w0 << 20u) >> 28u); p1 = p1 + x2 * f32(i32(w1 << 20u) >> 28u);
    p0 = p0 + x3 * f32(i32(w0 << 16u) >> 28u); p1 = p1 + x3 * f32(i32(w1 << 16u) >> 28u);
    p0 = p0 + x4 * f32(i32(w0 << 12u) >> 28u); p1 = p1 + x4 * f32(i32(w1 << 12u) >> 28u);
    p0 = p0 + x5 * f32(i32(w0 << 8u)  >> 28u); p1 = p1 + x5 * f32(i32(w1 << 8u)  >> 28u);
    p0 = p0 + x6 * f32(i32(w0 << 4u)  >> 28u); p1 = p1 + x6 * f32(i32(w1 << 4u)  >> 28u);
    p0 = p0 + x7 * f32(i32(w0)        >> 28u); p1 = p1 + x7 * f32(i32(w1)        >> 28u);
    
    acc0 = acc0 + p0 * sc0;
    acc1 = acc1 + p1 * sc1;
  }

  let ssum0 = subgroupAdd(acc0); let ssum1 = subgroupAdd(acc1);
  if (sgid == 0u) { partSum[tid / sgsz] = ssum0; partSum[32u + tid / sgsz] = ssum1; }
  workgroupBarrier();

  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; 
    var o0 = 0.0; var o1 = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o0 = o0 + partSum[i]; o1 = o1 + partSum[32u + i]; }
    
    o0 = o0 + bias[n0];
    o1 = o1 + bias[n1];

    if (isQ || isK) {
      let off = m.pos * m.headDim + rope_j;
      let c = cosT[off]; let s = sinT[off];
      let rl = o0 * c - o1 * s;
      let rh = o1 * c + o0 * s;
      o0 = rl; o1 = rh;
    }

    if (isQ) { qOut[out_idx0] = o0; qOut[out_idx1] = o1; }
    else if (isK) { kOut[out_idx0] = o0; kOut[out_idx1] = o1; }
    else { vOut[out_idx0] = o0; vOut[out_idx1] = o1; }
  }
}
`;

// src/qwgpu/dispatch_plan.js
function createDispatchPlan(schema) {
  return {
    embed: schema.embed,
    finalNorm: schema.finalNorm,
    layers: schema.layers.map((layer) => ({
      index: layer.index,
      inputNorm: layer.inputNorm.name,
      postAttentionNorm: layer.postAttentionNorm.name,
      q: {
        weight: layer.projections.q.name,
        bias: layer.biases.q?.name || null,
        loraKey: layer.projections.q.loraKey
      },
      k: {
        weight: layer.projections.k.name,
        bias: layer.biases.k?.name || null,
        loraKey: layer.projections.k.loraKey
      },
      v: {
        weight: layer.projections.v.name,
        bias: layer.biases.v?.name || null,
        loraKey: layer.projections.v.loraKey
      },
      o: {
        weight: layer.projections.o.name,
        bias: null,
        loraKey: layer.projections.o.loraKey
      },
      gate: {
        weight: layer.projections.gate.name,
        bias: null,
        loraKey: layer.projections.gate.loraKey
      },
      up: {
        weight: layer.projections.up.name,
        bias: null,
        loraKey: layer.projections.up.loraKey
      },
      down: {
        weight: layer.projections.down.name,
        bias: null,
        loraKey: layer.projections.down.loraKey
      }
    }))
  };
}
__name(createDispatchPlan, "createDispatchPlan");

// src/qwgpu/safetensors_loader.js
function decodeBf16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < numel; i++) o32[i] = u16[i] << 16;
  return out;
}
__name(decodeBf16ToF32, "decodeBf16ToF32");
function decodeF16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  for (let i = 0; i < numel; i++) {
    const h = u16[i], s = (h & 32768) >> 15, e = (h & 31744) >> 10, f = h & 1023;
    if (e === 0) out[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) out[i] = f ? NaN : s ? -Infinity : Infinity;
    else out[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}
__name(decodeF16ToF32, "decodeF16ToF32");
function decodeF32(u8, numel) {
  return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + numel * 4));
}
__name(decodeF32, "decodeF32");
var DECODERS = {
  BF16: decodeBf16ToF32,
  F16: decodeF16ToF32,
  FP16: decodeF16ToF32,
  F32: decodeF32,
  FP32: decodeF32
};
async function loadIndex(reader) {
  try {
    const idx = JSON.parse(await reader.text("model.safetensors.index.json"));
    return { weightMap: idx.weight_map || {}, shards: [...new Set(Object.values(idx.weight_map || {}))] };
  } catch {
    return { weightMap: null, shards: ["model.safetensors"] };
  }
}
__name(loadIndex, "loadIndex");
function shardPlan(shards, weightMap, names) {
  if (!weightMap || !names) return new Map(shards.map((shard) => [shard, null]));
  const plan = /* @__PURE__ */ new Map();
  for (const name of names) {
    const shard = weightMap[name];
    if (!shard) continue;
    if (!plan.has(shard)) plan.set(shard, /* @__PURE__ */ new Set());
    plan.get(shard).add(name);
  }
  return plan;
}
__name(shardPlan, "shardPlan");
async function streamSafetensors(source, { names = null, onTensor, onProgress = /* @__PURE__ */ __name(() => {
}, "onProgress") } = {}) {
  if (!onTensor) throw new Error("streamSafetensors requires onTensor");
  const reader = typeof source === "string" ? urlReader(source) : source;
  const { weightMap, shards } = await loadIndex(reader);
  const plan = shardPlan(shards, weightMap, names);
  let visited = 0;
  const total = names?.size || 0;
  for (const [shard, wantedInShard] of plan) {
    const lenBuf = await reader.range(shard, 0, 8);
    const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
    const hdrBuf = await reader.range(shard, 8, 8 + headerLen);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(hdrBuf)));
    const dataStart = 8 + headerLen;
    const allNames = Object.keys(header).filter((k) => k !== "__metadata__");
    const tensorNames = wantedInShard ? allNames.filter((n) => wantedInShard.has(n)) : names ? allNames.filter((n) => names.has(n)) : allNames;
    for (const name of tensorNames) {
      const t = header[name];
      if (!t) continue;
      const dtype = String(t.dtype || "").toUpperCase();
      const dec = DECODERS[dtype];
      if (!dec) throw new Error(`unsupported dtype ${dtype} for ${name}`);
      const numel = t.shape.reduce((a, b) => a * b, 1);
      const [s, e] = t.data_offsets;
      const buf = await reader.range(shard, dataStart + s, dataStart + e);
      const data = dec(new Uint8Array(buf), numel);
      await onTensor({ name, shape: t.shape, dtype, data, shard });
      visited++;
      onProgress(name, total ? Math.min(0.95, visited / total) : 0.3);
    }
  }
}
__name(streamSafetensors, "streamSafetensors");

// src/qwgpu/quantize.js
function quantizeInt8RowMajor(f322, outDim, inDim) {
  const scale = new Float32Array(outDim);
  const q = new Int8Array(outDim * inDim);
  for (let o = 0; o < outDim; o++) {
    const base = o * inDim;
    let amax = 0;
    for (let i = 0; i < inDim; i++) {
      const a = Math.abs(f322[base + i]);
      if (a > amax) amax = a;
    }
    const s = amax > 0 ? amax / 127 : 1;
    scale[o] = s;
    const inv = 1 / s;
    for (let i = 0; i < inDim; i++) {
      let v = Math.round(f322[base + i] * inv);
      if (v > 127) v = 127;
      else if (v < -128) v = -128;
      q[base + i] = v;
    }
  }
  const packed = new Uint32Array(outDim * inDim / 4);
  const u8 = new Uint8Array(q.buffer);
  for (let w = 0; w < packed.length; w++) {
    packed[w] = u8[w * 4] | u8[w * 4 + 1] << 8 | u8[w * 4 + 2] << 16 | u8[w * 4 + 3] << 24;
  }
  return { packed, scale, outDim, inDim };
}
__name(quantizeInt8RowMajor, "quantizeInt8RowMajor");
function quantizeInt4Group(f322, outDim, inDim, group = 128) {
  const groupsPerRow = inDim / group;
  const scale = new Float32Array(outDim * groupsPerRow);
  const q = new Int8Array(outDim * inDim);
  for (let o = 0; o < outDim; o++) {
    for (let g = 0; g < groupsPerRow; g++) {
      const base = o * inDim + g * group;
      let amax = 0;
      for (let i = 0; i < group; i++) {
        const a = Math.abs(f322[base + i]);
        if (a > amax) amax = a;
      }
      const s = amax > 0 ? amax / 7 : 1;
      scale[o * groupsPerRow + g] = s;
      const inv = 1 / s;
      for (let i = 0; i < group; i++) {
        let v = Math.round(f322[base + i] * inv);
        if (v > 7) v = 7;
        else if (v < -8) v = -8;
        q[base + i] = v;
      }
    }
  }
  const packed = new Uint32Array(outDim * inDim / 8);
  for (let w = 0; w < packed.length; w++) {
    let acc = 0;
    for (let j = 0; j < 8; j++) acc |= (q[w * 8 + j] & 15) << j * 4;
    packed[w] = acc >>> 0;
  }
  return { packed, scale, groupsPerRow };
}
__name(quantizeInt4Group, "quantizeInt4Group");

// src/qwgpu/model_uploader.js
var ModelUploader = class {
  static {
    __name(this, "ModelUploader");
  }
  constructor({ schema, q, q4, bufs, uploadF32, uploadU32, groupSize = 128 }) {
    this.schema = schema;
    this.q = q;
    this.q4 = q4;
    this.bufs = bufs;
    this.uploadF32 = uploadF32;
    this.uploadU32 = uploadU32;
    this.groupSize = groupSize;
    this.seen = /* @__PURE__ */ new Set();
  }
  visit({ name, shape, data }) {
    const desc = this.schema.validateTensor(name, shape);
    if (!desc) return;
    if (this.seen.has(name)) throw new Error(`duplicate tensor ${name}`);
    if (desc.quant === "int8") {
      const { packed, scale } = quantizeInt8RowMajor(data, shape[0], shape[1]);
      this.q[name] = { w: this.uploadU32(packed), scale: this.uploadF32(scale), N: shape[0], K: shape[1] };
    } else if (desc.quant === "int4") {
      const { packed, scale, groupsPerRow } = quantizeInt4Group(data, shape[0], shape[1], this.groupSize);
      this.q4[name] = {
        w: this.uploadU32(packed),
        scale: this.uploadF32(scale),
        N: shape[0],
        K: shape[1],
        gpr: groupsPerRow,
        desc
      };
    } else if (desc.quant === "f32") {
      this.bufs[name] = this.uploadF32(data);
    } else {
      throw new Error(`unsupported quant mode ${desc.quant} for ${name}`);
    }
    this.seen.add(name);
  }
  finalize() {
    this.schema.assertComplete(this.seen);
  }
};

// src/qwgpu/buffer_pool.js
var GPUBufferPool = class {
  static {
    __name(this, "GPUBufferPool");
  }
  constructor(device, { cacheBindGroups = true } = {}) {
    this.dev = device;
    this.cacheBindGroups = cacheBindGroups;
    this.uniformPool = [];
    this.uniformIdx = 0;
    this.staticUniforms = /* @__PURE__ */ new Map();
    this.bindGroups = /* @__PURE__ */ new Map();
    this.sensitiveBindGroups = /* @__PURE__ */ new Set();
    this.bufferIds = /* @__PURE__ */ new WeakMap();
    this.pipelineIds = /* @__PURE__ */ new WeakMap();
    this.nextBufferId = 1;
    this.nextPipelineId = 1;
    this._stats = this._emptyStats();
  }
  /*
   * TECHNIQUE: Bind group caching (opt-in per call site)
   *   Frequently reused (pipeline + buffer set) combinations are stored in a Map.
   *   Avoids repeated GPU bind group creation on the hot GEMV / attention paths.
   *   Sensitive / one-shot groups are deliberately not cached.
   */
  _emptyStats() {
    return {
      buffersCreated: 0,
      dynamicUniformWrites: 0,
      staticUniformHits: 0,
      staticUniformMisses: 0,
      bindGroupHits: 0,
      bindGroupMisses: 0,
      uncachedBindGroups: 0
    };
  }
  resetStats() {
    this._stats = this._emptyStats();
  }
  stats() {
    return {
      ...this._stats,
      uniformPoolSize: this.uniformPool.length,
      staticUniforms: this.staticUniforms.size,
      bindGroups: this.bindGroups.size
    };
  }
  buffer(size, usage) {
    this._stats.buffersCreated++;
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
    if (!b) {
      b = this.buffer(32, usage);
      this.uniformPool[this.uniformIdx] = b;
    }
    this.uniformIdx++;
    this._stats.dynamicUniformWrites++;
    this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  }
  resetUniforms() {
    this.uniformIdx = 0;
  }
  staticUniform(key, arr, usage) {
    let b = this.staticUniforms.get(key);
    if (!b) {
      this._stats.staticUniformMisses++;
      b = this.buffer(32, usage);
      this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
      this.staticUniforms.set(key, b);
    } else this._stats.staticUniformHits++;
    return b;
  }
  idForBuffer(buffer) {
    let id = this.bufferIds.get(buffer);
    if (!id) {
      id = this.nextBufferId++;
      this.bufferIds.set(buffer, id);
    }
    return id;
  }
  idForPipeline(pipe) {
    let id = this.pipelineIds.get(pipe);
    if (!id) {
      id = this.nextPipelineId++;
      this.pipelineIds.set(pipe, id);
    }
    return id;
  }
  uncachedBindGroup(pipe, buffers) {
    this._stats.uncachedBindGroups++;
    return this.dev.createBindGroup({
      label: pipe.__name ? `${pipe.__name}:bg:${buffers.length}` : void 0,
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } }))
    });
  }
  cachedBindGroup(pipe, buffers, key, { sensitive = false } = {}) {
    if (!this.cacheBindGroups || !key) return this.uncachedBindGroup(pipe, buffers);
    const fullKey = `${this.idForPipeline(pipe)}:${key}:${buffers.map((b) => this.idForBuffer(b)).join(",")}`;
    let bg = this.bindGroups.get(fullKey);
    if (!bg) {
      this._stats.bindGroupMisses++;
      bg = this.uncachedBindGroup(pipe, buffers);
      this.bindGroups.set(fullKey, bg);
      if (sensitive) this.sensitiveBindGroups.add(fullKey);
    } else this._stats.bindGroupHits++;
    return bg;
  }
  clearSensitiveBindGroups() {
    for (const key of this.sensitiveBindGroups) this.bindGroups.delete(key);
    this.sensitiveBindGroups.clear();
  }
};

// src/qwgpu/runtime.js
var STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
var UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
var QwenWGPU = class {
  static {
    __name(this, "QwenWGPU");
  }
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
    this.workgroupAutotunePromise = null;
    this._argmaxReadBusy = false;
    this._topKReadBusy = false;
  }
  _normalizeFeatures(opts = {}) {
    const prefillAttention = opts.prefillAttention || "block";
    if (!["row", "block"].includes(prefillAttention))
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
      pagedAttention: !!opts.pagedAttention
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
  }
  usingF16() {
    return !!this._useF16;
  }
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
    const useTS = this.hasTimestampQuery;
    const timeKernel = /* @__PURE__ */ __name(async (spec, pipe, label) => {
      const n = spec.n;
      const a = this._buf(n * 4);
      const g = this._buf(n * 4);
      const y = this._buf(n * 4);
      const buffers = spec.buffers(a, y, g);
      const imm = spec.imm(n);
      let gpuMs = 0;
      let usedGPU = false;
      if (useTS) {
        const qs = this.dev.createQuerySet({ type: "timestamp", count: 2 });
        const resolveBuf = this._buf(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
        const readBuf = this._buf(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const tWall0 = typeof performance !== "undefined" ? performance.now() : Date.now();
        for (let i = 0; i < iters; i++) {
          const enc = this.dev.createCommandEncoder();
          const bg = this._bg(pipe, buffers);
          const p = enc.beginComputePass({
            timestampWrites: {
              querySet: qs,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1
            }
          });
          p.setPipeline(pipe);
          if (bg) p.setBindGroup(0, bg);
          if (imm) p.setImmediates(0, imm);
          p.dispatchWorkgroups(Math.ceil(n / (pipe.__wg || 256)), 1);
          p.end();
          enc.resolveQuerySet(qs, 0, 2, resolveBuf, 0);
          enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16);
          this.dev.queue.submit([enc.finish()]);
          if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
          await readBuf.mapAsync(GPUMapMode.READ);
          const t = new BigInt64Array(readBuf.getMappedRange());
          const us = Number(t[1] - t[0]) / 1e3;
          gpuMs += us;
          readBuf.unmap();
        }
        const wallMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - tWall0;
        resolveBuf.destroy?.();
        readBuf.destroy?.();
        qs.destroy?.();
        usedGPU = true;
        a.destroy?.();
        g.destroy?.();
        y.destroy?.();
        return gpuMs / iters / 1e3;
      }
      const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
      for (let i = 0; i < iters; i++) {
        const enc = this.dev.createCommandEncoder();
        const bg = this._bg(pipe, buffers);
        this._dispatch(enc, pipe, bg, Math.ceil(n / (pipe.__wg || 256)), 1, label + ":bench", imm);
        this.dev.queue.submit([enc.finish()]);
        if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
      }
      const ms = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
      a.destroy?.();
      g.destroy?.();
      y.destroy?.();
      return ms / iters;
    }, "timeKernel");
    const kernels = [
      { name: "add", src: ADD, n: 8192, buffers: /* @__PURE__ */ __name((a, y) => [a, y], "buffers"), imm: /* @__PURE__ */ __name((n) => new Uint32Array([n]), "imm") },
      { name: "rms", src: RMSNORM, n: 4096, buffers: /* @__PURE__ */ __name((a, y, g) => [a, g, y], "buffers"), imm: /* @__PURE__ */ __name((n) => new Float32Array([n, this.cfg.rmsNormEps]), "imm") },
      { name: "silu", src: SILUMUL, n: 8192, buffers: /* @__PURE__ */ __name((a, y) => [a, y], "buffers"), imm: /* @__PURE__ */ __name((n) => new Uint32Array([n]), "imm") }
    ];
    for (const k of kernels) {
      try {
        let best = { wg: 256, ms: Infinity };
        for (const wg of cands) {
          const p = this._pipe(k.src, `${k.name}:autotune:${wg}`, { WG: wg });
          p.__wg = wg;
          const ms = await timeKernel(k, p, `${k.name}${wg}`);
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
    this.bestWorkgroupSizes = {
      add: results.bestAdd?.wg,
      rms: results.bestRms?.wg,
      silu: results.bestSilu?.wg,
      source: useTS ? "gpu-ts" : "wall"
    };
    console.log("[autotune] WG microbench results (ms/iter, source=" + (useTS ? "gpu-ts" : "wall") + "):", results);
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
    const processedCode = typeof code === "string" ? code.replaceAll("WG_SIZE", this.workgroupSize || 64) : code;
    const m = this.dev.createShaderModule({
      label: name || void 0,
      code: processedCode
    });
    const comp = { module: m, entryPoint: "main" };
    if (overrides && typeof overrides === "object") comp.constants = overrides;
    const pipe = this.dev.createComputePipeline({
      label: name ? `${name}-pipeline` : void 0,
      layout: "auto",
      compute: comp
    });
    if (overrides?.WG) pipe.__wg = overrides.WG;
    if (name) pipe.__name = name;
    return pipe;
  }
  /*
   * TECHNIQUE: Specialization via pipeline constants (overrides)
   *   Workgroup size and other small values are passed as pipeline-overridable
   *   constants instead of uniforms or JS branches. Allows the shader compiler
   *   to specialize the binary (better than runtime if).
   */
  // `source` is a base URL string OR a reader { range, text } (e.g. hfReader/fileReader).
  async build(source, onProgress = () => {
  }) {
    const shaderCompileStart = performance.now();
    const dev = this.dev, c = this.cfg;
    this.CHUNK = 128;
    this._initRuntimeOptions();
    this.maxCtx = this.opts.maxCtx || 8192;
    this.maxPrefillT = Math.min(this.opts.maxPrefillT || 8192, this.maxCtx);
    const isAppleSilicon = this.dev.limits.minStorageBufferOffsetAlignment === 4;
    const isIntelArc = this.dev.limits.minStorageBufferOffsetAlignment === 256;
    this.workgroupSize = isAppleSilicon || isIntelArc ? 32 : 64;
    onProgress && onProgress(`workgroup size chosen: ${this.workgroupSize} (apple/intel bias toward 32)`, 0);
    let hasDP4a = false;
    if (typeof navigator !== "undefined" && navigator.gpu?.wgslLanguageFeatures?.has?.("packed_4x8_integer_dot_product")) {
      dev.pushErrorScope("validation");
      try {
        dev.createShaderModule({
          code: `enable packed_4x8_integer_dot_product; @compute @workgroup_size(1) fn main() {}`
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
    const hasF16 = this.dev.features.has("shader-f16");
    this.hasF16 = hasF16;
    this.hasTimestampQuery = this.dev.features.has("timestamp-query");
    this.pam = new PagedAttentionManager(this.maxCtx);
    this.pipes = {
      gemv: this._pipe(GEMV, "gemv"),
      loraA: this._pipe(LORA_A, "loraA"),
      loraABatch: this._pipe(LORA_A_BATCH, "loraABatch"),
      loraBAdd: this._pipe(LORA_B_ADD, "loraBAdd"),
      loraBAddT: this._pipe(LORA_B_ADD_T, "loraBAddT"),
      rms: this._pipe(RMSNORM, "rms", { WG: this.workgroupSize || 256 }),
      rmsF16: hasF16 ? this._pipe(RMSNORM_F16, "rmsF16", { WG: this.workgroupSize || 256 }) : null,
      rope: this._pipe(ROPE, "rope"),
      ropeF16: hasF16 ? this._pipe(ROPE_F16, "ropeF16") : null,
      ropeQK: this._pipe(ROPE_QK, "ropeQK"),
      ropeQKF16: hasF16 ? this._pipe(ROPE_QK_F16, "ropeQKF16") : null,
      ropeT: this._pipe(ROPE_T, "ropeT"),
      ropeTF16: hasF16 ? this._pipe(ROPE_T_F16, "ropeTF16") : null,
      attnP: this._pipe(ATTN_PARTIAL, "attnP", { WG: 128 }),
      attnPF16: hasF16 ? this._pipe(ATTN_PARTIAL_F16, "attnPF16", { WG: 128 }) : null,
      attnC: this._pipe(ATTN_COMBINE, "attnC", { WG: 128 }),
      attnCF16: hasF16 ? this._pipe(ATTN_COMBINE_F16, "attnCF16", { WG: 128 }) : null,
      add: this._pipe(ADD, "add", { WG: this.workgroupSize || 256 }),
      silu: this._pipe(SILUMUL, "silu", { WG: this.workgroupSize || 256 }),
      addF16: hasF16 ? this._pipe(ADD_F16, "addF16", { WG: this.workgroupSize || 256 }) : null,
      siluF16: hasF16 ? this._pipe(SILUMUL_F16, "siluF16", { WG: this.workgroupSize || 256 }) : null,
      embed: this._pipe(EMBED, "embed"),
      embedBuf: this._pipe(EMBED_BUF, "embedBuf"),
      argmax: this._pipe(ARGMAX, "argmax"),
      gemv4: this._pipe(GEMV4, "gemv4"),
      gemv4Add: this._pipe(GEMV4_ADD, "gemv4Add"),
      qkvGemv4: this._pipe(QKV_GEMV4, "qkvGemv4"),
      gateUpSiluGemv4: this._pipe(GATE_UP_SILU_GEMV4, "gateUpSiluGemv4"),
      topkSelect: this._pipe(TOPK_SELECT, "topkSelect"),
      sampleTopK: this._pipe(SAMPLE_TOPK, "sampleTopK"),
      gemm4: this._pipe(GEMM4, "gemm4"),
      gemm4AddT: this._pipe(GEMM4_ADD_T, "gemm4AddT"),
      rmsT: this._pipe(RMSNORM_T, "rmsT", { WG: this.workgroupSize || 256 }),
      rmsTF16: hasF16 ? this._pipe(RMSNORM_T_F16, "rmsTF16", { WG: this.workgroupSize || 256 }) : null,
      embedT: this._pipe(EMBED_T, "embedT"),
      attnPrefill: this._pipe(ATTN_PREFILL, "attnPrefill"),
      attnPrefillBlock: this._pipe(ATTN_PREFILL_BLOCK, "attnPrefillBlock"),
      dynQuant: this._pipe(DYN_QUANT_X, "dynQuant"),
      dynQuantT: this._pipe(DYN_QUANT_X_T, "dynQuantT"),
      gemv4W4A8: this._pipe(GEMV4_W4A8(hasDP4a, this.workgroupSize), "gemv4W4A8"),
      gemv4AddW4A8: this._pipe(GEMV4_ADD_W4A8(hasDP4a, this.workgroupSize), "gemv4AddW4A8"),
      qkvGemv4W4A8: this._pipe(QKV_GEMV4_W4A8(hasDP4a, this.workgroupSize), "qkvGemv4W4A8"),
      gateUpSiluGemv4W4A8: this._pipe(GATE_UP_SILU_GEMV4_W4A8(hasDP4a, this.workgroupSize), "gateUpSiluGemv4W4A8"),
      gemm4W4A8: this._pipe(GEMM4_W4A8(hasDP4a), "gemm4W4A8"),
      gemm4AddTW4A8: this._pipe(GEMM4_ADD_T_W4A8(hasDP4a), "gemm4AddTW4A8"),
      rmsNormQkvRope: this._pipe(GEMV4_QKV_ROPE_RMS, "rmsNormQkvRope"),
      writeKvPage: this._pipe(WRITE_KV_PAGE, "writeKvPage"),
      writeKvPageBatch: this._pipe(WRITE_KV_PAGE_BATCH, "writeKvPageBatch"),
      attnPartialPaged: this._pipe(ATTN_PARTIAL_PAGED, "attnPartialPaged"),
      attnPrefillPaged: this._pipe(ATTN_PREFILL_PAGED, "attnPrefillPaged"),
      attnPrefillBlockPaged: this._pipe(ATTN_PREFILL_BLOCK_PAGED, "attnPrefillBlockPaged")
    };
    this.shaderCompileMs = performance.now() - shaderCompileStart;
    if (hasF16) {
      this.setUseF16(true);
      onProgress("f16 compute enabled (add/silu/rms/rope/attn-partial/combine paths)", 0);
    }
    if (this.hasTimestampQuery) {
      onProgress("timestamp-query available (precise GPU timing + autotune)", 0);
    }
    onProgress("streaming + quantizing weights", 0);
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
      uploadF32: /* @__PURE__ */ __name((arr) => this._f32(arr), "uploadF32"),
      uploadU32: /* @__PURE__ */ __name((arr) => this._u32(arr), "uploadU32")
    });
    if (source === "mock") {
      for (const name of this.schema.expectedNames) {
        const desc = this.schema.tensors.find((t) => t.name === name);
        const shape = desc.shape;
        const numel = shape.reduce((a, b) => a * b, 1);
        const type = desc.quant === "int8" ? "I8" : "F32";
        uploader.visit({ name, shape, data: new Uint8Array(numel * (type === "I8" ? 1 : 4)), type });
      }
    } else {
      await streamSafetensors(source, {
        names: this.schema.expectedNames,
        onProgress,
        onTensor: /* @__PURE__ */ __name(async (tensor) => {
          uploader.visit(tensor);
          if (uploader.seen.size % 48 === 0) await new Promise((r) => setTimeout(r, 0));
        }, "onTensor")
      });
    }
    uploader.finalize();
    await this._buildPackedProjectionBuffers();
    this._buildRope(this.maxCtx);
    this.kc = [], this.vc = [];
    const kvSize = c.numKVHeads * this.maxCtx * c.headDim * 4;
    for (let i = 0; i < c.numLayers; i++) {
      this.kc.push(this._buf(kvSize));
      this.vc.push(this._buf(kvSize));
    }
    const H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
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
      sampled: this._buf(4),
      // single u32 chosen by GPU sampler (Phase 5)
      x_q: this._buf(Math.max(qd, I) * 4),
      scale_x: this._buf(256 * 4),
      blockTableBuf: this._buf(this.pam.maxBlocksPerSeq * 4, STORAGE | GPUBufferUsage.COPY_DST)
    };
    this.idsRead = this._buf(this.decodeBatchCapacity * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.argmaxRead = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampleIdsRead = this._buf(this.maxSamplingTopK * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampleValsRead = this._buf(this.maxSamplingTopK * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampledRead = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sT = null;
    this.sTcap = 0;
    this._initStaticUniforms();
    if (this.decodeBatchMode === "auto") {
      onProgress("autotuning decode batch", 0.98);
      await this.autotuneDecodeBatch();
    }
    onProgress("ready", 1);
    if (!this._didAutoWG) {
      this._didAutoWG = true;
      this.workgroupAutotunePromise = this.autotuneWorkgroups({ iters: 2, apply: true }).catch((e) => ({
        error: String(e)
      }));
    }
    return this;
  }
  _initRuntimeOptions() {
    const opts = this.opts;
    this.decodeBatchMode = opts.decodeBatchSize === "auto" ? "auto" : "fixed";
    this.decodeBatchCandidates = (opts.decodeBatchCandidates || [1, 2, 4, 8, 16, 32]).map((x) => Math.max(1, Math.floor(Number(x) || 0))).filter(Boolean);
    const requested = opts.decodeBatchSize === void 0 || opts.decodeBatchSize === "auto" ? 16 : Math.max(1, Math.floor(Number(opts.decodeBatchSize)));
    this.maxDecodeBatchSize = Math.max(
      1,
      Math.floor(Number(opts.maxDecodeBatchSize || Math.max(requested, ...this.decodeBatchCandidates, 16)))
    );
    this.decodeBatchCapacity = Math.min(this.maxDecodeBatchSize, Math.max(requested, ...this.decodeBatchCandidates));
    this.MAXBATCH = Math.min(requested, this.decodeBatchCapacity);
    this.decodeBatchWarmupTokens = Math.max(0, Math.floor(Number(opts.decodeBatchWarmupTokens ?? 4)));
    this.decodeBatchWarmupSize = Math.min(
      this.decodeBatchCapacity,
      Math.max(1, Math.floor(Number(opts.decodeBatchWarmupSize ?? 4)))
    );
    this.decodeBatchMaxLatencyMs = Number(opts.decodeBatchMaxLatencyMs ?? 250);
    this.samplingTopK = Math.max(1, Math.floor(Number(opts.samplingTopK ?? 40)));
    this.maxSamplingTopK = Math.max(this.samplingTopK, Math.floor(Number(opts.maxSamplingTopK ?? 64)));
    this.decodeBatchTuning = {
      selected: this.MAXBATCH,
      candidates: [],
      reason: this.decodeBatchMode === "auto" ? "pending" : "fixed"
    };
  }
  _buildRope(maxSeq) {
    const { headDim, ropeTheta } = this.cfg;
    const half = headDim / 2;
    const cos = new Float32Array(maxSeq * headDim), sin = new Float32Array(maxSeq * headDim);
    for (let p = 0; p < maxSeq; p++)
      for (let i = 0; i < half; i++) {
        const a = p / Math.pow(ropeTheta, 2 * i / headDim);
        const cc = Math.cos(a), ss = Math.sin(a);
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
      argmax: this._staticUni(`argmax:${c.vocabSize}`, new Uint32Array([c.vocabSize]))
    };
  }
  async _buildPackedProjectionBuffers() {
    const enc = this.dev.createCommandEncoder();
    const copy = /* @__PURE__ */ __name((src, dst, dstOffset, bytes) => enc.copyBufferToBuffer(src, 0, dst, dstOffset, bytes), "copy");
    this.packedBytes = 0;
    for (const L of this.plan.layers) {
      const q = this.q4[L.q.weight], k = this.q4[L.k.weight], v = this.q4[L.v.weight];
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
      let wOff = 0, sOff = 0, bOff = 0;
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
      const gate = this.q4[L.gate.weight], up = this.q4[L.up.weight];
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
    const decodeScratchBytes = c.hiddenSize * 2 * 4 + (c.numHeads * c.headDim + 2 * c.numKVHeads * c.headDim + c.numHeads * c.headDim) * 4 + (Math.max(c.numHeads * c.headDim, c.intermediateSize) + c.intermediateSize + c.vocabSize) * 4;
    const prefillScratchBytes = this.sTcap ? this.sTcap * (3 * c.hiddenSize + c.numHeads * c.headDim + 2 * c.numKVHeads * c.headDim + c.numHeads * c.headDim + 2 * c.intermediateSize) * 4 : 0;
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
      bytes
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
      bytes
    };
  }
  setLora(adapter) {
    this.lora = adapter;
    this._loraEpoch++;
    this.pool.clearSensitiveBindGroups();
  }
  // {modules: {key:{A,B,rank,scale}}}  A:[K][rank], B:[rank][N] f32 GPUBuffers
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
      this.prof.cats.push(cat || "misc");
      ts = { querySet: this.prof.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
    }
    const p = enc.beginComputePass(ts ? { timestampWrites: ts } : void 0);
    p.setPipeline(pipe);
    if (bg) p.setBindGroup(0, bg);
    if (imm) {
      if (Array.isArray(imm)) {
        let off = 0;
        for (const part of imm) {
          p.setImmediates(off, part);
          off += part.byteLength || part.length * (part.BYTES_PER_ELEMENT || 4);
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
      qs: this.dev.createQuerySet({ type: "timestamp", count: cap * 2 }),
      cap,
      idx: 0,
      cats: [],
      resolve: this._buf(cap * 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
      read: this._buf(cap * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
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
      const us = Number(t[2 * i + 1] - t[2 * i]) / 1e3;
      const c = this.prof.cats[i];
      sums[c] = (sums[c] || 0) + us;
    }
    this.prof.read.unmap();
    return sums;
  }
  poolStats() {
    return this.pool.stats();
  }
  // Phase 4 observability: best workgroup sizes chosen by autotune (or null if not run).
  getBestWorkgroupSizes() {
    return this.bestWorkgroupSizes ? { ...this.bestWorkgroupSizes } : null;
  }
  resetPoolStats() {
    this.pool.resetStats();
  }
  estimateKvCacheBytes() {
    const c = this.cfg;
    return c.numLayers * 2 * c.numKVHeads * this.maxCtx * c.headDim * 4;
  }
  estimatePrefillScratchBytes(T, loraRank = this._activeMaxLoraRank()) {
    const c = this.cfg, H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    return T * H * 4 * 2 + T * qd * 4 * 2 + T * kvd * 4 * 2 + T * I * 4 * 2 + T * 4 + Math.max(1, T * Math.max(1, loraRank)) * 4;
  }
  greedyBatchSizeFor({ emitted = 0, remaining = Infinity, pos = 0 } = {}) {
    const interactive = emitted < this.decodeBatchWarmupTokens ? this.decodeBatchWarmupSize : this.MAXBATCH;
    return Math.max(0, Math.min(interactive, remaining, this.maxCtx - pos, this.decodeBatchCapacity));
  }
  async _resetAutotuneDecodeState(tokens, seedTokenId = 0) {
    const c = this.cfg, S = this.s, H = c.hiddenSize, hd = c.headDim, qd = c.numHeads * hd, kvd = c.numKVHeads * hd, I = c.intermediateSize;
    const nsplitMax = Math.ceil(this.maxCtx / this.CHUNK);
    const touchedTokens = Math.min(Math.max(0, Math.floor(tokens)), this.maxCtx);
    const enc = this.dev.createCommandEncoder();
    const clear = /* @__PURE__ */ __name((buf, bytes) => {
      if (bytes > 0) enc.clearBuffer(buf, 0, bytes);
    }, "clear");
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
    const candidates = [...new Set(this.decodeBatchCandidates)].filter((k) => k >= 1 && k <= this.decodeBatchCapacity && k <= this.maxCtx).sort((a, b) => a - b);
    const rows = [];
    const resetTokens = candidates.length ? Math.max(...candidates) : 0;
    let selected = candidates[0] ?? this.MAXBATCH, best = Infinity;
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
        selected = rows.reduce((a, b) => a.msPerToken <= b.msPerToken ? a : b).k;
      this.MAXBATCH = selected;
      this.decodeBatchTuning = {
        selected,
        candidates: rows,
        reason: "auto wall-clock decodeGreedyBatch with reset state"
      };
    } catch (e) {
      this.decodeBatchTuning = { selected: this.MAXBATCH, candidates: rows, reason: `auto failed: ${e.message}` };
    } finally {
      if (resetTokens > 0) {
        try {
          await this._resetAutotuneDecodeState(resetTokens);
        } catch {
        }
      }
    }
    return this.decodeBatchTuning;
  }
  // y = int8-GEMV(x, q) [+bias] [+lora]. q={w,scale,N,K}. moduleKey for LoRA lookup.
  gemv(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) {
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      const bgA = this._bgCached(
        this.pipes.loraA,
        [xBuf, mod.A, this.s.loraD, uA],
        `loraA:${moduleKey}:${this._loraEpoch}`,
        { sensitive: true }
      );
      this._dispatch(enc, this.pipes.loraA, bgA, mod.rank, 1, "loraA");
    }
    const meta = this._gemvMeta(q, biasBuf, mod);
    const key = `gemv:${moduleKey || "base"}:${q.K}:${q.N}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv, bg, meta.gx, meta.gy, `gemv:${q.N}x${q.K}`, meta.bytes);
  }
  gemv4(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (this.debugCapture) console.log("VWG gemv4: " + moduleKey + " mod=" + !!mod);
    if (mod) {
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      this._dispatch(
        enc,
        this.pipes.loraA,
        this._bgCached(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, uA], `loraA:${moduleKey}:${this._loraEpoch}`, {
          sensitive: true
        }),
        mod.rank,
        1,
        "loraA"
      );
      if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
        enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xSeq, this.debugStep * q.K * 4, q.K * 4);
        enc.copyBufferToBuffer(this.s.loraD, 0, this.debugBufs.dSeq, this.debugStep * mod.rank * 4, mod.rank * 4);
      }
    }
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4, bg, meta.gx, meta.gy, `g4:${q.N}x${q.K}`, meta.bytes);
    if (mod) {
      if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
        enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.ySeq, this.debugStep * q.N * 4, q.N * 4);
        this.debugStep++;
      }
    }
  }
  _loraA(enc, xBuf, q, mod, dBuf, moduleKey, label = "loraA") {
    const imm = new Uint32Array([q.K, mod.rank]);
    this._dispatch(
      enc,
      this.pipes.loraA,
      this._bgCached(this.pipes.loraA, [xBuf, mod.A, dBuf], `${label}:${moduleKey}:${this._loraEpoch}`, {
        sensitive: true
      }),
      mod.rank,
      1,
      label,
      imm
    );
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
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
      { sensitive: true }
    );
    this._dispatch(enc, this.pipes.loraBAdd, bg, Math.ceil(q.N / 256), 1, "loraB", new Uint8Array(meta));
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
      enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.ySeq, this.debugStep * q.N * 4, q.N * 4);
      this.debugStep++;
    }
  }
  gemv4Add(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4add:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4Add,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4Add, bg, meta.gx, meta.gy, `g4add:${q.N}x${q.K}`, meta.bytes);
  }
  dynQuant(enc, xBuf, x_qBuf, scale_xBuf, K) {
    const numGroups = Math.ceil(K / 128);
    const imm = new Uint32Array([K]);
    const bg = this._bg(this.pipes.dynQuant, [xBuf, x_qBuf, scale_xBuf]);
    this._dispatch(enc, this.pipes.dynQuant, bg, numGroups, 1, "dynQuant", imm);
  }
  dynQuantT(enc, xBuf, x_qBuf, scale_xBuf, K, T) {
    const numGroups = Math.ceil(K / 128);
    const imm = new Uint32Array([K, T]);
    const bg = this._bg(this.pipes.dynQuantT, [xBuf, x_qBuf, scale_xBuf]);
    this._dispatch(enc, this.pipes.dynQuantT, bg, numGroups, T, "dynQuantT", imm);
  }
  gemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) {
      const uA = this._staticUni(`loraA:${this._loraEpoch}:${q.K}:${mod.rank}`, new Uint32Array([q.K, mod.rank]));
      this._dispatch(
        enc,
        this.pipes.loraA,
        this._bgCached(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, uA], `loraA:${moduleKey}:${this._loraEpoch}`, {
          sensitive: true
        }),
        mod.rank,
        1,
        "loraA"
      );
    }
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4_w4a8:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
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
        yBuf
      ],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4W4A8, bg, meta.gx, meta.gy, `g4w4a8:${q.N}x${q.K}`, meta.bytes);
  }
  gemv4AddW4A8(enc, xBuf, x_qBuf, scale_xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4add_w4a8:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
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
        yBuf
      ],
      key,
      { sensitive: !!mod }
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
      { sensitive: false }
    );
    this._dispatch(
      enc,
      this.pipes.qkvGemv4W4A8,
      bg,
      gx,
      Math.ceil(packed.totalN / gx),
      `qkvw4a8:${packed.totalN}x${packed.K}`,
      imm
    );
    for (const [part, out] of [
      [L.q, qBuf],
      [L.k, kBuf],
      [L.v, vBuf]
    ]) {
      const mod = this.lora?.modules?.[part.loraKey];
      if (!mod) continue;
      const q = this.q4[part.weight];
      this._loraA(enc, xBuf, q, mod, this.s.loraD, part.loraKey);
      this._loraBAdd(enc, out, q, mod, this.s.loraD, part.loraKey);
    }
  }
  _gateUpImmediate(packed, gx, gateMod, upMod) {
    const imm = new Uint32Array(12);
    imm.set([
      packed.K,
      packed.N,
      packed.gpr,
      gx,
      gateMod ? gateMod.rank : 0,
      upMod ? upMod.rank : 0,
      gateMod ? 1 : 0,
      upMod ? 1 : 0
    ]);
    const f322 = new Float32Array(imm.buffer);
    f322[8] = gateMod ? gateMod.scale : 0;
    f322[9] = upMod ? upMod.scale : 0;
    return imm;
  }
  gateUpSiluGemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, packed, yBuf, L) {
    const gate = this.q4[L.gate.weight], up = this.q4[L.up.weight];
    const gateMod = this.lora?.modules?.[L.gate.loraKey];
    const upMod = this.lora?.modules?.[L.up.loraKey];
    if (gateMod) this._loraA(enc, xBuf, gate, gateMod, this.s.loraD, L.gate.loraKey, "loraA:gate");
    if (upMod) this._loraA(enc, xBuf, up, upMod, this.s.loraD2, L.up.loraKey, "loraA:up");
    const gx = Math.min(packed.N, 65535);
    const imm = this._gateUpImmediate(packed, gx, gateMod, upMod);
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
        upMod ? upMod.B : this.s.dummy
      ],
      `gu_w4a8:${L.index}:${this._loraEpoch}:${gateMod ? 1 : 0}:${upMod ? 1 : 0}`,
      { sensitive: !!(gateMod || upMod) }
    );
    this._dispatch(
      enc,
      this.pipes.gateUpSiluGemv4W4A8,
      bg,
      gx,
      Math.ceil(packed.N / gx),
      `guw4a8:${packed.N}x${packed.K}`,
      imm
    );
  }
  gemm4W4A8(enc, aBuf, a_qBuf, scale_xBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4W4A8, [a_qBuf, scale_xBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4W4A8, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4W4A8", imm);
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
      yBuf
    ]);
    this._dispatch(enc, this.pipes.gemm4AddTW4A8, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4AddTW4A8", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  rmsNormQkvRope(enc, xBuf, layerIndex, pos) {
    const c = this.cfg, L = this.plan.layers[layerIndex];
    const packed = this.qkv[L.index];
    const meta = new Uint32Array([
      packed.K,
      packed.totalN,
      packed.qN,
      packed.kN,
      packed.vN,
      packed.gpr,
      20,
      pos,
      c.headDim,
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
        this.s.v
      ]
    );
    this._dispatch(enc, this.pipes.rmsNormQkvRope, bg, 20, 1, "rmsNormQkvRope", meta);
    for (const [part, out] of [
      [L.q, this.s.q],
      [L.k, this.s.k],
      [L.v, this.s.v]
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
    this._dispatch(enc, this.pipes.writeKvPage, bg, Math.ceil(kvd / 256), 1, "writeKvPage", meta);
  }
  writeKvPageBatch(enc, kBuf, vBuf, kcBuf, vcBuf, T, off, layerIndex) {
    const c = this.cfg;
    const kvd = c.numKVHeads * c.headDim;
    this.pam.ensureBlocks(0, off + T);
    const btArr = this.pam.getBlockTableArray(0);
    this.dev.queue.writeBuffer(this.s.blockTableBuf, 0, btArr);
    const meta = new Uint32Array([T, 0, this.pam.maxBlocksPerSeq, kvd, off]);
    const bg = this._bg(this.pipes.writeKvPageBatch, [kBuf, vBuf, kcBuf, vcBuf, this.s.blockTableBuf]);
    this._dispatch(enc, this.pipes.writeKvPageBatch, bg, Math.ceil(T * kvd / 256), 1, "writeKvPageBatch", meta);
  }
  attnPaged(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg, S = this.s;
    const nsplit = Math.ceil(ctx / this.CHUNK);
    const bgP = this._bg(this.pipes.attnPartialPaged, [
      qBuf,
      kc,
      vc,
      S.pm,
      S.pz,
      S.po,
      S.blockTableBuf
    ]);
    const immP = new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim, nsplit, this.CHUNK, 0, this.pam.maxBlocksPerSeq]);
    this._dispatch(enc, this.pipes.attnPartialPaged, bgP, c.numHeads, nsplit, "attnP_paged", immP);
    const useF16C = this.usingF16() && this.pipes.attnCF16;
    const pipeC = useF16C ? this.pipes.attnCF16 : this.pipes.attnC;
    const bgC = this._bg(pipeC, [
      S.pm,
      S.pz,
      S.po,
      oBuf
    ]);
    const immC = new Uint32Array([c.numHeads, c.headDim, nsplit, 0]);
    this._dispatch(enc, pipeC, bgC, c.numHeads, 1, useF16C ? "attnCF16" : "attnC", immC);
  }
  attnPrefillPaged(enc, qBuf, kc, vc, oBuf, T, qStart = 0, ctx = T) {
    const c = this.cfg;
    if (this.features.prefillAttention === "block" || qStart !== 0 || ctx !== T) {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, qStart, ctx, 0, this.pam.maxBlocksPerSeq]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillBlockPaged,
        this._bg(this.pipes.attnPrefillBlockPaged, [qBuf, kc, vc, oBuf, this.s.blockTableBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        "attnPrefillBlockPaged",
        imm
      );
    } else {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, 0, this.pam.maxBlocksPerSeq, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillPaged,
        this._bg(this.pipes.attnPrefillPaged, [
          qBuf,
          kc,
          vc,
          oBuf,
          this.s.blockTableBuf
        ]),
        c.numHeads,
        T,
        "attnPrefillPaged",
        imm
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
      { sensitive: false }
    );
    this._dispatch(enc, this.pipes.qkvGemv4, bg, gx, Math.ceil(packed.totalN / gx), `qkv:${packed.totalN}x${packed.K}`, imm);
    for (const [part, out] of [
      [L.q, qBuf],
      [L.k, kBuf],
      [L.v, vBuf]
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
      packed.K,
      totalPairs,
      qPairs,
      kPairs,
      vPairs,
      packed.gpr,
      gx,
      pos,
      this.cfg.headDim,
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
        vBuf
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
    const gate = this.q4[L.gate.weight], up = this.q4[L.up.weight];
    const gateMod = this.lora?.modules?.[L.gate.loraKey];
    const upMod = this.lora?.modules?.[L.up.loraKey];
    if (gateMod) this._loraA(enc, xBuf, gate, gateMod, this.s.loraD, L.gate.loraKey, "loraA:gate");
    if (upMod) this._loraA(enc, xBuf, up, upMod, this.s.loraD2, L.up.loraKey, "loraA:up");
    const gx = Math.min(packed.N, 65535);
    const imm = this._gateUpImmediate(packed, gx, gateMod, upMod);
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
        upMod ? upMod.B : this.s.dummy
      ],
      `gu:${L.index}:${this._loraEpoch}:${gateMod ? 1 : 0}:${upMod ? 1 : 0}`,
      { sensitive: !!(gateMod || upMod) }
    );
    this._dispatch(enc, this.pipes.gateUpSiluGemv4, bg, gx, Math.ceil(packed.N / gx), `gu:${packed.N}x${packed.K}`, imm);
  }
  rms(enc, xBuf, gBuf, yBuf, K) {
    const imm = new Float32Array([K, this.cfg.rmsNormEps]);
    const useF16 = this.usingF16() && this.pipes.rmsF16;
    const pipe = useF16 ? this.pipes.rmsF16 : this.pipes.rms;
    const key = `rms:${K}${useF16 ? ":f16" : ""}`;
    this._dispatch(enc, pipe, this._bgCached(pipe, [xBuf, gBuf, yBuf], key), 1, 1, useF16 ? "rmsF16" : "rms", imm);
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
        this.ropeSin
      ]),
      Math.ceil(nHeads * (this.cfg.headDim / 2) / 256),
      1,
      useF16 ? "ropeF16" : "rope",
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
        this.ropeSin
      ]),
      Math.ceil(pairs / 256),
      1,
      useF16 ? "ropeQKF16" : "ropeQK",
      new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, pos])
    );
  }
  attn(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg, S = this.s;
    const nsplit = Math.ceil(ctx / this.CHUNK);
    const useF16P = this.usingF16() && this.pipes.attnPF16;
    const pipeP = useF16P ? this.pipes.attnPF16 : this.pipes.attnP;
    const bgP = this._bg(pipeP, [
      qBuf,
      kc,
      vc,
      S.pm,
      S.pz,
      S.po
    ]);
    const immP = new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim, nsplit, this.CHUNK]);
    this._dispatch(enc, pipeP, bgP, c.numHeads, nsplit, useF16P ? "attnPF16" : "attnP", immP);
    const useF16C = this.usingF16() && this.pipes.attnCF16;
    const pipeC = useF16C ? this.pipes.attnCF16 : this.pipes.attnC;
    const bgC = this._bg(pipeC, [
      S.pm,
      S.pz,
      S.po,
      oBuf
    ]);
    const immC = new Uint32Array([c.numHeads, c.headDim, nsplit, 0]);
    this._dispatch(enc, pipeC, bgC, c.numHeads, 1, useF16C ? "attnCF16" : "attnC", immC);
  }
  // Decode one token at absolute position `pos`. Writes logits to s.logits. Returns nothing.
  step(enc, tokenId, pos) {
    const c = this.cfg, S = this.s, hd = c.headDim, kvd = c.numKVHeads * hd;
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
          const hasQkvLora = this.lora && (this.lora.modules[L.q.loraKey] || this.lora.modules[L.k.loraKey] || this.lora.modules[L.v.loraKey]);
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
    const bg = this._bgCached(pipe, [aBuf, yBuf], `add:${n}${useF16 ? ":f16" : ""}`);
    const wg = pipe.__wg || 256;
    this._dispatch(enc, pipe, bg, Math.min(Math.ceil(n / wg), 65535), 1, useF16 ? "addF16" : "add", imm);
  }
  _siluMul(enc, gateBuf, upBuf, n) {
    const imm = new Uint32Array([n]);
    const useF16 = this.usingF16() && this.pipes.siluF16;
    const pipe = useF16 ? this.pipes.siluF16 : this.pipes.silu;
    const bg = this._bgCached(pipe, [gateBuf, upBuf], `silu:${n}${useF16 ? ":f16" : ""}`);
    const wg = pipe.__wg || 256;
    this._dispatch(enc, pipe, bg, Math.min(Math.ceil(n / wg), 65535), 1, useF16 ? "siluF16" : "silu", imm);
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
      "embed",
      imm
    );
  }
  async argmaxLogits() {
    if (this._argmaxReadBusy)
      throw new Error("argmaxLogits() is already in flight; concurrent generation is not supported");
    this._argmaxReadBusy = true;
    const enc = this.dev.createCommandEncoder();
    const n = this.cfg.vocabSize || 0;
    this._dispatch(
      enc,
      this.pipes.argmax,
      this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax], "argmax"),
      1,
      1,
      "argmax",
      new Uint32Array([n])
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
    if (this._topKReadBusy) throw new Error("topKLogits() is already in flight; concurrent sampling is not supported");
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
          "topk",
          imm
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
      if (this.sampleIdsRead.mapState !== "unmapped") this.sampleIdsRead.unmap();
      if (this.sampleValsRead.mapState !== "unmapped") this.sampleValsRead.unmap();
      this._topKReadBusy = false;
    }
  }
  // Phase 5: GPU-resident sampling (pure-GPU top-k + sample chaining).
  // Runs the iterative top-k selection dispatches directly into the GPU sampleIds/sampleVals
  // buffers, then immediately chains the SAMPLE_TOPK kernel in the same submission.
  // Only a single u32 (the chosen token) is ever read back from the GPU.
  // This eliminates the previous k-value readbacks for the sampling path.
  async sampleToken(temp = 1, r = typeof Math !== "undefined" ? Math.random() : 0.5) {
    if (this._topKReadBusy) throw new Error("sampleToken: top-k selection already in flight");
    this._topKReadBusy = true;
    const k = Math.min(this.samplingTopK, this.maxSamplingTopK, this.cfg.vocabSize);
    try {
      const enc = this.dev.createCommandEncoder();
      for (let i = 0; i < k; i++) {
        const imm2 = new Uint32Array([this.cfg.vocabSize, i]);
        this._dispatch(
          enc,
          this.pipes.topkSelect,
          this._bgCached(this.pipes.topkSelect, [this.s.logits, this.s.sampleIds, this.s.sampleVals], `topk:${i}`),
          1,
          1,
          "topk",
          imm2
        );
      }
      const bg = this._bg(this.pipes.sampleTopK, [
        this.s.sampleIds,
        this.s.sampleVals,
        this.s.sampled
      ]);
      const imm = new Uint32Array(4);
      imm[0] = k;
      const f322 = new Float32Array(imm.buffer);
      f322[2] = temp > 0 ? temp : 1;
      f322[3] = Math.max(0, Math.min(1, r));
      this._dispatch(enc, this.pipes.sampleTopK, bg, 1, 1, "sampleTopK", imm);
      enc.copyBufferToBuffer(this.s.sampled, 0, this.sampledRead, 0, 4);
      this.dev.queue.submit([enc.finish()]);
      if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
      await this.sampledRead.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(this.sampledRead.getMappedRange())[0];
      this.sampledRead.unmap();
      return id;
    } finally {
      this._topKReadBusy = false;
    }
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
      this._bgCached(this.pipes.embedBuf, [e.w, e.scale, this.s.hidden, this.s.amax], "embedBuf"),
      Math.ceil(this.cfg.hiddenSize / 256),
      1,
      "embed",
      imm
    );
  }
  // argmax(logits) -> s.amax, within the given encoder (no submit/readback)
  argmaxInto(enc) {
    const n = this.cfg.vocabSize || 0;
    this._dispatch(
      enc,
      this.pipes.argmax,
      this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax], "argmax"),
      1,
      1,
      "argmax",
      new Uint32Array([n])
    );
  }
  // GPU-resident batched GREEDY decode only: chains embed->step->argmax for K
  // tokens in ONE submit, reads back K ids once, and checks stop tokens only
  // after readback. It assumes s.amax already holds the current token id to
  // embed. Do not use for sampled decoding; sampled tokens must be written by
  // the CPU/GPU sampler one step at a time.
  async decodeBatch(startPos, K) {
    K = Math.min(K, this.decodeBatchCapacity, this.maxCtx - startPos);
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
    this._dispatch(enc, this.pipes.gemm4, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  gemm4AddT(enc, aBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4AddT, [aBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4AddT, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4AddT", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  loraBatchDelta(enc, xBuf, yBuf, q, T, mod, moduleKey) {
    if (this.debugCapture) console.log("VWG loraBatchDelta: " + moduleKey + " mod=" + !!mod);
    const imm = new Uint32Array([q.K, mod.rank, T, 0]);
    const bgA = this._bg(this.pipes.loraABatch, [xBuf, mod.A, this.sT.loraD]);
    this._dispatch(enc, this.pipes.loraABatch, bgA, mod.rank, T, "loraA:T", imm);
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj") {
      enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xBat, 0, T * q.K * 4);
      enc.copyBufferToBuffer(this.sT.loraD, 0, this.debugBufs.dBat, 0, T * mod.rank * 4);
    }
    const totalGroups = Math.ceil(T * q.N / 256);
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
    this._dispatch(enc, this.pipes.loraBAddT, bgB, gx, gy, "loraB:T", new Uint8Array(meta));
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj") {
      enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.yBat, 0, T * q.N * 4);
      this.debugCaptured = true;
    }
  }
  rmsT(enc, xBuf, gBuf, yBuf, T, K) {
    const imm = new Float32Array([K, this.cfg.rmsNormEps]);
    const useF16 = this.usingF16() && this.pipes.rmsTF16;
    const pipe = useF16 ? this.pipes.rmsTF16 : this.pipes.rmsT;
    this._dispatch(enc, pipe, this._bg(pipe, [xBuf, gBuf, yBuf]), T, 1, useF16 ? "rmsTF16" : "rmsT", imm);
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
      Math.ceil(T * nHeads * (hd / 2) / 256),
      1,
      useF16 ? "ropeTF16" : "ropeT",
      imm
    );
  }
  attnPrefill(enc, qBuf, kc, vc, oBuf, T, qStart = 0, ctx = T) {
    const c = this.cfg;
    if (this.features.prefillAttention === "block" || qStart !== 0 || ctx !== T) {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, qStart, ctx, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillBlock,
        this._bg(this.pipes.attnPrefillBlock, [qBuf, kc, vc, oBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        "attnPrefillBlock",
        imm
      );
    } else {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T]);
      this._dispatch(
        enc,
        this.pipes.attnPrefill,
        this._bg(this.pipes.attnPrefill, [qBuf, kc, vc, oBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        "attnPrefill",
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
        `prefill scratch ${Math.ceil(need / 1048576)}MiB exceeds maxPrefillScratchBytes; lower maxPrefillT or use shorter prompt chunks`
      );
    }
    if (this.sT) for (const k in this.sT) this.sT[k].destroy();
    const c = this.cfg, H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
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
      scale_x: this._buf(T * Math.max(H, I) / 128 * 4)
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
    const c = this.cfg, S = this.s, T = ids.length, hd = c.headDim, kvd = c.numKVHeads * hd, H = c.hiddenSize;
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
      Math.min(Math.ceil(T * H / 256), 65535),
      1,
      "embedT",
      imm
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
          L.q.loraKey
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
          L.k.loraKey
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
          L.v.loraKey
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
            L.down.loraKey
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
    const c = this.cfg, S = this.s, H = c.hiddenSize, hd = c.headDim, kvd = c.numKVHeads * hd;
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
        Math.min(Math.ceil(CT * H / 256), 65535),
        1,
        "embedT",
        new Uint32Array([CT, H, off, 0])
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
            L.q.loraKey
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
            L.k.loraKey
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
            L.v.loraKey
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
              L.down.loraKey
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
              L.down.loraKey
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
      const c = this.cfg, H = c.hiddenSize, kvd = c.numKVHeads * c.headDim;
      this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(draftCandidates));
      const enc = this.dev.createCommandEncoder();
      const e = this.q[this.plan.embed.name];
      const embedUni = new Uint32Array([T, H, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.embedT,
        this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
        Math.min(Math.ceil(T * H / 256), 65535),
        1,
        "embedT",
        embedUni
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
            L.q.loraKey
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
            L.k.loraKey
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
            L.v.loraKey
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
              L.down.loraKey
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
    const temp = opts.temp != null && opts.temp > 0 ? opts.temp : 1;
    await this.prefillBatch(promptIds);
    const generatedIds = [];
    let pos = promptIds.length;
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
      yBat: this._buf(T * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
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
      bufs.yBat.mapAsync(GPUMapMode.READ)
    ]);
    const res = {
      xSeq: new Float32Array(bufs.xSeq.getMappedRange()).slice(),
      dSeq: new Float32Array(bufs.dSeq.getMappedRange()).slice(),
      ySeq: new Float32Array(bufs.ySeq.getMappedRange()).slice(),
      xBat: new Float32Array(bufs.xBat.getMappedRange()).slice(),
      dBat: new Float32Array(bufs.dBat.getMappedRange()).slice(),
      yBat: new Float32Array(bufs.yBat.getMappedRange()).slice()
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
};
var PagedAttentionManager = class {
  static {
    __name(this, "PagedAttentionManager");
  }
  constructor(maxCtx, pageSize = 16) {
    this.pageSize = pageSize;
    this.maxCtx = maxCtx;
    this.maxBlocksPerSeq = Math.ceil(maxCtx / pageSize);
    this.freeBlocks = [];
    this.seqBlocks = /* @__PURE__ */ new Map();
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
        const newBlock = blocks.length + 1e3;
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
};

// src/services/device_service.js
async function initWebGPUDevice({ log: log2 = /* @__PURE__ */ __name(() => {
}, "log") } = {}) {
  log2("requesting WebGPU device\u2026");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter (use a WebGPU-capable browser)");
  if (!navigator.gpu.wgslLanguageFeatures?.has("immediate_address_space"))
    throw new Error("WGSL immediate_address_space is not available (upgrade to Chrome 149+)");
  if (!adapter.features.has("subgroups"))
    throw new Error(
      'GPU lacks the required "subgroups" feature. The current fast WGSL kernels require subgroups and no fallback kernel set is bundled.'
    );
  const hasSubgroupId = !!navigator.gpu.wgslLanguageFeatures?.has("subgroup_id");
  const hasLinearIndexing = !!navigator.gpu.wgslLanguageFeatures?.has("linear_indexing");
  const hasF16 = adapter.features.has("shader-f16");
  const hasTimestamp = adapter.features.has("timestamp-query");
  const reqFeatures = ["subgroups"];
  if (adapter.features.has("shader-f16")) reqFeatures.push("shader-f16");
  if (hasTimestamp) reqFeatures.push("timestamp-query");
  const dev = await adapter.requestDevice({
    requiredFeatures: reqFeatures,
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
    }
  });
  dev.addEventListener?.("uncapturederror", (e) => console.error("GPUERR", e.error.message));
  log2(`WebGPU ready. maxBuffer=${(Number(adapter.limits.maxBufferSize) / 1e9).toFixed(2)}GB subgroupId=${hasSubgroupId} linearIdx=${hasLinearIndexing} f16=${hasF16} tsQuery=${hasTimestamp}`);
  return dev;
}
__name(initWebGPUDevice, "initWebGPUDevice");

// src/services/prompt_formatter.js
function chatML(messages) {
  let s = messages[0]?.role === "system" ? "" : "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  for (const m of messages) s += `<|im_start|>${m.role}
${m.content}<|im_end|>
`;
  return s + "<|im_start|>assistant\n";
}
__name(chatML, "chatML");
function formatMessages(tokenizer, messages) {
  try {
    return tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
  } catch {
    return chatML(messages);
  }
}
__name(formatMessages, "formatMessages");

// src/services/model_session.js
async function buildTokenizer(reader) {
  const tj = JSON.parse(await reader.text("tokenizer.json"));
  const tc = JSON.parse(await reader.text("tokenizer_config.json"));
  const { PreTrainedTokenizer } = await import("@huggingface/transformers");
  return new PreTrainedTokenizer(tj, tc);
}
__name(buildTokenizer, "buildTokenizer");
function randomUnit() {
  if (globalThis.crypto?.getRandomValues) {
    const u = new Uint32Array(1);
    globalThis.crypto.getRandomValues(u);
    return u[0] / 4294967296;
  }
  return Math.random();
}
__name(randomUnit, "randomUnit");
function sampleTopK(candidates, { temperature, topP = 1 }) {
  if (!temperature || temperature <= 0) return candidates[0]?.id ?? 0;
  const best = candidates[0]?.logit ?? 0;
  const weighted = candidates.map((c2) => ({ id: c2.id, w: Math.exp((c2.logit - best) / temperature) }));
  let sum = weighted.reduce((a, c2) => a + c2.w, 0);
  if (topP > 0 && topP < 1 && weighted.length > 1 && sum > 0) {
    let csum = 0, keep = 0;
    for (; keep < weighted.length; keep++) {
      csum += weighted[keep].w / sum;
      if (csum >= topP) {
        keep++;
        break;
      }
    }
    weighted.length = Math.max(1, keep);
    sum = weighted.reduce((a, c2) => a + c2.w, 0);
  }
  let r = randomUnit() * sum, c = 0;
  for (const item of weighted) {
    c += item.w;
    if (r <= c) return item.id;
  }
  return weighted[weighted.length - 1]?.id ?? candidates[0]?.id ?? 0;
}
__name(sampleTopK, "sampleTopK");
var ModelSession = class {
  static {
    __name(this, "ModelSession");
  }
  constructor({ cfg = QWEN25_3B, log: log2 = /* @__PURE__ */ __name(() => {
  }, "log"), runtimeOptions = {} } = {}) {
    this.cfg = cfg;
    this.log = log2;
    this.runtimeOptions = { decodeBatchSize: "auto", samplingTopK: 40, ...runtimeOptions };
    this.dev = null;
    this.rt = null;
    this.tokenizer = null;
  }
  async loadWith(reader, label) {
    this.dev = await initWebGPUDevice({ log: this.log });
    this.log(`loading tokenizer from ${label}\u2026`);
    this.tokenizer = await buildTokenizer(reader);
    this.log(`tokenizer loaded. streaming + quantizing weights (int4) from ${label}\u2026`);
    const t0 = performance.now();
    this.rt = new QwenWGPU(this.dev, this.cfg, this.runtimeOptions);
    await this.rt.build(reader, (msg, frac) => this.log(`weights: ${msg} ${(frac * 100).toFixed(0)}%`));
    window.__rt = this.rt;
    window.__tokenizer = this.tokenizer;
    const tuning = this.rt.decodeBatchTuning;
    const tuned = tuning ? ` decodeBatch=${tuning.selected} (${tuning.reason})` : "";
    this.log(
      `READY in ${((performance.now() - t0) / 1e3).toFixed(1)}s \u2014 base loaded once; adapters hot-swap live.${tuned}`
    );
    return this;
  }
  async readLogits() {
    const n = this.cfg.vocabSize;
    const rb = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.rt.s.logits, 0, rb, 0, n * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice();
    rb.unmap();
    rb.destroy();
    return a;
  }
  async sampleNextToken({ temperature, topK = this.rt.samplingTopK, topP = 1 } = {}) {
    return sampleTopK(await this.rt.topKLogits(topK), { temperature, topP });
  }
  async *generate(messages, { maxTokens = 1024, temperature = 0, topK, topP = 1, stopIds = [151645, 151643] } = {}) {
    const rt = this.rt, tokenizer = this.tokenizer;
    const ids = tokenizer.encode(formatMessages(tokenizer, messages));
    if (ids.length <= rt.maxPrefillT) rt.prefillBatch(ids);
    else for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    let pos = ids.length;
    const emit = /* @__PURE__ */ __name((id) => tokenizer.decode([id], { skip_special_tokens: true }), "emit");
    if (temperature > 0) {
      let next = await this.sampleNextToken({ temperature, topK, topP });
      for (let step = 0; step < maxTokens; step++) {
        if (stopIds.includes(next)) break;
        const d = emit(next);
        if (d) yield d;
        rt.token(next, pos);
        pos++;
        next = await this.sampleNextToken({ temperature, topK, topP });
      }
      return;
    }
    const first = await rt.argmaxLogits();
    if (stopIds.includes(first)) return;
    {
      const d = emit(first);
      if (d) yield d;
    }
    let emitted = 1;
    while (emitted < maxTokens && pos < rt.maxCtx) {
      const K = rt.greedyBatchSizeFor({ emitted, remaining: maxTokens - emitted, pos });
      const batch = await rt.decodeGreedyBatch(pos, K);
      pos += batch.length;
      let stop = false;
      for (const id of batch) {
        if (stopIds.includes(id)) {
          stop = true;
          break;
        }
        const d = emit(id);
        if (d) yield d;
        emitted++;
        if (emitted >= maxTokens) {
          stop = true;
          break;
        }
      }
      if (stop) break;
    }
  }
};

// src/main.js
var $ = /* @__PURE__ */ __name((id) => document.getElementById(id), "$");
var log = /* @__PURE__ */ __name((m) => {
  const s = $("status");
  if (s) s.textContent = m;
  console.log("[harness]", m);
}, "log");
var SYS = `You are a senior bug bounty triage analyst. Read the submission and assign exactly ONE disposition from: valid_impactful, valid_low, corroborated_surge, likely_duplicate, out_of_scope, theoretical_no_poc, self_inflicted, accepted_risk, slop. Estimate severity_estimate (critical/high/medium/low/none). Think step by step, then output a SINGLE JSON object on the last line with keys: disposition, severity_estimate, is_duplicate_risk, reasoning, questions_for_researcher, confidence. Output only valid JSON for that object.`;
var session = new ModelSession({ cfg: QWEN25_3B, log });
var adapters = new AdapterRegistry();
var generation = new GenerationController({ session, adapters, systemPrompt: SYS, log });
async function loadWith(reader, label) {
  await session.loadWith(reader, label);
  $("go").disabled = false;
  $("loraFile").disabled = false;
}
__name(loadWith, "loadWith");
function addAdapterOption(name, modules, where) {
  adapters.add(name, modules);
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = `${name} (${Object.keys(modules).length} modules${where ? ", " + where : ""})`;
  $("adapter").appendChild(opt);
  $("adapter").value = name;
  log(`LoRA "${name}" loaded (${Object.keys(modules).length} modules) \u2014 Triage to hot-swap.`);
}
__name(addAdapterOption, "addAdapterOption");
async function runTriage() {
  $("go").disabled = true;
  try {
    await generation.runTriage({
      adapterName: $("adapter").value,
      report: $("report").value,
      outputNode: $("out")
    });
  } finally {
    $("go").disabled = false;
  }
}
__name(runTriage, "runTriage");
async function fetchHfAdapterFiles(repo, token) {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const grab = /* @__PURE__ */ __name(async (n) => {
    const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${n}`, { headers });
    if (!r.ok) return null;
    const buf = await r.arrayBuffer();
    return {
      name: n,
      async text() {
        return new TextDecoder().decode(buf);
      },
      async arrayBuffer() {
        return buf;
      }
    };
  }, "grab");
  const st = await grab("adapters.safetensors") || await grab("adapter_model.safetensors");
  if (!st) throw new Error("no adapters.safetensors / adapter_model.safetensors in " + repo);
  const cfg = await grab("adapter_config.json");
  return cfg ? [st, cfg] : [st];
}
__name(fetchHfAdapterFiles, "fetchHfAdapterFiles");
window.addEventListener("DOMContentLoaded", () => {
  $("load").onclick = () => loadWith(urlReader($("modelUrl").value.trim()), $("modelUrl").value.trim()).catch((e) => {
    log("ERROR: " + e.message);
    console.error(e);
  });
  const hfBtn = $("loadHF");
  if (hfBtn)
    hfBtn.onclick = () => {
      const repo = $("hfRepo").value.trim();
      const token = ($("hfToken")?.value || "").trim();
      if (!repo) return log("enter a Hugging Face repo id, e.g. WeiboAI/VibeThinker-3B");
      loadWith(hfReader(repo, token), "HF: " + repo).catch((e) => {
        log("ERROR: " + e.message + " (private/gated repo? add a token)");
        console.error(e);
      });
    };
  const mf = $("modelFiles");
  if (mf)
    mf.onchange = (ev) => {
      const files = [...ev.target.files];
      if (!files.length) return;
      const map = {};
      for (const f of files) map[f.name] = f;
      loadWith(fileReader(map), `${files.length} local files`).catch((e) => {
        log("ERROR: " + e.message);
        console.error(e);
      });
    };
  $("go").onclick = () => runTriage().catch((e) => {
    log("ERROR: " + e.message);
    console.error(e);
  });
  $("loraFile").onchange = async (ev) => {
    try {
      const { name, modules } = await loadLoraAdapterGPU(session.dev, [...ev.target.files], QWEN25_3B);
      addAdapterOption(name, modules);
    } catch (e) {
      log("LoRA load error: " + e.message);
      console.error(e);
    }
  };
  const hfLoraBtn = $("loadHFLora");
  if (hfLoraBtn)
    hfLoraBtn.onclick = async () => {
      if (!session.dev) return log("load a model first, then load a LoRA adapter");
      const repo = ($("hfLora")?.value || "").trim();
      const token = ($("hfToken")?.value || "").trim();
      if (!repo) return log("enter a Hugging Face LoRA adapter repo id");
      try {
        const files = await fetchHfAdapterFiles(repo, token);
        const { name, modules } = await loadLoraAdapterGPU(session.dev, files, QWEN25_3B);
        addAdapterOption(repo.split("/").pop() || name, modules, "HF");
      } catch (e) {
        log("HF LoRA error: " + e.message + (token ? "" : " (private/gated? add a token)"));
        console.error(e);
      }
    };
});
