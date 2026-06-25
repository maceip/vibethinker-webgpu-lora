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

// tf-free LoRA adapter loader for the custom WebGPU runtime. Parses a PEFT/MLX
// adapter (.safetensors + adapter_config.json) and uploads each module's A/B as
// GPUBuffers in the layout the GEMV kernels expect:
//   A -> [rank][in]  (transposed so loraA reads each rank-row contiguously)
//   B -> [rank][out] (GEMV reads loraB[r*N + n])
// Returns { name, modules: { "layers.I.sub.proj": {A,B,rank,scale} } } for setLora().
import { moduleKeyFromTensorName } from './qwgpu/model_schema.js';

function parseSt(buf) {
  const dv = new DataView(buf);
  const hl = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  return { header, dataStart: 8 + hl, u8: new Uint8Array(buf) };
}

/*
 * TECHNIQUE: Layout-matched LoRA upload for direct GEMV consumption
 *   A is stored transposed [rank][in] so the LORA_A kernel can read contiguous rows.
 *   B is [rank][out] so the main GEMV can do a simple indexed dot-product add.
 *   This avoids extra transposes on the GPU and keeps the hot path simple.
 */
function bf16f32(u8, off, n) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset + off, n);
  const o = new Float32Array(n);
  const o32 = new Uint32Array(o.buffer);
  for (let i = 0; i < n; i++) o32[i] = u16[i] << 16;
  return o;
}
function f32(u8, off, n) {
  return new Float32Array(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + n * 4));
}
function readTensor(st, name) {
  const t = st.header[name];
  const n = t.shape.reduce((a, b) => a * b, 1);
  const dt = t.dtype.toUpperCase();
  const arr =
    dt === 'BF16'
      ? bf16f32(st.u8, st.dataStart + t.data_offsets[0], n)
      : f32(st.u8, st.dataStart + t.data_offsets[0], n);
  return { arr, shape: t.shape };
}
const isA = (name) => /lora_a/i.test(name);
// [rows][cols] row-major -> [cols][rows] row-major
function transpose2d(arr, rows, cols) {
  const o = new Float32Array(arr.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = arr[r * cols + c];
  return o;
}

export async function loadLoraAdapterGPU(dev, files, cfg) {
  const stFile = files.find((f) => f.name.endsWith('.safetensors'));
  if (!stFile) throw new Error('no .safetensors in adapter files');
  const cfgFile = files.find((f) => /adapter_config\.json|config\.json/.test(f.name));
  let rankCfg = 16,
    scaleCfg = null;
  if (cfgFile) {
    const c = JSON.parse(await cfgFile.text());
    const lp = c.lora_parameters || {};
    rankCfg = c.r ?? c.rank ?? c.lora_rank ?? lp.rank ?? rankCfg;
    if (lp.scale != null)
      scaleCfg = lp.scale; // MLX: scale is a direct multiplier
    else if (c.lora_alpha != null)
      scaleCfg = c.lora_alpha / rankCfg; // PEFT: scale = alpha / rank
    else if (c.alpha != null) scaleCfg = c.alpha / rankCfg;
  }

  const st = parseSt(await stFile.arrayBuffer());
  const names = Object.keys(st.header).filter((k) => k !== '__metadata__' && /lora_[abAB]/.test(k));
  const groups = {};
  for (const nm of names) {
    const key = moduleKeyFromTensorName(nm);
    if (!key) continue;
    (groups[key] ||= {})[isA(nm) ? 'A' : 'B'] = readTensor(st, nm);
  }

  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = (arr) => {
    const b = dev.createBuffer({ size: arr.byteLength, usage: S });
    dev.queue.writeBuffer(b, 0, arr);
    return b;
  };
  const modules = {};
  for (const key of Object.keys(groups)) {
    const g = groups[key];
    if (!g.A || !g.B) continue;
    const r = Math.min(...g.A.shape, ...g.B.shape); // rank = smallest dim
    // want A as [r, in]; PEFT lora_A is usually [r, in] already, else [in, r]
    let Aarr = g.A.arr;
    if (g.A.shape[0] !== r) Aarr = transpose2d(g.A.arr, g.A.shape[0], g.A.shape[1]);
    // want B as [r, out]; PEFT lora_B is usually [out, r], transpose to [r, out]
    let Barr = g.B.arr;
    if (g.B.shape[0] !== r) Barr = transpose2d(g.B.arr, g.B.shape[0], g.B.shape[1]);
    const scale = scaleCfg != null ? scaleCfg : 2.0;
    modules[key] = { A: mk(Aarr), B: mk(Barr), rawA: Aarr, rawB: Barr, rank: r, scale };
  }
  if (!Object.keys(modules).length) throw new Error('no LoRA modules matched layers.*.{self_attn,mlp}.*_proj');
  const name = stFile.name.replace(/\.safetensors$/, '');
  return { name, modules };
}
