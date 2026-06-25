/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

// Export a trained LoRA adapter (QwenLoraTrainer state) back into the PEFT
// on-disk layout: a .safetensors blob + adapter_config.json. This is the inverse
// of src/lora_gpu.js's loader.
//
// Layout conversion (trainer -> PEFT):
//   trainer A is [rank][in]  (transposed for loraABatch)  -> PEFT lora_A.weight [rank][in] (same)
//   trainer B is [rank][out]                              -> PEFT lora_B.weight [out][rank] (transpose)

async function readBufferF32(dev, src, byteLen) {
  const rb = dev.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, byteLen);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}

// [rows][cols] row-major -> [cols][rows] row-major
function transpose2d(arr, rows, cols) {
  const o = new Float32Array(arr.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = arr[r * cols + c];
  return o;
}

/*
 * TECHNIQUE: Minimal safetensors writer
 *   Header is a JSON map of name -> {dtype, shape, data_offsets:[start,end]} with a
 *   leading u64 length. All tensors are F32 and concatenated after the header. The
 *   header is padded with spaces to an 8-byte boundary (HF reference behavior).
 */
export function buildSafetensors(tensors, metadata = { format: 'pt' }) {
  let offset = 0;
  const header = {};
  if (metadata) header.__metadata__ = metadata;
  for (const t of tensors) {
    const bytes = t.data.byteLength;
    header[t.name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  let headerStr = JSON.stringify(header);
  const enc = new TextEncoder();
  let headerBytes = enc.encode(headerStr);
  const pad = (8 - (headerBytes.length % 8)) % 8;
  if (pad) {
    headerStr += ' '.repeat(pad);
    headerBytes = enc.encode(headerStr);
  }
  const total = 8 + headerBytes.length + offset;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setBigUint64(0, BigInt(headerBytes.length), true);
  new Uint8Array(buf, 8, headerBytes.length).set(headerBytes);
  let p = 8 + headerBytes.length;
  for (const t of tensors) {
    new Uint8Array(buf, p, t.data.byteLength).set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    p += t.data.byteLength;
  }
  return new Uint8Array(buf);
}

// Read back A/B for every trained module and produce { safetensors, configJson, config }.
export async function exportLoraAdapter(trainer, opts = {}) {
  const rt = trainer.rt;
  const dev = rt.dev;
  const tensors = [];
  const targets = new Set();
  const rankByKey = {}; // peft module path -> rank (for rank_pattern when mixed)
  const alphaByKey = {}; // peft module path -> alpha
  for (const key of trainer.trainedKeys) {
    const st = trainer.state[key];
    const A = await readBufferF32(dev, st.mod.A, st.rank * st.K * 4); // [rank][K]
    const B = await readBufferF32(dev, st.mod.B, st.rank * st.N * 4); // [rank][N]
    const Bt = transpose2d(B, st.rank, st.N); // -> [N][rank]
    const base = `base_model.model.model.${key}`;
    tensors.push({ name: `${base}.lora_A.weight`, shape: [st.rank, st.K], data: A });
    tensors.push({ name: `${base}.lora_B.weight`, shape: [st.N, st.rank], data: Bt });
    rankByKey[key] = st.rank;
    alphaByKey[key] = st.scale * st.rank;
    targets.add(key.split('.').pop()); // q_proj, k_proj, ...
  }
  const safetensors = buildSafetensors(tensors);

  // Pick the modal rank/alpha as the top-level value; record any deviations in
  // rank_pattern/alpha_pattern (PEFT honors these) so mixed-rank adapters are
  // described accurately instead of silently inheriting the last module's values.
  const ranks = Object.values(rankByKey);
  const alphas = Object.values(alphaByKey);
  const r = opts.rank ?? mode(ranks) ?? 0;
  const alpha = opts.alpha ?? mode(alphas) ?? 0;
  const rankPattern = {};
  const alphaPattern = {};
  for (const key of Object.keys(rankByKey)) {
    if (rankByKey[key] !== r) rankPattern[key] = rankByKey[key];
    if (alphaByKey[key] !== alpha) alphaPattern[key] = alphaByKey[key];
  }
  const config = {
    peft_type: 'LORA',
    auto_mapping: null,
    base_model_name_or_path: opts.baseModel || 'WeiboAI/VibeThinker-3B',
    r,
    lora_alpha: alpha,
    target_modules: [...targets],
    lora_dropout: 0.0,
    bias: 'none',
    fan_in_fan_out: false,
    inference_mode: true,
    task_type: 'CAUSAL_LM',
    ...(Object.keys(rankPattern).length ? { rank_pattern: rankPattern } : {}),
    ...(Object.keys(alphaPattern).length ? { alpha_pattern: alphaPattern } : {}),
  };
  const configJson = JSON.stringify(config, null, 2);
  return { safetensors, config, configJson };
}

// most frequent value (ties -> first seen); undefined for empty input
function mode(arr) {
  if (!arr.length) return undefined;
  const counts = new Map();
  let best = arr[0],
    bestN = 0;
  for (const v of arr) {
    const n = (counts.get(v) || 0) + 1;
    counts.set(v, n);
    if (n > bestN) {
      bestN = n;
      best = v;
    }
  }
  return best;
}

// Browser convenience: trigger downloads of adapter_model.safetensors + adapter_config.json.
export async function downloadLoraAdapter(trainer, opts = {}) {
  const { safetensors, configJson } = await exportLoraAdapter(trainer, opts);
  const stem = opts.name || trainer.adapter?.name || 'adapter';
  triggerDownload(new Blob([safetensors], { type: 'application/octet-stream' }), `${stem}.safetensors`);
  triggerDownload(new Blob([configJson], { type: 'application/json' }), 'adapter_config.json');
}

function triggerDownload(blob, filename) {
  if (typeof document === 'undefined') return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
