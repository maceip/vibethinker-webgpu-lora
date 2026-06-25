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

// Streaming safetensors loader for the WebGPU runtime. It preserves the reader
// contract ({ range, text }) but avoids building a repository-wide Float32Array
// map: each requested tensor is range-fetched, decoded, visited, then released.
import { urlReader } from '../readers.js';

function decodeBf16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < numel; i++) o32[i] = u16[i] << 16;
  return out;
}

function decodeF16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  for (let i = 0; i < numel; i++) {
    const h = u16[i],
      s = (h & 0x8000) >> 15,
      e = (h & 0x7c00) >> 10,
      f = h & 0x03ff;
    if (e === 0) out[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    else if (e === 0x1f) out[i] = f ? NaN : s ? -Infinity : Infinity;
    else out[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

function decodeF32(u8, numel) {
  return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + numel * 4));
}

const DECODERS = {
  BF16: decodeBf16ToF32,
  F16: decodeF16ToF32,
  FP16: decodeF16ToF32,
  F32: decodeF32,
  FP32: decodeF32,
};

async function loadIndex(reader) {
  try {
    const idx = JSON.parse(await reader.text('model.safetensors.index.json'));
    return { weightMap: idx.weight_map || {}, shards: [...new Set(Object.values(idx.weight_map || {}))] };
  } catch {
    return { weightMap: null, shards: ['model.safetensors'] };
  }
}

function shardPlan(shards, weightMap, names) {
  if (!weightMap || !names) return new Map(shards.map((shard) => [shard, null]));
  const plan = new Map();
  for (const name of names) {
    const shard = weightMap[name];
    if (!shard) continue;
    if (!plan.has(shard)) plan.set(shard, new Set());
    plan.get(shard).add(name);
  }
  return plan;
}

/**
 * Stream requested tensors from a safetensors model.
 *
 * @param {string|{range:function,text:function}} source
 * @param {{names?:Set<string>, onTensor:function, onProgress?:function}} opts
 */
export async function streamSafetensors(source, { names = null, onTensor, onProgress = () => {} } = {}) {
  if (!onTensor) throw new Error('streamSafetensors requires onTensor');

  /*
   * TECHNIQUE: Streaming Range + visit + release (no full model in memory)
   *   Each tensor is fetched by exact byte range, decoded, passed to onTensor
   *   (which uploads to GPU and drops the JS array), then the source buffer
   *   can be GC'd. Essential for large models in the browser.
   */
  const reader = typeof source === 'string' ? urlReader(source) : source;
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
    const allNames = Object.keys(header).filter((k) => k !== '__metadata__');
    const tensorNames = wantedInShard
      ? allNames.filter((n) => wantedInShard.has(n))
      : names
        ? allNames.filter((n) => names.has(n))
        : allNames;

    for (const name of tensorNames) {
      const t = header[name];
      if (!t) continue;
      const dtype = String(t.dtype || '').toUpperCase();
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
