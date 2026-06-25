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

// Legacy tf.js weight loader (kept for reference / older test paths).
// The primary engine now uses the pure-WebGPU path (safetensors_loader + runtime).

// Load HF safetensors (sharded, BF16) into tf.js tensors via per-tensor Range
// requests. A whole 5.4GB shard can't be a single ArrayBuffer (V8 ~4.29GB cap),
// so we read the header, then Range-fetch each tensor's bytes (≤1.24GB) one at a
// time, decode BF16->F32, upload to the backend, and drop the JS array.
import * as tf from '@tensorflow/tfjs-core';

/*
 * TECHNIQUE: Streaming per-tensor Range fetch + immediate decode + drop
 *   Avoids ever holding a full multi-GB model in a single JS ArrayBuffer.
 *   Each tensor is decoded and turned into a tf.Tensor, then the raw bytes
 *   can be GC'd. Critical for browser memory limits.
 */
function decodeBf16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < numel; i++) u32[i] = u16[i] << 16;
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

/**
 * @param reader  { range(path,start,end)->ArrayBuffer, text(path)->string }
 * @param onProgress (msg, frac)=>void
 */
export async function loadModelWeights(reader, onProgress = () => {}) {
  onProgress('reading index', 0);
  let shards;
  try {
    const idx = JSON.parse(await reader.text('model.safetensors.index.json'));
    shards = [...new Set(Object.values(idx.weight_map))];
  } catch {
    shards = ['model.safetensors'];
  }

  const weights = {};
  for (const shard of shards) {
    const lenBuf = await reader.range(shard, 0, 8);
    const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
    const hdrBuf = await reader.range(shard, 8, 8 + headerLen);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(hdrBuf)));
    const dataStart = 8 + headerLen;
    const names = Object.keys(header).filter((k) => k !== '__metadata__');
    let done = 0;
    for (const name of names) {
      const t = header[name];
      const dtype = t.dtype.toUpperCase();
      const dec = DECODERS[dtype];
      if (!dec) throw new Error(`unsupported dtype ${dtype} for ${name}`);
      const numel = t.shape.reduce((a, b) => a * b, 1);
      const [s, e] = t.data_offsets;
      const buf = await reader.range(shard, dataStart + s, dataStart + e);
      const f32 = dec(new Uint8Array(buf), numel);
      weights[name] = tf.tensor(f32, t.shape, 'float32');
      done++;
      if (done % 8 === 0) {
        onProgress(`${shard} ${done}/${names.length}`, done / names.length);
        await tf.nextFrame?.();
      }
    }
  }
  onProgress('weights loaded', 1);
  return weights;
}

// Readers live in readers.js (tf-free) so the app/runtime bundles don't pull tf.
export { urlReader, hfReader, fileReader } from './readers.js';
