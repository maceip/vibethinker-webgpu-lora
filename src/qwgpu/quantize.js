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

// Per-output-channel symmetric int8 quantization of a [out, in] weight matrix.
// W[o, :] is quantized with one scale per output row: q = round(w / scale),
// scale = max(|W[o,:]|) / 127. Dequant: w ≈ q * scale.
// Stored row-major [out][in] as int8 (so a GEMV thread reads one output row
// contiguously) — packed 4 int8 per u32 for WGSL (no i8 type).

/*
 * TECHNIQUE: Per-row symmetric int8 + 4-packed u32 storage
 *   One scale per output channel gives good quality with 4x smaller weights.
 *   Packing 4 int8 per u32 lets the WGSL kernel do one load + unpack4xI8 per 4
 *   elements, maximizing memory bandwidth utilization in the GEMV inner loop.
 */
export function quantizeInt8RowMajor(f32, outDim, inDim) {
  // f32 is HF layout [out, in] row-major (out rows, each in long).
  const scale = new Float32Array(outDim);
  const q = new Int8Array(outDim * inDim);
  for (let o = 0; o < outDim; o++) {
    const base = o * inDim;
    let amax = 0;
    for (let i = 0; i < inDim; i++) {
      const a = Math.abs(f32[base + i]);
      if (a > amax) amax = a;
    }
    const s = amax > 0 ? amax / 127 : 1;
    scale[o] = s;
    const inv = 1 / s;
    for (let i = 0; i < inDim; i++) {
      let v = Math.round(f32[base + i] * inv);
      if (v > 127) v = 127;
      else if (v < -128) v = -128;
      q[base + i] = v;
    }
  }
  // pack int8 -> u32 (4 per word), row-major. inDim assumed multiple of 4 (2048/11008? 11008%4=0, 2048%4=0).
  const packed = new Uint32Array((outDim * inDim) / 4);
  const u8 = new Uint8Array(q.buffer);
  for (let w = 0; w < packed.length; w++) {
    packed[w] = u8[w * 4] | (u8[w * 4 + 1] << 8) | (u8[w * 4 + 2] << 16) | (u8[w * 4 + 3] << 24);
  }
  return { packed, scale, outDim, inDim };
}

// Max relative error of int8 round-trip vs original (quality sanity check).
export function quantError(f32, outDim, inDim) {
  const { packed, scale } = quantizeInt8RowMajor(f32, outDim, inDim);
  const i8 = new Int8Array(new Uint8Array(packed.buffer));
  let maxAbs = 0,
    maxRel = 0,
    denom = 0,
    num = 0;
  for (let o = 0; o < outDim; o++)
    for (let i = 0; i < inDim; i++) {
      const idx = o * inDim + i;
      const deq = i8[idx] * scale[o];
      const err = Math.abs(deq - f32[idx]);
      maxAbs = Math.max(maxAbs, err);
      num += err * err;
      denom += f32[idx] * f32[idx];
    }
  return { maxAbs, rms: Math.sqrt(num / denom) };
}

// Per-output-channel, per-group int4 (group=128). Symmetric: q in [-8,7].
// Returns { packed:Uint32Array (8 nibbles/word, row-major [out][in]),
//           scale:Float32Array [out*(in/group)] }.
export function quantizeInt4Group(f32, outDim, inDim, group = 128) {
  const groupsPerRow = inDim / group;
  const scale = new Float32Array(outDim * groupsPerRow);
  const q = new Int8Array(outDim * inDim); // nibble values -8..7 stored as int8 temporarily
  for (let o = 0; o < outDim; o++) {
    for (let g = 0; g < groupsPerRow; g++) {
      const base = o * inDim + g * group;
      let amax = 0;
      for (let i = 0; i < group; i++) {
        const a = Math.abs(f32[base + i]);
        if (a > amax) amax = a;
      }
      const s = amax > 0 ? amax / 7 : 1;
      scale[o * groupsPerRow + g] = s;
      const inv = 1 / s;
      for (let i = 0; i < group; i++) {
        let v = Math.round(f32[base + i] * inv);
        if (v > 7) v = 7;
        else if (v < -8) v = -8;
        q[base + i] = v;
      }
    }
  }
  const packed = new Uint32Array((outDim * inDim) / 8);
  for (let w = 0; w < packed.length; w++) {
    let acc = 0;
    for (let j = 0; j < 8; j++) acc |= (q[w * 8 + j] & 0xf) << (j * 4);
    packed[w] = acc >>> 0;
  }
  return { packed, scale, groupsPerRow };
}
