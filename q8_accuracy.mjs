// Does int8 quantization preserve the merged model's output? Quantize every
// projection weight to int8 + dequant back to f32, run decode, compare gen_ids
// to the f32/HF reference. If they match (or nearly), int8 is safe for speed.
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import fs from 'fs';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights } from './weights.js';
import { quantizeInt8RowMajor, quantError } from './qwgpu/quantize.js';

await tf.setBackend('cpu'); await tf.ready();
const MD = 'model'; const fds = {};
const reader = { async range(p, s, e) { const l = e - s; const b = Buffer.allocUnsafe(l); fs.readSync(fds[p] ??= fs.openSync(`${MD}/${p}`, 'r'), b, 0, l, s); return b.buffer.slice(b.byteOffset, b.byteOffset + l); }, async text(p) { return fs.readFileSync(`${MD}/${p}`, 'utf8'); } };
const ref = JSON.parse(fs.readFileSync('ref.json')); const ids = ref.ids;

console.log('loading f32 weights…');
const weights = await loadModelWeights(reader);

// quantize+dequant each projection weight in place (int8 round-trip)
const suffixes = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'];
let worstRms = 0;
for (let i = 0; i < QWEN25_3B.numLayers; i++) {
  for (const s of suffixes) {
    const name = `model.layers.${i}.${s}.weight`;
    const W = weights[name]; const [outDim, inDim] = W.shape;
    const f32 = await W.data();
    if (i === 0) { const e = quantError(f32, outDim, inDim); worstRms = Math.max(worstRms, e.rms); if (s === 'self_attn.q_proj' || s === 'mlp.down_proj') console.log(`  L0 ${s}: rms=${(e.rms * 100).toFixed(2)}%`); }
    const { packed, scale } = quantizeInt8RowMajor(f32, outDim, inDim);
    const i8 = new Int8Array(new Uint8Array(packed.buffer));
    const deq = new Float32Array(outDim * inDim);
    for (let o = 0; o < outDim; o++) { const sc = scale[o]; for (let k = 0; k < inDim; k++) deq[o * inDim + k] = i8[o * inDim + k] * sc; }
    W.dispose();
    weights[name] = tf.tensor(deq, [outDim, inDim], 'float32');
  }
}
console.log(`worst L0 per-tensor rms quant error: ${(worstRms * 100).toFixed(2)}%`);

const model = new QwenModel(QWEN25_3B, weights);
async function topArg(l) { const am = tf.argMax(tf.reshape(l, [-1])); const d = await am.data(); am.dispose(); return d[0]; }
const idsT = tf.tensor2d([ids], [1, ids.length], 'int32');
const emb = model.embed(idsT); let pf = model.forward(emb, 0, null); emb.dispose(); idsT.dispose();
let kv = pf.kvCaches, pos = ids.length, nxt = await topArg(pf.logits); pf.logits.dispose();
const got = [];
for (let s = 0; s < 16; s++) { got.push(nxt); const tt = tf.tensor2d([[nxt]], [1, 1], 'int32'); const ee = model.embed(tt); const r = model.forward(ee, pos, kv); ee.dispose(); tt.dispose(); kv = r.kvCaches; pos++; nxt = await topArg(r.logits); r.logits.dispose(); }
model.disposeKV(kv);
console.log('int8 gen_ids:', JSON.stringify(got));
console.log('f32  gen_ids:', JSON.stringify(ref.gen_ids));
const exact = JSON.stringify(got) === JSON.stringify(ref.gen_ids);
let agree = 0; for (let i = 0; i < got.length; i++) if (got[i] === ref.gen_ids[i]) agree++;
console.log(`RESULT: ${exact ? 'EXACT' : agree + '/16 tokens agree'} — int8 ${exact || agree >= 14 ? 'PRESERVES output' : 'DEGRADES output'}`);
