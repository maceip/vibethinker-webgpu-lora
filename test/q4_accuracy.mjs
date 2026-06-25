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

import * as tf from '@tensorflow/tfjs-core'; import '@tensorflow/tfjs-backend-cpu'; import fs from 'fs'; import { fileURLToPath } from 'url';
import { QwenModel, QWEN25_3B } from '../src/qwen25.js'; import { loadModelWeights } from '../src/weights.js';
import { quantizeInt4Group } from '../src/qwgpu/quantize.js';
await tf.setBackend('cpu'); await tf.ready();
const MD=fileURLToPath(new URL('../model', import.meta.url)); const fds={}; const reader={async range(p,s,e){const l=e-s;const b=Buffer.allocUnsafe(l);fs.readSync(fds[p]??=fs.openSync(`${MD}/${p}`,'r'),b,0,l,s);return b.buffer.slice(b.byteOffset,b.byteOffset+l);},async text(p){return fs.readFileSync(`${MD}/${p}`,'utf8');}};
const ref=JSON.parse(fs.readFileSync(new URL('./ref.json', import.meta.url))); const ids=ref.ids;
const weights=await loadModelWeights(reader);
const suf=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'];
for(let i=0;i<QWEN25_3B.numLayers;i++) for(const s of suf){
  const name=`model.layers.${i}.${s}.weight`; const W=weights[name]; const [o,k]=W.shape; const f=await W.data();
  const {packed,scale,groupsPerRow}=quantizeInt4Group(f,o,k,128);
  // dequant
  const deq=new Float32Array(o*k);
  for(let r=0;r<o;r++) for(let c=0;c<k;c++){ const wi=(r*k+c); const word=packed[wi>>3]; let nib=(word>>((wi&7)*4))&0xF; if(nib>7)nib-=16; deq[wi]=nib*scale[r*groupsPerRow+(c>>7)]; }
  W.dispose(); weights[name]=tf.tensor(deq,[o,k],'float32');
}
const model=new QwenModel(QWEN25_3B,weights);
const topArg=async l=>{const am=tf.argMax(tf.reshape(l,[-1]));const d=await am.data();am.dispose();return d[0];};
const idsT=tf.tensor2d([ids],[1,ids.length],'int32'); const emb=model.embed(idsT); let pf=model.forward(emb,0,null); emb.dispose(); idsT.dispose();
let kv=pf.kvCaches,pos=ids.length,nxt=await topArg(pf.logits); pf.logits.dispose(); const got=[];
for(let s=0;s<16;s++){got.push(nxt);const tt=tf.tensor2d([[nxt]],[1,1],'int32');const ee=model.embed(tt);const r=model.forward(ee,pos,kv);ee.dispose();tt.dispose();kv=r.kvCaches;pos++;nxt=await topArg(r.logits);r.logits.dispose();}
model.disposeKV(kv);
console.log('int4 gen:',JSON.stringify(got)); console.log('ref  gen:',JSON.stringify(ref.gen_ids));
let agree=0; for(let i=0;i<16;i++) if(got[i]===ref.gen_ids[i]) agree++;
console.log(`RESULT int4-g128: ${agree}/16 agree ${agree>=14?'PRESERVES':'DEGRADES'}`);
