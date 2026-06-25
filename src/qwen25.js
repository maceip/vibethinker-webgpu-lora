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

// Qwen2.5 forward pass on WebGPU via TensorFlow.js, with runtime-swappable LoRA.
//
// Every linear projection goes through proj(), which adds an optional LoRA delta
// (scale * (x @ A) @ B). The active adapter is held in this.lora and can be
// swapped at runtime with setLora()/clearLora() WITHOUT reloading base weights —
// that is the in-browser LoRA hot-swap capability.
//
// Weights follow HF naming/layout: Linear weight is [out, in] and y = x @ Wᵀ,
// so we use tf.matMul(x, W, /*transposeA=*/false, /*transposeB=*/true).

import * as tf from '@tensorflow/tfjs-core';

export { QWEN25_3B } from './config.js';

/*
 * TECHNIQUE: Runtime LoRA hot-swap via delta injection in proj()
 *   LoRA A/B live in this.lora. proj() does y = x@W^T + (scale * (x@A)@B)
 *   when an adapter is active. No base weight reload required.
 *   (This file is the tf.js reference; the pure WebGPU path replicates the idea.)
 */

/**
 * A LoRA adapter is a flat map of module key -> {A, B, scale}, where
 *   A: tf.Tensor [in, rank], B: tf.Tensor [rank, out], scale: number.
 * Module keys match the ones passed to proj() below, e.g.
 *   "layers.0.self_attn.q_proj", "layers.0.mlp.gate_proj", ...
 */
export class LoraAdapter {
  constructor(name, modules = {}) {
    this.name = name;
    this.modules = modules; // key -> {A, B, scale}
  }
  dispose() {
    for (const k of Object.keys(this.modules)) {
      const m = this.modules[k];
      m.A?.dispose?.();
      m.B?.dispose?.();
    }
    this.modules = {};
  }
}

export class QwenModel {
  /**
   * @param config  one of the QWEN25_* config objects
   * @param weights map of HF tensor name -> tf.Tensor (already on the active
   *                backend). Expected keys per layer i:
   *   model.layers.i.input_layernorm.weight
   *   model.layers.i.self_attn.{q,k,v,o}_proj.weight  (+ q,k,v .bias)
   *   model.layers.i.post_attention_layernorm.weight
   *   model.layers.i.mlp.{gate,up,down}_proj.weight
   *   model.norm.weight, model.embed_tokens.weight  (lm_head tied)
   */
  constructor(config, weights) {
    this.cfg = config;
    this.w = weights;
    this.lora = null; // active LoraAdapter (or null)
    this._pretransposeProjections(); // work around tf.js webgpu transposeB bug
    this._precomputeRope(8192); // max context for the cos/sin tables
  }

  // tf.js webgpu matMul(_, W, false, /*transposeB=*/true) is BROKEN for small
  // output dims (128/256) — k_proj/v_proj produced garbage. Pre-transpose every
  // projection weight [out,in]->[in,out] so proj() uses transposeB=false
  // (correct at all sizes). embed_tokens is left as-is (gather + lm_head matmul,
  // whose 151936 output is unaffected by the bug).
  _pretransposeProjections() {
    const suffixes = [
      'self_attn.q_proj',
      'self_attn.k_proj',
      'self_attn.v_proj',
      'self_attn.o_proj',
      'mlp.gate_proj',
      'mlp.up_proj',
      'mlp.down_proj',
    ];
    for (let i = 0; i < this.cfg.numLayers; i++) {
      for (const s of suffixes) {
        const name = `model.layers.${i}.${s}.weight`;
        const W = this.w[name];
        if (!W) continue;
        const Wt = tf.transpose(W); // [out,in] -> [in,out]
        W.dispose();
        this.w[name] = Wt;
      }
    }
  }

  // ---- runtime LoRA hot-swap ------------------------------------------------
  setLora(adapter) {
    this.lora = adapter;
  } // swap with no base reload
  clearLora() {
    this.lora = null;
  }
  get activeLora() {
    return this.lora?.name ?? null;
  }

  // ---- core ops -------------------------------------------------------------

  /** Linear projection with optional LoRA delta merged in at runtime. */
  proj(x, weightName, moduleKey, bias = null) {
    return tf.tidy(() => {
      let y = tf.matMul(x, this.w[weightName], false, false); // W pre-transposed to [in,out]
      if (bias) y = tf.add(y, bias);
      const a = this.lora?.modules?.[moduleKey];
      if (a) {
        // delta = scale * (x @ A) @ B   ; A:[in,r] B:[r,out]
        const delta = tf.mul(tf.matMul(tf.matMul(x, a.A), a.B), a.scale);
        y = tf.add(y, delta);
      }
      return y;
    });
  }

  rmsNorm(x, weightName) {
    return tf.tidy(() => {
      const variance = tf.mean(tf.square(x), -1, true);
      const normed = tf.mul(x, tf.rsqrt(tf.add(variance, this.cfg.rmsNormEps)));
      return tf.mul(normed, this.w[weightName]);
    });
  }

  _precomputeRope(maxSeq) {
    const { headDim, ropeTheta } = this.cfg;
    const half = headDim / 2;
    const inv = new Float32Array(half);
    for (let i = 0; i < half; i++) inv[i] = 1.0 / Math.pow(ropeTheta, (2 * i) / headDim);
    const cos = new Float32Array(maxSeq * headDim);
    const sin = new Float32Array(maxSeq * headDim);
    for (let p = 0; p < maxSeq; p++) {
      for (let i = 0; i < half; i++) {
        const ang = p * inv[i];
        const c = Math.cos(ang),
          s = Math.sin(ang);
        // layout matches rotate-half: [first half | second half] share freq i
        cos[p * headDim + i] = c;
        cos[p * headDim + half + i] = c;
        sin[p * headDim + i] = s;
        sin[p * headDim + half + i] = s;
      }
    }
    this._ropeCos = tf.tensor2d(cos, [maxSeq, headDim]); // [maxSeq, headDim]
    this._ropeSin = tf.tensor2d(sin, [maxSeq, headDim]);
  }

  /** Apply RoPE. x: [B, nHeads, T, headDim], positions: int over [0,maxSeq). */
  applyRope(x, startPos, T) {
    return tf.tidy(() => {
      const { headDim } = this.cfg;
      const half = headDim / 2;
      const cos = tf.reshape(tf.slice(this._ropeCos, [startPos, 0], [T, headDim]), [1, 1, T, headDim]);
      const sin = tf.reshape(tf.slice(this._ropeSin, [startPos, 0], [T, headDim]), [1, 1, T, headDim]);
      const x1 = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, half]);
      const x2 = tf.slice(x, [0, 0, 0, half], [-1, -1, -1, half]);
      const rotated = tf.concat([tf.neg(x2), x1], 3); // rotate-half
      return tf.add(tf.mul(x, cos), tf.mul(rotated, sin));
    });
  }

  /** One decoder layer. kv = {k,v} cache tensors [B,nKV,Tpast,hd] or null. */
  layer(i, hidden, startPos, kvCache) {
    const { numHeads, numKVHeads, headDim } = this.cfg;
    const B = hidden.shape[0],
      T = hidden.shape[1];
    const p = `model.layers.${i}`;

    // Single tidy returning [h2, newK, newV]. tf.tidy auto-keeps tensors in the
    // returned array, so no tf.keep/clone side-effects (those mishandled GPU
    // buffer lifetimes on the webgpu backend and corrupted the output).
    const [h2, newK, newV] = tf.tidy(() => {
      const normed = this.rmsNorm(hidden, `${p}.input_layernorm.weight`);
      let q = this.proj(
        normed,
        `${p}.self_attn.q_proj.weight`,
        `layers.${i}.self_attn.q_proj`,
        this.w[`${p}.self_attn.q_proj.bias`],
      );
      let k = this.proj(
        normed,
        `${p}.self_attn.k_proj.weight`,
        `layers.${i}.self_attn.k_proj`,
        this.w[`${p}.self_attn.k_proj.bias`],
      );
      let v = this.proj(
        normed,
        `${p}.self_attn.v_proj.weight`,
        `layers.${i}.self_attn.v_proj`,
        this.w[`${p}.self_attn.v_proj.bias`],
      );

      q = tf.transpose(tf.reshape(q, [B, T, numHeads, headDim]), [0, 2, 1, 3]);
      k = tf.transpose(tf.reshape(k, [B, T, numKVHeads, headDim]), [0, 2, 1, 3]);
      v = tf.transpose(tf.reshape(v, [B, T, numKVHeads, headDim]), [0, 2, 1, 3]);

      q = this.applyRope(q, startPos, T);
      k = this.applyRope(k, startPos, T);

      let kFull = k,
        vFull = v;
      if (kvCache && kvCache.k) {
        kFull = tf.concat([kvCache.k, k], 2);
        vFull = tf.concat([kvCache.v, v], 2);
      }

      // GQA: repeat KV heads to match query heads
      const groups = numHeads / numKVHeads;
      let kRep = kFull,
        vRep = vFull;
      if (groups > 1) {
        kRep = tf.reshape(tf.tile(tf.expandDims(kFull, 2), [1, 1, groups, 1, 1]), [
          B,
          numHeads,
          kFull.shape[2],
          headDim,
        ]);
        vRep = tf.reshape(tf.tile(tf.expandDims(vFull, 2), [1, 1, groups, 1, 1]), [
          B,
          numHeads,
          vFull.shape[2],
          headDim,
        ]);
      }

      const scale = 1 / Math.sqrt(headDim);
      const kT = tf.transpose(kRep, [0, 1, 3, 2]); // [B,H,hd,Tfull]
      let scores = tf.mul(tf.matMul(q, kT, false, false), scale); // [B,H,T,Tfull]
      // Decode (T=1) needs no causal mask: the single query attends to all cached
      // keys. Only prefill (T>1) needs the upper-triangular mask. Skipping the
      // per-layer CPU->GPU mask upload is a big decode speedup.
      if (T > 1) scores = tf.add(scores, this._causalMask(T, kRep.shape[2], startPos));
      const attn = tf.softmax(scores, -1);
      let ao = tf.matMul(attn, vRep); // [B,H,T,hd]
      ao = tf.reshape(tf.transpose(ao, [0, 2, 1, 3]), [B, T, numHeads * headDim]);
      const attnProj = this.proj(ao, `${p}.self_attn.o_proj.weight`, `layers.${i}.self_attn.o_proj`);
      const h = tf.add(hidden, attnProj);

      const normed2 = this.rmsNorm(h, `${p}.post_attention_layernorm.weight`);
      const gate = this.proj(normed2, `${p}.mlp.gate_proj.weight`, `layers.${i}.mlp.gate_proj`);
      const up = this.proj(normed2, `${p}.mlp.up_proj.weight`, `layers.${i}.mlp.up_proj`);
      const act = tf.mul(tf.mul(gate, tf.sigmoid(gate)), up); // silu(gate) * up
      const mlpO = this.proj(act, `${p}.mlp.down_proj.weight`, `layers.${i}.mlp.down_proj`);
      const out = tf.add(h, mlpO);

      return [out, kFull, vFull];
    });

    this._newKV = { k: newK, v: newV };
    return h2;
  }

  /** Debug: layer-0 step by step, returns every intermediate (no tidy; leaks). */
  debugLayer0(hidden, startPos = 0) {
    const { numHeads, numKVHeads, headDim } = this.cfg;
    const B = hidden.shape[0],
      T = hidden.shape[1];
    const p = 'model.layers.0';
    const ln1 = this.rmsNorm(hidden, `${p}.input_layernorm.weight`);
    let q = this.proj(
      ln1,
      `${p}.self_attn.q_proj.weight`,
      'layers.0.self_attn.q_proj',
      this.w[`${p}.self_attn.q_proj.bias`],
    );
    let k = this.proj(
      ln1,
      `${p}.self_attn.k_proj.weight`,
      'layers.0.self_attn.k_proj',
      this.w[`${p}.self_attn.k_proj.bias`],
    );
    let v = this.proj(
      ln1,
      `${p}.self_attn.v_proj.weight`,
      'layers.0.self_attn.v_proj',
      this.w[`${p}.self_attn.v_proj.bias`],
    );
    const qproj = q;
    const kproj = k;
    q = tf.transpose(tf.reshape(q, [B, T, numHeads, headDim]), [0, 2, 1, 3]);
    k = tf.transpose(tf.reshape(k, [B, T, numKVHeads, headDim]), [0, 2, 1, 3]);
    v = tf.transpose(tf.reshape(v, [B, T, numKVHeads, headDim]), [0, 2, 1, 3]);
    const qr = this.applyRope(q, startPos, T),
      kr = this.applyRope(k, startPos, T);
    const groups = numHeads / numKVHeads;
    const kRep = tf.reshape(tf.tile(tf.expandDims(kr, 2), [1, 1, groups, 1, 1]), [B, numHeads, kr.shape[2], headDim]);
    const vRep = tf.reshape(tf.tile(tf.expandDims(v, 2), [1, 1, groups, 1, 1]), [B, numHeads, v.shape[2], headDim]);
    const kT = tf.transpose(kRep, [0, 1, 3, 2]);
    let scores = tf.mul(tf.matMul(qr, kT, false, false), 1 / Math.sqrt(headDim));
    scores = tf.add(scores, this._causalMask(T, kRep.shape[2], startPos));
    const attn = tf.softmax(scores, -1);
    let ao = tf.matMul(attn, vRep);
    ao = tf.reshape(tf.transpose(ao, [0, 2, 1, 3]), [B, T, numHeads * headDim]);
    const attnProj = this.proj(ao, `${p}.self_attn.o_proj.weight`, 'layers.0.self_attn.o_proj');
    const h = tf.add(hidden, attnProj);
    const ln2 = this.rmsNorm(h, `${p}.post_attention_layernorm.weight`);
    const gate = this.proj(ln2, `${p}.mlp.gate_proj.weight`, 'layers.0.mlp.gate_proj');
    const up = this.proj(ln2, `${p}.mlp.up_proj.weight`, 'layers.0.mlp.up_proj');
    const act = tf.mul(tf.mul(gate, tf.sigmoid(gate)), up);
    const mlpO = this.proj(act, `${p}.mlp.down_proj.weight`, 'layers.0.mlp.down_proj');
    const out = tf.add(h, mlpO);
    return { ln1, qproj, kproj, qr, kr, kRep, kT, scores, attnProj, h, ln2, mlpO, out };
  }

  _causalMask(T, Tfull, startPos) {
    return tf.tidy(() => {
      // allowed if keyPos <= queryAbsPos = startPos + qi ; keyPos in [0,Tfull)
      const buf = new Float32Array(T * Tfull);
      for (let qi = 0; qi < T; qi++) {
        const qAbs = startPos + qi;
        for (let ki = 0; ki < Tfull; ki++) buf[qi * Tfull + ki] = ki <= qAbs ? 0 : -1e9;
      }
      return tf.tensor(buf, [1, 1, T, Tfull]);
    });
  }

  /**
   * Forward over a token window. embeds: [B,T,hidden] (from embedding lookup).
   * kvCaches: array length numLayers of {k,v}|null. Returns {logits, kvCaches}.
   * Only the last position's logits are returned (for decoding).
   */
  forward(embeds, startPos, kvCaches) {
    let hidden = embeds;
    const newCaches = new Array(this.cfg.numLayers);
    for (let i = 0; i < this.cfg.numLayers; i++) {
      this._newKV = null;
      const next = this.layer(i, hidden, startPos, kvCaches ? kvCaches[i] : null);
      if (hidden !== embeds) hidden.dispose();
      hidden = next;
      // free previous cache for this layer, keep the fresh one
      if (kvCaches && kvCaches[i]) {
        kvCaches[i].k.dispose();
        kvCaches[i].v.dispose();
      }
      newCaches[i] = this._newKV;
    }
    const logits = tf.tidy(() => {
      const normed = this.rmsNorm(hidden, 'model.norm.weight');
      const T = normed.shape[1];
      const last = tf.slice(normed, [0, T - 1, 0], [-1, 1, -1]); // [B,1,hidden]
      // tied embeddings: lm_head == embed_tokens [vocab, hidden]
      return tf.matMul(last, this.w['model.embed_tokens.weight'], false, true); // [B,1,vocab]
    });
    hidden.dispose();
    return { logits, kvCaches: newCaches };
  }

  /** Embedding lookup: ids int32 [B,T] -> [B,T,hidden]. */
  embed(ids) {
    return tf.tidy(() => tf.gather(this.w['model.embed_tokens.weight'], ids));
  }

  disposeKV(kvCaches) {
    if (!kvCaches) return;
    for (const kv of kvCaches) {
      kv?.k?.dispose();
      kv?.v?.dispose();
    }
  }
}
