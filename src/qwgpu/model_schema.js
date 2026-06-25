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

// Centralized tensor schema and naming helpers for the custom Qwen2.5 WebGPU
// runtime. Keeping names, expected shapes, quantization mode, and LoRA module
// keys in one table makes model-loading and dispatch fail fast instead of
// silently depending on duplicated string templates.

/*
 * TECHNIQUE: Single source of truth schema
 *   All tensor names, shapes, quant modes and LoRA keys are derived here.
 *   Prevents drift between loader, uploader, and the many _dispatch call sites.
 */
const arrEq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

function projDesc(layer, subpath, outDim, inDim, { bias = false } = {}) {
  const name = `model.layers.${layer}.${subpath}.weight`;
  const m = subpath.match(/^(self_attn|mlp)\.(.+)$/);
  const loraKey = `layers.${layer}.${m[1]}.${m[2]}`;
  return {
    name,
    role: 'projection',
    quant: 'int4',
    shape: [outDim, inDim],
    loraKey,
    biasName: bias ? name.replace(/\.weight$/, '.bias') : null,
  };
}

function f32Desc(name, shape, role = 'f32') {
  return { name, role, quant: 'f32', shape };
}

/**
 * Build the exact Qwen2.5-3B tensor schema consumed by this runtime.
 *
 * Public dispatch paths continue to use the same tensor names and LoRA keys as
 * before; this schema is the single source of truth for build-time validation
 * and uploader decisions.
 */
export function createQwenSchema(cfg) {
  if (!cfg.tieWordEmbeddings && cfg.tieWordEmbeddings !== undefined) {
    throw new Error('QwenWGPU currently requires tied input/output embeddings');
  }
  const H = cfg.hiddenSize;
  const QD = cfg.numHeads * cfg.headDim;
  const KVD = cfg.numKVHeads * cfg.headDim;
  const I = cfg.intermediateSize;
  const tensors = [];
  const layers = [];

  const add = (d) => {
    tensors.push(d);
    return d;
  };
  const embed = add({ name: 'model.embed_tokens.weight', role: 'embedding', quant: 'int8', shape: [cfg.vocabSize, H] });
  const finalNorm = add(f32Desc('model.norm.weight', [H], 'final_norm'));

  for (let i = 0; i < cfg.numLayers; i++) {
    const p = `model.layers.${i}`;
    const layer = {
      index: i,
      inputNorm: add(f32Desc(`${p}.input_layernorm.weight`, [H], 'input_norm')),
      postAttentionNorm: add(f32Desc(`${p}.post_attention_layernorm.weight`, [H], 'post_attention_norm')),
      projections: {},
      biases: {},
    };
    layer.projections.q = add(projDesc(i, 'self_attn.q_proj', QD, H, { bias: !!cfg.attentionBias }));
    layer.projections.k = add(projDesc(i, 'self_attn.k_proj', KVD, H, { bias: !!cfg.attentionBias }));
    layer.projections.v = add(projDesc(i, 'self_attn.v_proj', KVD, H, { bias: !!cfg.attentionBias }));
    layer.projections.o = add(projDesc(i, 'self_attn.o_proj', H, QD));
    layer.projections.gate = add(projDesc(i, 'mlp.gate_proj', I, H));
    layer.projections.up = add(projDesc(i, 'mlp.up_proj', I, H));
    layer.projections.down = add(projDesc(i, 'mlp.down_proj', H, I));
    for (const key of ['q', 'k', 'v']) {
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
    projectionDescs: tensors.filter((t) => t.role === 'projection'),
    validateTensor(name, shape) {
      const desc = byName.get(name);
      if (!desc) return null;
      if (!arrEq(shape, desc.shape)) {
        throw new Error(`shape mismatch for ${name}: got [${shape.join(',')}], expected [${desc.shape.join(',')}]`);
      }
      return desc;
    },
    assertComplete(seen) {
      const missing = [];
      for (const name of expectedNames) if (!seen.has(name)) missing.push(name);
      if (missing.length) {
        const sample = missing.slice(0, 12).join(', ');
        throw new Error(`missing ${missing.length} required tensor(s): ${sample}${missing.length > 12 ? ', …' : ''}`);
      }
    },
  };
}

// Match common PEFT/MLX LoRA safetensors names and normalize them to runtime
// module keys: layers.I.{self_attn,mlp}.foo_proj.
export function moduleKeyFromTensorName(name) {
  const m = name.match(/layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+?)(_proj)?\.(lora_[ABab])/i);
  if (!m) return null;
  return `layers.${m[1]}.${m[2]}.${m[3].replace(/_proj$/, '')}_proj`;
}
