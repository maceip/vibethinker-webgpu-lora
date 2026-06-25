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

// Schema-derived operation descriptors used by runtime dispatch. This keeps the
// hot decode/prefill paths from rebuilding safetensors names and LoRA keys with
// ad-hoc string templates.

/*
 * TECHNIQUE: Precomputed dispatch plan
 *   Instead of string concatenation on every token, we build a static plan
 *   object once at build time. The generate/step loops just index into it.
 *   Reduces per-token JS work and keeps call sites simple/monomorphic.
 */
export function createDispatchPlan(schema) {
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
        loraKey: layer.projections.q.loraKey,
      },
      k: {
        weight: layer.projections.k.name,
        bias: layer.biases.k?.name || null,
        loraKey: layer.projections.k.loraKey,
      },
      v: {
        weight: layer.projections.v.name,
        bias: layer.biases.v?.name || null,
        loraKey: layer.projections.v.loraKey,
      },
      o: {
        weight: layer.projections.o.name,
        bias: null,
        loraKey: layer.projections.o.loraKey,
      },
      gate: {
        weight: layer.projections.gate.name,
        bias: null,
        loraKey: layer.projections.gate.loraKey,
      },
      up: {
        weight: layer.projections.up.name,
        bias: null,
        loraKey: layer.projections.up.loraKey,
      },
      down: {
        weight: layer.projections.down.name,
        bias: null,
        loraKey: layer.projections.down.loraKey,
      },
    })),
  };
}
