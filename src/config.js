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

// Qwen2.5-3B architecture config.
// Standalone (no tf.js) so the WebGPU runtime and app can import it
// without pulling in the reference implementation's dependencies.

/*
 * TECHNIQUE: Minimal static architecture table
 *   All dimensions, counts and flags live in one small object.
 *   Eliminates magic numbers scattered across kernels, dispatch plan,
 *   buffer sizing and schema generation. Makes the entire engine
 *   easy to retarget to other model sizes.
 */
export const QWEN25_3B = {
  hiddenSize: 2048,
  numLayers: 36,
  numHeads: 16,
  numKVHeads: 2,
  headDim: 128,
  intermediateSize: 11008,
  vocabSize: 151936,
  rmsNormEps: 1e-6,
  ropeTheta: 1000000.0,

  /*
   * TECHNIQUE: Tie word embeddings
   *   input embedding == output head.
   *   Simplifies loading (one tensor), schema, and final projection math.
   *   Required by the current model_uploader + schema.
   */
  tieWordEmbeddings: true,

  // QKV projections carry a bias in Qwen2.5; o_proj and the MLP do not.
  attentionBias: true,
};
