// Qwen2.5-3B architecture config. Standalone (no tf.js) so the WebGPU runtime and
// app import it without pulling in the reference implementation's dependencies.
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
  tieWordEmbeddings: true,
  // qkv projections have a bias in Qwen2.5; o_proj and mlp do not.
  attentionBias: true,
};
