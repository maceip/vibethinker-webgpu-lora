/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

// TrainingController: turns chat/reasoning examples into masked, shifted-label
// training micro-batches and drives QwenLoraTrainer's accumulate -> optimizer loop.
// Reuses the session tokenizer + ChatML formatter so training data is tokenized
// exactly like inference prompts (critical for VibeThinker-3B reasoning traces).

import { QwenLoraTrainer, createTrainableAdapter } from '../qwgpu/trainer.js';
import { formatMessages } from './prompt_formatter.js';

const IM_END = 151645; // <|im_end|>

export class TrainingController {
  // session: a loaded ModelSession (rt + tokenizer). adapters: AdapterRegistry.
  constructor({ session, adapters, log = () => {}, trainerOptions = {} } = {}) {
    this.session = session;
    this.adapters = adapters;
    this.log = log;
    this.trainerOptions = trainerOptions;
    this.trainer = null;
    this.adapter = null;
  }

  get rt() {
    return this.session.rt;
  }
  get tokenizer() {
    return this.session.tokenizer;
  }

  // Create + register a fresh trainable adapter and attach the trainer to it.
  initAdapter(name = 'trainable', { rank = 16, alpha = 32, targetModules } = {}) {
    const adapter = createTrainableAdapter(this.rt, { name, rank, alpha, targetModules });
    this.adapters.adapters[name] = adapter;
    this.adapter = adapter;
    this.trainer = new QwenLoraTrainer(this.rt, this.trainerOptions);
    this.trainer.attach(adapter);
    this.log(`init adapter "${name}" rank=${rank} alpha=${alpha} modules=${Object.keys(adapter.modules).length}`);
    return adapter;
  }

  // Attach to an already-registered adapter (e.g. continue training a loaded one).
  attachAdapter(name) {
    const adapter = this.adapters.get(name);
    if (!adapter) throw new Error(`adapter "${name}" not found`);
    this.adapter = adapter;
    this.trainer = new QwenLoraTrainer(this.rt, this.trainerOptions);
    this.trainer.attach(adapter);
    return adapter;
  }

  /*
   * TECHNIQUE: Completion-only loss masking with shifted labels
   *   Tokenize prompt (with assistant generation prompt) and completion separately.
   *   mask[t]=1 trains the prediction of tokens[t+1] from position t — so we mask
   *   positions whose NEXT token is part of the completion (incl. the final EOS).
   *   Prompt tokens get mask=0, so the model is only graded on what it should write.
   */
  prepareExample({ messages, prompt, completion, trainPromptToo = false }) {
    const tk = this.tokenizer;
    let promptIds;
    if (messages) {
      promptIds = tk.encode(formatMessages(tk, messages));
    } else {
      promptIds = tk.encode(prompt);
    }
    const compIds = tk.encode(completion, { add_special_tokens: false });
    const tokens = [...promptIds, ...compIds, IM_END];
    const T = tokens.length;
    const lossMask = new Array(T).fill(0);
    const firstTrainPos = trainPromptToo ? 0 : Math.max(0, promptIds.length - 1);
    for (let t = firstTrainPos; t < T - 1; t++) lossMask[t] = 1;
    return { tokens, lossMask };
  }

  prepareBatch(examples) {
    return examples.map((e) => this.prepareExample(e));
  }

  // One optimizer step over `microBatches` (array of {tokens, lossMask}); grads
  // accumulate across them, then a single AdamW update is applied.
  async step(microBatches) {
    if (!this.trainer) throw new Error('call initAdapter()/attachAdapter() first');
    return this.trainer.trainStep(microBatches);
  }

  // Full training run over a dataset of examples. Honors gradAccumSteps by grouping
  // examples into accumulation windows. Calls onStep({step, loss, lr, gradNorm}).
  async train(examples, { epochs = 1, onStep = () => {}, maxTrainSeq } = {}) {
    if (!this.trainer) this.initAdapter();
    const accum = this.trainer.opts.gradAccumSteps;
    const cap = maxTrainSeq ?? this.trainer.opts.maxTrainSeq;
    let globalStep = 0;
    for (let ep = 0; ep < epochs; ep++) {
      const order = shuffle([...Array(examples.length).keys()]);
      let window = [];
      for (const idx of order) {
        let mb = this.prepareExample(examples[idx]);
        if (mb.tokens.length > cap) mb = truncate(mb, cap);
        window.push(mb);
        if (window.length === accum) {
          const r = await this.step(window);
          globalStep++;
          this.log(`step ${globalStep} epoch ${ep} loss=${r.loss.toFixed(4)} lr=${r.lr.toExponential(2)} |g|=${r.gradNorm.toFixed(3)}`);
          onStep({ step: globalStep, epoch: ep, ...r });
          window = [];
        }
      }
      if (window.length) {
        const r = await this.step(window);
        globalStep++;
        onStep({ step: globalStep, epoch: ep, ...r });
      }
    }
    // adapter A/B already mutated in place; inference hot-swap is live.
    this.adapters.applyToRuntime(this.adapter.name, this.rt);
    return { steps: globalStep, adapter: this.adapter };
  }
}

function truncate(mb, cap) {
  return { tokens: mb.tokens.slice(0, cap), lossMask: mb.lossMask.slice(0, cap) };
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
