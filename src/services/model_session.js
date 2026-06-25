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

import { QwenWGPU } from '../qwgpu/runtime.js';
import { QWEN25_3B } from '../config.js';
import { initWebGPUDevice } from './device_service.js';
import { formatMessages } from './prompt_formatter.js';

async function buildTokenizer(reader) {
  const tj = JSON.parse(await reader.text('tokenizer.json'));
  const tc = JSON.parse(await reader.text('tokenizer_config.json'));
  /*
   * TECHNIQUE: Dynamic import for heavy optional dependency
   *   @huggingface/transformers is externalized in esbuild and only loaded
   *   when a tokenizer is actually needed. Keeps the core inference bundle lean.
   */
  const { PreTrainedTokenizer } = await import('@huggingface/transformers');
  return new PreTrainedTokenizer(tj, tc);
}

function randomUnit() {
  if (globalThis.crypto?.getRandomValues) {
    const u = new Uint32Array(1);
    globalThis.crypto.getRandomValues(u);
    return u[0] / 4294967296;
  }
  return Math.random();
}

function sampleTopK(candidates, { temperature, topP = 1.0 }) {
  if (!temperature || temperature <= 0) return candidates[0]?.id ?? 0;
  const best = candidates[0]?.logit ?? 0;
  const weighted = candidates.map((c) => ({ id: c.id, w: Math.exp((c.logit - best) / temperature) }));
  let sum = weighted.reduce((a, c) => a + c.w, 0);
  if (topP > 0 && topP < 1 && weighted.length > 1 && sum > 0) {
    let csum = 0,
      keep = 0;
    for (; keep < weighted.length; keep++) {
      csum += weighted[keep].w / sum;
      if (csum >= topP) {
        keep++;
        break;
      }
    }
    weighted.length = Math.max(1, keep);
    sum = weighted.reduce((a, c) => a + c.w, 0);
  }
  let r = randomUnit() * sum,
    c = 0;
  for (const item of weighted) {
    c += item.w;
    if (r <= c) return item.id;
  }
  return weighted[weighted.length - 1]?.id ?? candidates[0]?.id ?? 0;
}

export class ModelSession {
  constructor({ cfg = QWEN25_3B, log = () => {}, runtimeOptions = {} } = {}) {
    this.cfg = cfg;
    this.log = log;
    this.runtimeOptions = { decodeBatchSize: 'auto', samplingTopK: 40, ...runtimeOptions };
    this.dev = null;
    this.rt = null;
    this.tokenizer = null;
  }

  async loadWith(reader, label) {
    this.dev = await initWebGPUDevice({ log: this.log });
    this.log(`loading tokenizer from ${label}…`);
    this.tokenizer = await buildTokenizer(reader);
    this.log(`tokenizer loaded. streaming + quantizing weights (int4) from ${label}…`);
    const t0 = performance.now();
    this.rt = new QwenWGPU(this.dev, this.cfg, this.runtimeOptions);
    await this.rt.build(reader, (msg, frac) => this.log(`weights: ${msg} ${(frac * 100).toFixed(0)}%`));
    window.__rt = this.rt;
    window.__tokenizer = this.tokenizer;
    const tuning = this.rt.decodeBatchTuning;
    const tuned = tuning ? ` decodeBatch=${tuning.selected} (${tuning.reason})` : '';
    this.log(
      `READY in ${((performance.now() - t0) / 1000).toFixed(1)}s — base loaded once; adapters hot-swap live.${tuned}`,
    );
    return this;
  }

  async readLogits() {
    const n = this.cfg.vocabSize;
    const rb = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.rt.s.logits, 0, rb, 0, n * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice();
    rb.unmap();
    rb.destroy();
    return a;
  }

  async sampleNextToken({ temperature, topK = this.rt.samplingTopK, topP = 1.0 } = {}) {
    return sampleTopK(await this.rt.topKLogits(topK), { temperature, topP });
  }

  async *generate(
    messages,
    { maxTokens = 1024, temperature = 0.0, topK, topP = 1.0, stopIds = [151645, 151643] } = {},
  ) {
    const rt = this.rt,
      tokenizer = this.tokenizer;
    const ids = tokenizer.encode(formatMessages(tokenizer, messages));
    if (ids.length <= rt.maxPrefillT) rt.prefillBatch(ids);
    else for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    let pos = ids.length;
    const emit = (id) => tokenizer.decode([id], { skip_special_tokens: true });

    if (temperature > 0) {
      let next = await this.sampleNextToken({ temperature, topK, topP });
      for (let step = 0; step < maxTokens; step++) {
        if (stopIds.includes(next)) break;
        const d = emit(next);
        if (d) yield d;
        rt.token(next, pos);
        pos++;
        next = await this.sampleNextToken({ temperature, topK, topP });
      }
      return;
    }

    const first = await rt.argmaxLogits();
    if (stopIds.includes(first)) return;
    {
      const d = emit(first);
      if (d) yield d;
    }
    let emitted = 1;
    while (emitted < maxTokens && pos < rt.maxCtx) {
      // Start with small batches for interactivity/EOS, then use the tuned
      // greedy batch size. decodeGreedyBatch is greedy-only: sampled decoding
      // stays one token at a time so it can feed the sampled id back into KV.
      const K = rt.greedyBatchSizeFor({ emitted, remaining: maxTokens - emitted, pos });
      const batch = await rt.decodeGreedyBatch(pos, K);
      pos += batch.length;
      let stop = false;
      for (const id of batch) {
        if (stopIds.includes(id)) {
          stop = true;
          break;
        }
        const d = emit(id);
        if (d) yield d;
        emitted++;
        if (emitted >= maxTokens) {
          stop = true;
          break;
        }
      }
      if (stop) break;
    }
  }
}
