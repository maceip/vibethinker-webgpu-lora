import { QwenWGPU } from '../qwgpu/runtime.js';
import { QWEN25_3B } from '../config.js';
import { initWebGPUDevice } from './device_service.js';
import { formatMessages } from './prompt_formatter.js';

async function buildTokenizer(reader) {
  const tj = JSON.parse(await reader.text('tokenizer.json'));
  const tc = JSON.parse(await reader.text('tokenizer_config.json'));
  const { PreTrainedTokenizer } = await import('@huggingface/transformers');
  return new PreTrainedTokenizer(tj, tc);
}

function sample(logits, temperature) {
  let best = 0, bv = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  if (!temperature || temperature <= 0) return best;
  let sum = 0; const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { const e = Math.exp((logits[i] - bv) / temperature); p[i] = e; sum += e; }
  let r = Math.random() * sum, c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (r <= c) return i; } return p.length - 1;
}

export class ModelSession {
  constructor({ cfg = QWEN25_3B, log = () => {} } = {}) {
    this.cfg = cfg;
    this.log = log;
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
    this.rt = new QwenWGPU(this.dev, this.cfg);
    await this.rt.build(reader, (msg, frac) => this.log(`weights: ${msg} ${(frac * 100).toFixed(0)}%`));
    window.__rt = this.rt; window.__tokenizer = this.tokenizer;
    this.log(`READY in ${((performance.now() - t0) / 1000).toFixed(1)}s — base loaded once; adapters hot-swap live.`);
    return this;
  }

  async readLogits() {
    const n = this.cfg.vocabSize;
    const rb = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.rt.s.logits, 0, rb, 0, n * 4); this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const a = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy(); return a;
  }

  async *generate(messages, { maxTokens = 1024, temperature = 0.0, stopIds = [151645, 151643] } = {}) {
    const rt = this.rt, tokenizer = this.tokenizer;
    const ids = tokenizer.encode(formatMessages(tokenizer, messages));
    if (ids.length <= rt.maxPrefillT) rt.prefillBatch(ids);
    else for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    let pos = ids.length;
    const emit = (id) => tokenizer.decode([id], { skip_special_tokens: true });

    if (temperature > 0) {
      let next = sample(await this.readLogits(), temperature);
      for (let step = 0; step < maxTokens; step++) {
        if (stopIds.includes(next)) break;
        const d = emit(next); if (d) yield d;
        rt.token(next, pos); pos++; next = sample(await this.readLogits(), temperature);
      }
      return;
    }

    const first = await rt.argmaxLogits();
    if (stopIds.includes(first)) return;
    { const d = emit(first); if (d) yield d; }
    let emitted = 1;
    while (emitted < maxTokens && pos < rt.maxCtx) {
      const K = Math.min(rt.MAXBATCH, maxTokens - emitted, rt.maxCtx - pos);
      const batch = await rt.decodeBatch(pos, K); pos += K;
      let stop = false;
      for (const id of batch) {
        if (stopIds.includes(id)) { stop = true; break; }
        const d = emit(id); if (d) yield d; emitted++;
        if (emitted >= maxTokens) { stop = true; break; }
      }
      if (stop) break;
    }
  }
}
