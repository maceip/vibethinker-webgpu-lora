// Browser harness: load our Qwen2.5-3B bug-bounty triage model and run it on a
// custom pure-WebGPU runtime (int4 weights, GPU-resident KV cache), with runtime
// LoRA adapter hot-swap (no base reload). BYO model: served under /model/.
import { QwenWGPU } from './qwgpu/runtime.js';
import { QWEN25_3B } from './config.js';
import { loadLoraAdapterGPU } from './lora_gpu.js';
import { urlReader, hfReader, fileReader } from './readers.js';

const $ = id => document.getElementById(id);
const log = (m) => { const s = $('status'); if (s) s.textContent = m; console.log('[harness]', m); };

const SYS = `You are a senior bug bounty triage analyst. Read the submission and assign exactly ONE disposition from: valid_impactful, valid_low, corroborated_surge, likely_duplicate, out_of_scope, theoretical_no_poc, self_inflicted, accepted_risk, slop. Estimate severity_estimate (critical/high/medium/low/none). Think step by step, then output a SINGLE JSON object on the last line with keys: disposition, severity_estimate, is_duplicate_risk, reasoning, questions_for_researcher, confidence. Output only valid JSON for that object.`;

let rt = null, dev = null, tokenizer = null;
const adapters = { none: null }; // name -> {modules} | null

async function initDevice() {
  log('requesting WebGPU device…');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter (use a WebGPU-capable browser)');
  if (!adapter.features.has('subgroups')) throw new Error('GPU lacks the "subgroups" feature (needed by the fast GEMV kernels)');
  dev = await adapter.requestDevice({ requiredFeatures: ['subgroups'], requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  dev.addEventListener?.('uncapturederror', e => console.error('GPUERR', e.error.message));
  log(`WebGPU ready. maxBuffer=${(Number(adapter.limits.maxBufferSize) / 1e9).toFixed(2)}GB`);
}

async function buildTokenizer(reader) {
  const tj = JSON.parse(await reader.text('tokenizer.json'));
  const tc = JSON.parse(await reader.text('tokenizer_config.json'));
  const { PreTrainedTokenizer } = await import('@huggingface/transformers');
  return new PreTrainedTokenizer(tj, tc);
}

// load from any reader (same-origin URL, Hugging Face, or BYO files)
async function loadWith(reader, label) {
  await initDevice();
  log(`loading tokenizer from ${label}…`);
  tokenizer = await buildTokenizer(reader);
  log(`tokenizer loaded. streaming + quantizing weights (int4) from ${label}…`);
  const t0 = performance.now();
  rt = new QwenWGPU(dev, QWEN25_3B);
  await rt.build(reader, (msg, frac) => log(`weights: ${msg} ${(frac * 100).toFixed(0)}%`));
  window.__rt = rt; window.__tokenizer = tokenizer;
  log(`READY in ${((performance.now() - t0) / 1000).toFixed(1)}s — base loaded once; adapters hot-swap live.`);
  $('go').disabled = false; $('loraFile').disabled = false;
}

// read full logits (for temperature sampling); greedy uses the GPU argmax path.
async function readLogits() {
  const n = QWEN25_3B.vocabSize;
  const rb = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(rt.s.logits, 0, rb, 0, n * 4); dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const a = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy(); return a;
}
function sample(logits, temperature) {
  let best = 0, bv = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  if (!temperature || temperature <= 0) return best;
  let sum = 0; const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { const e = Math.exp((logits[i] - bv) / temperature); p[i] = e; sum += e; }
  let r = Math.random() * sum, c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (r <= c) return i; } return p.length - 1;
}

// Faithful Qwen2.5 ChatML (matches VibeThinker-3B's chat_template for non-tool messages,
// incl. injecting the default system prompt when none is given). Used as the fallback when
// the tokenizer has no embedded chat_template (e.g. it lives in a separate chat_template.jinja).
function chatML(messages) {
  let s = (messages[0]?.role === 'system') ? '' : '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n';
  for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  return s + '<|im_start|>assistant\n';
}
async function* generate(messages, { maxTokens = 1024, temperature = 0.0, stopIds = [151645, 151643] } = {}) {
  let promptText;
  try { promptText = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true }); }
  catch { promptText = chatML(messages); }
  const ids = tokenizer.encode(promptText);
  // prefill: batched tiled GEMM fast path, with LoRA deltas fused after each projection
  // when an adapter is active; sequential fallback only for prompts above maxPrefillT.
  if (ids.length <= rt.maxPrefillT) rt.prefillBatch(ids);
  else for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
  let pos = ids.length;
  const emit = (id) => tokenizer.decode([id], { skip_special_tokens: true }); // byte-level BPE: per-token decode is exact for ASCII/JSON

  if (temperature > 0) {                                              // sampling: per-token (needs CPU)
    let next = sample(await readLogits(), temperature);
    for (let step = 0; step < maxTokens; step++) {
      if (stopIds.includes(next)) break;
      const d = emit(next); if (d) yield d;
      rt.token(next, pos); pos++; next = sample(await readLogits(), temperature);
    }
    return;
  }
  // greedy: GPU-resident batched decode — argmax->embed stays on the GPU, sync once per batch.
  const first = await rt.argmaxLogits();                             // leaves s.amax = first
  if (stopIds.includes(first)) return;
  { const d = emit(first); if (d) yield d; }
  let emitted = 1;
  while (emitted < maxTokens && pos < rt.maxCtx) {                   // stop before the KV cache fills
    const K = Math.min(rt.MAXBATCH, maxTokens - emitted, rt.maxCtx - pos);
    const batch = await rt.decodeBatch(pos, K); pos += K;            // K new tokens, one readback
    let stop = false;
    for (const id of batch) {
      if (stopIds.includes(id)) { stop = true; break; }
      const d = emit(id); if (d) yield d; emitted++;
      if (emitted >= maxTokens) { stop = true; break; }
    }
    if (stop) break;
  }
}

async function runTriage() {
  if (!rt) return;
  $('go').disabled = true; $('out').textContent = '';
  const node = document.createTextNode(''); $('out').appendChild(node);   // O(n) streaming, not O(n^2)
  const adapterName = $('adapter').value;
  if (adapters[adapterName]) rt.setLora(adapters[adapterName]); else rt.clearLora();
  log(`generating (adapter=${adapterName})…`);
  const messages = [{ role: 'system', content: SYS }, { role: 'user', content: $('report').value }];
  const t0 = performance.now(); let n = 0;
  // reasoning model thinks long; let it use the context window (decode stops at EOS or maxCtx).
  for await (const delta of generate(messages, { maxTokens: rt.maxCtx, temperature: 0.0 })) { node.appendData(delta); n++; }
  const dt = (performance.now() - t0) / 1000;
  log(`done: ${n} tokens in ${dt.toFixed(1)}s (${(n / dt).toFixed(1)} tok/s) adapter=${adapterName}`);
  $('go').disabled = false;
}

window.addEventListener('DOMContentLoaded', () => {
  $('load').onclick = () => loadWith(urlReader($('modelUrl').value.trim()), $('modelUrl').value.trim())
    .catch(e => { log('ERROR: ' + e.message); console.error(e); });
  const hfBtn = $('loadHF');
  if (hfBtn) hfBtn.onclick = () => {
    const repo = $('hfRepo').value.trim(); const token = ($('hfToken')?.value || '').trim();
    if (!repo) return log('enter a Hugging Face repo id, e.g. WeiboAI/VibeThinker-3B');
    loadWith(hfReader(repo, token), 'HF: ' + repo).catch(e => { log('ERROR: ' + e.message + ' (private/gated repo? add a token)'); console.error(e); });
  };
  const mf = $('modelFiles');
  if (mf) mf.onchange = (ev) => {
    const files = [...ev.target.files]; if (!files.length) return;
    const map = {}; for (const f of files) map[f.name] = f;   // fileReader picks by basename
    loadWith(fileReader(map), `${files.length} local files`).catch(e => { log('ERROR: ' + e.message); console.error(e); });
  };
  $('go').onclick = () => runTriage().catch(e => { log('ERROR: ' + e.message); console.error(e); });
  const addAdapter = (name, modules, where) => {
    adapters[name] = { modules };
    const opt = document.createElement('option'); opt.value = name; opt.textContent = `${name} (${Object.keys(modules).length} modules${where ? ', ' + where : ''})`;
    $('adapter').appendChild(opt); $('adapter').value = name;
    log(`LoRA "${name}" loaded (${Object.keys(modules).length} modules) — Triage to hot-swap.`);
  };
  $('loraFile').onchange = async (ev) => {
    try { const { name, modules } = await loadLoraAdapterGPU(dev, [...ev.target.files], QWEN25_3B); addAdapter(name, modules); }
    catch (e) { log('LoRA load error: ' + e.message); console.error(e); }
  };
  const hfLoraBtn = $('loadHFLora');
  if (hfLoraBtn) hfLoraBtn.onclick = async () => {
    if (!dev) return log('load a model first, then load a LoRA adapter');
    const repo = ($('hfLora')?.value || '').trim(); const token = ($('hfToken')?.value || '').trim();
    if (!repo) return log('enter a Hugging Face LoRA adapter repo id');
    try {
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const grab = async (n) => { const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${n}`, { headers }); if (!r.ok) return null; const buf = await r.arrayBuffer(); return { name: n, async text() { return new TextDecoder().decode(buf); }, async arrayBuffer() { return buf; } }; };
      const st = (await grab('adapters.safetensors')) || (await grab('adapter_model.safetensors'));
      if (!st) throw new Error('no adapters.safetensors / adapter_model.safetensors in ' + repo);
      const cfg = await grab('adapter_config.json');
      const { name, modules } = await loadLoraAdapterGPU(dev, cfg ? [st, cfg] : [st], QWEN25_3B);
      addAdapter(repo.split('/').pop() || name, modules, 'HF');
    } catch (e) { log('HF LoRA error: ' + e.message + (token ? '' : ' (private/gated? add a token)')); console.error(e); }
  };
});
