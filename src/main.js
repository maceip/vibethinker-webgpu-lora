// Browser harness wiring for the custom WebGPU Qwen2.5 runtime. Core behavior
// lives in small services so UI changes do not accidentally change loading,
// adapter state, prompt formatting, or generation semantics.
import { QWEN25_3B } from './config.js';
import { loadLoraAdapterGPU } from './lora_gpu.js';
import { urlReader, hfReader, fileReader } from './readers.js';
import { AdapterRegistry } from './services/adapter_registry.js';
import { GenerationController } from './services/generation_controller.js';
import { ModelSession } from './services/model_session.js';

const $ = id => document.getElementById(id);
const log = (m) => { const s = $('status'); if (s) s.textContent = m; console.log('[harness]', m); };

const SYS = `You are a senior bug bounty triage analyst. Read the submission and assign exactly ONE disposition from: valid_impactful, valid_low, corroborated_surge, likely_duplicate, out_of_scope, theoretical_no_poc, self_inflicted, accepted_risk, slop. Estimate severity_estimate (critical/high/medium/low/none). Think step by step, then output a SINGLE JSON object on the last line with keys: disposition, severity_estimate, is_duplicate_risk, reasoning, questions_for_researcher, confidence. Output only valid JSON for that object.`;

const session = new ModelSession({ cfg: QWEN25_3B, log });
const adapters = new AdapterRegistry();
const generation = new GenerationController({ session, adapters, systemPrompt: SYS, log });

async function loadWith(reader, label) {
  await session.loadWith(reader, label);
  $('go').disabled = false; $('loraFile').disabled = false;
}

function addAdapterOption(name, modules, where) {
  adapters.add(name, modules);
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = `${name} (${Object.keys(modules).length} modules${where ? ', ' + where : ''})`;
  $('adapter').appendChild(opt); $('adapter').value = name;
  log(`LoRA "${name}" loaded (${Object.keys(modules).length} modules) — Triage to hot-swap.`);
}

async function runTriage() {
  $('go').disabled = true;
  try {
    await generation.runTriage({
      adapterName: $('adapter').value,
      report: $('report').value,
      outputNode: $('out'),
    });
  } finally {
    $('go').disabled = false;
  }
}

async function fetchHfAdapterFiles(repo, token) {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const grab = async (n) => {
    const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${n}`, { headers });
    if (!r.ok) return null;
    const buf = await r.arrayBuffer();
    return { name: n, async text() { return new TextDecoder().decode(buf); }, async arrayBuffer() { return buf; } };
  };
  const st = (await grab('adapters.safetensors')) || (await grab('adapter_model.safetensors'));
  if (!st) throw new Error('no adapters.safetensors / adapter_model.safetensors in ' + repo);
  const cfg = await grab('adapter_config.json');
  return cfg ? [st, cfg] : [st];
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
    const map = {}; for (const f of files) map[f.name] = f;
    loadWith(fileReader(map), `${files.length} local files`).catch(e => { log('ERROR: ' + e.message); console.error(e); });
  };

  $('go').onclick = () => runTriage().catch(e => { log('ERROR: ' + e.message); console.error(e); });

  $('loraFile').onchange = async (ev) => {
    try {
      const { name, modules } = await loadLoraAdapterGPU(session.dev, [...ev.target.files], QWEN25_3B);
      addAdapterOption(name, modules);
    } catch (e) { log('LoRA load error: ' + e.message); console.error(e); }
  };

  const hfLoraBtn = $('loadHFLora');
  if (hfLoraBtn) hfLoraBtn.onclick = async () => {
    if (!session.dev) return log('load a model first, then load a LoRA adapter');
    const repo = ($('hfLora')?.value || '').trim(); const token = ($('hfToken')?.value || '').trim();
    if (!repo) return log('enter a Hugging Face LoRA adapter repo id');
    try {
      const files = await fetchHfAdapterFiles(repo, token);
      const { name, modules } = await loadLoraAdapterGPU(session.dev, files, QWEN25_3B);
      addAdapterOption(repo.split('/').pop() || name, modules, 'HF');
    } catch (e) { log('HF LoRA error: ' + e.message + (token ? '' : ' (private/gated? add a token)')); console.error(e); }
  };
});
