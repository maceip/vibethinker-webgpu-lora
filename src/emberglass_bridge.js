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

// Embeddable Emberglass runtime for third-party demos (e.g. VibeBounty GitHub Pages).
// Exports a small chat-completions-shaped API over the custom WebGPU harness.
import { QWEN25_3B } from './config.js';
import { hfReader, urlReader } from './readers.js';
import { loadLoraAdapterGPU } from './lora_gpu.js';
import { ModelSession } from './services/model_session.js';

function fileLike(name, buf) {
  return {
    name,
    async text() {
      return new TextDecoder().decode(buf);
    },
    async arrayBuffer() {
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    },
  };
}

async function fetchAdapterFiles(baseUrl, headers = {}) {
  const base = baseUrl.replace(/\/$/, '');
  const names = ['adapter_config.json', 'adapters.safetensors', 'adapter_model.safetensors'];
  const out = [];
  for (const name of names) {
    const res = await fetch(`${base}/${name}`, { headers });
    if (!res.ok) continue;
    out.push(fileLike(name, new Uint8Array(await res.arrayBuffer())));
  }
  if (!out.some((f) => f.name.endsWith('.safetensors'))) throw new Error(`no adapter weights under ${base}`);
  if (!out.some((f) => f.name === 'adapter_config.json')) throw new Error(`no adapter_config.json under ${base}`);
  return out;
}

/**
 * @param {object} opts
 * @param {string} [opts.hfRepo] - Hugging Face model repo (default WeiboAI/VibeThinker-3B)
 * @param {string} [opts.hfToken] - optional HF token for gated repos
 * @param {string} [opts.modelUrl] - same-origin model base (overrides hfRepo when set)
 * @param {string} [opts.loraUrl] - same-origin LoRA adapter directory
 * @param {string} [opts.loraRepo] - Hugging Face LoRA repo id
 * @param {(msg:string)=>void} [opts.log]
 * @param {(msg:string, frac:number)=>void} [opts.onProgress]
 * @param {object} [opts.runtimeOptions] - passed to QwenWGPU
 */
export async function createEmberglassEngine(opts = {}) {
  const log = opts.log || (() => {});
  const onProgress = opts.onProgress || (() => {});
  const session = new ModelSession({
    cfg: QWEN25_3B,
    log,
    runtimeOptions: {
      decodeBatchSize: 'auto',
      decodeBatchMinTokPerSec: 20,
      samplingTopK: 40,
      ...(opts.runtimeOptions || {}),
    },
  });

  let reader, label;
  if (opts.modelUrl) {
    reader = urlReader(opts.modelUrl);
    label = opts.modelUrl;
  } else {
    const repo = opts.hfRepo || 'WeiboAI/VibeThinker-3B';
    reader = hfReader(repo, opts.hfToken || '');
    label = `HF:${repo}`;
  }

  onProgress('streaming model weights', 0.05);
  await session.loadWith(reader, label);

  if (opts.loraUrl) {
    onProgress('loading LoRA adapter', 0.92);
    const files = await fetchAdapterFiles(opts.loraUrl);
    const lora = await loadLoraAdapterGPU(session.dev, files, QWEN25_3B);
    session.rt.setLora(lora);
    log(`LoRA loaded from ${opts.loraUrl} (${Object.keys(lora.modules).length} modules)`);
  } else if (opts.loraRepo) {
    onProgress('loading LoRA from Hugging Face', 0.92);
    const headers = opts.hfToken ? { Authorization: `Bearer ${opts.hfToken}` } : {};
    const grab = async (n) => {
      const r = await fetch(`https://huggingface.co/${opts.loraRepo}/resolve/main/${n}`, { headers });
      if (!r.ok) return null;
      return fileLike(n, new Uint8Array(await r.arrayBuffer()));
    };
    const st = (await grab('adapters.safetensors')) || (await grab('adapter_model.safetensors'));
    if (!st) throw new Error(`no adapter weights in HF repo ${opts.loraRepo}`);
    const cfg = await grab('adapter_config.json');
    const lora = await loadLoraAdapterGPU(session.dev, cfg ? [st, cfg] : [st], QWEN25_3B);
    session.rt.setLora(lora);
    log(`LoRA loaded from HF:${opts.loraRepo} (${Object.keys(lora.modules).length} modules)`);
  }

  onProgress('ready', 1);

  return {
    label: 'emberglass (custom webgpu)',
    async chatComplete(messages, { maxTokens = 4096, temperature = 0 } = {}) {
      let out = '';
      for await (const delta of session.generate(messages, { maxTokens, temperature })) out += delta;
      return out;
    },
    dispose() {
      session.rt = null;
      session.dev = null;
    },
  };
}
