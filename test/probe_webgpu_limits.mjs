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

import { chromium } from 'playwright';
import http from 'http';
const CANARY = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';

// navigator.gpu only exists in a secure context (a loaded page on localhost)
const server = http.createServer((req, res) => { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end('<!doctype html><html><body>probe</body></html>'); });
await new Promise(r => server.listen(8011, r));

const browser = await chromium.launch({
  executablePath: CANARY, headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run', '--no-default-browser-check'],
});
const page = await browser.newPage();
page.on('console', m => { const t = m.text(); if (/error|fail|gpu/i.test(t)) console.log('[console]', t.slice(0, 160)); });
await page.goto('http://localhost:8011/', { waitUntil: 'domcontentloaded' });

const res = await page.evaluate(async () => {
  if (!navigator.gpu) return { err: 'no navigator.gpu' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { err: 'no adapter' };
  const L = adapter.limits;
  const out = {
    maxBufferSize_GB: (Number(L.maxBufferSize) / 1e9).toFixed(2),
    maxStorageBufferBindingSize_GB: (Number(L.maxStorageBufferBindingSize) / 1e9).toFixed(2),
  };
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    },
  });
  // single big buffer test: embed_tokens f32 = 151936*2048*4 = 1.24GB
  async function tryAlloc(sz) {
    device.pushErrorScope('out-of-memory');
    let b = null;
    try { b = device.createBuffer({ size: sz, usage: GPUBufferUsage.STORAGE }); } catch (e) { return { ok: false, err: String(e).slice(0, 80) }; }
    const err = await device.popErrorScope();
    return { ok: !err, buf: b };
  }
  const embedSz = 151936 * 2048 * 4;
  const big = await tryAlloc(embedSz);
  out.embed_1_24GB_buffer = big.ok;
  if (big.buf) big.buf.destroy();

  // cumulative allocation toward ~14GB in 512MB chunks (f32 3B ≈ 12GB)
  const chunk = 512 * 1024 * 1024;
  const bufs = []; let allocated = 0; let failed = false;
  for (let i = 0; i < 28 && !failed; i++) {
    const a = await tryAlloc(chunk);
    if (!a.ok) { failed = true; break; }
    bufs.push(a.buf); allocated += chunk;
  }
  out.cumulative_allocated_GB = (allocated / 1e9).toFixed(1);
  out.hit_limit_before_14GB = failed;
  for (const b of bufs) b.destroy();
  return out;
});

console.log('WEBGPU_LIMITS ' + JSON.stringify(res, null, 2));
await browser.close(); server.close();
