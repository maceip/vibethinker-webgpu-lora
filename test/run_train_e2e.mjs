/*
 * Driver for the end-to-end in-browser LoRA training ladder (test/train_e2e.js).
 * Loads the real model in Chrome Canary, runs the four rungs, captures TRAIN/RUNG
 * lines, and saves the emitted Rung-1 token cases to test/rung1_cases.json so the
 * torch reference (test/torch_ce_ref.py) can cross-check the loss.
 */
import { chromium } from 'playwright';
import { existsSync, writeFileSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linuxChrome = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linuxChrome) ? linuxChrome : existsSync(macCanary) ? macCanary : undefined);
const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'],
});
const p = await b.newPage();
const L = [];
p.on('console', (m) => {
  const t = m.text();
  L.push(t);
  if (t.startsWith('RUNG1_CASES ')) {
    try {
      writeFileSync('test/rung1_cases.json', t.slice('RUNG1_CASES '.length));
      console.log('DRIVER: saved test/rung1_cases.json');
    } catch (e) {
      console.log('DRIVER: failed to save cases', e.message);
    }
    return;
  }
  if (/^(TRAIN|RUNG|PAGEERR)/.test(t)) console.log('BROWSER:', t.slice(0, 400));
});
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 400)));
await p.goto('http://localhost:8013/test/train.html');
const MAX = Number(process.env.LADDER_TIMEOUT_MS || 1800000); // 30 min
const t0 = Date.now();
while (Date.now() - t0 < MAX) {
  if (L.some((l) => l.includes('TRAIN DONE') || l.startsWith('TRAIN FATAL'))) break;
  await p.waitForTimeout(1000);
}
await b.close();
const fatal = L.some((l) => l.startsWith('TRAIN FATAL'));
console.log(fatal ? 'LADDER FATAL' : 'LADDER FINISHED');
process.exit(fatal ? 1 : 0);
