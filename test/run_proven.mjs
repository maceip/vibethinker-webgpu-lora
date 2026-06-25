/*
 * Driver for the held-out generalization ("proven") run (test/train_proven.js).
 * Loads the real model in Chrome Canary, trains on the SFT train split, evaluates on
 * a disjoint held-out split, and prints all PROVEN lines. Exits 0 on "PROVEN DONE".
 */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
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
  if (/^(PROVEN|PAGEERR)/.test(t)) console.log('BROWSER:', t.slice(0, 500));
});
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 500)));
await p.goto('http://localhost:8013/test/proven.html');
const MAX = Number(process.env.PROVEN_TIMEOUT_MS || 3600000); // 60 min
const t0 = Date.now();
while (Date.now() - t0 < MAX) {
  if (L.some((l) => l.includes('PROVEN DONE') || l.startsWith('PROVEN FATAL'))) break;
  await p.waitForTimeout(1000);
}
await b.close();
const fatal = L.some((l) => l.startsWith('PROVEN FATAL'));
console.log(fatal ? 'PROVEN RUN FATAL' : 'PROVEN RUN FINISHED');
process.exit(fatal ? 1 : 0);
