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
import { existsSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linuxChrome = '/usr/local/bin/google-chrome';
const winChrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const CHROME =
  process.env.CHROME_PATH ||
  (existsSync(linuxChrome) ? linuxChrome : existsSync(macCanary) ? macCanary : existsSync(winChrome) ? winChrome : undefined);
const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run'],
});
const p = await b.newPage();
const L = [];
p.on('console', (m) => {
  const t = m.text();
  L.push(t);
  console.log('BROWSER:', t);
});
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 300)));
await p.goto('http://localhost:8013/test/backward.html');
const t0 = Date.now();
while (Date.now() - t0 < 120000) {
  if (L.some((l) => l.includes('BWD-GPU DONE') || l.includes('BWD-GPU ERROR'))) break;
  await p.waitForTimeout(300);
}
await b.close();
const failed = L.some((l) => l.includes('BWD-GPU FAILED') || l.includes('BWD-GPU ERROR') || l.includes('BWD-GPU FAIL '));
process.exit(failed ? 1 : 0);
