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

import { existsSync } from 'node:fs';
import { chromium } from 'playwright';

const winDev = process.env.LOCALAPPDATA + '\\Google\\Chrome Dev\\Application\\chrome.exe';
const CHROME = process.env.CHROME_PATH || (existsSync(winDev) ? winDev : undefined);
const browser = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run'],
});
const page = await browser.newPage();
const lines = [];
page.on('console', m => { const t = m.text(); if (t.startsWith('VWG')) { lines.push(t); console.log(t); } });
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 300)));
await page.goto('http://localhost:8013/test/perf_ab.html', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
while (Date.now() - t0 < 900000) {
  if (lines.some(l => l.includes('VWG DONE') || l.startsWith('VWG ERROR'))) break;
  await page.waitForTimeout(1000);
}
await browser.close();
if (!lines.some(l => l.includes('VWG DONE'))) process.exitCode = 1;
