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

const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linuxChrome = '/usr/local/bin/google-chrome';
const executablePath = process.env.CHROME_PATH || (existsSync(linuxChrome) ? linuxChrome : (existsSync(macCanary) ? macCanary : undefined));
const browser = await chromium.launch({
  ...(executablePath ? { executablePath } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run'],
});
const page = await browser.newPage();
const lines = [];
page.on('console', m => { const t = m.text(); if (t.startsWith('VWG')) { lines.push(t); console.log(t); } });
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 300)));
await page.goto('http://localhost:8013/test/sampling.html', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
while (Date.now() - t0 < 180000) {
  if (lines.some(l => l.includes('VWG DONE') || l.startsWith('VWG ERROR'))) break;
  await page.waitForTimeout(500);
}
await browser.close();
if (!lines.some(l => l.includes('SAMPLING PASS'))) process.exitCode = 1;
