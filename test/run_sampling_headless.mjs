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
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=WebGPU',
    '--no-first-run',
    '--use-gpu-in-tests'
  ],
});

const page = await browser.newPage();
const lines = [];
page.on('console', m => {
  const t = m.text();
  if (t.startsWith('VWG')) {
    lines.push(t);
    console.log(t);
  }
});
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 300)));

console.log('Navigating to test page in headless mode...');
await page.goto('http://localhost:8013/test/sampling.html', { waitUntil: 'domcontentloaded' });

const t0 = Date.now();
// Wait for up to 5 minutes
while (Date.now() - t0 < 300000) {
  if (lines.some(l => l.includes('VWG DONE') || l.startsWith('VWG ERROR'))) break;
  await page.waitForTimeout(1000);
}

await browser.close();
if (!lines.some(l => l.includes('SAMPLING PASS'))) {
  console.log('Test did not report SAMPLING PASS');
  process.exitCode = 1;
} else {
  console.log('Headless sampling test passed!');
}
