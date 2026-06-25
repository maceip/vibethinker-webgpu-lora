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
const CHROME = process.env.CHROME_PATH || (existsSync(linuxChrome) ? linuxChrome : (existsSync(macCanary) ? macCanary : (existsSync(winChrome) ? winChrome : undefined)));
const browser = await chromium.launch({ 
  ...(CHROME ? { executablePath: CHROME } : {}), 
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run'] 
});
const page = await browser.newPage();
const lines = [];
page.on('console', m => { const t = m.text(); if (t.startsWith('VWG')) { lines.push(t); console.log(t); } });
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0,400)));
await page.goto('http://localhost:8013/test/deepkernels.html', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
while (Date.now() - t0 < 300000) { if (lines.some(l => l.includes('VWG DONE') || l.startsWith('VWG ERROR'))) break; await page.waitForTimeout(1000); }
await browser.close();
