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
const CHROME = process.env.CHROME_PATH || (existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary');
const browser = await chromium.launch({ executablePath: CHROME, headless: false,
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU','--use-angle=metal','--no-first-run',
         '--disable-dawn-features=timestamp_quantization'] });
const page = await browser.newPage();
const lines = [];
page.on('console', m => { const t = m.text(); if (t.startsWith("VWG")) { lines.push(t); console.log(t); } });
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0,300)));
await page.goto('http://localhost:8013/test/prof.html', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
while (Date.now() - t0 < 300000) { if (lines.some(l => l.includes('VWG DONE') || l.startsWith('VWG ERROR'))) break; await page.waitForTimeout(1000); }
await browser.close();
