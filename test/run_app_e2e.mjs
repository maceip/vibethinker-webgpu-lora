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
const CANARY = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const b = await chromium.launch({ executablePath: CANARY, headless: false, args:['--enable-unsafe-webgpu','--enable-features=WebGPU','--use-angle=metal','--no-first-run'] });
const p = await b.newPage();
p.on('console', m => { const t=m.text(); if(/GPUERR|ERROR/.test(t)) console.log('PAGE:', t.slice(0,200)); });
p.on('pageerror', e => console.log('PAGEERR', String(e).slice(0,200)));
await p.goto('http://localhost:8013/index.html', { waitUntil:'domcontentloaded' });
await p.click('#load');
const waitStatus = async (pred, max=180000) => { const t0=Date.now(); while(Date.now()-t0<max){ const s=await p.$eval('#status',e=>e.textContent); if(pred(s)) return s; await p.waitForTimeout(1500);} throw new Error('timeout waiting; last status='+await p.$eval('#status',e=>e.textContent)); };
await waitStatus(s=>s.startsWith('READY'));
console.log('E2E', await p.$eval('#status', e=>e.textContent));
// base triage with progress polling
await p.click('#go');
const t0=Date.now();
while(Date.now()-t0<150000){ const s=await p.$eval('#status',e=>e.textContent); const olen=(await p.$eval('#out',e=>e.textContent)).length; process.stdout.write(`\r  base: ${s.slice(0,60)} | out=${olen} chars   `); if(s.startsWith('done')) break; await p.waitForTimeout(2000);} console.log();
const baseOut = await p.$eval('#out', e=>e.textContent);
console.log('E2E base done:', await p.$eval('#status',e=>e.textContent));
const jsonLine = baseOut.trim().split('\n').filter(l=>l.trim().startsWith('{')).pop() || baseOut.slice(-200);
console.log('E2E base JSON:', jsonLine.slice(0,260));
await b.close();
