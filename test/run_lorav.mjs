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
const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run']
});
const p = await b.newPage(); const L=[];
p.on('console',m=>{const t=m.text(); L.push(t); console.log('BROWSER:', t);});
p.on('pageerror',e=>console.log('PAGEERR',String(e).slice(0,300)));
await p.goto('http://localhost:8013/test/lorav.html');
const t0=Date.now(); while(Date.now()-t0<240000){ if(L.some(l=>l.includes('VWG DONE')||l.includes('VWG ERROR')))break; await p.waitForTimeout(500);} await b.close();
