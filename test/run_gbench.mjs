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
const b = await chromium.launch({ executablePath:'/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary', headless:false, args:['--enable-unsafe-webgpu','--enable-features=WebGPU','--use-angle=metal','--no-first-run','--disable-dawn-features=timestamp_quantization'] });
const p = await b.newPage(); const L=[];
p.on('console',m=>{const t=m.text(); if(t.startsWith('VWG')){L.push(t);console.log(t);}});
p.on('pageerror',e=>console.log('PAGEERR',String(e).slice(0,300)));
await p.goto('http://localhost:8013/test/gbench.html');
const t0=Date.now(); while(Date.now()-t0<60000){ if(L.some(l=>l.includes('DONE')||l.includes('VWG ERROR')))break; await p.waitForTimeout(500);} await b.close();
