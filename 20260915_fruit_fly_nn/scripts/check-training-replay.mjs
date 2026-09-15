import {chromium} from '@playwright/test';
import {writeFileSync,mkdirSync}from'node:fs';
const url=process.argv[2]??'http://127.0.0.1:5173';
// This checker connects to an existing server; it never launches one.
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1600,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(url);await page.getByRole('button',{name:'Load real training run + 3D neurons',exact:true}).click();
await page.waitForFunction(()=>window.__cbc?.getMetrics().graphSource==='malecns');
await page.getByRole('button',{name:'Inspect 3D neurons',exact:true}).click();
await page.waitForFunction(()=>window.__cbc.getMetrics().replayFinished,undefined,{timeout:60000});
const before=await page.evaluate(()=>window.__cbc.getMetrics());
await page.getByRole('button',{name:'After training',exact:true}).click();
await page.waitForFunction(()=>window.__cbc.getMetrics().replayFinished,undefined,{timeout:60000});
const after=await page.evaluate(()=>window.__cbc.getMetrics());
mkdirSync('outputs/review-malecns-phase-lock-20260915',{recursive:true});
await page.screenshot({path:'outputs/review-malecns-phase-lock-20260915/neural-replay.png'});
writeFileSync('outputs/review-malecns-phase-lock-20260915/browser-replay.json',JSON.stringify({before,after,errors},null,2));
await browser.close();console.log({before,after,errors});
if(errors.length||after.neural3dNodes!==4268||after.replayMeanIntensity<before.replayMeanIntensity+.3)throw Error('Replay validation failed');
