import { chromium } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1200,height:850}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:8766/assets/preview/');await page.waitForFunction(()=>window.assetReady,{timeout:60000});
mkdirSync('assets/generated/flybody/validation',{recursive:true});
for(const view of ['side','top','front']){await page.locator('#'+view).click();await page.waitForTimeout(200);await page.screenshot({path:`assets/generated/flybody/validation/${view}.png`});}
await page.locator('#animate').click();await page.waitForTimeout(250);await page.screenshot({path:'assets/generated/flybody/validation/wing-cycle.png'});
const info=await page.evaluate(()=>window.assetInfo);writeFileSync('assets/generated/flybody/validation/browser-report.json',JSON.stringify({info,errors,browser:browser.version()},null,2));await browser.close();if(errors.length)throw Error(errors.join('\n'));console.log(info);
