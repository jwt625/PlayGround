/** Public manufacturer reference cache. No runtime redistribution rights implied. */
import {chromium} from '@playwright/test';
import {mkdirSync,writeFileSync,readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
const root='assets/reference/hardware', dir=root+'/vendor';mkdirSync(dir,{recursive:true});
const records=existsSync(root+'/manifest.json')?JSON.parse(readFileSync(root+'/manifest.json','utf8')).records.filter(r=>r.file):[];
function record(file,url,kind){const b=readFileSync(file);const relative=file.replace(root+'/','');const existing=records.findIndex(r=>r.file===relative);if(existing>=0)records.splice(existing,1);records.push({file:relative,url,kind,bytes:b.length,sha256:createHash('sha256').update(b).digest('hex'),rights:'Manufacturer reference only; redistribution license not established',retrieved:new Date().toISOString()});}
const browser=await chromium.launch({headless:true});
for(const part of (process.argv.includes('--pdf-only')?[]:['ADAFCPM2','EDFA100P','F220APC-1550','30126C3','LN65S-FC'])){
 const page=await browser.newPage({acceptDownloads:true});const url='https://www.thorlabs.com/item/'+part;
 try{await page.goto(url,{waitUntil:'networkidle',timeout:45000});const body=await page.locator('body').innerText();writeFileSync(dir+'/'+part+'.txt',body);record(dir+'/'+part+'.txt',url,'rendered manufacturer page');
 const buttons=page.getByRole('button',{name:/Open .* as part of (CAD PDF|Manual)|Download .* as part of Step/});
 for(let i=0;i<await buttons.count();i++){
  const button=buttons.nth(i);const label=await button.getAttribute('aria-label');
  const downloadPromise=page.waitForEvent('download',{timeout:5000}).catch(()=>null);
  const responses=[];const listener=r=>{if(/\.pdf(?:\?|$)|\.step(?:\?|$)/i.test(r.url()))responses.push(r)};page.on('response',listener);
  await button.click();const download=await downloadPromise;
  if(download){const file=dir+'/'+part+'-'+download.suggestedFilename();await download.saveAs(file);record(file,url,'manufacturer downloadable CAD; use product-page Step button');}
  for(const response of responses){const b=await response.body().catch(()=>null);if(b?.subarray(0,5).toString()==='%PDF-'){const file=dir+'/'+part+'-'+(label.includes('Manual')?'manual':'drawing')+'.pdf';writeFileSync(file,b);record(file,response.url(),'manufacturer PDF');}}
  page.off('response',listener);
 }
 console.log(part,'done');
 }catch(e){records.push({url,error:String(e)});console.log(part,String(e))}await page.close();
}
const requestContext=await browser.newContext();
for(const [name,url]of[
 ['phase-modulator-lab-fact.pdf','https://media.thorlabs.com/contentassets/5924f547c0be4d98a41aee67e7a83d0c/fiber_eo_phase_modulator_lab_fact.pdf?v=1116113902'],
 ['hammond-1455N1601.pdf','https://www.hammfg.com/files/parts/pdf/1455N1601.pdf'],
 ['hammond-1455NHD1601.pdf','https://www.hammfg.com/files/parts/pdf/1455NHD1601.pdf']]){
 try{const response=await requestContext.request.get(url,{timeout:30000});const b=await response.body();if(!response.ok()||b.subarray(0,5).toString()!=='%PDF-')throw Error('Not a valid PDF: '+response.status());const file=dir+'/'+name;writeFileSync(file,b);record(file,url,'manufacturer PDF');console.log(name,b.length)}catch(e){records.push({url,error:String(e)});console.log(name,String(e))}
}
await browser.close();
// Preserve already cached CAD from a previous successful collection.
for(const part of ['ADAFCPM2','EDFA100P']){const file=dir+'/'+part+'-'+part+'-Step.step';if(existsSync(file)&&!records.some(r=>r.file===file.replace(root+'/','')))record(file,'https://www.thorlabs.com/item/'+part,'manufacturer STEP downloaded through product-page Step button');}
writeFileSync(root+'/manifest.json',JSON.stringify({purpose:'Local design reference cache; original generic runtime assets are separate',records},null,2)+'\n');
