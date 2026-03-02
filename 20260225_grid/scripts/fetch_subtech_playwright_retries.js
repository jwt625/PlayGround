#!/usr/bin/env node
const fs=require('fs'); const path=require('path');
const { firefox } = require('playwright');

const targets=[
  ['bess','tesla_q4_2025_deployments','https://ir.tesla.com/press-release/tesla-fourth-quarter-2025-production-deliveries-deployments'],
  ['bess','fluence_q3_2025','https://ir.fluenceenergy.com/news-releases/news-release-details/fluence-energy-inc-reports-third-quarter-2025-results-reaffirms/'],
  ['solar','first_solar_2024_results_2025_guidance','https://www.businesswire.com/news/home/20250225031936/en/First-Solar-Inc.-Announces-Fourth-Quarter-and-Full-Year-2024-Financial-Results-and-2025-Guidance'],
  ['bess','tesla_megafactory','https://www.tesla.com/megafactory']
];

function clean(s){return (s||'').replace(/\u00a0/g,' ').replace(/[ \t]+/g,' ').replace(/\n{3,}/g,'\n\n').trim();}

(async()=>{
  const browser=await firefox.launch({headless:true});
  const context=await browser.newContext({ignoreHTTPSErrors:true, viewport:{width:1440,height:1800}});
  const out=[];
  for (const [cat,key,url] of targets){
    const ts=new Date().toISOString();
    const rawDir=path.join('references','raw','subtech',cat); fs.mkdirSync(rawDir,{recursive:true});
    const mdDir=path.join('references','md','subtech'); fs.mkdirSync(mdDir,{recursive:true});
    const page=await context.newPage();
    try{
      const resp=await page.goto(url,{waitUntil:'domcontentloaded',timeout:90000});
      await page.waitForTimeout(2500);
      const finalUrl=page.url();
      const status=resp?resp.status():null;
      const html=await page.content();
      const text=await page.evaluate(()=>{
        const root=document.querySelector('article')||document.querySelector('main')||document.body;
        return (root?.innerText)||document.body?.innerText||'';
      });
      const rawPath=path.join(rawDir,`${key}.html`); fs.writeFileSync(rawPath,html,'utf8');
      const mdPath=path.join(mdDir,`${cat}__${key}.md`);
      fs.writeFileSync(mdPath,[
        '---',
        `category: ${cat}`,
        `key: ${key}`,
        `url: ${url}`,
        `final_url: ${finalUrl}`,
        `retrieved_at_utc: ${ts}`,
        `source_type: html`,
        `source_method: playwright_chromium`,
        `http_status: ${status}`,
        `raw_path: ${rawPath}`,
        '---','',clean(text),''].join('\n'),'utf8');
      out.push({category:cat,key,url,status:'ok',http_status:status,raw_path:rawPath,md_path:mdPath,error:null});
      console.log('[ok]',cat,key,status);
    }catch(e){
      const mdPath=path.join('references','md','subtech',`${cat}__${key}.md`);
      fs.writeFileSync(mdPath,[
        '---',`category: ${cat}`,`key: ${key}`,`url: ${url}`,`retrieved_at_utc: ${ts}`,'status: failed',`error: ${String(e).replace(/\n/g,' ')}`,'---',''].join('\n'),'utf8');
      out.push({category:cat,key,url,status:'failed',http_status:null,raw_path:null,md_path:mdPath,error:String(e)});
      console.log('[failed]',cat,key,e.message);
    }finally{await page.close();}
  }
  await browser.close();
  const manifest='references/manifest.subtech.playwright_retries.json';
  fs.writeFileSync(manifest,JSON.stringify({generated_at_utc:new Date().toISOString(),total:out.length,ok:out.filter(x=>x.status==='ok').length,failed:out.filter(x=>x.status!=='ok').length,items:out},null,2));
  console.log('Manifest written:',manifest);
})();
