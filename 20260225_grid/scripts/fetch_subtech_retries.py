#!/usr/bin/env python3
import datetime as dt, os, re, ssl, json
from html import unescape
from urllib.request import Request, urlopen

TARGETS=[
 ("bess","tesla_q4_2025_deployments","https://ir.tesla.com/press-release/tesla-fourth-quarter-2025-production-deliveries-deployments"),
 ("bess","fluence_q3_2025","https://ir.fluenceenergy.com/news-releases/news-release-details/fluence-energy-inc-reports-third-quarter-2025-results-reaffirms/"),
 ("solar","first_solar_2024_results_2025_guidance","https://www.businesswire.com/news/home/20250225031936/en/First-Solar-Inc.-Announces-Fourth-Quarter-and-Full-Year-2024-Financial-Results-and-2025-Guidance"),
 ("solar","canadian_solar_20f","https://www.sec.gov/Archives/edgar/data/1375877/000141057825001046/csiq-20241231x20f.htm"),
]

RAW='references/raw/subtech'
MD='references/md/subtech'
LOG='references/manifest.subtech.retries.json'


def h2t(b):
    h=b.decode('utf-8','ignore')
    h=re.sub(r'<script[\\s\\S]*?</script>',' ',h,flags=re.I)
    h=re.sub(r'<style[\\s\\S]*?</style>',' ',h,flags=re.I)
    h=re.sub(r'</(p|div|h1|h2|h3|h4|h5|h6|li|tr|section|article|br)>','\n',h,flags=re.I)
    t=re.sub(r'<[^>]+>',' ',h)
    t=unescape(t)
    t=re.sub(r'[ \t\r\f\v]+',' ',t)
    t=re.sub(r'\n\s*\n+','\n\n',t)
    return '\n\n'.join([x.strip() for x in t.splitlines() if x.strip()])

out=[]
for c,k,u in TARGETS:
    ts=dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(os.path.join(RAW,c),exist_ok=True)
    os.makedirs(MD,exist_ok=True)
    md=os.path.join(MD,f'{c}__{k}.md')
    try:
        headers={
            'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
            'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        if 'sec.gov' in u:
            headers['User-Agent']='wentaojiang research project wentao@example.com'
        req=Request(u,headers=headers)
        with urlopen(req,timeout=30,context=ssl.create_default_context()) as r:
            data=r.read(); status=getattr(r,'status',None); final=r.geturl()
        rp=os.path.join(RAW,c,f'{k}.html')
        open(rp,'wb').write(data)
        txt=h2t(data)
        open(md,'w',encoding='utf-8').write(f"---\ncategory: {c}\nkey: {k}\nurl: {u}\nfinal_url: {final}\nretrieved_at_utc: {ts}\nsource_type: html\nraw_path: {rp}\n---\n\n{txt}\n")
        out.append({'category':c,'key':k,'url':u,'status':'ok','http_status':status,'raw_path':rp,'md_path':md,'error':None})
        print('[ok]',c,k)
    except Exception as e:
        out.append({'category':c,'key':k,'url':u,'status':'failed','http_status':None,'raw_path':None,'md_path':md,'error':f'{type(e).__name__}: {e}'})
        print('[failed]',c,k,type(e).__name__,e)

json.dump({'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'total':len(out),'ok':sum(1 for x in out if x['status']=='ok'),'failed':sum(1 for x in out if x['status']!='ok'),'items':out},open(LOG,'w'),indent=2)
print('Manifest written:',LOG)
