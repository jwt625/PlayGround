#!/usr/bin/env python3
import concurrent.futures, datetime as dt, json, os, re, ssl, time
from dataclasses import dataclass
from html import unescape
from urllib.request import Request, urlopen
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

SOURCES=[
 ("bess","nrel_atb_utility_battery_2024","https://atb.nrel.gov/electricity/2024/2023/utility-scale_battery_storage"),
 ("solar","nrel_atb_utility_pv_2024","https://atb.nrel.gov/electricity/2024/utility-scale_pv"),
 ("bess","nrel_battery_cost_proj_2025","https://research-hub.nrel.gov/en/publications/cost-projections-for-utility-scale-battery-storage-2025-update/"),
 ("turbine","ge_lm6000_fast_start","https://www.gevernova.com/gas-power/services/gas-turbines/upgrades/lm6000-fast-start"),
 ("turbine","mhi_m501j_specs","https://power.mhi.com/products/gasturbines/lineup/m501j"),
 ("turbine","siemens_sgt800_press","https://press.siemens.com/global/en/feature/siemens-launches-performance-enhancement-select-sgt-800-gas-turbines"),
 ("sofc","bloom_hydrogen_sofc_efficiency","https://investor.bloomenergy.com/press-releases/press-release-details/2024/Bloom-Energy-Announces-Hydrogen-Solid-Oxide-Fuel-Cell-with-60-Electrical-Efficiency-and-90-High-Temperature-Combined-Heat-and-Power-Efficiency/default.aspx"),
 ("sofc","bloom_energy_server_datasheet_page","https://www.bloomenergy.com/resource/bloom-energy-server/"),
]
RAW='references/raw/subtech'
MD='references/md/subtech'
MAN='references/manifest.subtech.additional.json'

@dataclass
class R:
    category:str; key:str; url:str; status:str; http_status:int|None; source_type:str; raw_path:str|None; md_path:str|None; error:str|None

def html2txt(b:bytes)->str:
    h=b.decode('utf-8','ignore')
    h=re.sub(r'<script[\s\S]*?</script>',' ',h,flags=re.I)
    h=re.sub(r'<style[\s\S]*?</style>',' ',h,flags=re.I)
    h=re.sub(r'</(p|div|h1|h2|h3|h4|h5|h6|li|tr|section|article|br)>','\n',h,flags=re.I)
    t=re.sub(r'<[^>]+>',' ',h)
    t=unescape(t)
    t=re.sub(r'[ \t\r\f\v]+',' ',t)
    t=re.sub(r'\n\s*\n+','\n\n',t)
    return '\n\n'.join([x.strip() for x in t.splitlines() if x.strip()])

def pdf2txt(p):
    if PdfReader is None: return ''
    try:
      rd=PdfReader(p); return '\n\n'.join([(pg.extract_text() or '').strip() for pg in rd.pages if (pg.extract_text() or '').strip()])
    except: return ''

def fetch(u):
    ua='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36'
    e=None
    for i in range(2):
      try:
        req=Request(u,headers={'User-Agent':ua,'Accept':'text/html,application/pdf,*/*'})
        with urlopen(req,timeout=20,context=ssl.create_default_context()) as r:
          return getattr(r,'status',None),(r.headers.get('Content-Type','') or '').lower(),r.geturl(),r.read()
      except Exception as ex:
        e=ex; time.sleep(1+i)
    raise e

def proc(item):
    c,k,u=item; ts=dt.datetime.now(dt.timezone.utc).isoformat(); os.makedirs(os.path.join(RAW,c),exist_ok=True); os.makedirs(MD,exist_ok=True)
    try:
      s,ct,fu,d=fetch(u); isp=('application/pdf' in ct) or fu.lower().endswith('.pdf'); ext='pdf' if isp else 'html'
      rp=os.path.join(RAW,c,f'{k}.{ext}')
      open(rp,'wb').write(d)
      txt=pdf2txt(rp) if isp else html2txt(d)
      mp=os.path.join(MD,f'{c}__{k}.md')
      open(mp,'w',encoding='utf-8').write(f"---\ncategory: {c}\nkey: {k}\nurl: {u}\nfinal_url: {fu}\nretrieved_at_utc: {ts}\nsource_type: {'pdf' if isp else 'html'}\nraw_path: {rp}\n---\n\n{txt or 'Text extraction unavailable.'}\n")
      return R(c,k,u,'ok',s,'pdf' if isp else 'html',rp,mp,None)
    except Exception as ex:
      mp=os.path.join(MD,f'{c}__{k}.md')
      open(mp,'w',encoding='utf-8').write(f"---\ncategory: {c}\nkey: {k}\nurl: {u}\nretrieved_at_utc: {ts}\nstatus: failed\nerror: {type(ex).__name__}: {ex}\n---\n")
      return R(c,k,u,'failed',None,'unknown',None,mp,f'{type(ex).__name__}: {ex}')

out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  fs=[ex.submit(proc,s) for s in SOURCES]
  for f in concurrent.futures.as_completed(fs):
    r=f.result(); out.append(r); print(f'[{r.status}] {r.category}/{r.key}')
out.sort(key=lambda x:(x.category,x.key))
json.dump({'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'total':len(out),'ok':sum(1 for x in out if x.status=='ok'),'failed':sum(1 for x in out if x.status!='ok'),'items':[x.__dict__ for x in out]},open(MAN,'w'),indent=2)
print('Manifest written:',MAN)
