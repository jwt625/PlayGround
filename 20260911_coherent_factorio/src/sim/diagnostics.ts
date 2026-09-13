import type {World} from './world';
import {DEFS} from './definitions';
import {ports,routeMetrics,footprint} from './geometry';
export interface Issue {id:string;severity:'error'|'warning'|'info';object?:string;title:string;remedy:string}
export function diagnose(w:World):Issue[]{
 const issues:Issue[]=[];const add=(object:string,title:string,remedy:string,severity:Issue['severity']='warning')=>issues.push({id:`${object}:${title}`,object,title,remedy,severity});
 if(w.stats.error)issues.push({id:'network',severity:'error',title:'Field network failed',remedy:w.stats.error});
 for(const e of w.entities){const d=DEFS[e.kind];if(e.health<=0){add(e.id,'Equipment destroyed','Repair for 3 assemblies or recover the wreck.','error');continue;}
  if(e.health<100)add(e.id,'Equipment damaged','Inspect integrity. Repair for 3 assemblies; keep a powered sentry covering this machine before commissioning.');
  if(e.tripped)add(e.id,'Thermal protection tripped','Disconnect the live field input, then repair/reset.','error');
  else if(d.watts>0&&!e.powered)add(e.id,'No electrical power','Use Wire: generator BUS OUT → this machine POWER IN. Check generator capacity.','error');
  if(e.temperature>65&&!e.tripped)add(e.id,'Equipment heating up','Inspect incident power and the cooled dump; keep protection enabled.');
  if(e.kind==='assembler'&&e.ore<2)add(e.id,'Waiting for ore','Connect extractor ORE OUT → assembler ORE IN. Two ore make one assembly.','info');
  if(e.kind==='extractor'){const f=footprint(e);if(!w.deposits.some(p=>p.remaining>0&&e.x<p.x+p.w&&e.x+f.w>p.x&&e.y<p.y+p.h&&e.y+f.h>p.y))add(e.id,'Deposit exhausted','Recover and relocate this extractor to a deposit with material.');if(e.ore>=20)add(e.id,'Output buffer full','Connect a material belt and check the receiving assembler.');}
  for(const [i,p] of ports(e).entries())if(!w.links.some(l=>l.type==='field'&&[l.a,l.b].some(v=>v.node===e.id&&v.port===i))&&(w.stats.network.ports[e.id]?.[i]?.outgoing??0)>.1)add(e.id,`Open field port ${p.id}`,'Connect it to a useful load or a cooled dump to reduce stray radiation.');
 }
 for(const l of w.links){if(l.type==='field'&&routeMetrics(l.path,l.radius).bends.some(b=>b.bad))add(l.a.node,`Tight bends on ${l.id}`,'Inspect this route. Disconnect and rebuild with rounded bends and longer straight approaches.');if(l.type==='material'&&l.packets.length&&l.packets[0]>=routeMetrics(l.path).length)add(l.b.node,'Belt receiver blocked','Restore receiver power or free its ore buffer.');}
 if(w.ecology.defenseReady&&w.ecology.threat>=20)issues.push({id:'wildlife',severity:'warning',object:w.ecology.creatures.find(c=>c.health>0)?.id,title:w.ecology.grace>0?'Wildlife exposure rising — defense grace period':'Wildlife attracted by stray field',remedy:'Reduce open ports and sharp bends. Keep perimeter sentries powered near exposed machines.'});
 return issues.sort((a,b)=>({error:0,warning:1,info:2}[a.severity]-{error:0,warning:1,info:2}[b.severity]));
}
