import {DEFS,center,type Entity,type World} from './sim/world';
import {footprint} from './sim/geometry';
export type Condition=0|1|2|3;
export type WorkState='running'|'ready'|'no-power'|'starved'|'blocked'|'exhausted'|'no-field'|'uncooled'|'tripped'|'destroyed'|'firing';
export interface EquipmentState {condition:Condition;work:WorkState;powered:boolean;passive:boolean;field:boolean;moving:boolean;hot:boolean;smoke:boolean;label:string;damageLabel:string}
/** Presentation only: independent service, work, thermal and integrity signals. */
export function equipmentState(w:World,e:Entity):EquipmentState {
 const condition:Condition=e.health<=0?3:e.health<65?2:e.health<100?1:0;
 const passive=e.kind==='junction'||e.kind==='tuner';
 const field=(w.stats.network.ports[e.id]??[]).some(p=>p.incoming>.1||p.outgoing>.1);
 let work:WorkState='ready',moving=false;
 if(condition===3)work='destroyed';else if(e.tripped)work='tripped';
 else if(!e.powered&&!passive)work=e.kind==='dump'&&field?'uncooled':'no-power';
 else if(e.kind==='extractor'){
  const f=footprint(e),deposit=w.deposits.find(d=>d.remaining>0&&(d.kind==='ore'||w.frontier)&&e.x<d.x+d.w&&e.x+f.w>d.x&&e.y<d.y+d.h&&e.y+f.h>d.y);
  work=!deposit?'exhausted':deposit.kind==='ore'&&e.ore>=20?'blocked':'running';moving=work==='running';
 }else if(e.kind==='assembler'){work=e.ore<2?'starved':'running';moving=work==='running';}
 else if(e.kind==='generator'){moving=w.links.some(l=>l.type==='power'&&l.a.node===e.id&&w.entities.some(load=>load.id===l.b.node&&load.powered&&DEFS[load.kind].watts>0));work=moving?'running':'ready';}
 else if(e.kind==='sentry'){const p=center(e);moving=w.ecology.shots.some(s=>s.ttl>0&&Math.hypot(s.from.x-p.x,s.from.y-p.y)<.01);work=moving?'firing':'ready';}
 else if(e.kind==='dump'){moving=field||e.temperature>30;work=moving?'running':'ready';}
 else work=field?'running':'no-field';
 const labels:Record<WorkState,string>={running:passive?'FIELD ACTIVE':e.kind==='dump'?'COOLING':'RUNNING',ready:'READY',starved:'WAITING FOR ORE',blocked:'OUTPUT FULL',exhausted:'NO RESOURCE', 'no-power':'POWER OFF','no-field':'NO FIELD',uncooled:'UNCOOLED',tripped:'TRIPPED',destroyed:'WRECK',firing:'FIRING'};
 return {condition,work,powered:e.powered&&!e.tripped&&condition!==3,passive,field:field&&condition!==3,moving,hot:e.temperature>65&&condition!==3,smoke:e.temperature>105&&condition!==3,label:labels[work],damageLabel:condition===1?'LIGHT DAMAGE':condition===2?'SEVERE DAMAGE':condition===3?'DESTROYED':''};
}
