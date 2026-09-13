import type {World,Entity} from './world';
import {footprint,ports,routeMetrics,type Point} from './geometry';
export interface Creature extends Point {id:string;health:number;state:'patrol'|'investigate'|'attack'|'flee';heading:number;target:Point;exposure:number}
export interface Ecology {creatures:Creature[];defenseReady:boolean;grace:number;threat:number;shots:{from:Point;to:Point;ttl:number}[]}
export function newEcology():Ecology {return {creatures:[{id:'c1',x:23,y:19,health:30,state:'patrol',heading:0,target:{x:26,y:23},exposure:0},{id:'c2',x:33,y:16,health:30,state:'patrol',heading:2,target:{x:31,y:21},exposure:0},{id:'c3',x:10,y:24,health:30,state:'patrol',heading:0,target:{x:15,y:22},exposure:0}],defenseReady:false,grace:30,threat:0,shots:[]};}
const center=(e:Entity)=>({x:e.x+footprint(e).w/2,y:e.y+footprint(e).h/2});
const dist=(a:Point,b:Point)=>Math.hypot(a.x-b.x,a.y-b.y);
function blocked(w:World,p:Point){return p.x<.5||p.y<.5||p.x>63.5||p.y>35.5||w.entities.some(e=>{const f=footprint(e);return p.x>e.x-.25&&p.x<e.x+f.w+.25&&p.y>e.y-.25&&p.y<e.y+f.h+.25;})||(!w.frontier&&dist(p,w.target)<2.4);}
/** Bounded, deterministic creature ecology. Exposure is a game signal, not irradiance. */
export function stepEcology(w:World,dt:number):string[]{
 const eco=w.ecology,messages:string[]=[];eco.shots=eco.shots.filter(s=>(s.ttl-=dt)>0);
 const defenses=w.entities.filter(e=>e.kind==='sentry'&&e.powered);
 if(defenses.length&&!eco.defenseReady){eco.defenseReady=true;messages.push('Perimeter defense online. Wildlife may become aggressive after 30 seconds of sustained field exposure.');}
 if(eco.defenseReady)eco.grace=Math.max(0,eco.grace-dt);
 const sources:{point:Point;strength:number}[]=[];
 for(const l of w.links.filter(l=>l.type==='field')){const m=routeMetrics(l.path,l.radius),out=(w.stats.network.ports[l.a.node]?.[l.a.port]?.outgoing??0)+(w.stats.network.ports[l.b.node]?.[l.b.port]?.outgoing??0);for(const b of m.bends)sources.push({point:b.point,strength:out*(1-Math.exp(-m.bendExponent))/Math.max(1,m.bends.length)});}
 for(const e of w.entities){const p=center(e);if(e.kind==='emitter'&&e.powered)sources.push({point:p,strength:(w.stats.network.absorbed[e.id]??0)*.2});for(const [i,port] of (w.stats.network.ports[e.id]??[]).entries())if(port.outgoing>0&&!w.links.some(l=>l.type==='field'&&[l.a,l.b].some(v=>v.node===e.id&&v.port===i)))sources.push({point:ports(e)[i].position,strength:port.outgoing*.25});}
 eco.threat=0;
 for(const c of eco.creatures){if(c.health<=0)continue;let local=0,best:{point:Point;strength:number}|undefined;
  for(const source of sources){const strength=source.strength/(1+dist(source.point,c)**2*.08);local+=strength;if(!best||strength>best.strength)best={point:source.point,strength};}
  c.exposure=Math.max(0,Math.min(60,c.exposure+dt*(local>.2?1:-2)));eco.threat=Math.max(eco.threat,c.exposure);
  const hostile=eco.defenseReady&&eco.grace===0&&c.exposure>=30;
  if(hostile&&c.state!=='attack'){messages.push(`${c.id.toUpperCase()} agitated by sustained stray field. Perimeter sentries engaging.`);c.state='attack';}
  if(!hostile)c.state=best&&best.strength>.1?'investigate':'patrol';
  const machine=hostile?w.entities.filter(e=>e.health>0).sort((a,b)=>dist(c,center(a))-dist(c,center(b)))[0]:undefined;
  if(machine)c.target=center(machine);else if(best&&best.strength>.1){const angle=w.time*.12+Number(c.id.slice(1))*2;c.target={x:best.point.x+Math.cos(angle)*4,y:best.point.y+Math.sin(angle)*4};}else if(dist(c,c.target)<.5){const i=Number(c.id.slice(1)),angle=w.time*.31+i;c.target={x:20+Math.cos(angle)*12,y:22+Math.sin(angle)*7};}
  // Defense is conventional electrical ordnance; field ledger remains unchanged.
  for(const sentry of defenses)if(dist(c,center(sentry))<8){if(hostile){c.health=Math.max(0,c.health-12*dt);eco.shots.push({from:center(sentry),to:{x:c.x,y:c.y},ttl:.12});}else if(dist(c,center(sentry))<3){c.state='flee';const a=Math.atan2(c.y-center(sentry).y,c.x-center(sentry).x);c.target={x:c.x+Math.cos(a)*4,y:c.y+Math.sin(a)*4};}}
  if(c.health<=0){messages.push(`${c.id.toUpperCase()} neutralized by perimeter defense.`);continue;}
  if(machine&&dist(c,center(machine))<Math.max(footprint(machine).w,footprint(machine).h)/2+.7){machine.health=Math.max(0,machine.health-3*dt);if(machine.health===0){w.stock.scrap+=machine.ore;machine.ore=0;messages.push(`${machine.id.toUpperCase()} destroyed by wildlife.`);}continue;}
  const angle=Math.atan2(c.target.y-c.y,c.target.x-c.x),speed=c.state==='attack'?1.1:.65;
  for(const turn of [0,.6,-.6,1.2,-1.2,Math.PI/2,-Math.PI/2,Math.PI]){const a=angle+turn,p={x:c.x+Math.cos(a)*speed*dt,y:c.y+Math.sin(a)*speed*dt};if(!blocked(w,p)){c.x=p.x;c.y=p.y;c.heading=((Math.round(a/(Math.PI/2))%4)+4)%4;break;}}
 }
 return messages;
}
