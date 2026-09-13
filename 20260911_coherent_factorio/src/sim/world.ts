import {newEcology,stepEcology,type Ecology} from './ecology';
import {footprint,ports,inside,routeMetrics,type Rotation,type Point,type Transport} from './geometry';
import {findRoute,segmentClear,type RouteOptions} from './routing';
export {footprint,ports,routeMetrics} from './geometry';
import {c,polar,mul,add,power,type Complex} from './complex';
import {source,hybrid,through,matched,solveNetwork,type NetworkResult,type Component,type Endpoint,type WaveLink} from './network';
import {DEFS,type Kind} from './definitions';
export {DEFS,type Kind} from './definitions';
export interface Entity {id:string;kind:Kind;x:number;y:number;rotation:Rotation;phase:number;temperature:number;health:number;tripped:boolean;protection:boolean;ore:number;progress:number;powered:boolean}
export interface Connection {id:string;a:Endpoint;b:Endpoint;type:Transport;path:Point[];diagonal:boolean;radius:number;packets:number[]}
export interface Deposit {id:string;x:number;y:number;w:number;h:number;remaining:number;kind:'ore'|'crystal'}
export interface Blueprint {entities:Entity[];links:Connection[];width:number;height:number}
export interface Commission {state:'idle'|'testing'|'qualified'|'failed';elapsed:number;minimum:number;rating:number;reason:string;revision:number}
export interface World {version:3;ecology:Ecology;time:number;nextId:number;entities:Entity[];links:Connection[];deposits:Deposit[];stock:{assemblies:number;crystal:number;scrap:number};produced:number;target:{x:number;y:number;health:number};frontier:boolean;controller:boolean;revision:number;commission:Commission;blueprint:Blueprint|null;events:{time:number;text:string}[];stats:Stats}
export interface Stats {network:NetworkResult;targetPower:number;offTarget:number;radiated:number;heat:number;leaked:number;supply:number;demand:number;bendRadiation:number;propagationLoss:number;error:string;emitterFields:Record<string,Record<string,Complex>>}
export const WIDTH=64,HEIGHT=36,DT=.1;
const emptyNet=():NetworkResult=>({ports:{},absorbed:{},sourcePower:0,linkLoss:0,escaped:0,residual:0});
const emptyStats=():Stats=>({network:emptyNet(),targetPower:0,offTarget:0,radiated:0,heat:0,leaked:0,supply:0,demand:0,bendRadiation:0,propagationLoss:0,error:'',emitterFields:{}});
const blankCommission=():Commission=>({state:'idle',elapsed:0,minimum:Infinity,rating:0,reason:'Not commissioned',revision:0});
export const center=(e:Entity)=>({x:e.x+footprint(e).w/2,y:e.y+footprint(e).h/2});
export function event(w:World,text:string){w.events.unshift({time:w.time,text});w.events=w.events.slice(0,30);}
export function entity(w:World,id:string){return w.entities.find(e=>e.id===id);}
export function portPosition(e:Entity,p:number,type:Transport='field'){return ports(e,type)[p].position;}
export function newEntity(kind:Kind,x:number,y:number,id:string,rotation:Rotation=0):Entity{return{id,kind,x,y,rotation,phase:0,temperature:25,health:100,tripped:false,protection:true,ore:0,progress:0,powered:false};}
export function invalidate(w:World,reason:string){w.revision++;if(w.commission.state==='testing'||w.commission.state==='qualified'){w.commission.state='failed';w.commission.reason=reason;event(w,`Qualification invalidated: ${reason}`);}}
function overlap(x:number,y:number,aw:number,ah:number,b:{x:number;y:number;w:number;h:number}){return x<b.x+b.w&&x+aw>b.x&&y<b.y+b.h&&y+ah>b.y;}
export function placementError(w:World,kind:Kind,x:number,y:number,extra:Entity[]=[],rotation:Rotation=0):string {
 const d={...DEFS[kind],...footprint({kind,rotation})};if(![0,1,2,3].includes(rotation))return 'Invalid rotation';if(w.entities.length+extra.length>=40)return 'Prototype limit: 40 machines';if([...w.entities,...extra].reduce((n,e)=>n+DEFS[e.kind].ports.length,0)+d.ports.length>128)return 'Prototype limit: 128 field ports';if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||y<0||x+d.w>WIDTH||y+d.h>HEIGHT)return 'Outside the build area';
 if(!w.frontier&&x+d.w>25)return 'Clear the armored organism to open the eastern frontier';
 if([...w.entities,...extra].some(e=>overlap(x,y,d.w,d.h,{x:e.x,y:e.y,...footprint(e)})))return 'Footprint occupied';
 if(w.links.some(l=>l.path.some((p,i)=>inside(p,{kind,x,y,rotation})||(i>0&&!segmentClear(l.path[i-1],p,[{kind,x,y,rotation}])))))return 'Route occupies footprint';
 if(kind==='extractor'&&!w.deposits.some(dep=>dep.remaining>0&&overlap(x,y,d.w,d.h,dep)))return 'An extractor needs a resource deposit';
 return '';
}
export function place(w:World,kind:Kind,x:number,y:number,rotation:Rotation=0):string {
 const error=placementError(w,kind,x,y,[],rotation);if(error)return error;if(w.stock.assemblies<DEFS[kind].cost)return 'Not enough assemblies';
 w.stock.assemblies-=DEFS[kind].cost;w.entities.push(newEntity(kind,x,y,`e${w.nextId++}`,rotation));invalidate(w,'Installation changed');event(w,`${DEFS[kind].name} built`);return '';
}
export function remove(w:World,id:string):void {const e=entity(w,id);if(!e)return;w.stock.assemblies+=e.health>0?DEFS[e.kind].cost:0;w.stock.scrap+=e.ore;e.ore=0;w.entities=w.entities.filter(x=>x.id!==id);for(const l of w.links.filter(l=>l.a.node===id||l.b.node===id))disconnect(w,l.id);invalidate(w,'Equipment removed');event(w,e.health>0?'Equipment recovered; buffered ore becomes scrap':'Wreck cleared');}
export function connect(w:World,type:Connection['type'],a:Endpoint,b:Endpoint,options:RouteOptions={}):string {
 const ea=entity(w,a.node),eb=entity(w,b.node);if(!ea||!eb||a.node===b.node)return 'Choose two different machines';
 if(!['field','material','power'].includes(type)||!Number.isInteger(a.port)||!Number.isInteger(b.port)||a.port<0||b.port<0)return 'Invalid port index';
 const pa=ports(ea,type)[a.port],pb=ports(eb,type)[b.port];if(!pa||!pb)return 'Select a compatible port';
 if(type!=='field'&&(pa.role!=='output'||pb.role!=='input'))return 'Belts and wires run from output to input';
 if(w.links.some(l=>l.type===type&&[l.a,l.b].some(p=>(type!=='power'&&p.node===a.node&&p.port===a.port)||(p.node===b.node&&p.port===b.port))))return 'Port occupied — disconnect its existing route first';
 const radius=options.radius??.5;if(!Number.isFinite(radius)||radius<0||radius>4)return 'Invalid bend radius';
 const path=findRoute(pa,pb,w.entities,WIDTH,HEIGHT,options);if(!path)return 'No valid grid route: check clearance and port direction';
 w.links.push({id:`l${w.nextId++}`,type,a:{...a},b:{...b},path,diagonal:options.diagonal??false,radius,packets:[]});invalidate(w,'Routing changed');return '';
}
export function disconnect(w:World,id:string){const l=w.links.find(l=>l.id===id);if(!l)return;w.stock.scrap+=l.packets.length;w.links=w.links.filter(l=>l.id!==id);invalidate(w,'Routing changed');}
export function rotateEntity(w:World,id:string):string {
 const e=entity(w,id);if(!e)return 'Select equipment';if(w.links.some(l=>l.a.node===id||l.b.node===id))return 'Disconnect routes before rotating equipment';
 const rotation=((e.rotation+1)%4) as Rotation;
 const error=placementError({...w,entities:w.entities.filter(v=>v.id!==id)},e.kind,e.x,e.y,[],rotation);if(error)return error;e.rotation=rotation;invalidate(w,'Equipment rotated');return '';
}
export function setPhase(w:World,id:string,degrees:number){const e=entity(w,id);if(!e||e.kind!=='tuner'||!Number.isFinite(degrees))return;e.phase=Math.max(-180,Math.min(180,degrees));invalidate(w,'Manual phase adjustment');}
export function repair(w:World,id:string):string{const e=entity(w,id);if(!e)return 'Select equipment';if(w.stock.assemblies<3)return 'Repair needs 3 assemblies';w.stock.assemblies-=3;e.health=100;e.temperature=25;e.tripped=false;invalidate(w,'Equipment repaired');event(w,`${DEFS[e.kind].name} repaired and reset`);return '';}
export function createWorld():World {
 const w:World={version:3,ecology:newEcology(),time:0,nextId:1,entities:[],links:[],deposits:[{id:'starter',x:3,y:5,w:4,h:4,remaining:1400,kind:'ore'},{id:'reserve',x:3,y:17,w:4,h:3,remaining:900,kind:'ore'},{id:'remote-ore',x:38,y:5,w:4,h:4,remaining:1200,kind:'ore'},{id:'frontier',x:29,y:7,w:4,h:4,remaining:500,kind:'crystal'}],stock:{assemblies:28,crystal:0,scrap:0},produced:0,target:{x:28,y:13,health:600},frontier:false,controller:false,revision:0,commission:blankCommission(),blueprint:null,events:[],stats:emptyStats()};
 const seed=(kind:Kind,x:number,y:number)=>{const e=newEntity(kind,x,y,`e${w.nextId++}`);w.entities.push(e);return e.id;};
 const gen=seed('generator',9,3),ex=seed('extractor',3,5),as=seed('assembler',3,11),ref=seed('reference',10,9),j=seed('junction',14,9),em=seed('emitter',20,5),dump=seed('dump',14,15);
 connect(w,'material',{node:ex,port:0},{node:as,port:0});connect(w,'field',{node:ref,port:0},{node:j,port:0});connect(w,'field',{node:j,port:2},{node:em,port:0});connect(w,'field',{node:j,port:1},{node:dump,port:0});
 for(const e of w.entities.filter(e=>DEFS[e.kind].watts>0))connect(w,'power',{node:gen,port:0},{node:e.id,port:0});
 w.events=[];w.revision=0;event(w,'Expedition bus online. Build a tuner and a second emitter to concentrate the field.');evaluate(w);return w;
}
function powerGrid(w:World){
 const generators=w.entities.filter(e=>e.kind==='generator'&&e.health>0&&!e.tripped);const supply=generators.length*240;
 const demand=w.entities.reduce((sum,e)=>sum+(e.health>0&&!e.tripped?DEFS[e.kind].watts:0),0);
 for(const e of w.entities)e.powered=e.health>0&&!e.tripped&&DEFS[e.kind].watts===0;
 for(const g of generators){const loads=w.links.filter(l=>l.type==='power'&&l.a.node===g.id).map(l=>entity(w,l.b.node)!).filter(e=>e.health>0&&!e.tripped);const localDemand=loads.reduce((sum,e)=>sum+DEFS[e.kind].watts,0);if(localDemand<=240)for(const e of loads)e.powered=true;}
 return {supply,demand};
}
/** Geometry uses effective phase rad/tile, not real optical wavelength. */
export function waveLinks(w:World):WaveLink[]{return w.links.filter(l=>l.type==='field').map(l=>{const m=routeMetrics(l.path,l.radius);return {a:l.a,b:l.b,amplitude:Math.exp(-.5*(m.propagationExponent+m.bendExponent)),phase:.31*m.length};});}
export function evaluate(w:World):Stats {
 const grid=powerGrid(w);const comps:Component[]=w.entities.filter(e=>DEFS[e.kind].ports.length).map(e=>{
  if(e.health<=0)return matched(e.id,DEFS[e.kind].ports.length);
  if(e.tripped&&e.kind!=='reference')return {id:e.id,s:DEFS[e.kind].ports.map((_,i)=>DEFS[e.kind].ports.map((_,j)=>c(i===j?1:0)))};
  if(e.tripped)return matched(e.id,DEFS[e.kind].ports.length);
  if(e.kind==='reference')return source(e.id,e.powered?100:0,e.id);
  if(e.kind==='junction')return hybrid(e.id);
  if(e.kind==='tuner')return through(e.id,e.phase*Math.PI/180+(e.temperature-25)*.045);
  if(e.kind==='emitter')return {id:e.id,s:[[c(.08)]]};return matched(e.id);
 });
 const stats=emptyStats();Object.assign(stats,grid);
 try{stats.network=solveNetwork(comps,waveLinks(w));}catch(err){stats.error=err instanceof Error?err.message:'Wave solve failed';w.stats=stats;return stats;}
 for(const l of w.links.filter(l=>l.type==='field')){const m=routeMetrics(l.path,l.radius),total=m.propagationExponent+m.bendExponent;const outgoing=(stats.network.ports[l.a.node]?.[l.a.port]?.outgoing??0)+(stats.network.ports[l.b.node]?.[l.b.port]?.outgoing??0);const loss=outgoing*(1-Math.exp(-total));stats.bendRadiation+=total?loss*m.bendExponent/total:0;stats.propagationLoss+=total?loss*m.propagationExponent/total:0;}
 let heat=0,radiated=0;const fields:Record<string,Complex>={};const emitters=w.entities.filter(e=>e.kind==='emitter'&&e.powered);const n=Math.max(2,emitters.length);
 for(const e of w.entities){const absorbed=Math.max(0,stats.network.absorbed[e.id]??0);
  if(e.kind==='emitter'&&e.powered){const p=center(e),distance=Math.hypot(p.x-w.target.x,p.y-w.target.y);const capture=Math.min(.88,50/(distance*distance+30));const rad=absorbed*.92;radiated+=rad;heat+=absorbed-rad;
   stats.emitterFields[e.id]={};for(const [group,value] of Object.entries(stats.network.ports[e.id]?.[0]?.fields??{})){const field=mul(value.a,polar(Math.sqrt((1-.08**2)*.92*capture/n),distance*.23));fields[group]=add(fields[group]??c(0),field);stats.emitterFields[e.id][group]=field;}
  }else heat+=absorbed;
 }
 stats.targetPower=Object.values(fields).reduce((s,v)=>s+power(v),0);stats.radiated=radiated;stats.offTarget=Math.max(0,radiated-stats.targetPower);stats.heat=heat;stats.leaked=stats.network.escaped+stats.network.linkLoss+stats.offTarget;w.stats=stats;return stats;
}
function automaticControl(w:World){if(!w.controller||!w.entities.some(e=>e.kind==='reference'&&e.powered))return;for(const e of w.entities.filter(e=>e.kind==='tuner'&&e.health>0&&!e.tripped)){const original=e.phase;const base=evaluate(w).targetPower;e.phase=original+2;const plus=evaluate(w).targetPower;e.phase=original-2;const minus=evaluate(w).targetPower;e.phase=original;if(Math.max(plus,minus)>base+1e-6)e.phase=wrap(original+(plus>minus?2:-2));}evaluate(w);}
const wrap=(x:number)=>((x+180)%360+360)%360-180;
export function setController(w:World,on:boolean){w.controller=on;invalidate(w,'Controller mode changed');event(w,on?'Automatic phase control enabled':'Automatic phase control disabled');}
export function beginCommission(w:World):string{evaluate(w);if(!w.frontier)return 'Clear the frontier first';if(!w.controller)return 'Enable automatic phase control first';if(w.stats.targetPower<35)return 'Establish at least 35 target power before testing';w.commission={state:'testing',elapsed:0,minimum:Infinity,rating:0,reason:'20 s thermal drift test · minimum 32 target power',revision:w.revision};event(w,'Commissioning started: 20 s drift profile');return '';}
export function cancelCommission(w:World){w.commission.state='idle';w.commission.reason='Cancelled; not qualified';w.commission.rating=0;}
export function step(w:World,dt=DT){if(!Number.isFinite(dt)||dt<=0||dt>.25)throw new Error('Step must be between 0 and 0.25 seconds');w.time+=dt;evaluate(w);
 for(const e of w.entities){if(!e.powered)continue;
  if(e.kind==='extractor'){const d=w.deposits.find(d=>d.remaining>0&&overlap(e.x,e.y,footprint(e).w,footprint(e).h,d)&&(d.kind==='ore'||w.frontier));if(d){e.progress+=dt;while(e.progress>=.65&&d.remaining>0&&(d.kind==='crystal'||e.ore<20)){e.progress-=.65;d.remaining--;if(d.kind==='crystal')w.stock.crystal++;else e.ore++;}e.progress=Math.min(e.progress,.65);}}
  if(e.kind==='assembler'){if(e.ore>=2){e.progress+=dt;if(e.progress>=1.4){e.progress-=1.4;e.ore-=2;w.stock.assemblies++;w.produced++;}}else e.progress=0;}
 }
 for(const l of w.links.filter(l=>l.type==='material')){
  const a=entity(w,l.a.node)!,b=entity(w,l.b.node)!,length=routeMetrics(l.path).length;
  // Front packet moves first; a full or stopped receiver propagates a queue upstream.
  let limit=length;for(let i=0;i<l.packets.length;i++){l.packets[i]=Math.min(limit,l.packets[i]+dt*2);limit=l.packets[i]-.5;}
  if(l.packets.length&&l.packets[0]>=length&&b.powered&&b.ore<20){b.ore++;l.packets.shift();}
  if(a.powered&&a.ore>=1&&(!l.packets.length||l.packets.at(-1)!>=.5)){a.ore--;l.packets.push(0);}
 }
 automaticControl(w);
 for(const e of w.entities){if(e.health<=0)continue;const absorbed=Math.max(0,w.stats.network.absorbed[e.id]??0);const heating=e.kind==='emitter'&&e.powered?absorbed*.08:absorbed;const drift=e.kind==='tuner'?3+2*Math.sin(w.time*.12)+(w.commission.state==='testing'?4*Math.sin(w.commission.elapsed*.3):0):0;const cooling=e.kind==='dump'?(e.powered?.28:.04):.18;e.temperature+=dt*(heating*.32+drift-cooling*(e.temperature-25));e.temperature=Math.max(25,e.temperature);
  if(e.temperature>85&&e.protection&&!e.tripped){e.tripped=true;invalidate(w,'Thermal protection tripped');event(w,`${DEFS[e.kind].name} tripped at 85°C. Disconnect input and repair.`);}
  if(e.temperature>105){e.health=Math.max(0,e.health-(e.temperature-105)*dt*.45);if(e.health===0){w.stock.scrap+=DEFS[e.kind].cost+e.ore;e.ore=0;invalidate(w,'Equipment destroyed');event(w,`${DEFS[e.kind].name} destroyed by heat`);}}
 }
 evaluate(w);
 if(!w.frontier){w.target.health=Math.max(0,w.target.health-Math.max(0,w.stats.targetPower-32)*dt*3);if(w.target.health===0){w.frontier=true;event(w,'Frontier cleared. Crystal access and commissioning unlocked.');}}
 const integrity=w.entities.reduce((sum,e)=>sum+e.health,0);for(const message of stepEcology(w,dt))event(w,message);if(w.entities.reduce((sum,e)=>sum+e.health,0)<integrity){invalidate(w,'Wildlife damaged equipment');evaluate(w);}
 const test=w.commission;if(test.state==='testing'){test.elapsed+=dt;test.minimum=Math.min(test.minimum,w.stats.targetPower);if(w.stats.error||w.stats.targetPower<32||w.entities.some(e=>e.health<=0||e.tripped)){test.state='failed';test.reason='Output fell below 32 or equipment protection failed';event(w,'Commissioning failed. Inspect the network and retry.');}else if(test.elapsed>=20){test.state='qualified';test.rating=test.minimum;test.reason='Passed 20 s drift profile; rating valid for this topology';event(w,`Module qualified at ${test.rating.toFixed(1)} target power. Blueprint ready.`);}}
 if(test.state==='qualified'&&(w.stats.targetPower<32||w.stats.error)){test.state='failed';test.reason='Operating conditions left the qualified range';event(w,'Qualification lost: output below operating limit');}
}
export function captureBlueprint(w:World):string {
 if(w.commission.state!=='qualified')return 'Commission the installation before recording a blueprint';
 const points=w.links.flatMap(l=>l.path),minX=Math.floor(Math.min(...w.entities.map(e=>e.x),...points.map(p=>p.x))),minY=Math.floor(Math.min(...w.entities.map(e=>e.y),...points.map(p=>p.y)));
 w.blueprint={entities:w.entities.map(e=>({...e,x:e.x-minX,y:e.y-minY,ore:0,progress:0})),links:w.links.map(l=>({...structuredClone(l),path:l.path.map(p=>({x:p.x-minX,y:p.y-minY})),packets:[]})),width:Math.ceil(Math.max(...w.entities.map(e=>e.x+footprint(e).w),...points.map(p=>p.x)))-minX,height:Math.ceil(Math.max(...w.entities.map(e=>e.y+footprint(e).h),...points.map(p=>p.y)))-minY};event(w,'Qualified outpost blueprint recorded');return '';
}
export function blueprintCost(w:World){return w.blueprint?.entities.reduce((s,e)=>s+DEFS[e.kind].cost,0)??0;}
export function stampBlueprint(w:World,x:number,y:number):string {
 const bp=w.blueprint;if(!bp)return 'Record a blueprint first';const cost=blueprintCost(w);if(w.stock.assemblies<cost)return `Blueprint requires ${cost} assemblies`;
 const staged:Entity[]=[];for(const e of bp.entities){const err=placementError(w,e.kind,x+e.x,y+e.y,staged,e.rotation);if(err)return err;staged.push({...newEntity(e.kind,x+e.x,y+e.y,`e${w.nextId+staged.length}`,e.rotation),phase:e.phase,protection:e.protection});}
 const ids=new Map(bp.entities.map((e,i)=>[e.id,staged[i].id]));
 const draft={...w,entities:[...w.entities,...staged],links:[...w.links],events:[],commission:{...w.commission},nextId:w.nextId+staged.length};
 for(const l of bp.links){const error=connect(draft,l.type,{node:ids.get(l.a.node)!,port:l.a.port},{node:ids.get(l.b.node)!,port:l.b.port},{path:l.path.map(p=>({x:p.x+x,y:p.y+y})),radius:l.radius,diagonal:l.diagonal});if(error)return error;}
 w.nextId=draft.nextId;w.entities=draft.entities;w.links=draft.links;w.stock.assemblies-=cost;invalidate(w,'Blueprint placed — local commissioning required');event(w,'Blueprint placed; recheck power, deposits and phase at this site');return '';
}
export function serialize(w:World){const {stats,...save}=w;return JSON.stringify(save);}
/** Reject malformed saves rather than allowing NaNs, missing endpoints or unbounded solves. */
export function deserialize(raw:string):World {
 const s=JSON.parse(raw);const fail=()=>{throw new Error('Invalid or unsupported save');};
 const finite=(v:unknown)=>typeof v==='number'&&Number.isFinite(v);const nonnegative=(v:unknown)=>finite(v)&&(v as number)>=0;
 const legacy=s?.version===1;if(legacy&&Array.isArray(s.entities)){s.version=2;for(const e of s.entities)if(e)e.rotation=0;}
 if(s?.version===2){s.version=3;s.ecology=newEcology();}
 if(s?.version!==3||!Array.isArray(s.entities)||s.entities.length>40||!Array.isArray(s.links)||s.links.length>120||!Array.isArray(s.deposits)||s.deposits.length>20||!nonnegative(s.time)||!Number.isSafeInteger(s.nextId)||s.nextId<1||!s.stock||!['assemblies','crystal','scrap'].every(k=>nonnegative(s.stock[k]))||!nonnegative(s.produced)||!s.target||!nonnegative(s.target.health)||!finite(s.target.x)||!finite(s.target.y)||typeof s.frontier!=='boolean'||typeof s.controller!=='boolean'||!nonnegative(s.revision))fail();
 const eco=s.ecology;
 if(!eco||typeof eco.defenseReady!=='boolean'||!nonnegative(eco.grace)||eco.grace>30||!nonnegative(eco.threat)||eco.threat>60||!Array.isArray(eco.creatures)||eco.creatures.length>20)fail();
 const creatureIds=new Set<string>();for(const c of eco.creatures){if(!c||typeof c.id!=='string'||!/^c\d+$/.test(c.id)||creatureIds.has(c.id)||!['patrol','investigate','attack','flee'].includes(c.state)||![0,1,2,3].includes(c.heading)||!['x','y','health','exposure'].every(k=>nonnegative(c[k]))||c.x>WIDTH||c.y>HEIGHT||c.health>30||c.exposure>60||!c.target||!finite(c.target.x)||!finite(c.target.y))fail();creatureIds.add(c.id);}eco.shots=[];
 const ids=new Set<string>();for(const e of s.entities){if(!e||typeof e.id!=='string'||!/^e\d+$/.test(e.id)||ids.has(e.id)||!Object.hasOwn(DEFS,e.kind)||![0,1,2,3].includes(e.rotation)||!Number.isInteger(e.x)||!Number.isInteger(e.y)||!['phase','temperature','health','ore','progress'].every(k=>finite(e[k]))||e.health<0||e.health>100||e.ore<0||e.progress<0||Math.abs(e.phase)>180||e.temperature<0||typeof e.protection!=='boolean'||typeof e.tripped!=='boolean'||typeof e.powered!=='boolean')fail();ids.add(e.id);const d=footprint(e);if(e.x<0||e.y<0||e.x+d.w>WIDTH||e.y+d.h>HEIGHT)fail();}
 for(let i=0;i<s.entities.length;i++)for(let j=0;j<i;j++){const a=s.entities[i],b=s.entities[j],ad=footprint(a),bd=footprint(b);if(overlap(a.x,a.y,ad.w,ad.h,{...b,w:bd.w,h:bd.h}))fail();}
 const clean={...s,links:[],stats:emptyStats(),events:[],commission:blankCommission(),blueprint:null} as World;
 for(const dep of s.deposits)if(!dep||!['ore','crystal'].includes(dep.kind)||!['x','y','w','h','remaining'].every(k=>nonnegative(dep[k]))||dep.w<1||dep.h<1||dep.x+dep.w>WIDTH||dep.y+dep.h>HEIGHT||!Number.isInteger(dep.remaining))fail();
 const linkIds=new Set<string>();
 for(const l of s.links){
  if(!l||typeof l.id!=='string'||!/^l\d+$/.test(l.id)||linkIds.has(l.id)||!['field','material','power'].includes(l.type)||!l.a||!l.b)fail();linkIds.add(l.id);
  if(!legacy&&(!Array.isArray(l.path)||typeof l.diagonal!=='boolean'||!finite(l.radius)||!Array.isArray(l.packets)))fail();
  if(connect(clean,l.type,l.a,l.b,legacy?{}:{path:l.path,diagonal:l.diagonal,radius:l.radius}))fail();clean.links.at(-1)!.id=l.id;
  if(!legacy){const link=clean.links.at(-1)!,length=routeMetrics(link.path).length;if(l.packets.length>length*2+1||l.packets.some((v:unknown,i:number)=>!nonnegative(v)||(v as number)>length||(i>0&&l.packets[i-1]-(v as number)<.5-1e-8))||(l.type!=='material'&&l.packets.length))fail();link.packets=[...l.packets];}
 }
 clean.nextId=Math.max(clean.nextId,...clean.links.map(l=>Number(l.id.slice(1))+1));
 if(legacy){for(const e of clean.entities.filter(e=>DEFS[e.kind].watts>0)){const g=clean.entities.find(g=>g.kind==='generator'&&Math.hypot(center(e).x-center(g).x,center(e).y-center(g).y)<=16);if(g&&connect(clean,'power',{node:g.id,port:0},{node:e.id,port:0}))fail();}}
 clean.nextId=Math.max(s.nextId,...s.entities.map((e:Entity)=>Number(e.id.slice(1))+1),clean.nextId);clean.revision=s.revision;
 // Blueprint geometry/topology is validated through the same bounded world schema.
 if(s.blueprint!=null){
  if(!Array.isArray(s.blueprint.entities)||!s.blueprint.entities.length||!Array.isArray(s.blueprint.links))fail();
  if(legacy)for(const e of s.blueprint.entities){e.x+=1;e.y+=1;}
  const bp=deserialize(JSON.stringify({...clean,version:legacy?1:3,entities:s.blueprint.entities,links:s.blueprint.links,blueprint:null,frontier:true}));
  clean.blueprint={entities:bp.entities,links:bp.links,width:Math.ceil(Math.max(...bp.entities.map(e=>e.x+footprint(e).w),...bp.links.flatMap(l=>l.path.map(p=>p.x)))),height:Math.ceil(Math.max(...bp.entities.map(e=>e.y+footprint(e).h),...bp.links.flatMap(l=>l.path.map(p=>p.y))))};
 }
 // Saved qualifications are re-tested; external conditions may have changed.
 if(Array.isArray(s.events))clean.events=s.events.filter((e:{time:unknown;text:unknown})=>e&&nonnegative(e.time)&&typeof e.text==='string'&&e.text.length<=1000).slice(0,30);
 clean.commission.reason='Loaded installation — recommission to verify rating';event(clean,'Save loaded. Installation restored; qualification requires a fresh test.');evaluate(clean);return clean;
}
