import {c,polar,mul,add,power,type Complex} from './complex';
import {source,hybrid,through,matched,solveNetwork,type NetworkResult,type Component,type Endpoint,type WaveLink} from './network';
export type Kind='extractor'|'assembler'|'generator'|'reference'|'junction'|'tuner'|'emitter'|'dump';
export interface Definition {name:string;asset:string;w:number;h:number;cost:number;watts:number;ports:string[];description:string}
export const DEFS:Record<Kind,Definition>={
 extractor:{name:'Extractor',asset:'starter-extractor',w:3,h:3,cost:8,watts:8,ports:[],description:'Place over ore. Route extracted material to an assembler.'},
 assembler:{name:'Assembler',asset:'compact-assembler',w:3,h:3,cost:10,watts:6,ports:[],description:'Consumes 2 ore per assembly. Outputs to the shared construction stock.'},
 generator:{name:'Power unit',asset:'power-unit',w:3,h:2,cost:14,watts:0,ports:[],description:'240 power units; supplies a wired expedition bus within 16 tiles. Fuel is abstracted.'},
 reference:{name:'Reference station',asset:'reference-control-station',w:2,h:2,cost:12,watts:125,ports:['OUT'],description:'Produces 100 field units. Also houses the automatic controller and commissioning tools.'},
 junction:{name:'Four-port junction',asset:'four-port-junction',w:2,h:2,cost:5,watts:0,ports:['A','B','C','D'],description:'A/B combine into C/D; C/D combine into A/B. Unconnected outputs leak to the environment.'},
 tuner:{name:'Phase tuner',asset:'phase-tuner',w:2,h:1,cost:6,watts:0,ports:['IN','OUT'],description:'Fine path adjustment ±180°. Its phase changes slowly with temperature.'},
 emitter:{name:'Field emitter',asset:'field-emitter',w:3,h:3,cost:10,watts:4,ports:['IN'],description:'Directs its incident field toward the frontier target. Two phased emitters fill the target mode.'},
 dump:{name:'Cooled dump',asset:'cooled-dump',w:2,h:2,cost:5,watts:2,ports:['IN'],description:'Absorbs unused output as heat. Protection trips at 85°C; bypassing it risks destruction.'}
};
export interface Entity {id:string;kind:Kind;x:number;y:number;phase:number;temperature:number;health:number;tripped:boolean;protection:boolean;ore:number;progress:number;powered:boolean}
export interface Connection {id:string;a:Endpoint;b:Endpoint;type:'field'|'material'}
export interface Deposit {id:string;x:number;y:number;w:number;h:number;remaining:number;kind:'ore'|'crystal'}
export interface Blueprint {entities:Entity[];links:Connection[];width:number;height:number}
export interface Commission {state:'idle'|'testing'|'qualified'|'failed';elapsed:number;minimum:number;rating:number;reason:string;revision:number}
export interface World {version:1;time:number;nextId:number;entities:Entity[];links:Connection[];deposits:Deposit[];stock:{assemblies:number;crystal:number;scrap:number};produced:number;target:{x:number;y:number;health:number};frontier:boolean;controller:boolean;revision:number;commission:Commission;blueprint:Blueprint|null;events:{time:number;text:string}[];stats:Stats}
export interface Stats {network:NetworkResult;targetPower:number;offTarget:number;radiated:number;heat:number;leaked:number;supply:number;demand:number;error:string;emitterFields:Record<string,Record<string,Complex>>}
export const WIDTH=64,HEIGHT=36,DT=.1;
const emptyNet=():NetworkResult=>({ports:{},absorbed:{},sourcePower:0,linkLoss:0,escaped:0,residual:0});
const emptyStats=():Stats=>({network:emptyNet(),targetPower:0,offTarget:0,radiated:0,heat:0,leaked:0,supply:0,demand:0,error:'',emitterFields:{}});
const blankCommission=():Commission=>({state:'idle',elapsed:0,minimum:Infinity,rating:0,reason:'Not commissioned',revision:0});
export const center=(e:Entity)=>({x:e.x+DEFS[e.kind].w/2,y:e.y+DEFS[e.kind].h/2});
export function event(w:World,text:string){w.events.unshift({time:w.time,text});w.events=w.events.slice(0,30);}
export function entity(w:World,id:string){return w.entities.find(e=>e.id===id);}
export function portPosition(e:Entity,p:number){const d=DEFS[e.kind];if(d.ports.length===4){return [{x:e.x,y:e.y+d.h*.3},{x:e.x,y:e.y+d.h*.7},{x:e.x+d.w,y:e.y+d.h*.3},{x:e.x+d.w,y:e.y+d.h*.7}][p];}if(d.ports.length===2)return {x:e.x+(p?d.w:0),y:e.y+d.h/2};return {x:e.x+(e.kind==='reference'?d.w:0),y:e.y+d.h/2};}
export function newEntity(kind:Kind,x:number,y:number,id:string):Entity{return{id,kind,x,y,phase:0,temperature:25,health:100,tripped:false,protection:true,ore:0,progress:0,powered:false};}
export function invalidate(w:World,reason:string){w.revision++;if(w.commission.state==='testing'||w.commission.state==='qualified'){w.commission.state='failed';w.commission.reason=reason;event(w,`Qualification invalidated: ${reason}`);}}
function overlap(x:number,y:number,aw:number,ah:number,b:{x:number;y:number;w:number;h:number}){return x<b.x+b.w&&x+aw>b.x&&y<b.y+b.h&&y+ah>b.y;}
export function placementError(w:World,kind:Kind,x:number,y:number,extra:Entity[]=[]):string {
 const d=DEFS[kind];if(w.entities.length+extra.length>=40)return 'Prototype limit: 40 machines';if([...w.entities,...extra].reduce((n,e)=>n+DEFS[e.kind].ports.length,0)+d.ports.length>128)return 'Prototype limit: 128 field ports';if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||y<0||x+d.w>WIDTH||y+d.h>HEIGHT)return 'Outside the build area';
 if(!w.frontier&&x+d.w>25)return 'Clear the armored organism to open the eastern frontier';
 if([...w.entities,...extra].some(e=>overlap(x,y,d.w,d.h,{x:e.x,y:e.y,w:DEFS[e.kind].w,h:DEFS[e.kind].h})))return 'Footprint occupied';
 if(kind==='extractor'&&!w.deposits.some(dep=>dep.remaining>0&&overlap(x,y,d.w,d.h,dep)))return 'An extractor needs a resource deposit';
 return '';
}
export function place(w:World,kind:Kind,x:number,y:number):string {
 const error=placementError(w,kind,x,y);if(error)return error;if(w.stock.assemblies<DEFS[kind].cost)return 'Not enough assemblies';
 w.stock.assemblies-=DEFS[kind].cost;w.entities.push(newEntity(kind,x,y,`e${w.nextId++}`));invalidate(w,'Installation changed');event(w,`${DEFS[kind].name} built`);return '';
}
export function remove(w:World,id:string):void {const e=entity(w,id);if(!e)return;w.stock.assemblies+=e.health>0?DEFS[e.kind].cost:0;w.stock.scrap+=e.ore;e.ore=0;w.entities=w.entities.filter(x=>x.id!==id);w.links=w.links.filter(l=>l.a.node!==id&&l.b.node!==id);invalidate(w,'Equipment removed');event(w,e.health>0?'Equipment recovered; buffered ore becomes scrap':'Wreck cleared');}
export function connect(w:World,type:Connection['type'],a:Endpoint,b:Endpoint):string {
 const ea=entity(w,a.node),eb=entity(w,b.node);if(!ea||!eb||a.node===b.node)return 'Choose two different machines';
 if(!Number.isInteger(a.port)||!Number.isInteger(b.port)||a.port<0||b.port<0)return 'Invalid port index';
 if(type==='material'){if(a.port!==0||b.port!==0)return 'Material endpoints use the material interface';if(ea.kind!=='extractor'||eb.kind!=='assembler')return 'Material routes go from extractor to assembler';if(w.links.some(l=>l.type==='material'&&l.a.node===a.node&&l.b.node===b.node))return 'This material route already exists';}
 else {if(!DEFS[ea.kind].ports[a.port]||!DEFS[eb.kind].ports[b.port])return 'Select a field port';if(w.links.some(l=>l.type==='field'&&[l.a,l.b].some(p=>(p.node===a.node&&p.port===a.port)||(p.node===b.node&&p.port===b.port))))return 'Port occupied — disconnect its existing route first';}
 w.links.push({id:`l${w.nextId++}`,type,a:{...a},b:{...b}});invalidate(w,'Routing changed');return '';
}
export function disconnect(w:World,id:string){w.links=w.links.filter(l=>l.id!==id);invalidate(w,'Routing changed');}
export function setPhase(w:World,id:string,degrees:number){const e=entity(w,id);if(!e||e.kind!=='tuner'||!Number.isFinite(degrees))return;e.phase=Math.max(-180,Math.min(180,degrees));invalidate(w,'Manual phase adjustment');}
export function repair(w:World,id:string):string{const e=entity(w,id);if(!e)return 'Select equipment';if(w.stock.assemblies<3)return 'Repair needs 3 assemblies';w.stock.assemblies-=3;e.health=100;e.temperature=25;e.tripped=false;invalidate(w,'Equipment repaired');event(w,`${DEFS[e.kind].name} repaired and reset`);return '';}
export function createWorld():World {
 const w:World={version:1,time:0,nextId:1,entities:[],links:[],deposits:[{id:'starter',x:3,y:5,w:4,h:4,remaining:1400,kind:'ore'},{id:'reserve',x:3,y:17,w:4,h:3,remaining:900,kind:'ore'},{id:'remote-ore',x:38,y:5,w:4,h:4,remaining:1200,kind:'ore'},{id:'frontier',x:29,y:7,w:4,h:4,remaining:500,kind:'crystal'}],stock:{assemblies:28,crystal:0,scrap:0},produced:0,target:{x:28,y:13,health:600},frontier:false,controller:false,revision:0,commission:blankCommission(),blueprint:null,events:[],stats:emptyStats()};
 const seed=(kind:Kind,x:number,y:number)=>{const e=newEntity(kind,x,y,`e${w.nextId++}`);w.entities.push(e);return e.id;};
 const gen=seed('generator',9,3),ex=seed('extractor',3,5),as=seed('assembler',3,11),ref=seed('reference',10,9),j=seed('junction',14,9),em=seed('emitter',20,5),dump=seed('dump',14,15);
 connect(w,'material',{node:ex,port:0},{node:as,port:0});connect(w,'field',{node:ref,port:0},{node:j,port:0});connect(w,'field',{node:j,port:2},{node:em,port:0});connect(w,'field',{node:j,port:1},{node:dump,port:0});
 w.events=[];w.revision=0;event(w,'Expedition bus online. Build a tuner and a second emitter to concentrate the field.');evaluate(w);return w;
}
function powerGrid(w:World){const generators=w.entities.filter(e=>e.kind==='generator'&&e.health>0&&!e.tripped);const supply=generators.length*240;let demand=0;const covered=new Set<string>();for(const e of w.entities){const p=center(e);const inRange=generators.some(g=>Math.hypot(p.x-center(g).x,p.y-center(g).y)<=16);if(inRange||DEFS[e.kind].watts===0){covered.add(e.id);if(e.health>0&&!e.tripped)demand+=DEFS[e.kind].watts;}}
 for(const e of w.entities)e.powered=e.health>0&&!e.tripped&&covered.has(e.id)&&(demand<=supply||DEFS[e.kind].watts===0);
 return {supply,demand};}
/** Geometry uses effective phase rad/tile, not real optical wavelength. */
export function waveLinks(w:World):WaveLink[]{return w.links.filter(l=>l.type==='field').map(l=>{const a=portPosition(entity(w,l.a.node)!,l.a.port),b=portPosition(entity(w,l.b.node)!,l.b.port),distance=Math.hypot(a.x-b.x,a.y-b.y);return {a:l.a,b:l.b,amplitude:Math.exp(-.006*distance),phase:.31*distance};});}
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
  if(e.kind==='extractor'){const d=w.deposits.find(d=>d.remaining>0&&overlap(e.x,e.y,DEFS[e.kind].w,DEFS[e.kind].h,d)&&(d.kind==='ore'||w.frontier));if(d){e.progress+=dt;while(e.progress>=.65&&d.remaining>0&&(d.kind==='crystal'||e.ore<20)){e.progress-=.65;d.remaining--;if(d.kind==='crystal')w.stock.crystal++;else e.ore++;}e.progress=Math.min(e.progress,.65);}}
  if(e.kind==='assembler'){if(e.ore>=2){e.progress+=dt;if(e.progress>=1.4){e.progress-=1.4;e.ore-=2;w.stock.assemblies++;w.produced++;}}else e.progress=0;}
 }
 for(const l of w.links.filter(l=>l.type==='material')){const a=entity(w,l.a.node)!,b=entity(w,l.b.node)!;if(a.powered&&b.powered&&a.ore>=1&&b.ore<20){a.ore--;b.ore++;}}
 automaticControl(w);
 for(const e of w.entities){if(e.health<=0)continue;const absorbed=Math.max(0,w.stats.network.absorbed[e.id]??0);const heating=e.kind==='emitter'&&e.powered?absorbed*.08:absorbed;const drift=e.kind==='tuner'?3+2*Math.sin(w.time*.12)+(w.commission.state==='testing'?4*Math.sin(w.commission.elapsed*.3):0):0;const cooling=e.kind==='dump'?(e.powered?.28:.04):.18;e.temperature+=dt*(heating*.32+drift-cooling*(e.temperature-25));e.temperature=Math.max(25,e.temperature);
  if(e.temperature>85&&e.protection&&!e.tripped){e.tripped=true;invalidate(w,'Thermal protection tripped');event(w,`${DEFS[e.kind].name} tripped at 85°C. Disconnect input and repair.`);}
  if(e.temperature>105){e.health=Math.max(0,e.health-(e.temperature-105)*dt*.45);if(e.health===0){w.stock.scrap+=DEFS[e.kind].cost+e.ore;e.ore=0;invalidate(w,'Equipment destroyed');event(w,`${DEFS[e.kind].name} destroyed by heat`);}}
 }
 evaluate(w);
 if(!w.frontier){w.target.health=Math.max(0,w.target.health-Math.max(0,w.stats.targetPower-32)*dt*3);if(w.target.health===0){w.frontier=true;event(w,'Frontier cleared. Crystal access and commissioning unlocked.');}}
 const test=w.commission;if(test.state==='testing'){test.elapsed+=dt;test.minimum=Math.min(test.minimum,w.stats.targetPower);if(w.stats.error||w.stats.targetPower<32||w.entities.some(e=>e.health<=0||e.tripped)){test.state='failed';test.reason='Output fell below 32 or equipment protection failed';event(w,'Commissioning failed. Inspect the network and retry.');}else if(test.elapsed>=20){test.state='qualified';test.rating=test.minimum;test.reason='Passed 20 s drift profile; rating valid for this topology';event(w,`Module qualified at ${test.rating.toFixed(1)} target power. Blueprint ready.`);}}
 if(test.state==='qualified'&&(w.stats.targetPower<32||w.stats.error)){test.state='failed';test.reason='Operating conditions left the qualified range';event(w,'Qualification lost: output below operating limit');}
}
export function captureBlueprint(w:World):string{if(w.commission.state!=='qualified')return 'Commission the installation before recording a blueprint';const minX=Math.min(...w.entities.map(e=>e.x)),minY=Math.min(...w.entities.map(e=>e.y));w.blueprint={entities:w.entities.map(e=>({...e,x:e.x-minX,y:e.y-minY})),links:structuredClone(w.links),width:Math.max(...w.entities.map(e=>e.x+DEFS[e.kind].w))-minX,height:Math.max(...w.entities.map(e=>e.y+DEFS[e.kind].h))-minY};event(w,'Qualified outpost blueprint recorded');return '';}
export function blueprintCost(w:World){return w.blueprint?.entities.reduce((s,e)=>s+DEFS[e.kind].cost,0)??0;}
export function stampBlueprint(w:World,x:number,y:number):string{const bp=w.blueprint;if(!bp)return 'Record a blueprint first';const cost=blueprintCost(w);if(w.stock.assemblies<cost)return `Blueprint requires ${cost} assemblies`;const staged:Entity[]=[];for(const e of bp.entities){const err=placementError(w,e.kind,x+e.x,y+e.y,staged);if(err)return err;staged.push({...newEntity(e.kind,x+e.x,y+e.y,`e${w.nextId+staged.length}`),phase:e.phase,protection:e.protection});}const ids=new Map(bp.entities.map((e,i)=>[e.id,staged[i].id]));w.nextId+=staged.length;w.entities.push(...staged);w.stock.assemblies-=cost;for(const l of bp.links)w.links.push({id:`l${w.nextId++}`,type:l.type,a:{node:ids.get(l.a.node)!,port:l.a.port},b:{node:ids.get(l.b.node)!,port:l.b.port}});invalidate(w,'Blueprint placed — local commissioning required');event(w,'Blueprint placed; recheck power, deposits and phase at this site');return '';}
export function serialize(w:World){const {stats,...save}=w;return JSON.stringify(save);}
/** Reject malformed saves rather than allowing NaNs, missing endpoints or unbounded solves. */
export function deserialize(raw:string):World {
 const s=JSON.parse(raw);const fail=()=>{throw new Error('Invalid or unsupported save');};
 const finite=(v:unknown)=>typeof v==='number'&&Number.isFinite(v);const nonnegative=(v:unknown)=>finite(v)&&(v as number)>=0;
 if(s?.version!==1||!Array.isArray(s.entities)||s.entities.length>40||!Array.isArray(s.links)||s.links.length>120||!Array.isArray(s.deposits)||s.deposits.length>20||!nonnegative(s.time)||!Number.isSafeInteger(s.nextId)||s.nextId<1||!s.stock||!['assemblies','crystal','scrap'].every(k=>nonnegative(s.stock[k]))||!nonnegative(s.produced)||!s.target||!nonnegative(s.target.health)||!finite(s.target.x)||!finite(s.target.y)||typeof s.frontier!=='boolean'||typeof s.controller!=='boolean'||!nonnegative(s.revision))fail();
 const ids=new Set<string>();for(const e of s.entities){if(!e||typeof e.id!=='string'||!/^e\d+$/.test(e.id)||ids.has(e.id)||!Object.hasOwn(DEFS,e.kind)||!Number.isInteger(e.x)||!Number.isInteger(e.y)||!['phase','temperature','health','ore','progress'].every(k=>finite(e[k]))||e.health<0||e.health>100||e.ore<0||e.progress<0||Math.abs(e.phase)>180||e.temperature<0||typeof e.protection!=='boolean'||typeof e.tripped!=='boolean'||typeof e.powered!=='boolean')fail();ids.add(e.id);const d=DEFS[e.kind as Kind];if(e.x<0||e.y<0||e.x+d.w>WIDTH||e.y+d.h>HEIGHT)fail();}
 for(let i=0;i<s.entities.length;i++)for(let j=0;j<i;j++){const a=s.entities[i],b=s.entities[j],ad=DEFS[a.kind as Kind],bd=DEFS[b.kind as Kind];if(overlap(a.x,a.y,ad.w,ad.h,{...b,w:bd.w,h:bd.h}))fail();}
 const clean={...s,links:[],stats:emptyStats(),events:[],commission:blankCommission(),blueprint:null} as World;
 for(const dep of s.deposits)if(!dep||!['ore','crystal'].includes(dep.kind)||!['x','y','w','h','remaining'].every(k=>nonnegative(dep[k]))||dep.w<1||dep.h<1||dep.x+dep.w>WIDTH||dep.y+dep.h>HEIGHT||!Number.isInteger(dep.remaining))fail();
 for(const l of s.links){if(!l||!['field','material'].includes(l.type)||!l.a||!l.b||connect(clean,l.type,l.a,l.b))fail();}
 clean.nextId=Math.max(s.nextId,...s.entities.map((e:Entity)=>Number(e.id.slice(1))+1),clean.nextId);clean.revision=s.revision;
 // Blueprint geometry/topology is validated through the same bounded world schema.
 if(s.blueprint!=null){
  if(!Array.isArray(s.blueprint.entities)||!s.blueprint.entities.length||!Array.isArray(s.blueprint.links))fail();
  const bp=deserialize(JSON.stringify({...clean,entities:s.blueprint.entities,links:s.blueprint.links,blueprint:null,frontier:true}));
  clean.blueprint={entities:bp.entities,links:bp.links,width:Math.max(...bp.entities.map(e=>e.x+DEFS[e.kind].w)),height:Math.max(...bp.entities.map(e=>e.y+DEFS[e.kind].h))};
 }
 // Saved qualifications are re-tested; external conditions may have changed.
 clean.commission.reason='Loaded installation — recommission to verify rating';event(clean,'Save loaded. Installation restored; qualification requires a fresh test.');evaluate(clean);return clean;
}
