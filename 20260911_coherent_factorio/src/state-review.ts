import {loadSprites,sprites} from './assets';
import {Renderer,type View} from './renderer';
import {createWorld,newEntity,DEFS,center,type Kind,type World,type Entity} from './sim/world';
import {footprint} from './sim/geometry';
import {equipmentState} from './equipment-state';
import {equipmentEffects} from './equipment-effects';
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const kinds=Object.keys(DEFS) as Kind[];
$('state-cards').innerHTML=kinds.map(kind=>`<article class="card" data-kind="${kind}"><h2>${DEFS[kind].name}</h2><canvas width="360" height="300" aria-label="${DEFS[kind].name} state preview"></canvas><div class="state-label"></div><p>${DEFS[kind].description}</p><a class="source" href="./${kind}-condition-v1.png">Condition atlas ↗</a></article>`).join('');
$('item-cards').innerHTML=['Ore nugget','Finished assembly','Precision crystal','Scrap'].map((name,i)=>`<div class="item"><div class="item-art" style="background-position:${i%2*100}% ${Math.floor(i/2)*100}%"></div>${name}</div>`).join('');
const views=kinds.map(kind=>{const card=document.querySelector<HTMLElement>(`[data-kind="${kind}"]`)!,canvas=card.querySelector('canvas')!,view:View={rotation:0,diagonal:false,radius:.5,waypoints:[],selected:null,build:null,tool:'select',pending:null,overlay:false,grid:false,mouse:null,panX:0,panY:0,zoom:1};return {kind,card,renderer:new Renderer(canvas,view)};});
function resize(){for(const {renderer:r} of views){r.view.zoom=1;r.resize();r.view.zoom=Math.min(r.width,r.height)/5.5/r.scale;r.resize();}}
let paused=false,time=0,last=performance.now();$('pause-preview').onclick=()=>{paused=!paused;$('pause-preview').textContent=paused?'Resume animation':'Pause animation';};
function controls(){return {power:$<HTMLInputElement>('power-state').checked,field:$<HTMLInputElement>('field-state').checked,work:$<HTMLSelectElement>('work-state').value,health:Number($<HTMLInputElement>('integrity').value),temperature:Number($<HTMLInputElement>('temperature').value),trip:$<HTMLInputElement>('tripped').checked,rotation:Number($<HTMLSelectElement>('rotation').value) as 0|1|2|3};}
document.querySelector('.presets')!.addEventListener('click',ev=>{const preset=(ev.target as HTMLElement).closest<HTMLButtonElement>('[data-preset]')?.dataset.preset;if(!preset)return;$<HTMLInputElement>('power-state').checked=preset!=='off'&&preset!=='wreck';$<HTMLInputElement>('field-state').checked=preset!=='off'&&preset!=='wreck'&&preset!=='waiting';$<HTMLSelectElement>('work-state').value=preset==='waiting'?'waiting':'working';$<HTMLInputElement>('integrity').value=preset==='light'?'85':preset==='severe'?'35':preset==='wreck'?'0':'100';$<HTMLInputElement>('temperature').value=preset==='trip'?'95':'25';$<HTMLInputElement>('tripped').checked=preset==='trip';});
const base=createWorld();
function sample(kind:Kind):{world:World;entity:Entity}{const v=controls(),e=newEntity(kind,0,0,'e1',v.rotation),f=footprint(e);e.x=(5.5-f.w)/2;e.y=(5.5-f.h)/2;e.health=v.health;e.temperature=v.temperature;e.tripped=v.trip;e.powered=(v.power||kind==='junction'||kind==='tuner')&&!v.trip&&v.health>0;e.ore=kind==='assembler'?(v.work==='working'?4:0):v.work==='blocked'?20:0;
 e.progress=v.power&&!v.trip&&v.health>0&&v.work==='working'?time%(kind==='extractor'?.65:1.4):0;
 const w=structuredClone(base);w.time=time;w.entities=[e];w.links=[];w.deposits=[{id:'sample',x:0,y:0,w:10,h:10,kind:'ore',remaining:100}];w.stats.network.ports={[e.id]:DEFS[kind].ports.map(()=>({incoming:v.field?20:0,outgoing:v.field?10:0,fields:{}}))};w.ecology.shots=[];
 if(kind==='generator'&&v.work==='working'){const load=newEntity('assembler',10,10,'e2');load.powered=v.power;w.entities.push(load);w.links.push({id:'l1',type:'power',a:{node:e.id,port:0},b:{node:load.id,port:0},path:[],packets:[],radius:.5,diagonal:false});}
 if(kind==='sentry'&&v.work==='working'&&e.powered)w.ecology.shots=[{from:center(e),to:{x:e.x+4,y:e.y},ttl:.1}];return {world:w,entity:e};}
function draw(now:number){const dt=Math.min(.1,(now-last)/1000);last=now;if(!paused&&!document.hidden)time+=dt;const values=controls();$('integrity-value').textContent=values.health+'%';$('temperature-value').textContent=values.temperature+'°C';
 for(const {kind,card,renderer:r} of views){const {world:w,entity:e}=sample(kind),state=equipmentState(w,e),c=r.ctx,f=footprint(e);c.clearRect(0,0,r.width,r.height);c.fillStyle='#142325';c.fillRect(0,0,r.width,r.height);r.view.panX=(r.width-r.scale*5.5)/2;const p=r.screen(e.x,e.y);c.strokeStyle='#5d775338';c.strokeRect(p.x,p.y,f.w*r.scale,f.h*r.scale);r.machine(e,1,state,time);equipmentEffects(c,state,p.x,p.y,f.w*r.scale,f.h*r.scale,time);card.querySelector('.state-label')!.textContent=`${state.label} · ${state.damageLabel||'INTACT'}${state.hot?' · HOT':''}`;card.dataset.work=state.work;card.dataset.condition=String(state.condition);card.dataset.moving=String(state.moving);card.dataset.powered=String(state.powered);}
 requestAnimationFrame(draw);
}
window.addEventListener('resize',resize);
await loadSprites();resize();$('review-status').textContent=`State assets loaded · ${kinds.filter(k=>sprites[`${k}-condition-v1`]).length}/9 condition atlases · synthetic state review`;requestAnimationFrame(draw);
