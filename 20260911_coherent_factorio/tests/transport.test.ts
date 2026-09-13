import {test} from 'node:test';
import assert from 'node:assert/strict';
import {captureBlueprint,stampBlueprint,waveLinks,createWorld,newEntity,connect,disconnect,place,rotateEntity,step,evaluate,serialize,deserialize,ports,footprint,routeMetrics,type World} from '../src/sim/world';
import {inside,distance} from '../src/sim/geometry';
import {findRoute,validPath} from '../src/sim/routing';
const ticks=(w:World,n:number)=>{for(let i=0;i<n;i++)step(w);};

test('all equipment has stable typed physical interfaces and four-turn geometry',()=>{
 const w=createWorld();for(const e of w.entities){const original=ports(e);for(let r=0;r<4;r++){e.rotation=r as 0|1|2|3;for(const type of ['field','material','power'] as const)for(const p of ports(e,type)){assert.equal(Number.isInteger(p.position.x*2),true);assert.equal(Number.isInteger(p.position.y*2),true);assert.equal(Math.hypot(p.normal.x,p.normal.y),1);}assert.deepEqual(ports(e).map(p=>p.id),original.map(p=>p.id));}e.rotation=0;assert.deepEqual(ports(e),original);}
 const tuner=newEntity('tuner',3,3,'e90',1);assert.deepEqual(footprint(tuner),{w:1,h:2});assert.deepEqual(ports(tuner)[0].normal,{x:0,y:-1});
});
test('routes avoid obstacles, use port normals, and allow explicit diagonals and waypoints',()=>{
 const a=newEntity('reference',1,4,'e1'),b=newEntity('dump',10,8,'e2'),obstacle=newEntity('emitter',5,4,'e3');const bodies=[a,b,obstacle];
 for(const diagonal of [false,true]){const path=findRoute(ports(a)[0],ports(b)[0],bodies,20,20,{diagonal,waypoints:[{x:8,y:3}]});assert.ok(path);assert.ok(validPath(path,ports(a)[0],ports(b)[0],bodies,20,20,diagonal));assert.ok(path.some(p=>p.x===8&&p.y===3));assert.ok(!path.some(p=>inside(p,obstacle)));if(diagonal)assert.ok(path.some((p,i)=>i&&p.x!==path[i-1].x&&p.y!==path[i-1].y));}
});
test('wave phase uses installed route length and bend losses close the power ledger',()=>{
 const w=createWorld(),link=w.links.find(l=>l.type==='field')!;const before=evaluate(w);assert.equal(waveLinks(w)[0].phase,.31*routeMetrics(link.path).length);assert.ok(Math.abs(before.propagationLoss+before.bendRadiation-before.network.linkLoss)<1e-9);
 const bent=w.links.find(l=>l.type==='field'&&routeMetrics(l.path,l.radius).bends.length)!;const good=routeMetrics(bent.path,.5),bad=routeMetrics(bent.path,0);assert.ok(bad.bendExponent>good.bendExponent);bent.radius=0;const after=evaluate(w);assert.ok(after.bendRadiation>before.bendRadiation);assert.ok(Math.abs(after.network.residual)<1e-9);assert.ok(routeMetrics(link.path).length>=distance(link.path[0],link.path.at(-1)!));
});
test('belts have transit delay, capacity, backpressure, and conserve buffered items on disconnect',()=>{
 const w=createWorld(),belt=w.links.find(l=>l.type==='material')!,a=w.entities.find(e=>e.id===belt.a.node)!,b=w.entities.find(e=>e.id===belt.b.node)!;a.ore=10;step(w);assert.equal(b.ore,0);assert.equal(belt.packets.length,1);assert.equal(belt.packets[0],0);
 b.ore=20;b.tripped=true;ticks(w,100);const length=routeMetrics(belt.path).length;assert.equal(belt.packets[0],length);assert.ok(belt.packets.length<=Math.floor(length*2)+1);for(let i=1;i<belt.packets.length;i++)assert.ok(belt.packets[i-1]-belt.packets[i]>=.5-1e-8);
 const inFlight=belt.packets.length,scrap=w.stock.scrap;disconnect(w,belt.id);assert.equal(w.stock.scrap,scrap+inFlight);
});
test('power requires a wire, each receiver has one feed, and wires rotate with equipment',()=>{
 const w=createWorld(),load=w.entities.find(e=>e.kind==='assembler')!,g=w.entities.find(e=>e.kind==='generator')!,wire=w.links.find(l=>l.type==='power'&&l.b.node===load.id)!;disconnect(w,wire.id);evaluate(w);assert.equal(load.powered,false);
 assert.equal(connect(w,'power',{node:g.id,port:0},{node:load.id,port:0}),'');assert.ok(connect(w,'power',{node:g.id,port:0},{node:load.id,port:0}));assert.ok(connect(w,'material',{node:g.id,port:0},{node:load.id,port:0}));evaluate(w);assert.equal(load.powered,true);assert.ok(rotateEntity(w,g.id));
 assert.equal(place(w,'tuner',17,12,1),'');const tuner=w.entities.at(-1)!;assert.equal(tuner.rotation,1);assert.equal(rotateEntity(w,tuner.id),'');assert.equal(tuner.rotation,2);
});
test('new equipment cannot overwrite installed routes',()=>{
 const w=createWorld();w.frontier=true;w.stock.assemblies=100;const a=newEntity('reference',30,20,'e90'),b=newEntity('dump',40,20,'e91');w.entities.push(a,b);assert.equal(connect(w,'field',{node:a.id,port:0},{node:b.id,port:0}),'');assert.match(place(w,'dump',35,20),/Route/);
});
test('save preserves routes, orientation and packets; rejects broken geometry and overpacked belts',()=>{
 const w=createWorld();ticks(w,20);const restored=deserialize(serialize(w));assert.deepEqual(restored.links.map(l=>l.id),w.links.map(l=>l.id));assert.deepEqual(restored.links.map(l=>l.path),w.links.map(l=>l.path));assert.deepEqual(restored.links.map(l=>l.packets),w.links.map(l=>l.packets));ticks(w,10);ticks(restored,10);assert.deepEqual(restored.stock,w.stock);
 const bad=JSON.parse(serialize(w));bad.links[0].path[1].x+=.25;assert.throws(()=>deserialize(JSON.stringify(bad)),/Invalid/);
 const packed=JSON.parse(serialize(w));packed.links.find((l:any)=>l.type==='material').packets=[1,1];assert.throws(()=>deserialize(JSON.stringify(packed)),/Invalid/);
});
test('legacy v1 saves migrate to physical routes and explicit power wires',()=>{
 const w=createWorld(),old=JSON.parse(serialize(w));old.version=1;for(const e of old.entities)delete e.rotation;old.links=old.links.filter((l:any)=>l.type!=='power').map(({id,a,b,type}:any)=>({id,a,b,type}));const migrated=deserialize(JSON.stringify(old));assert.equal(migrated.version,3);assert.ok(migrated.links.some(l=>l.type==='power'));assert.ok(migrated.entities.every(e=>e.powered));assert.ok(migrated.links.every(l=>l.path.length>=3));
});

test('a generator overload removes power until its excess load is disconnected',()=>{
 const w=createWorld();w.stock.assemblies=100;assert.equal(place(w,'reference',20,20),'');const load=w.entities.at(-1)!,g=w.entities.find(e=>e.kind==='generator')!;assert.equal(connect(w,'power',{node:g.id,port:0},{node:load.id,port:0}),'');evaluate(w);assert.ok(w.stats.demand>240);assert.equal(w.entities.find(e=>e.kind==='assembler')!.powered,false);disconnect(w,w.links.at(-1)!.id);evaluate(w);assert.equal(w.entities.find(e=>e.kind==='assembler')!.powered,true);assert.equal(load.powered,false);
});


test('diagonal segments cannot cut through a newly placed footprint corner',()=>{
 const w=createWorld();w.links=[{id:'l99',type:'field',a:{node:'e4',port:0},b:{node:'e5',port:0},path:[{x:17,y:12.5},{x:17.5,y:12}],diagonal:true,radius:.5,packets:[]}];assert.match(place(w,'tuner',17,12),/Route/);
});
test('blueprint route validation fails atomically before spending or placing',()=>{
 const w=createWorld();w.frontier=true;w.commission.state='qualified';assert.equal(captureBlueprint(w),'');w.stock.assemblies=300;w.blueprint!.links[0].path[1].x+=.25;const before=serialize(w);assert.ok(stampBlueprint(w,38,3));assert.equal(serialize(w),before);
});
