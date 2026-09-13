import {DEFS,type Kind} from './definitions';
export type Rotation=0|1|2|3;
export type Transport='field'|'material'|'power';
export interface Point {x:number;y:number}
export interface Body extends Point {kind:Kind;rotation:Rotation}
export interface Port {id:string;role:'input'|'output'|'bidirectional';position:Point;normal:Point}
export function footprint(e:Pick<Body,'kind'|'rotation'>){const d=DEFS[e.kind];return e.rotation%2?{w:d.h,h:d.w}:{w:d.w,h:d.h};}
function rotate(p:Point,r:Rotation):Point {for(let i=0;i<r;i++)p={x:-p.y||0,y:p.x||0};return p;}
export function ports(e:Body,type:Transport='field'):Port[]{
 const d=DEFS[e.kind];let local:Port[]=[];
 const make=(id:string,x:number,y:number,nx:number,ny:number,role:Port['role']='bidirectional'):Port=>({id,position:{x,y},normal:{x:nx,y:ny},role});
 if(type==='field')local=d.ports.map((name,i)=>d.ports.length===4?make(name,i<2?0:d.w,i%2?1.5:.5,i<2?-1:1,0):make(name,d.ports.length===2?(i?d.w:0):e.kind==='reference'?d.w:0,d.h/2,d.ports.length===2?(i?1:-1):e.kind==='reference'?1:-1,0));
 if(type==='material'&&e.kind==='extractor')local=[make('ORE OUT',d.w/2,d.h,0,1,'output')];
 if(type==='material'&&e.kind==='assembler')local=[make('ORE IN',d.w/2,0,0,-1,'input')];
 if(type==='power'&&(d.watts>0||e.kind==='generator'))local=[make(e.kind==='generator'?'BUS OUT':'POWER IN',d.w/2,0,0,-1,e.kind==='generator'?'output':'input')];
 const f=footprint(e);
 return local.map(p=>{const q=rotate({x:p.position.x-d.w/2,y:p.position.y-d.h/2},e.rotation);return {...p,position:{x:e.x+f.w/2+q.x,y:e.y+f.h/2+q.y},normal:rotate(p.normal,e.rotation)};});
}
export const inside=(p:Point,e:Body)=>{const f=footprint(e);return p.x>e.x&&p.x<e.x+f.w&&p.y>e.y&&p.y<e.y+f.h;};
export const same=(a:Point,b:Point)=>a.x===b.x&&a.y===b.y;
export const distance=(a:Point,b:Point)=>Math.hypot(a.x-b.x,a.y-b.y);
export function routeMetrics(path:Point[],radius=.5){
 const corners:Point[]=[];for(const p of path){while(corners.length>=2){const a=corners.at(-2)!,b=corners.at(-1)!;if(Math.abs((b.x-a.x)*(p.y-b.y)-(b.y-a.y)*(p.x-b.x))>1e-8)break;corners.pop();}corners.push(p);}
 let length=0,bendExponent=0;const bends:{point:Point;radius:number;angle:number;bad:boolean}[]=[];
 for(let i=1;i<corners.length;i++)length+=distance(corners[i-1],corners[i]);
 for(let i=1;i<corners.length-1;i++){const a=corners[i-1],b=corners[i],c=corners[i+1],u=distance(a,b),v=distance(b,c);const angle=Math.acos(Math.max(-1,Math.min(1,((b.x-a.x)*(c.x-b.x)+(b.y-a.y)*(c.y-b.y))/(u*v))));const effective=Math.min(radius,Math.min(u,v)/2/Math.tan(angle/2));const bad=effective<.5-1e-8;bends.push({point:b,radius:effective,angle,bad});bendExponent+=.001*(angle/(Math.PI/2))*(1+8*Math.max(0,1-effective/.5)**2);}
 return {length,bends,propagationExponent:.012*length,bendExponent};
}
export function pointAt(path:Point[],travel:number):Point {for(let i=1;i<path.length;i++){const n=distance(path[i-1],path[i]);if(travel<=n){const t=Math.max(0,travel/n);return {x:path[i-1].x+(path[i].x-path[i-1].x)*t,y:path[i-1].y+(path[i].y-path[i-1].y)*t};}travel-=n;}return path.at(-1)!;}
