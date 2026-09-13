import {type Body,type Point,type Port,inside,same} from './geometry';
export interface RouteOptions {diagonal?:boolean;radius?:number;path?:Point[];waypoints?:Point[]}
const key=(p:Point)=>`${p.x},${p.y}`;
const grid=(p:Point)=>Number.isInteger(p.x*2)&&Number.isInteger(p.y*2);
export function segmentClear(a:Point,b:Point,bodies:Body[]){return !bodies.some(e=>inside(a,e)||inside(b,e)||inside({x:(a.x+b.x)/2,y:(a.y+b.y)/2},e));}
export function validPath(path:Point[],a:Port,b:Port,bodies:Body[],width:number,height:number,diagonal=false):boolean {
 if(!Array.isArray(path)||path.length<3||path.length>20000||!path.every(p=>p&&grid(p)&&p.x>=0&&p.y>=0&&p.x<=width&&p.y<=height)||!same(path[0],a.position)||!same(path.at(-1)!,b.position))return false;
 if(!same(path[1],{x:a.position.x+a.normal.x*.5,y:a.position.y+a.normal.y*.5})||!same(path.at(-2)!,{x:b.position.x+b.normal.x*.5,y:b.position.y+b.normal.y*.5}))return false;
 const seen=new Set<string>();for(let i=0;i<path.length;i++){const p=path[i];if(seen.has(key(p)))return false;seen.add(key(p));if(i){const q=path[i-1],dx=Math.abs(p.x-q.x),dy=Math.abs(p.y-q.y);if(Math.max(dx,dy)!==.5||(!diagonal&&dx!==0&&dy!==0)||!segmentClear(q,p,bodies))return false;}}
 return true;
}
/** Deterministic breadth-first route on the half-tile lattice. Crossings do not connect. */
export function findRoute(a:Port,b:Port,bodies:Body[],width:number,height:number,options:RouteOptions={}):Point[]|null {
 if(options.path)return validPath(options.path,a,b,bodies,width,height,options.diagonal)?structuredClone(options.path):null;
 const start={x:a.position.x+a.normal.x*.5,y:a.position.y+a.normal.y*.5},end={x:b.position.x+b.normal.x*.5,y:b.position.y+b.normal.y*.5};
 const directions=[[1,0],[0,1],[-1,0],[0,-1],...(options.diagonal?[[1,1],[-1,1],[-1,-1],[1,-1]]:[])];
 const result=[a.position,start];
 for(const goal of [...options.waypoints??[],end]){
  if(!grid(goal))return null;
  const queue=[result.at(-1)!],previous=new Map<string,Point|null>([[key(queue[0]),null]]);let found=false;
  for(let i=0;i<queue.length;i++){const p=queue[i];if(same(p,goal)){found=true;break;}for(const [dx,dy] of directions){const q={x:p.x+dx*.5,y:p.y+dy*.5};if(q.x<0||q.y<0||q.x>width||q.y>height||previous.has(key(q))||same(q,a.position)||same(q,b.position)||!segmentClear(p,q,bodies))continue;previous.set(key(q),p);queue.push(q);}}
  if(!found)return null;const section:Point[]=[];for(let p:Point|null=goal;p;p=previous.get(key(p))!)section.push(p);result.push(...section.reverse().slice(1));
 }
 result.push(b.position);return validPath(result,a,b,bodies,width,height,options.diagonal)?result:null;
}
