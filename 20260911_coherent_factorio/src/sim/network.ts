import { type Complex, ZERO, c, add, sub, mul, polar, power, solveLinear } from './complex';
export interface Component {id:string; s:Complex[][]; emission?:{group:string; fields:Complex[]}}
export interface Endpoint {node:string; port:number}
export interface WaveLink {a:Endpoint;b:Endpoint; amplitude:number; phase:number}
export interface PortResult {incoming:number;outgoing:number; fields:Record<string,{a:Complex;b:Complex}>}
export interface NetworkResult {ports:Record<string,PortResult[]>; absorbed:Record<string,number>; sourcePower:number;linkLoss:number;escaped:number;residual:number}
export const key=(p:Endpoint)=>`${p.node}:${p.port}`;
export const matched=(id:string,ports=1):Component=>({id,s:Array.from({length:ports},()=>Array.from({length:ports},()=>ZERO))});
export function source(id:string,p:number,group=id):Component {if(!Number.isFinite(p)||p<0)throw new Error('Invalid source power');return {...matched(id),emission:{group,fields:[c(Math.sqrt(p))]}};}
export const through=(id:string,phase=0):Component=>({id,s:[[ZERO,polar(1,phase)],[polar(1,phase),ZERO]]});
/** Reciprocal, lossless 180-degree hybrid: ports 0/1 on one side, 2/3 on the other. */
export function hybrid(id:string):Component {const r=c(Math.SQRT1_2),m=c(-Math.SQRT1_2);return {id,s:[[ZERO,ZERO,r,r],[ZERO,ZERO,r,m],[r,r,ZERO,ZERO],[r,m,ZERO,ZERO]]};}
export function solveNetwork(components:Component[],links:WaveLink[]):NetworkResult {
 const offsets=new Map<string,number>(),indices=new Map<string,number>();let n=0;
 for(const comp of components){if(offsets.has(comp.id))throw new Error('Duplicate component ID');offsets.set(comp.id,n);const k=comp.s.length;if(!k||comp.s.some(r=>r.length!==k||r.some(z=>!z.every(Number.isFinite))))throw new Error('Invalid scattering matrix');for(let j=0;j<k;j++)indices.set(`${comp.id}:${j}`,n++);if(comp.emission && (comp.emission.fields.length!==k || comp.emission.fields.some(z=>!z.every(Number.isFinite)) || comp.s.some(r=>r.some(z=>power(z)>0))))throw new Error('Sources must be matched with one emission per port');}
 if(n>128)throw new Error('Prototype limit: 128 wave ports');
 const peers=new Map<number,{index:number;gain:Complex;amplitude:number}>();
 for(const link of links){const a=indices.get(key(link.a)),b=indices.get(key(link.b));if(a===undefined||b===undefined||a===b||peers.has(a)||peers.has(b))throw new Error('Invalid or occupied wave port');if(!Number.isFinite(link.phase)||!Number.isFinite(link.amplitude)||link.amplitude<0||link.amplitude>1)throw new Error('Invalid passive link gain');const gain=polar(link.amplitude,link.phase);peers.set(a,{index:b,gain,amplitude:link.amplitude});peers.set(b,{index:a,gain,amplitude:link.amplitude});}
 const matrix:Complex[][]=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>c(i===j?1:0)));
 for(const comp of components){const start=offsets.get(comp.id)!;for(let i=0;i<comp.s.length;i++)for(let j=0;j<comp.s.length;j++){const peer=peers.get(start+j);if(peer)matrix[start+i][peer.index]=sub(matrix[start+i][peer.index],mul(comp.s[i][j],peer.gain));}}
 const result:NetworkResult={ports:{},absorbed:{},sourcePower:0,linkLoss:0,escaped:0,residual:0};
 for(const comp of components){result.ports[comp.id]=comp.s.map(()=>({incoming:0,outgoing:0,fields:{}}));result.absorbed[comp.id]=0;}
 const groups=[...new Set(components.flatMap(comp=>comp.emission?[comp.emission.group]:[]))];
 for(const group of groups){const rhs:Complex[]=Array.from({length:n},()=>ZERO);let supplied=0;
  for(const comp of components)if(comp.emission?.group===group){const start=offsets.get(comp.id)!;comp.emission.fields.forEach((v,i)=>{rhs[start+i]=v;supplied+=power(v);});}
  const outgoing=solveLinear(matrix,rhs);result.sourcePower+=supplied;
  for(const comp of components){const start=offsets.get(comp.id)!;let absorbed=0;
   for(let i=0;i<comp.s.length;i++){const index=start+i,peer=peers.get(index),b=outgoing[index],a=peer?mul(peer.gain,outgoing[peer.index]):ZERO;const pin=power(a),pout=power(b);const entry=result.ports[comp.id][i];entry.incoming+=pin;entry.outgoing+=pout;entry.fields[group]={a,b};absorbed+=pin-pout+power(rhs[index]);if(peer)result.linkLoss+=pout*(1-peer.amplitude**2);else result.escaped+=pout;}
   result.absorbed[comp.id]+=absorbed;
  }
 }
 result.residual=result.sourcePower-result.linkLoss-result.escaped-Object.values(result.absorbed).reduce((a,b)=>a+b,0);
 return result;
}
