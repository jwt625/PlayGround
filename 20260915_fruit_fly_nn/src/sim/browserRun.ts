import type { RunSpec } from './runner';
import type { ReservoirConfig } from '../connectome/reservoir';
import type { ReservoirGraph } from '../connectome/graph';

export interface BrowserRun {
  version:number;
  spec:RunSpec;
  reservoir:ReservoirConfig;
  graph:{source:string;n:number;note:string;bodyIds:number[];pre:number[];post:number[];weights:number[]};
  readouts:{before:number[];after:number[]};
}
export function graphFromBrowserRun(run:BrowserRun):ReservoirGraph{
  const g=run.graph;
  if(g.source!=='malecns'||g.bodyIds.length!==g.n||g.pre.length!==g.post.length||g.pre.length!==g.weights.length)throw new Error('Invalid MaleCNS graph bundle');
  for(let i=0;i<g.pre.length;i++)if(!Number.isInteger(g.pre[i])||!Number.isInteger(g.post[i])||g.pre[i]<0||g.post[i]<0||g.pre[i]>=g.n||g.post[i]>=g.n||!Number.isFinite(g.weights[i]))throw new Error('Invalid graph endpoint or weight');
  return {n:g.n,source:'malecns',note:g.note,bodyIds:Int32Array.from(g.bodyIds),edges:{pre:Int32Array.from(g.pre),post:Int32Array.from(g.post),weight:Float64Array.from(g.weights)}};
}
