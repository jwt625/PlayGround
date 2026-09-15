import {describe,it,expect} from 'vitest';
import {graphFromBrowserRun,type BrowserRun}from'../src/sim/browserRun';
const fixture=()=>({graph:{source:'malecns',n:2,note:'scaled export',bodyIds:[12,43],pre:[0],post:[1],weights:[.003]}} as BrowserRun);
describe('saved browser graph',()=>{
 it('preserves source IDs, directed endpoints and scaled weights exactly',()=>{const g=graphFromBrowserRun(fixture());expect(Array.from(g.bodyIds!)).toEqual([12,43]);expect(g.edges.pre[0]).toBe(0);expect(g.edges.post[0]).toBe(1);expect(g.edges.weight[0]).toBe(.003);});
 it('rejects out-of-range endpoints instead of changing their identity',()=>{const f=fixture();f.graph.post=[2];expect(()=>graphFromBrowserRun(f)).toThrow('endpoint');});
});
