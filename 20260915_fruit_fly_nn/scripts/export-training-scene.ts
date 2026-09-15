/** Export the actual graph/readout used by a saved run for browser playback. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { buildRun, type RunSpec } from '../src/sim/runner';
const runDir=process.argv[2];if(!runDir)throw new Error('Provide saved run directory');
const read=(name:string)=>JSON.parse(readFileSync(join(runDir,name),'utf8'));
const spec:RunSpec=read('run-spec.json');if(!spec.malecns)throw new Error('Expected real MaleCNS run');
const built=buildRun(spec);const graph=built.graph;
if(!graph.bodyIds)throw new Error('Graph missing original body IDs');
const out='public/training';mkdirSync(out,{recursive:true});
writeFileSync(join(out,'selected-body-ids.json'),JSON.stringify(Array.from(graph.bodyIds)));
const py=spawnSync('.asset-venv/bin/python',['scripts/export-neuron-locations.py',spec.malecns.cacheDir??'data/cache/malecns-v1.0'],{encoding:'utf8'});
if(py.status!==0)throw new Error(py.stderr);console.log(py.stdout);
const bundle={version:1,runDir,spec,graph:{source:graph.source,n:graph.n,note:graph.note,bodyIds:Array.from(graph.bodyIds),pre:Array.from(graph.edges.pre),post:Array.from(graph.edges.post),weights:Array.from(graph.edges.weight)},reservoir:built.reservoir.config,readouts:{before:Array.from(built.readout.getParams()),after:read('readout.json').params},history:read('learning-history.json'),evaluation:read('evaluation.json'),notes:['Graph weights are exported after the same scaling used in headless training; do not scale again.','Initial readout regenerated from saved seed/configuration; final readout is saved checkpoint.','Intermediate-generation policies were not saved.','Positive placeholder transmitter signs and generic input/output mapping remain approximations.']};
writeFileSync(join(out,'scene-run.json'),JSON.stringify(bundle));console.log({nodes:graph.n,edges:graph.edges.pre.length,parameters:built.readout.parameterCount});
