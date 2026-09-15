/** Publish saved headless results for the inspector; never substitutes a browser policy. */
import { readFileSync, writeFileSync, copyFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { buildRun, type RunSpec } from '../src/sim/runner';
const runDir=process.argv[2];
if (!runDir) throw new Error('Usage: tsx scripts/publish-training-review.ts <run-dir>');
const read=(p:string)=>JSON.parse(readFileSync(join(runDir,p),'utf8'));
const spec:RunSpec=read('run-spec.json');const progress=read('progress.json');const evaluation=read('evaluation.json');
if(progress.status!=='complete')throw new Error('Run is incomplete');
const built=buildRun(spec);
const hashes:Record<string,string>={};
function hashSources(path:string):void{for(const item of readdirSync(path,{withFileTypes:true})){const file=join(path,item.name);if(item.isDirectory())hashSources(file);else hashes[file]=createHash('sha256').update(readFileSync(file)).digest('hex');}}
hashSources('src');hashSources('tests');
const metadata={graph:{nodes:built.graph.n,edges:built.graph.edges.pre.length,source:built.graph.source,note:built.graph.note},sourceHashesAtReview:hashes,cacheManifest:JSON.parse(readFileSync('data/manifests/malecns-v1.0.json','utf8')),note:'Graph reconstructed deterministically from saved run-spec. Source hashes captured at review after CLI/frontend changes, not at original process launch.'};
writeFileSync(join(runDir,'review-metadata.json'),JSON.stringify(metadata,null,2));
const summary={runDir,nodes:built.graph.n,edges:built.graph.edges.pre.length,generations:spec.train.generations,initial:progress.initialFitness,final:progress.finalFitness,transfer:evaluation.transferToNewArrays.meanStrehl,sameArray:evaluation.sameArray.meanStrehl,graphSource:built.graph.source,notes:built.graph.note};
mkdirSync('public/training',{recursive:true});writeFileSync('public/training/latest.json',JSON.stringify(summary,null,2));copyFileSync(join(runDir,'learning-curve.svg'),'public/training/learning-curve.svg');console.log(summary);
