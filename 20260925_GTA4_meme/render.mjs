import {createCanvas,loadImage} from '@napi-rs/canvas';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import './config.js';import './draw.js';
process.chdir(path.dirname(fileURLToPath(import.meta.url)));
const args=process.argv.slice(2),value=(flag,def)=>{const i=args.indexOf(flag);return i<0?def:args[i+1];};
const config=value('--config',null),c=config?JSON.parse(await readFile(config,'utf8')):globalThis.DEFAULT_MEME;
if(args.includes('--clean'))c.labels=false;
const output=value('--output','output/v3/ai-industry.mp4'),fps=30,width=Number(value('--width',1440)),height=Math.round(width*900/1440/2)*2;
if(!c.cards?.length||c.hold<=0||c.fade<=0)throw new Error('Need cards, positive hold and fade.');
const audioStart=c.audioStart??0;
if(!Number.isFinite(audioStart)||audioStart<0)throw new Error('Audio start must be a nonnegative number of seconds.');
const images={};await Promise.all(globalThis.MemeRenderer.assetPaths(c).map(async p=>images[p]=await loadImage(p)));
const canvas=createCanvas(width,height),ctx=canvas.getContext('2d'),R=globalThis.MemeRenderer;
await mkdir(path.dirname(output),{recursive:true});
for(let i=0;i<c.cards.length;i++){R.draw(ctx,c,images,c.intro+i*c.hold+c.hold/2);await writeFile(path.join(path.dirname(output),`${c.labels?'card':'clean'}-${c.cards[i].id}.png`),canvas.toBuffer('image/png'));}
const ff=spawn('ffmpeg',['-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgba','-s',`${width}x${height}`,'-r',String(fps),'-i','pipe:0','-ss',String(audioStart),'-i',c.audio,'-map','0:v','-map','1:a','-t',String(R.duration(c)),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-af',`afade=t=out:st=${R.duration(c)-2}:d=2`,'-movflags','+faststart',output],{stdio:['pipe','inherit','inherit']});
const done=once(ff,'close');let failure;ff.stdin.on('error',e=>failure=e);
const frames=Math.round(R.duration(c)*fps);
for(let f=0;f<frames;f++){
  if(failure)throw failure;
  R.draw(ctx,c,images,f/fps);
  if(!ff.stdin.write(canvas.data()))await once(ff.stdin,'drain');
  if(f%(fps*10)===0)console.log(`Rendering ${f/fps}s / ${frames/fps}s`);
}
ff.stdin.end();const [code]=await done;if(code)throw new Error(`ffmpeg exited ${code}`);
console.log(`Saved ${output}: ${width}×${height}, ${fps}fps, ${R.duration(c)} seconds`);
