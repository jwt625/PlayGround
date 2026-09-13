import {sprites} from './assets';
import {WIDTH,HEIGHT} from './sim/world';
/** Static presentation cache. Biomes and concrete do not affect collision, resources or routing. */
export function createTerrain():HTMLCanvasElement {
 const scale=32,tile=8*scale,canvas=document.createElement('canvas');canvas.width=WIDTH*scale;canvas.height=HEIGHT*scale;
 const c=canvas.getContext('2d')!;c.fillStyle='#17282b';c.fillRect(0,0,canvas.width,canvas.height);
 const base=sprites['terrain-basalt-v1'];if(base){c.globalAlpha=.42;for(let x=0;x<canvas.width;x+=tile)for(let y=0;y<canvas.height;y+=tile)c.drawImage(base,x,y,tile,tile);c.globalAlpha=1;}
 const patches=[
  {asset:'dust-basin-v1',x:10,y:27,rx:15,ry:9,alpha:.35},
  {asset:'dust-basin-v1',x:52,y:29,rx:17,ry:11,alpha:.3},
  {asset:'fractured-bedrock-v1',x:42,y:16,rx:17,ry:13,alpha:.4},
  {asset:'fractured-bedrock-v1',x:8,y:3,rx:12,ry:8,alpha:.3},
  {asset:'crystal-substrate-v1',x:31,y:8,rx:9,ry:8,alpha:.4},
  {asset:'industrial-paving-v1',x:13,y:10,rx:13,ry:10,alpha:.2}
 ];
 for(const p of patches){const im=sprites[p.asset];if(!im)continue;const layer=document.createElement('canvas');layer.width=p.rx*2*scale;layer.height=p.ry*2*scale;const l=layer.getContext('2d')!;
  for(let x=0;x<layer.width;x+=tile)for(let y=0;y<layer.height;y+=tile)l.drawImage(im,x,y,tile,tile);
  l.globalCompositeOperation='destination-in';l.scale(layer.width,layer.height);const fade=l.createRadialGradient(.5,.5,.16,.5,.5,.5);fade.addColorStop(0,'#fff');fade.addColorStop(1,'#fff0');l.fillStyle=fade;l.fillRect(0,0,1,1);
  c.globalAlpha=p.alpha;c.drawImage(layer,(p.x-p.rx)*scale,(p.y-p.ry)*scale);c.globalAlpha=1;
 }
 return canvas;
}
