import manifest from '../assets/manifest.json';
const urls=import.meta.glob('../assets/sprites/*.png',{eager:true,query:'?url',import:'default'}) as Record<string,string>;
export const spriteUrls:Record<string,string>=Object.fromEntries(manifest.assets.map(a=>[a.id,urls[`../assets/${a.path}`]]));
export const sprites:Record<string,HTMLImageElement>={};
export async function loadSprites(){await Promise.all(Object.entries(spriteUrls).map(([id,url])=>new Promise<void>((resolve,reject)=>{const im=new Image();im.onload=()=>{sprites[id]=im;resolve();};im.onerror=()=>reject(new Error(`Could not load ${id}`));im.src=url;})));}
