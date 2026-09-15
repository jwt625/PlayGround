/** Offline GLB contact sheet using an intercepted browser origin; no listening port. */
import {chromium} from '@playwright/test';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,extname,sep} from 'node:path';
const root=process.cwd(), out='assets/generated/hardware-v2';
const entries=JSON.parse(readFileSync(out+'/manifest.json','utf8')).components;
const html=`<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#111923;color:#e1e9ef;font:14px system-ui}header{height:44px;padding-left:20px;display:flex;align-items:center;font-size:18px}.grid{display:grid;grid-template-columns:repeat(3,320px)}.cell{height:250px;position:relative}.label{position:absolute;bottom:12px;left:20px}canvas{display:block}</style><header>CBC hardware v2 · original generic display assets</header><div class="grid">${entries.map((e,i)=>`<div class="cell"><canvas id="c${i}"></canvas><div class="label">${e.name}</div></div>`).join('')}</div><script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script><script type="module">
import * as T from 'three';import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const entries=${JSON.stringify(entries)};for(let i=0;i<entries.length;i++){
 const renderer=new T.WebGLRenderer({canvas:document.getElementById('c'+i),antialias:true,preserveDrawingBuffer:true});renderer.setSize(320,228);renderer.setPixelRatio(1);renderer.setClearColor(0x202c39);renderer.outputColorSpace=T.SRGBColorSpace;
 const scene=new T.Scene();scene.add(new T.HemisphereLight(0xffffff,0x596779,3));const light=new T.DirectionalLight(0xffffff,3);light.position.set(2,-3,4);scene.add(light);
 const gltf=await new GLTFLoader().loadAsync('/'+${JSON.stringify(out)}+'/'+entries[i].file);const object=gltf.scene;object.updateMatrixWorld(true);let b=new T.Box3().setFromObject(object),size=b.getSize(new T.Vector3()),center=b.getCenter(new T.Vector3());object.position.sub(center);const g=new T.Group();g.add(object);g.scale.setScalar(1/Math.max(size.x,size.y,size.z));scene.add(g);
 const camera=new T.OrthographicCamera(-.78,.78,.56,-.56,.01,20);camera.up.set(0,0,1);camera.position.set(1.8,-2.7,2.1);camera.lookAt(0,0,0);renderer.render(scene,camera);
}window.previewReady=true;
</script>`;
const b=await chromium.launch();const page=await b.newPage({viewport:{width:960,height:800},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.route('**/*',async route=>{const url=new URL(route.request().url());if(url.hostname!=='hardware-preview.local')return route.abort();if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:html});const path=resolve(root,'.'+decodeURIComponent(url.pathname));if(!path.startsWith(root+sep))return route.abort();try{const body=readFileSync(path);await route.fulfill({contentType:extname(path)==='.js'?'text/javascript':extname(path)==='.glb'?'model/gltf-binary':'application/octet-stream',body});}catch{return route.abort()}});
await page.goto('http://hardware-preview.local/');await page.waitForFunction(()=>window.previewReady===true);await page.screenshot({path:out+'/contact-sheet.png'});await b.close();if(errors.length)throw Error(errors.join('\n'));writeFileSync(out+'/preview-validation.json',JSON.stringify({method:'Offline Playwright request interception; Three.js GLTFLoader; nine rendered contexts',pageErrors:errors,loadedGlbs:entries.length},null,2)+'\n');console.log('Rendered all nine GLBs without page errors; no server opened.');
