import * as THREE from 'three';
import type { ReservoirGraph } from '../connectome/graph';

export interface NeuronLocation { bodyId:number; type:string|null; superclass:string|null; soma:number[]|null }

// Per-point shader so each soma can grow and brighten with its own activation.
// A uniform PointsMaterial cannot vary size per point, which made activation
// changes hard to see.
const POINT_VERTEX = `
  attribute float aActivation;
  attribute vec3 aTint;
  varying vec3 vColor;
  varying float vAct;
  void main(){
    vColor = aTint;
    vAct = aActivation;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = (1.5 + 13.0 * aActivation) * (95.0 / max(1.0, -mv.z));
    gl_Position = projectionMatrix * mv;
  }
`;
const POINT_FRAGMENT = `
  varying vec3 vColor;
  varying float vAct;
  void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) discard;
    float core = smoothstep(0.5, 0.0, r);
    vec3 c = vColor * (0.35 + 1.65 * vAct);
    gl_FragColor = vec4(c, core * (0.07 + 0.93 * vAct));
  }
`;

const DIM: [number, number, number] = [0.03, 0.05, 0.08];
const POSITIVE: [number, number, number] = [0.30, 1.00, 1.00];
const NEGATIVE: [number, number, number] = [1.00, 0.52, 0.10];

function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** Real soma positions + sampled real edges; colors/size come from the active reservoir. */
export class NeuralActivityView {
  readonly group=new THREE.Group();
  private points:THREE.Points;
  private pointIndices:number[]=[];
  private edgeSources:number[]=[];
  private pointTint:Float32Array;
  private pointActivation:Float32Array;
  private edgeColors:Float32Array;
  private lineGeometry:THREE.BufferGeometry;
  readonly renderedNodes:number;
  readonly renderedEdges:number;
  readonly locations:NeuronLocation[];
  /** Slowly adapting activity reference so contrast holds at any activity level. */
  private activityRef = 0.4;
  /** Per-soma slow baseline; deviations from it are what "blink". */
  private baseline:Float32Array;

  constructor(graph:ReservoirGraph,locations:NeuronLocation[]){
    this.locations=locations;
    if(locations.length!==graph.n)throw new Error('Soma metadata/graph size mismatch');
    locations.forEach((n,i)=>{if(n.bodyId!==graph.bodyIds?.[i])throw new Error('Soma metadata/body-ID order mismatch');});
    const valid=locations.filter(n=>n.soma);const box=new THREE.Box3();
    valid.forEach(n=>box.expandByPoint(new THREE.Vector3(...n.soma! as [number,number,number])));
    const center=box.getCenter(new THREE.Vector3()),extent=box.getSize(new THREE.Vector3());
    const scale=42/Math.max(extent.x,extent.y,extent.z,1); // rescaled for visibility
    const positions:number[]=[];const byIndex=new Map<number,THREE.Vector3>();
    locations.forEach((n,i)=>{if(!n.soma)return;
      const p=new THREE.Vector3((n.soma[0]-center.x)*scale,-(n.soma[2]-center.z)*scale,(n.soma[1]-center.y)*scale);
      byIndex.set(i,p);positions.push(p.x,p.y,p.z);this.pointIndices.push(i);
    });
    this.pointTint=new Float32Array(positions.length);
    this.pointActivation=new Float32Array(this.pointIndices.length);
    this.baseline=new Float32Array(this.pointIndices.length);
    const geom=new THREE.BufferGeometry();
    geom.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    geom.setAttribute('aTint',new THREE.BufferAttribute(this.pointTint,3));
    geom.setAttribute('aActivation',new THREE.BufferAttribute(this.pointActivation,1));
    this.points=new THREE.Points(geom,new THREE.ShaderMaterial({
      vertexShader:POINT_VERTEX,fragmentShader:POINT_FRAGMENT,
      transparent:true,depthWrite:false,blending:THREE.NormalBlending,
    }));
    this.group.add(this.points);
    const edgePositions:number[]=[];
    const stride=Math.max(1,Math.ceil(graph.edges.pre.length/2000));
    for(let i=0;i<graph.edges.pre.length;i+=stride){const a=byIndex.get(graph.edges.pre[i]),b=byIndex.get(graph.edges.post[i]);if(!a||!b)continue;edgePositions.push(a.x,a.y,a.z,b.x,b.y,b.z);this.edgeSources.push(graph.edges.pre[i]);}
    this.edgeColors=new Float32Array(edgePositions.length);
    this.lineGeometry=new THREE.BufferGeometry();this.lineGeometry.setAttribute('position',new THREE.Float32BufferAttribute(edgePositions,3));this.lineGeometry.setAttribute('color',new THREE.BufferAttribute(this.edgeColors,3));
    this.group.add(new THREE.LineSegments(this.lineGeometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.2,depthWrite:false})));
    this.group.position.set(-42,16,15);
    this.renderedNodes=this.pointIndices.length;this.renderedEdges=this.edgeSources.length;
    const canvas=document.createElement('canvas');canvas.width=768;canvas.height=64;const ctx=canvas.getContext('2d')!;
    ctx.fillStyle='#101925';ctx.fillRect(0,0,768,64);ctx.font='26px sans-serif';ctx.fillStyle='#d8f8ff';ctx.fillText(`MaleCNS · ${this.renderedNodes.toLocaleString()} mapped somata`,18,41);
    const title=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),depthTest:false}));title.position.set(0,21,0);title.scale.set(40,3.4,1);this.group.add(title);
  }

  update(activity:Float64Array):void{
    // Robust, slowly adapting reference from the observed peak so a handful of
    // active neurons stand out against a quiet background.
    const stride=Math.max(1,Math.floor(this.pointIndices.length/800));
    let sampleMax=0;
    for(let i=0;i<this.pointIndices.length;i+=stride){const v=Math.abs(activity[this.pointIndices[i]]);if(v>sampleMax)sampleMax=v;}
    const target=Math.max(0.12,sampleMax);
    this.activityRef=this.activityRef*0.9+target*0.1;
    const ref=this.activityRef;
    const floor=0.2*ref;
    const burstScale=0.5*ref;

    this.pointIndices.forEach((source,i)=>{
      const value=activity[source];
      const mag=Math.abs(value);
      const base=this.baseline[i];
      // Fast baseline tracks the local mean; deviations above it are bursts.
      this.baseline[i]=base+(mag-base)*0.08;
      const steady=Math.max(0,Math.min(1,(mag-floor)/Math.max(1e-6,ref-floor)));
      const burst=Math.max(0,Math.min(1,(mag-base)/Math.max(1e-6,burstScale)));
      const a=Math.max(0,Math.min(1,0.2*steady+1.25*burst));
      this.pointActivation[i]=a;
      this.pointTint.set(mix(DIM,value>=0?POSITIVE:NEGATIVE,Math.min(1,0.45*steady+0.8*a)),i*3);
    });
    this.edgeSources.forEach((source,i)=>{
      const mag=Math.abs(activity[source]);
      const norm=Math.max(0,Math.min(1,(mag-floor)/Math.max(1e-6,ref-floor)));
      const a=Math.pow(norm,1.6);
      const c=mix(DIM,POSITIVE,a);
      for(let end=0;end<2;end++)this.edgeColors.set(c,i*6+end*3);
    });
    this.points.geometry.attributes.aTint.needsUpdate=true;
    this.points.geometry.attributes.aActivation.needsUpdate=true;
    this.lineGeometry.attributes.color.needsUpdate=true;
  }

  pick(ray:THREE.Raycaster):NeuronLocation|undefined{
    ray.params.Points={threshold:.4};const hit=ray.intersectObject(this.points)[0];return hit?.index===undefined?undefined:this.locations[this.pointIndices[hit.index]];
  }
}
