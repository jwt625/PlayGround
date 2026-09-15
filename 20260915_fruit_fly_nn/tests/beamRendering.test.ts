import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { OpticalBench } from '../src/renderer/bench';
import { DISPLAY } from '../src/renderer/scene';
import { createArrayConfig } from '../src/optics/geometry';
import { defaultChannelCommand, type ChannelActual } from '../src/optics/types';
import { channelFieldFast } from '../src/optics/gaussian';

describe('individual beam envelope geometry',()=>{
  const array=createArrayConfig();
  const states=()=>array.channelIds.map(()=>({...defaultChannelCommand(),hiddenPiston_rad:0,hiddenGain:1} as ChannelActual));
  const beams=(bench:OpticalBench)=>bench.group.children.filter((o):o is THREE.Mesh=>o instanceof THREE.Mesh && (o.geometry as THREE.CylinderGeometry).parameters?.heightSegments===32);
  const update=(bench:OpticalBench,actual:ChannelActual[])=>bench.update({actual,selectedIndex:-1,showBeams:true,beamRange_m:1,beamTarget:new THREE.Vector3(0,0,100)});
  it('uses the field evaluator Gaussian radius instead of converging every tube on the centroid',()=>{
    const bench=new OpticalBench(array),actual=states();actual[0].curvature_per_m=3;update(bench,actual);
    const p=beams(bench)[0].geometry.attributes.position;
    const x=p.getX(0)/DISPLAY.scale,y=p.getY(0)/DISPLAY.scale,z=p.getZ(0)/DISPLAY.targetDistance;
    const edge=channelFieldFast(array,0,actual[0],x,y,z);
    const center=channelFieldFast(array,0,actual[0],array.x_m[0],array.y_m[0],z);
    expect((edge.re**2+edge.im**2)/(center.re**2+center.im**2)).toBeCloseTo(Math.exp(-2),5);
  });
  it('piston leaves an envelope unchanged; individual tilt changes only its channel',()=>{
    const bench=new OpticalBench(array),actual=states();update(bench,actual);
    const meshes=beams(bench);const before=meshes.map(m=>Array.from(m.geometry.attributes.position.array));
    actual[0].piston_rad=2;update(bench,actual);expect(Array.from(meshes[0].geometry.attributes.position.array)).toEqual(before[0]);
    actual[0].tiltX=.01;update(bench,actual);expect(Array.from(meshes[0].geometry.attributes.position.array)).not.toEqual(before[0]);
    expect(Array.from(meshes[1].geometry.attributes.position.array)).toEqual(before[1]);
  });
});
