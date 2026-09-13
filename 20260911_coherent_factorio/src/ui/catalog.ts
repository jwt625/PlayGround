import {DEFS,type Kind} from '../sim/definitions';
export const categories=['All','Extraction','Production','Power','Field','Defense','Logistics'] as const;
export const catalog:Record<Kind,{category:string;needs:string;output:string}>={
 extractor:{category:'Extraction',needs:'Over a resource deposit • 8 electrical power • ORE OUT belt',output:'1 ore / 0.65 s; crystal enters shared stock after frontier clearance'},
 assembler:{category:'Production',needs:'2 ore via ORE IN • 6 electrical power',output:'1 assembly / 1.4 s into shared construction stock'},
 generator:{category:'Power',needs:'Free ground • connect BUS OUT to each load',output:'240 electrical power; fuel abstracted'},
 reference:{category:'Field',needs:'125 electrical power • terminate or use field OUT',output:'100 coherent field units; automatic phase controller'},
 junction:{category:'Field',needs:'Field inputs A/B or C/D • terminate unused outputs',output:'Passive coherent split/combine; no electrical power'},
 tuner:{category:'Field',needs:'Field IN and OUT • room for two ports',output:'±180° phase trim; no electrical power; thermal drift'},
 emitter:{category:'Field',needs:'Field IN • 4 electrical power',output:'Directed field toward frontier target; stray light attracts wildlife'},
 dump:{category:'Field',needs:'Field IN • 2 electrical power for cooling',output:'Absorbs unused field; monitor temperature'},
 sentry:{category:'Defense',needs:'6 electrical power via POWER IN • perimeter location',output:'8-tile range; 12 damage/s to aggressive wildlife; ammunition abstracted'}
};
export function requirements(kind:Kind){const d=DEFS[kind],c=catalog[kind];return `<b>${d.name}</b> · ${d.cost} assemblies · ${d.w} × ${d.h} tiles<br><span>Requires: ${c.needs}</span><br><span>Produces: ${c.output}</span>`;}
