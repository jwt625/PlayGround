export type Kind='extractor'|'assembler'|'generator'|'reference'|'junction'|'tuner'|'emitter'|'dump'|'sentry';
export interface Definition {name:string;w:number;h:number;cost:number;watts:number;ports:string[];description:string}
export const DEFS:Record<Kind,Definition>={
 extractor:{name:'Extractor',w:3,h:3,cost:8,watts:8,ports:[],description:'Place over ore. Route extracted material to an assembler.'},
 assembler:{name:'Assembler',w:3,h:3,cost:10,watts:6,ports:[],description:'Consumes 2 ore per assembly. Outputs to the shared construction stock.'},
 generator:{name:'Power unit',w:3,h:2,cost:14,watts:0,ports:[],description:'240 power units. Connect BUS OUT to each machine POWER IN using grid wires. Fuel is abstracted.'},
 reference:{name:'Reference station',w:2,h:2,cost:12,watts:125,ports:['OUT'],description:'Produces 100 field units. Also houses the automatic controller and commissioning tools.'},
 junction:{name:'Four-port junction',w:2,h:2,cost:5,watts:0,ports:['A','B','C','D'],description:'A/B combine into C/D; C/D combine into A/B. Unconnected outputs leak to the environment.'},
 tuner:{name:'Phase tuner',w:2,h:1,cost:6,watts:0,ports:['IN','OUT'],description:'Fine path adjustment ±180°. Its phase changes slowly with temperature.'},
 emitter:{name:'Field emitter',w:3,h:3,cost:10,watts:4,ports:['IN'],description:'Directs its incident field toward the frontier target. Two phased emitters fill the target mode.'},
 sentry:{name:'Perimeter sentry',w:2,h:2,cost:8,watts:6,ports:[],description:'Electrical perimeter defense. Requires a power wire. Range 8 tiles; 12 damage/s against aggressive wildlife. Ammunition is abstracted.'},
 dump:{name:'Cooled dump',w:2,h:2,cost:5,watts:2,ports:['IN'],description:'Absorbs unused output as heat. Protection trips at 85°C; bypassing it risks destruction.'}
};
