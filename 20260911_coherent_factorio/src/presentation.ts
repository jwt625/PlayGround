import type {Kind} from './sim/definitions';
/** Art can be remapped without changing saves, ports, or simulation. */
export interface Appearance {asset:string;sheet?:string;views?:Partial<Record<0|1|2|3,string>>;scale?:number;offset?:{x:number;y:number}}
export const APPEARANCE:Record<Kind,Appearance>={
 extractor:{asset:'starter-extractor',sheet:'starter-extractor-turnaround-v2'},
 assembler:{asset:'compact-assembler',sheet:'compact-assembler-turnaround-v2'},
 generator:{asset:'power-unit',sheet:'power-unit-turnaround-v2'},
 reference:{asset:'reference-control-station',sheet:'reference-control-station-turnaround-v2'},
 junction:{asset:'four-port-junction',sheet:'four-port-junction-turnaround-v2'},
 tuner:{asset:'phase-tuner',sheet:'phase-tuner-turnaround-v2'},
 emitter:{asset:'field-emitter',sheet:'field-emitter-turnaround-v2'},
 sentry:{asset:'power-unit',sheet:'perimeter-sentry-v1'},
 dump:{asset:'cooled-dump',sheet:'cooled-dump-turnaround-v2'},
};
