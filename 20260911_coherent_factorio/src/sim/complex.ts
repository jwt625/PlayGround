/** Complex field amplitude; squared magnitude is power in game units. */
export type Complex = readonly [number, number];
export const ZERO: Complex = [0, 0];
export const c = (re: number, im = 0): Complex => [re, im];
export const add = (a: Complex, b: Complex): Complex => [a[0] + b[0], a[1] + b[1]];
export const sub = (a: Complex, b: Complex): Complex => [a[0] - b[0], a[1] - b[1]];
export const mul = (a: Complex, b: Complex): Complex => [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]];
export const power = (a: Complex) => a[0]*a[0]+a[1]*a[1];
export const polar = (amplitude: number, phase: number): Complex => [amplitude*Math.cos(phase), amplitude*Math.sin(phase)];
export function div(a: Complex, b: Complex): Complex { const d=power(b); if(d<1e-24) throw new Error('Singular wave network'); return [(a[0]*b[0]+a[1]*b[1])/d,(a[1]*b[0]-a[0]*b[1])/d]; }
/** Partial-pivot Gaussian elimination, intentionally bounded to small prototype networks. */
export function solveLinear(matrix: Complex[][], rhs: Complex[]): Complex[] {
 const n=rhs.length;
 if(matrix.length!==n || matrix.some(row=>row.length!==n)) throw new Error('Invalid matrix dimensions');
 const a=matrix.map((row,i)=>[...row,rhs[i]]);
 for(let k=0;k<n;k++) {
  let pivot=k; for(let i=k+1;i<n;i++) if(power(a[i][k])>power(a[pivot][k])) pivot=i;
  if(power(a[pivot][k])<1e-18) throw new Error('Singular wave network: undamped feedback loop');
  [a[k],a[pivot]]=[a[pivot],a[k]];
  const v=a[k][k]; for(let j=k;j<=n;j++) a[k][j]=div(a[k][j],v);
  for(let i=k+1;i<n;i++) {const f=a[i][k]; for(let j=k;j<=n;j++) a[i][j]=sub(a[i][j],mul(f,a[k][j]));}
 }
 const x:Complex[]=Array.from({length:n},()=>ZERO);
 for(let i=n-1;i>=0;i--) {let v=a[i][n];for(let j=i+1;j<n;j++)v=sub(v,mul(a[i][j],x[j]));x[i]=v;}
 return x;
}
