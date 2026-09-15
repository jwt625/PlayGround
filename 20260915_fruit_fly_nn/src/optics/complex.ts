/**
 * Minimal complex-field containers and a radix-2 2D FFT used by the numerical
 * optical reference. Forward transform uses exp(-i 2 pi f x), matching the
 * phasor convention Re{U exp(-i omega t)} with propagation exp(+i k z).
 */

export class ComplexField {
  readonly n: number;
  readonly re: Float64Array;
  readonly im: Float64Array;

  constructor(n: number) {
    if (!Number.isInteger(n) || n <= 0) throw new Error(`ComplexField: bad size ${n}`);
    this.n = n;
    this.re = new Float64Array(n);
    this.im = new Float64Array(n);
  }

  get length(): number {
    return this.re.length;
  }

  clone(): ComplexField {
    const out = new ComplexField(this.n);
    out.re.set(this.re);
    out.im.set(this.im);
    return out;
  }
}

/** In-place complex 1D FFT. Length must be a power of two. */
export function fft1d(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  if (n !== im.length) throw new Error("fft1d: mismatched real/imag lengths");
  if (n === 0) return;
  if ((n & (n - 1)) !== 0) throw new Error(`fft1d: length ${n} is not a power of two`);
  if (n === 1) return;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; (j & bit) !== 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  const sign = inverse ? 2 : -2;
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (sign * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j;
        const b = a + half;
        const vRe = re[b] * curRe - im[b] * curIm;
        const vIm = re[b] * curIm + im[b] * curRe;
        re[b] = re[a] - vRe;
        im[b] = im[a] - vIm;
        re[a] += vRe;
        im[a] += vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/**
 * In-place 2D FFT on a square n x n row-major field. `n` must be a power of two.
 * `inverse` applies the 1/n^2 normalization.
 */
export function fft2d(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const total = re.length;
  const n = Math.round(Math.sqrt(total));
  if (n * n !== total) throw new Error("fft2d: field is not square");
  if ((n & (n - 1)) !== 0) throw new Error(`fft2d: size ${n} is not a power of two`);

  const bufRe = new Float64Array(n);
  const bufIm = new Float64Array(n);

  for (let r = 0; r < n; r++) {
    const off = r * n;
    bufRe.set(re.subarray(off, off + n));
    bufIm.set(im.subarray(off, off + n));
    fft1d(bufRe, bufIm, inverse);
    re.set(bufRe, off);
    im.set(bufIm, off);
  }

  for (let c = 0; c < n; c++) {
    for (let r = 0; r < n; r++) {
      bufRe[r] = re[r * n + c];
      bufIm[r] = im[r * n + c];
    }
    fft1d(bufRe, bufIm, inverse);
    for (let r = 0; r < n; r++) {
      re[r * n + c] = bufRe[r];
      im[r * n + c] = bufIm[r];
    }
  }
}

/** Wrapped frequency index for a length-n FFT, cycles per sample. */
export function fftFrequency(index: number, n: number, dx: number): number {
  const i = index <= n / 2 ? index : index - n;
  return i / (n * dx);
}
