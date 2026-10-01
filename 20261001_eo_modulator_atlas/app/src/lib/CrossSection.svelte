<script lang="ts">
  import type { Geometry } from '../../../engine/src/config.mjs';
  let { geometry }: { geometry: Geometry } = $props();
  let zoom = $state(false);
  const bounds = $derived(zoom && geometry.optical_window ? geometry.optical_window : geometry.domain);
  const width = $derived(bounds.x[1] - bounds.x[0]);
  const height = $derived(bounds.y[1] - bounds.y[0]);
  const colors: Record<string, string> = { air: '#192534', silicon: '#666979', silicon_dioxide: '#447a99', lithium_niobate: '#b26dbb', gold: '#e4b65d' };
  const points = (poly: number[][]) => poly.map(([x, y]) => `${x},${-y}`).join(' ');
</script>

<div class="caption">
  <span>Geometry · x lateral, y up · µm</span>
  <label><input type="checkbox" bind:checked={zoom} disabled={!geometry.optical_window} /> Optical window</label>
</div>
<svg viewBox={`${bounds.x[0]} ${-bounds.y[1]} ${width} ${height}`} aria-label="Configured cross-section geometry" role="img">
  {#each geometry.regions as r}
    <polygon points={points(r.poly)} fill={colors[r.material] ?? '#779988'}><title>{r.name} · {r.material}</title></polygon>
  {/each}
  {#each geometry.electrodes as e}
    <polygon points={points(e.poly)} fill="#e4b65d" stroke="#ffe7a1" stroke-width={width / 700}><title>{e.name} · voltage weight {e.weight}</title></polygon>
  {/each}
</svg>
<div class="caption muted"><span>x: {bounds.x.join(' to ')} µm</span><span>y: {bounds.y.join(' to ')} µm</span></div>

<style>
  svg { width: 100%; height: 280px; display: block; background: #192534; }
  .caption { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; padding: 8px 0; }
  label { display: flex; align-items: center; gap: 6px; }
</style>
