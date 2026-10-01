<script lang="ts">
  import { store } from '../../lib/state.svelte';
  import { link } from '../../lib/paths';
</script>

<svelte:head><title>About · EO Modulator Atlas</title></svelte:head>
<article>
  <h1>About the atlas</h1>
  <p>A literature database of electro-optic modulator demonstrations, with traceable device metrics, comparison plots and simulation inputs.</p>
  {#if store.atlas}<p>The current verified collection contains {store.atlas.papers.length} papers and {store.atlas.devices.length} device or operating-point records. Candidate papers are kept separately until verified.</p>{/if}
  <h2>Read the evidence with the number</h2>
  <p>Each reported metric carries its source locator and basis: measured, simulated, extracted from a figure, derived, author estimate or design target. Open a device’s details in the <a href={link('/table')}>table</a> to inspect them.</p>
  <p>An empty cell means the value is not reported. Bounds (&lt;, &gt;) and approximations (~) remain attached to their values. On-chip insertion loss and fiber-to-fiber loss are separate fields. Voltage comparisons depend on the recorded drive and Vπ convention.</p>
  <h2>One representative per paper</h2>
  <p>The default representative is the device with the most complete core metrics. Ties use the bandwidth, voltage and on-chip loss figure of merit when its inputs are available, then lower VπL. Expand a paper to see every variant, or change the representative selector.</p>
  {#if store.atlas}
    <p class="formula">{store.atlas.meta.fom_definition}</p>
    <p>{store.atlas.meta.completeness_definition}</p>
  {/if}
  <h2>Simulation status</h2>
  <p>The <a href={link('/sim')}>cross-section simulator</a> runs two-dimensional electrostatics and optional scalar optical modes in a browser worker. Analytic tests cover parallel-plate capacitance, differential voltage normalization and TE/TM slab modes. These tests do not establish reproduction of a paper device.</p>
  <p>EO overlap, RF loss, periodic loaded-line response and EO bandwidth remain unfinished. Unsupported targets appear as “not evaluated.” Inputs, targets, tolerances and provenance are tracked; solver results stay in memory. The current Chen optical window intersects metal and needs further modelling work.</p>
  <h2>Coverage and reproducibility</h2>
  <p>Grade A indicates disclosed geometry and materials, B requires extraction or inference, and C supports metrics only. A grade describes disclosure, not solver accuracy. Simulation configs list missing inputs and modelling assumptions explicitly.</p>
  <p>The repository includes source PDFs, extracted text and figure images under its recorded reference policy. The static app packages database views and simulation input YAML only. Redistribution rights remain recorded per source.</p>
</article>

<style>
  article { height: 100%; overflow-y: auto; padding: 24px; max-width: 940px; line-height: 1.7; }
  h1 { margin-top: 0; font-size: 24px; } h2 { margin-top: 26px; font-size: 16px; }
  p { color: var(--ink-2); } .formula { font-family: var(--mono); background: var(--surface); padding: 12px; }
</style>
