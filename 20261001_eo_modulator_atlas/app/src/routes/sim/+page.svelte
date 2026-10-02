<script lang="ts">
  import { onDestroy } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { inspectConfig } from '../../../../engine/src/config.mjs';
  import type { SimulationResult } from '../../../../engine/src/run.mjs';
  import { store } from '../../lib/state.svelte';
  import { staticUrl, link } from '../../lib/paths';
  import { fmt } from '../../lib/logic';
  import CrossSection from '../../lib/CrossSection.svelte';

  let text = $state('');
  let section = $state('geometry');
  let optical = $state(false);
  let meshScale = $state(1);
  let busy = $state(false);
  let loading = $state(false);
  let progress = $state('');
  let error = $state('');
  let result = $state<SimulationResult | null>(null);
  let worker: Worker | null = null;
  let runInput = $state('');
  const sims = $derived(store.atlas?.sims ?? []);
  const requested = $derived(page.url.searchParams.get('id') ?? page.url.searchParams.get('config'));
  const chosen = $derived(requested ? sims.find(s => s.id === requested || s.path === requested) : sims[0]);
  const parsed = $derived.by(() => {
    if (!text) return { config: null, error: '' };
    try { const { preview, solveError } = inspectConfig(text); return { config: preview, error: solveError }; }
    catch (e) { return { config: null, error: e instanceof Error ? e.message : String(e) }; }
  });
  const geometry = $derived(parsed.config?.geometries[section]);
  const currentInput = $derived(JSON.stringify([text, section, optical, meshScale]));
  const stale = $derived(result !== null && currentInput !== runInput);
  const labels: Record<string, string> = {
    c_pul_pf_per_m: "C′ (pF/m)", c0_pul_pf_per_m: "C₀′ (pF/m)", l_pul_nh_per_m: "L′ (nH/m)",
    n_rf: 'RF index (section)', z0_ohm: 'Z₀ (Ω, section)', n_eff: 'Effective optical index', ng_opt: 'Optical group index'
  };

  function stopWorker() {
    if (worker) { worker.onmessage = null; worker.onerror = null; worker.terminate(); }
    worker = null; busy = false;
  }
  function cancel() { stopWorker(); progress = 'Cancelled'; }
  onDestroy(stopWorker);

  $effect(() => {
    const sim = chosen;
    stopWorker(); result = null; text = ''; error = ''; loading = false; section = 'geometry'; progress = '';
    if (!sim) return;
    const controller = new AbortController();
    loading = true;
    fetch(staticUrl(sim.path), { signal: controller.signal })
      .then(async res => { if (!res.ok) throw new Error(`Config: HTTP ${res.status}`); return res.text(); })
      .then(value => { if (!controller.signal.aborted) { text = value; loading = false; progress = ''; } })
      .catch(e => { if (!controller.signal.aborted) { error = String(e); loading = false; } });
    return () => controller.abort();
  });

  function run() {
    stopWorker(); error = ''; result = null; busy = true; progress = 'Starting solver'; runInput = currentInput;
    try {
      const activeWorker = new Worker(new URL('../../lib/sim.worker.ts', import.meta.url), { type: 'module' });
      worker = activeWorker;
      activeWorker.onmessage = event => {
        if (worker !== activeWorker) return;
        if (event.data.type === 'progress') progress = event.data.message;
        else {
          if (event.data.type === 'result') { result = event.data.result; progress = 'Cross-section solve complete'; }
          else { error = event.data.message; progress = 'Solve stopped'; }
          stopWorker();
        }
      };
      activeWorker.onerror = event => {
        if (worker !== activeWorker) return;
        error = event.message || 'Worker failed'; stopWorker(); progress = 'Solve stopped';
      };
      activeWorker.postMessage({ text, section, optical, meshScale });
    } catch (e) { error = String(e); stopWorker(); progress = 'Solve stopped'; }
  }
</script>

<svelte:head><title>Simulation · EO Modulator Atlas</title></svelte:head>
<div class="sim-page">
  <div class="intro">
    <h1>Cross-section simulator</h1>
    <p>Run electrostatics and optional scalar optical modes locally in your browser. Results stay in memory.</p>
    <p class="muted">EO overlap, RF loss, periodic loaded-line response and bandwidth are still pending. Section values do not establish reproduction of a complete modulator.</p>
  </div>
  {#if store.error}<p class="error" role="alert">{store.error}</p>{/if}
  {#if store.loading}<p>Loading atlas…</p>
  {:else if !sims.length}<p>No simulation configs are available.</p>
  {:else}
    <div class="toolbar">
      <label>Config <select aria-label="Simulation config" value={chosen?.id ?? ''} onchange={e => void goto(link(`/sim?id=${encodeURIComponent(e.currentTarget.value)}`))}>
        {#if !chosen}<option value="">Unknown config</option>{/if}
        {#each sims as s}<option value={s.id}>{s.id} · {s.paper_id}</option>{/each}
      </select></label>
      {#if chosen}<a href={staticUrl(chosen.path)} download="config.yaml">Download input YAML</a>{/if}
    </div>
    {#if !chosen}<p class="error">The requested config is not in this atlas.</p>{/if}
    {#if loading}<p>Loading config…</p>{/if}
    <div class="workspace">
      <section class="inputs">
        <h2>{parsed.config?.raw.title ?? 'Simulation input'}</h2>
        <div class="toolbar">
          <label>Cross-section <select aria-label="Cross-section" bind:value={section} disabled={busy}>{#each Object.keys(parsed.config?.geometries ?? {}) as name}<option value={name}>{name}</option>{/each}</select></label>
          <label>Mesh <select aria-label="Mesh resolution" bind:value={meshScale} disabled={busy}><option value={2}>Coarse (2× edge)</option><option value={1}>Configured</option><option value={0.5}>Fine (½ edge)</option></select></label>
          <label><input type="checkbox" bind:checked={optical} disabled={busy} /> Include optical mode</label>
        </div>
        {#if geometry}<CrossSection {geometry} />{/if}
        <label class="editor-label" for="config-yaml">Input YAML · edits apply to this session</label>
        <textarea id="config-yaml" bind:value={text} spellcheck="false" disabled={busy || loading}></textarea>
        {#if parsed.error}<p class="error" role="alert">{`${parsed.config ? 'Preview only. Solve blocked: ' : ''}${parsed.error}`}</p>{/if}
        <div class="toolbar actions">
          <button class="on" onclick={run} disabled={busy || loading || !geometry || !!parsed.error}>Run cross-section</button>
          {#if busy}<button onclick={cancel}>Cancel</button>{/if}
          <span role="status">{#if busy}<span class="spinner"></span> {/if}{progress}</span>
        </div>
        {#if optical}<p class="muted">The scalar optical solver requires dielectric material throughout the optical window. Metal inside that window is unsupported; a failed requested stage stops the run.</p>{/if}
        {#if error}<p class="error" role="alert">{error}</p>{/if}
      </section>
      <section class="outputs">
        <h2>Results</h2>
        {#if result}
          {#if stale}<p class="notice">Inputs changed. These results belong to the previous run.</p>{/if}
          <p>{result.id} · {result.section} · {fmt(result.elapsed_ms / 1000)} s</p>
          <div class="metrics">{#each Object.entries(result.metrics) as [key, value]}<div><span>{labels[key] ?? key}</span><strong class="num" data-metric={key} data-value={value}>{fmt(value, 5)}</strong></div>{/each}</div>
          <p class="muted">{result.electrostatics.nNodes.toLocaleString()} nodes · {result.electrostatics.nTris.toLocaleString()} triangles · charge/energy relative difference {fmt(result.electrostatics.energyChargeRelativeError)}</p>
          <h3>Paper targets</h3>
          <p class="target-summary">{result.targetSummary.evaluated} of {result.targetSummary.total} targets evaluated · {result.targetSummary.passed} passed · {result.targetSummary.failed} failed</p>
          <p class="muted">Pass/fail applies only to an independently evaluated metric. “Not evaluated” is never counted as a pass.</p>
          <div class="table-scroll"><table><thead><tr><th>Metric</th><th>Target</th><th>Actual</th><th>Status</th></tr></thead><tbody>
            {#each result.targets as t}<tr title={t.reason ?? JSON.stringify(t.source)}><td>{t.metric}{t.at_ghz ? ` @ ${t.at_ghz} GHz` : ''}</td><td class="num">{fmt(t.value)} ± {fmt(t.tolerance)}</td><td class="num">{fmt(t.actual)}</td><td class:failed={t.status === 'fail'}>{t.status.replaceAll('_', ' ')}</td></tr>{/each}
          </tbody></table></div>
          {#each [...new Set(result.targets.map(t => t.reason).filter(Boolean))] as reason}<p class="muted">{reason}</p>{/each}
          <h3>Model limitations</h3>
          <ul>{#each result.warnings as warning}<li>{warning.replaceAll('_', ' ')}</li>{/each}</ul>
          <p>Pending stages: {result.pendingStages.join(', ').replaceAll('_', ' ') || 'none'}.</p>
        {:else}<p class="muted">Choose a cross-section and run the solver to calculate its capacitance and quasi-static RF properties.</p>{/if}
        {#if parsed.config}
          <details open><summary>Disclosure and assumptions</summary>
            <p>Config validation status: {parsed.config.raw.validation_status ?? 'unvalidated'}</p>
            <ul>{#each parsed.config.raw.limitations ?? [] as note}<li>{note}</li>{/each}</ul>
            {#if parsed.config.raw.missing?.length}<h3>Missing or inferred inputs</h3><ul>{#each parsed.config.raw.missing as item}<li>{typeof item === 'string' ? item : JSON.stringify(item)}</li>{/each}</ul>{/if}
          </details>
          <details><summary>Parameter provenance</summary><dl>{#each Object.entries(parsed.config.raw.provenance ?? {}) as [key, evidence]}<dt class="mono">{key}</dt><dd>{evidence.class} · {evidence.locator ?? evidence.citation ?? ''}{evidence.note ? ` — ${evidence.note}` : ''}</dd>{/each}</dl></details>
        {/if}
      </section>
    </div>
  {/if}
</div>

<style>
  .sim-page { height: 100%; overflow: auto; padding: 18px; }
  h1 { font-size: 22px; margin: 0; } h2 { font-size: 15px; margin: 0 0 12px; } h3 { font-size: 12px; margin-top: 18px; }
  p { line-height: 1.6; } .intro { max-width: 900px; }
  .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 10px 0; }
  label { display: flex; align-items: center; gap: 6px; }
  .workspace { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
  section { background: var(--surface); border: 1px solid var(--border); padding: 14px; min-width: 0; }
  textarea { width: 100%; height: 250px; resize: vertical; color: var(--ink); background: var(--bg); border: 1px solid var(--border); font: 11px/1.5 var(--mono); padding: 10px; white-space: pre; }
  .editor-label { margin: 10px 0 6px; } .actions button { height: 32px; }
  button:disabled { opacity: 0.5; cursor: default; }
  .error, .failed { color: var(--crit); } .error { overflow-wrap: anywhere; } .notice { color: var(--warn); }
  .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .metrics div { background: var(--bg); padding: 10px; } .metrics span { display: block; color: var(--ink-3); }
  .metrics strong { display: block; font-size: 22px; font-weight: 500; margin-top: 4px; }
  .table-scroll { overflow: auto; } table { width: 100%; border-collapse: collapse; }
  th, td { padding: 6px; text-align: left; border-bottom: 1px solid var(--border); } th { color: var(--ink-3); }
  details { border-top: 1px solid var(--border); margin-top: 16px; padding-top: 12px; overflow-wrap: anywhere; }
  summary { cursor: pointer; font-weight: 600; } li { margin: 6px 0; } ul { padding-left: 18px; }
  dt { margin-top: 12px; color: var(--ink-2); } dd { margin: 4px 0 0; line-height: 1.6; color: var(--ink-3); }
  @media (max-width: 950px) { .workspace { grid-template-columns: 1fr; } .sim-page { padding: 10px; } }
</style>
