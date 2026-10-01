#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { runCrossSection } from './src/run.mjs';

try {
  const { values, positionals } = parseArgs({ allowPositionals: true, options: {
    optical: { type: 'boolean', default: false }, section: { type: 'string', default: 'geometry' },
    'mesh-scale': { type: 'string', default: '1' }, help: { type: 'boolean', short: 'h' },
  } });
  if (values.help) console.log('Usage: node engine/cli.mjs CONFIG.yaml [--optical] [--section unloaded] [--mesh-scale 1]\nRuns cross-section stages only; JSON results go to stdout and are not saved.');
  else {
    if (positionals.length !== 1) throw new Error('Provide one config path; use --help for usage');
    const result = runCrossSection(await readFile(positionals[0], 'utf8'), {
      optical: values.optical, section: values.section, meshScale: Number(values['mesh-scale']),
      onProgress: (message) => console.error(message),
    });
    console.log(JSON.stringify(result, null, 2));
    // Completion is not a literature pass. A failed evaluated target is a distinct exit status.
    if (result.targets.some((t) => t.status === 'fail')) process.exitCode = 2;
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
