/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */

// Drop-in config for the qwen-webgpu-lora browser compute engine.
// Usage:
//   node esbuild.config.mjs            # writes bundle.js + docs/bundle.js
//   node esbuild.config.mjs --minify   # prod

/*
 * TECHNIQUE: Centralized build config with prod flags
 *   Single place for target, external, minify, metafile, drop console.
 *   Enables reproducible builds and easy analysis of the driver bundle.
 */
import { build } from 'esbuild';
import { readFileSync, writeFileSync } from 'fs';

const minify = process.argv.includes('--minify') || process.argv.includes('-m');
const metafile = process.argv.includes('--metafile');

const base = {
  entryPoints: ['src/main.js'],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  // Critical: transformers is ONLY used via dynamic import() for the tokenizer.
  // Without this, a plain "await import()" can still cause esbuild to try to resolve it.
  external: ['@huggingface/transformers'],
  // Target: esnext is fine; we run in modern browsers with WebGPU.
  // If you need to support older Chromium for testing, pin e.g. chrome120.
  target: 'esnext',
  legalComments: 'none',
  // drop/pure are safe here; we don't rely on console.* side effects for semantics.
  drop: minify ? ['console'] : [],
  minify,
  metafile,
  sourcemap: false, // flip to 'inline' or true only when debugging bundles
};

async function go() {
  const t0 = Date.now();
  const res = await build({
    ...base,
    outfile: 'bundle.js',
  });
  await build({
    ...base,
    outfile: 'docs/bundle.js',
  });

  const bytes = readFileSync('bundle.js').length;
  console.log(
    `build ${minify ? '(min)' : '(dev)'} -> ${(bytes / 1024).toFixed(1)} KB in ${Date.now() - t0}ms`
  );

  if (metafile && res.metafile) {
    writeFileSync('meta.json', JSON.stringify(res.metafile, null, 2));
    console.log('wrote meta.json (use https://esbuild.github.io/analyze/ or source-map-explorer)');
  }
}

go().catch((e) => {
  console.error(e);
  process.exit(1);
});
