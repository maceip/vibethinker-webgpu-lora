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

// esbuild.config.mjs
// Riced-out bundler config for the Emberglass / qwen-webgpu-lora WebGPU inference engine.
//
// Goals:
// - Fast, reproducible builds
// - Consistent browser + WebGPU-friendly output (ESM, modern target)
// - Keep @huggingface/transformers external (loaded via dynamic import only for tokenizer)
// - Easy prod minification + bundle analysis
// - Reusable for main app and test harness bundles
//
// Usage examples:
//   node esbuild.config.mjs
//   node esbuild.config.mjs --minify
//   node esbuild.config.mjs --minify --metafile
//   node esbuild.config.mjs --entry test/f16_vs_f32_diff.js --outfile test/f16diff_bundle.js
//   node esbuild.config.mjs --entry test/benchmark_wgpu.js --outfile test/bench_bundle.js --minify false

import { build } from 'esbuild';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const jsBanner = readFileSync(new URL(import.meta.url), 'utf8').match(/^\/\*[\s\S]*?\*\//)?.[0] ?? '';

// Optional gzip size (not a hard dependency)
let gzipSizeFn = null;
try {
  // dynamic so it doesn't break if the package isn't installed
  const mod = await import('gzip-size');
  gzipSizeFn = mod.gzipSize || mod.default?.gzipSize;
} catch {}

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(name);
  if (i !== -1 && args[i + 1] && !args[i + 1].startsWith('--')) return args[i + 1];
  return fallback;
};

const entry = getArg('--entry', 'src/main.js');
const outfile = getArg('--outfile', entry === 'src/main.js' ? 'bundle.js' : null);
const minify = args.includes('--minify') || args.includes('-m') || process.env.MINIFY === '1';
const metafile = args.includes('--metafile') || args.includes('--analyze');
const sourcemap = args.includes('--sourcemap') ? 'inline' : (getArg('--sourcemap', false) || false);
const noConsoleDrop = args.includes('--keep-console');

const isMainBuild = entry === 'src/main.js';

const base = {
  entryPoints: [entry],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'esnext',           // Chrome Canary + WebGPU audience; change to chrome120+ if needed for older canaries
  legalComments: 'none',
  external: ['@huggingface/transformers'], // Only ever used via dynamic import() for tokenizer
  minify,
  sourcemap: sourcemap || false,
  metafile,
  banner: jsBanner ? { js: jsBanner } : undefined,
  // Drop console.* in minified builds (safe for this engine; no user-facing console semantics)
  drop: (minify && !noConsoleDrop) ? ['console'] : [],
  // Keep names for better stack traces in non-min builds; drop in prod
  keepNames: !minify,
  // Tree shaking is on by default for ESM
  treeShaking: true,
};

async function getGzSize(path) {
  try {
    if (!existsSync(path) || !gzipSizeFn) return null;
    const buf = readFileSync(path);
    const gz = await gzipSizeFn(buf);
    return gz / 1024;
  } catch {}
  return null;
}

async function go() {
  const t0 = Date.now();

  if (!outfile) {
    console.error('Missing --outfile when using custom --entry');
    process.exit(1);
  }

  const res = await build({
    ...base,
    outfile,
  });

  const bytes = readFileSync(outfile).length;
  const kb = (bytes / 1024).toFixed(1);
  let gzb = null;
  try { gzb = await getGzSize(outfile); } catch {}

  const label = isMainBuild ? (minify ? 'prod' : 'dev') : 'bundle';
  console.log(
    `esbuild ${label} ${entry} -> ${outfile}  ${kb} KB${gzb ? ` (gz ~${gzb.toFixed(1)} KB)` : ''}  in ${Date.now() - t0}ms`
  );

  if (metafile && res.metafile) {
    const metaPath = isMainBuild ? 'meta.json' : outfile.replace(/\.js$/, '.meta.json');
    writeFileSync(metaPath, JSON.stringify(res.metafile, null, 2));
    console.log(`  wrote ${metaPath} (analyze with https://esbuild.github.io/analyze/ or source-map-explorer)`);
  }

  // For the two main outputs, also write a docs/ copy when building main
  if (isMainBuild && outfile === 'bundle.js') {
    await build({
      ...base,
      outfile: 'docs/bundle.js',
    });
    const docsBytes = readFileSync('docs/bundle.js').length;
    console.log(`  + docs/bundle.js  ${(docsBytes / 1024).toFixed(1)} KB`);
  }
}

go().catch((e) => {
  console.error(e);
  process.exit(1);
});
