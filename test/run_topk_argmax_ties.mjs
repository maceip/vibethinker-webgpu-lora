import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { chromium } from 'playwright';

const root = process.cwd();
const types = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
]);

const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://127.0.0.1').pathname);
    const file = normalize(join(root, pathname === '/' ? 'index.html' : pathname));
    if (!file.startsWith(root)) {
      res.writeHead(403);
      res.end('forbidden');
      return;
    }
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': types.get(extname(file)) ?? 'application/octet-stream' });
    res.end(body);
  } catch (err) {
    res.writeHead(err.code === 'ENOENT' ? 404 : 500);
    res.end(String(err.message ?? err));
  }
});

await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const executablePath = process.env.CHROME_PATH || (existsSync(macCanary) ? macCanary : undefined);
const browser = await chromium.launch({
  ...(executablePath ? { executablePath } : {}),
  headless: process.env.HEADED !== '1',
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run', '--disable-gpu-sandbox'],
});

try {
  const page = await browser.newPage();
  const lines = [];
  page.on('console', m => {
    const text = m.text();
    if (text.startsWith('VWG')) {
      lines.push(text);
      console.log(text);
    }
  });
  page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 300)));

  const { port } = server.address();
  await page.goto(`http://127.0.0.1:${port}/test/topk_argmax_ties.html`, { waitUntil: 'domcontentloaded' });
  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    if (lines.some(l => l.includes('VWG DONE') || l.startsWith('VWG ERROR'))) break;
    await page.waitForTimeout(250);
  }
  if (!lines.some(l => l.includes('TOPK_ARGMAX_TIE PASS'))) process.exitCode = 1;
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
