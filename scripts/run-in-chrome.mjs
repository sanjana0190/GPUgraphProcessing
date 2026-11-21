// Drives the real app in your installed Google Chrome (WebGPU -> Metal on a Mac):
//   node scripts/run-in-chrome.mjs verify          correctness check
//   node scripts/run-in-chrome.mjs bench           CPU vs GPU benchmark -> bench/results.{json,md}
//   node scripts/run-in-chrome.mjs shot <out.png>  screenshot after the layout settles
// Options via env: GRAPH=barabasi-albert NODES=20000 ENGINE=gpu SETTLE_MS=8000 HEADLESS=0
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const CHROME =
  process.env.CHROME_PATH ||
  {
    darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    linux: '/usr/bin/google-chrome',
    win32: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  }[process.platform];

const [cmd = 'verify', arg] = process.argv.slice(2);
const server = await createServer({ server: { port: 0 }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0] + (process.env.QUERY ? `?${process.env.QUERY}` : '');

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: process.env.HEADLESS === '0' ? false : 'new',
  args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--window-size=1500,950', '--no-first-run'],
  defaultViewport: { width: 1500, height: 950, deviceScaleFactor: 2 },
  protocolTimeout: 30 * 60 * 1000,
});

try {
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (['error', 'warn'].includes(m.type())) console.log(`[browser ${m.type()}]`, m.text());
  });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => window.app?.state?.engine && !window.app.state.busy, { timeout: 60000 });
  const adapter = await page.$eval('#adapter', (e) => e.textContent);
  console.log(adapter);

  const setGraph = async () => {
    await page.evaluate(
      async ({ graph, nodes, engine, edgeMode }) => {
        const s = window.app.state;
        if (graph) document.getElementById('graph-type').value = graph;
        if (nodes) document.getElementById('node-count').value = nodes;
        if (edgeMode) document.getElementById('edge-mode').value = edgeMode;
        if (engine && engine !== s.engineKind) document.querySelector(`[data-engine="${engine}"]`).click();
        else document.getElementById('btn-generate').click();
        await new Promise((r) => setTimeout(r, 50));
        while (s.busy) await new Promise((r) => setTimeout(r, 50));
      },
      { graph: process.env.GRAPH, nodes: process.env.NODES, engine: process.env.ENGINE, edgeMode: process.env.EDGE_MODE },
    );
  };

  if (cmd === 'verify') {
    const r = await page.evaluate(() => window.app.runVerify());
    console.log(JSON.stringify(r, null, 2));
    if (!r?.pass) process.exitCode = 1;
  } else if (cmd === 'bench') {
    const sizes = process.env.SIZES ? process.env.SIZES.split(',').map(Number) : undefined;
    const rows = await page.evaluate((sizes) => window.app.runBench(sizes), sizes);
    const { markdown } = await page.evaluate(() => window.__benchResults);
    const ua = await browser.version();
    fs.mkdirSync('bench', { recursive: true });
    fs.writeFileSync('bench/results.json', JSON.stringify({ date: new Date().toISOString(), adapter, browser: ua, rows }, null, 2));
    fs.writeFileSync('bench/results.md', markdown.replace(/Browser: .*\./, `Browser: ${ua}.`) + '\n');
    console.log(markdown);
  } else if (cmd === 'shot') {
    await setGraph();
    await new Promise((r) => setTimeout(r, Number(process.env.SETTLE_MS || 8000)));
    const out = arg || 'docs/screenshot.png';
    fs.mkdirSync(path.dirname(out), { recursive: true });
    await page.screenshot({ path: out });
    const stats = await page.$eval('#stats', (e) => e.innerText);
    console.log(stats.replace(/\n/g, ' | '));
    console.log('saved', out);
  }
} finally {
  await browser.close();
  await server.close();
}
