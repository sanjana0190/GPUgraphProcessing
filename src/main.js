// App entry: builds graphs, owns the render loop, wires up the controls.
import { initWebGPU } from './gpu/device.js';
import { GPURenderer } from './gpu/renderer.js';
import { GPUEngine, CPUEngine } from './engines.js';
import { GENERATORS, randomPositions } from './graph/generators.js';
import { buildCSR } from './graph/csr.js';
import { loadEdgeListURL } from './graph/loaders.js';
import { degreeColorValues } from './graph/colors.js';
import { layoutParams, paramOverridesFromQuery } from './layout/params.js';
import { Camera } from './ui/camera.js';
import { runBenchmark, toMarkdown } from './bench/benchmark.js';
import { verifyGPU } from './bench/verify.js';

const $ = (id) => document.getElementById(id);
const els = {
  gpuCanvas: $('gpu-canvas'),
  cpuCanvas: $('cpu-canvas'),
  stage: $('stage'),
  message: $('message'),
  stats: $('stats'),
  graphType: $('graph-type'),
  nodeCount: $('node-count'),
  iters: $('iters'),
  itersOut: $('iters-out'),
  edgeMode: $('edge-mode'),
  toolStatus: $('tool-status'),
  toolOutput: $('tool-output'),
  adapter: $('adapter'),
};

const state = {
  gpu: null,
  gpuRenderer: null,
  engine: null,
  engineKind: 'gpu',
  scene: null,
  camera: new Camera(),
  running: true,
  busy: false,
  fps: null,
  lastFrame: performance.now(),
  error: null,
};

function showMessage(html) {
  els.message.innerHTML = html;
  els.message.hidden = !html;
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------
async function buildScene() {
  const type = els.graphType.value;
  let graph;
  if (type === 'snap-facebook') {
    graph = await loadEdgeListURL(`${import.meta.env.BASE_URL}data/facebook_combined.txt`, 'SNAP facebook_combined');
  } else {
    graph = GENERATORS[type](Number(els.nodeCount.value), 1);
  }
  const csr = buildCSR(graph.n, graph.edges);
  const params = layoutParams(graph.n, paramOverridesFromQuery(location.search));
  return {
    graph,
    csr,
    params,
    colors: degreeColorValues(csr.degree),
    positions: randomPositions(graph.n, params.initialSize, 7),
  };
}

function styleFor(scene) {
  const dpr = window.devicePixelRatio || 1;
  const n = scene.graph.n;
  const m = scene.graph.edges.length / 2;
  return {
    nodeRadiusPx: dpr * (n <= 2000 ? 3 : n <= 10000 ? 2.2 : n <= 50000 ? 1.6 : 1.2),
    edgeAlpha: Math.min(0.5, Math.max(0.035, 2500 / Math.max(1, m))),
  };
}

async function loadEngine() {
  const kind = state.engineKind;
  els.gpuCanvas.hidden = kind !== 'gpu';
  els.cpuCanvas.hidden = kind !== 'cpu';
  resize();
  state.engine?.destroy();
  state.engine = kind === 'gpu' ? new GPUEngine(state.gpu, state.gpuRenderer) : new CPUEngine(els.cpuCanvas);
  await state.engine.load(state.scene);
  if (kind === 'gpu') state.engine.setEdgeMode(Number(els.edgeMode.value));
  const half = state.scene.params.initialSize / 2;
  state.camera.autoFit = true;
  state.camera.fit({ minX: -half, minY: -half, maxX: half, maxY: half });
  if (kind === 'cpu' && state.scene.graph.n > 10000) {
    els.toolStatus.textContent = 'CPU mode at this size is slow on purpose: the O(n²) repulsion runs on one JS thread.';
  }
}

async function regenerate() {
  if (state.busy) return;
  state.busy = true;
  try {
    showMessage('');
    state.scene = await buildScene();
    await loadEngine();
  } catch (err) {
    console.error(err);
    showMessage(`<strong>Error:</strong> ${err.message}`);
  } finally {
    state.busy = false;
  }
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(els.stage.clientWidth * dpr));
  const h = Math.max(1, Math.floor(els.stage.clientHeight * dpr));
  for (const c of [els.gpuCanvas, els.cpuCanvas]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
  state.camera.setViewport(w, h);
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = now - state.lastFrame;
  state.lastFrame = now;
  state.fps = state.fps === null ? 1000 / dt : state.fps + 0.05 * (1000 / dt - state.fps);
  if (!state.engine || state.busy || state.error) return;

  resize();
  if (state.camera.autoFit) state.camera.fit(state.engine.bounds(), 0.12);
  state.engine.frame({
    iters: Number(els.iters.value),
    running: state.running,
    camera: state.camera,
    style: styleFor(state.scene),
  });
  if (Math.floor(now / 250) !== Math.floor((now - dt) / 250)) renderStats();
}

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(x) ? '–' : x.toFixed(d));

function renderStats() {
  const s = state.engine.stats();
  const g = state.scene.graph;
  const rows = [
    ['Engine', state.engineKind === 'gpu' ? 'WebGPU compute' : 'CPU (JS)'],
    ['Nodes', g.n.toLocaleString()],
    ['Edges', (g.edges.length / 2).toLocaleString()],
    ['Layout ms / iter', fmt(s.layoutMs, 3), 'big'],
    ['Render ms', fmt(s.renderMs, 3)],
  ];
  if (s.passes) {
    rows.push(['· repulsion', fmt(s.passes.repulsion, 3)]);
    rows.push(['· attraction', fmt(s.passes.attraction, 3)]);
    rows.push(['· integrate', fmt(s.passes.integrate, 3)]);
  }
  if (state.engineKind === 'gpu') rows.push(['CPU encode ms / frame', fmt(s.encodeMs, 3)]);
  if (state.engineKind === 'gpu' && s.layoutMs === null) rows.push(['GPU timers', 'timestamp-query unavailable']);
  rows.push(['Frame rate', `${fmt(state.fps, 0)} fps`]);
  rows.push(['Iteration', s.iteration.toLocaleString()]);
  rows.push(['Temperature', fmt(s.temperature, 2)]);
  els.stats.innerHTML = rows
    .map(([k, v, cls]) => `<dt>${k}</dt><dd${cls ? ` class="${cls}"` : ''}>${v}</dd>`)
    .join('');
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------
async function runVerify() {
  if (!state.gpu) return;
  els.toolStatus.textContent = 'Verifying GPU forces against the CPU reference…';
  const r = await verifyGPU(state.gpu);
  const e = (x) => x.toExponential(2);
  els.toolStatus.innerHTML = `Verification <span class="${r.pass ? 'pass' : 'fail'}">${r.pass ? 'passed' : 'FAILED'}</span>`;
  els.toolOutput.innerHTML = `<pre>Graph: ${r.n} nodes, ${r.m} edges
Force error, normalized by the largest force:
  CSR gather       max ${e(r.csr.maxErr)}   mean ${e(r.csr.meanErr)}
  atomics (fixed)  max ${e(r.atomic.maxErr)}   mean ${e(r.atomic.meanErr)}   scale ${r.fixedScale.toFixed(1)}
One full iteration, position error:
  max ${e(r.step.maxPosErr)} units (${e(r.step.relToTemperature)} of the step size)</pre>`;
  console.log('verify', r);
  return r;
}

async function runBench(sizes) {
  if (!state.gpu) return;
  const wasRunning = state.running;
  state.running = false;
  const type = els.graphType.value === 'snap-facebook' ? 'barabasi-albert' : els.graphType.value;
  const rows = [];
  const table = () =>
    `<table><tr><th>Nodes</th><th>CPU ms</th><th>GPU ms</th><th>Speedup</th></tr>${rows
      .map(
        (r) =>
          `<tr><td>${r.n.toLocaleString()}</td><td>${r.cpuMs.toFixed(1)}</td><td>${r.gpuMs.toFixed(2)}</td><td class="speed">${r.speedup.toFixed(1)}×</td></tr>`,
      )
      .join('')}</table>`;
  els.toolOutput.innerHTML = '';
  try {
    await runBenchmark(state.gpu, {
      sizes,
      type,
      onStatus: (s) => (els.toolStatus.textContent = `Benchmarking ${s}`),
      onRow: (r) => {
        rows.push(r);
        els.toolOutput.innerHTML = table();
      },
    });
    const md = toMarkdown(rows, { type, gpu: state.gpu.adapterLabel, browser: navigator.userAgent });
    els.toolStatus.textContent = 'Benchmark done. Metric: wall-clock ms per layout iteration, rendering excluded.';
    els.toolOutput.innerHTML = table() + `<pre>${md}</pre>`;
    console.log(md);
    window.__benchResults = { rows, markdown: md };
    return rows;
  } finally {
    state.running = wasRunning;
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function setEngine(kind) {
  if (kind === 'gpu' && !state.gpu) return;
  state.engineKind = kind;
  document.querySelectorAll('[data-engine]').forEach((b) => b.classList.toggle('active', b.dataset.engine === kind));
  $('edge-mode-wrap').hidden = kind !== 'gpu';
  regenerate();
}

function wireControls() {
  document.querySelectorAll('[data-engine]').forEach((b) => b.addEventListener('click', () => setEngine(b.dataset.engine)));
  $('btn-generate').addEventListener('click', regenerate);
  els.graphType.addEventListener('change', () => {
    els.nodeCount.disabled = els.graphType.value === 'snap-facebook';
    regenerate();
  });
  els.nodeCount.addEventListener('change', regenerate);
  els.iters.addEventListener('input', () => (els.itersOut.textContent = els.iters.value));
  els.edgeMode.addEventListener('change', () => state.engine?.setEdgeMode?.(Number(els.edgeMode.value)));
  $('btn-pause').addEventListener('click', (e) => {
    state.running = !state.running;
    e.target.textContent = state.running ? 'Pause' : 'Resume';
  });
  $('btn-reheat').addEventListener('click', () => state.engine?.reheat());
  $('btn-fit').addEventListener('click', () => (state.camera.autoFit = true));
  $('btn-verify').addEventListener('click', runVerify);
  $('btn-bench').addEventListener('click', () => runBench());

  // Pan / zoom
  let drag = null;
  els.stage.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY };
    els.stage.setPointerCapture(e.pointerId);
  });
  els.stage.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dpr = window.devicePixelRatio || 1;
    const c = els.gpuCanvas.hidden ? els.cpuCanvas : els.gpuCanvas;
    state.camera.autoFit = false;
    state.camera.panPixels((e.clientX - drag.x) * dpr, (e.clientY - drag.y) * dpr, c.width, c.height);
    drag = { x: e.clientX, y: e.clientY };
  });
  els.stage.addEventListener('pointerup', () => (drag = null));
  els.stage.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const dpr = window.devicePixelRatio || 1;
      const rect = els.stage.getBoundingClientRect();
      const c = els.gpuCanvas.hidden ? els.cpuCanvas : els.gpuCanvas;
      state.camera.autoFit = false;
      state.camera.zoomAt((e.clientX - rect.left) * dpr, (e.clientY - rect.top) * dpr, c.width, c.height, Math.exp(e.deltaY * 0.0015));
    },
    { passive: false },
  );
  els.stage.addEventListener('dblclick', () => (state.camera.autoFit = true));
}

async function main() {
  wireControls();
  try {
    state.gpu = await initWebGPU({
      onLost: (info) => {
        state.error = info;
        showMessage(`<strong>GPU device lost</strong> (${info.reason || 'unknown'}). ${info.message || ''}<br>Reload the page to start again.`);
      },
    });
    state.gpuRenderer = await new GPURenderer(state.gpu, els.gpuCanvas).init();
    els.adapter.textContent = `GPU: ${state.gpu.adapterLabel}${state.gpu.hasTimestamps ? ' · timestamp-query on' : ''}`;
  } catch (err) {
    console.warn(err);
    els.adapter.textContent = 'WebGPU unavailable: CPU mode only.';
    $('engine-gpu').disabled = true;
    $('btn-verify').disabled = true;
    $('btn-bench').disabled = true;
    state.engineKind = 'cpu';
    els.nodeCount.value = '1000';
    showMessage(`<strong>WebGPU is not available.</strong> ${err.message}<br>Falling back to the CPU engine.`);
    setTimeout(() => showMessage(''), 5000);
  }
  setEngine(state.engineKind);
  requestAnimationFrame(frame);
}

// Exposed for automated benchmarking (scripts/bench.mjs) and the console.
window.app = { state, runBench, runVerify };
main();
