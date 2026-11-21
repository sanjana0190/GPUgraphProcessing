// CPU vs GPU benchmark: milliseconds per layout iteration.
//
// Methodology (this is the answer to "how did you measure 10x?"):
//   - identical graph (same generator + seed) and identical initial positions
//   - identical algorithm and constants (src/layout/params.js)
//   - metric: wall-clock ms per FULL layout iteration (repulsion + attraction
//     + integrate), rendering excluded on both sides
//   - CPU: warm up (>= 3 iters or 300 ms, for the JIT), then run until
//     >= cpuBudgetMs and >= 3 iterations
//   - GPU: warm up 10 iterations, then submit iterations (one submit each,
//     like the live app) until >= gpuBudgetMs, and wait for
//     queue.onSubmittedWorkDone() so the GPU has really finished
import { GENERATORS, randomPositions } from '../graph/generators.js';
import { buildCSR } from '../graph/csr.js';
import { layoutParams } from '../layout/params.js';
import { CPULayout } from '../cpu/layout.js';
import { GPULayout } from '../gpu/layout.js';

const yieldToBrowser = () => new Promise((r) => setTimeout(r, 0));

export const DEFAULT_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000];

export async function benchmarkGPU(gpu, scene, budgetMs = 1000) {
  const layout = await new GPULayout(gpu, scene.graph, scene.csr, scene.positions, scene.params).init();
  const { queue } = gpu.device;
  layout.runIterations(10);
  await queue.onSubmittedWorkDone();
  let iters = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < budgetMs && iters < 2000) {
    layout.runIterations(10);
    await queue.onSubmittedWorkDone();
    iters += 10;
  }
  const ms = (performance.now() - t0) / iters;
  layout.destroy();
  return { ms, iters };
}

export async function benchmarkCPU(scene, budgetMs = 1500) {
  const layout = new CPULayout(scene.graph, scene.csr, scene.positions, scene.params);
  // JIT warm-up: at least 3 iterations or 300 ms, so V8 has optimized the hot loop.
  const w0 = performance.now();
  for (let i = 0; i < 3 || (performance.now() - w0 < 300 && i < 50); i++) layout.step();
  // Only time spent inside step() counts. The periodic yield (so the page
  // stays responsive) is excluded: browsers clamp setTimeout to >= 1-4 ms,
  // which would unfairly inflate small-graph CPU numbers.
  let iters = 0;
  let busy = 0;
  let lastYield = performance.now();
  while (iters < 3 || (busy < budgetMs && iters < 200)) {
    const t = performance.now();
    layout.step();
    busy += performance.now() - t;
    iters++;
    if (performance.now() - lastYield > 250) {
      await yieldToBrowser();
      lastYield = performance.now();
    }
  }
  return { ms: busy / iters, iters };
}

export function makeScene(type, n, seed = 1) {
  const graph = GENERATORS[type](n, seed);
  const params = layoutParams(graph.n);
  return {
    graph,
    csr: buildCSR(graph.n, graph.edges),
    params,
    positions: randomPositions(graph.n, params.initialSize, seed + 6),
  };
}

export async function runBenchmark(gpu, { sizes = DEFAULT_SIZES, type = 'barabasi-albert', onRow, onStatus } = {}) {
  const rows = [];
  for (const n of sizes) {
    const scene = makeScene(type, n);
    onStatus?.(`n = ${n.toLocaleString()}: GPU…`);
    await yieldToBrowser();
    const gpuRes = await benchmarkGPU(gpu, scene);
    onStatus?.(`n = ${n.toLocaleString()}: CPU…`);
    await yieldToBrowser();
    const cpuRes = await benchmarkCPU(scene);
    const row = {
      n: scene.graph.n,
      m: scene.graph.edges.length / 2,
      cpuMs: cpuRes.ms,
      gpuMs: gpuRes.ms,
      speedup: cpuRes.ms / gpuRes.ms,
      cpuIters: cpuRes.iters,
      gpuIters: gpuRes.iters,
    };
    rows.push(row);
    onRow?.(row);
  }
  return rows;
}

export function toMarkdown(rows, meta) {
  const lines = [
    `Graph: ${meta.type}. Hardware: ${meta.gpu}. Browser: ${meta.browser}.`,
    '',
    '| Nodes | Edges | CPU ms/iter | GPU ms/iter | Speedup |',
    '|---:|---:|---:|---:|---:|',
    ...rows.map(
      (r) =>
        `| ${r.n.toLocaleString()} | ${r.m.toLocaleString()} | ${r.cpuMs.toFixed(2)} | ${r.gpuMs.toFixed(3)} | ${r.speedup.toFixed(1)}x |`,
    ),
  ];
  return lines.join('\n');
}
