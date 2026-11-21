// Correctness check: the GPU must compute the same forces as the CPU.
//
// Runs the force passes on the GPU for a fixed graph and fixed positions,
// reads the displacement buffer back (the ONLY place the app reads node data
// back from the GPU), and compares it with the CPU reference.
import { makeScene } from './benchmark.js';
import { CPULayout } from '../cpu/layout.js';
import { GPULayout, EDGE_MODES } from '../gpu/layout.js';

function compare(ref, got) {
  let maxRef = 0;
  for (let i = 0; i < ref.length; i += 2) maxRef = Math.max(maxRef, Math.hypot(ref[i], ref[i + 1]));
  let maxErr = 0;
  let sumErr = 0;
  for (let i = 0; i < ref.length; i += 2) {
    const e = Math.hypot(ref[i] - got[i], ref[i + 1] - got[i + 1]);
    maxErr = Math.max(maxErr, e);
    sumErr += e;
  }
  const count = ref.length / 2;
  return { maxErr: maxErr / maxRef, meanErr: sumErr / count / maxRef };
}

export async function verifyGPU(gpu, { n = 2000, type = 'barabasi-albert', tolerance = 1e-4 } = {}) {
  const scene = makeScene(type, n, 3);
  const cpu = new CPULayout(scene.graph, scene.csr, scene.positions, scene.params);
  const cpuForces = Float64Array.from(cpu.computeForces());

  const gpuLayout = await new GPULayout(gpu, scene.graph, scene.csr, scene.positions, scene.params).init();

  gpuLayout.edgeMode = EDGE_MODES.csr;
  const csr = compare(cpuForces, await gpuLayout.debugForces());

  gpuLayout.edgeMode = EDGE_MODES.atomic;
  const atomic = compare(cpuForces, await gpuLayout.debugForces());

  // One full iteration (forces + integrate) from the same start, same temperature.
  gpuLayout.edgeMode = EDGE_MODES.csr;
  const t = scene.params.t0;
  cpu.integrate(t);
  await gpuLayout.stepWithTemperature(t);
  const gpuPos = await gpuLayout.readPositions();
  let maxMove = 0;
  for (let i = 0; i < gpuPos.length; i++) maxMove = Math.max(maxMove, Math.abs(gpuPos[i] - cpu.pos[i]));
  gpuLayout.destroy();

  const step = { maxPosErr: maxMove, relToTemperature: maxMove / t };
  return {
    n: scene.graph.n,
    m: scene.graph.edges.length / 2,
    fixedScale: gpuLayout.fixedScale,
    csr,
    atomic,
    step,
    pass: csr.maxErr < tolerance && step.relToTemperature < 1e-3,
  };
}
