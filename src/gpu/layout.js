// GPU force-directed layout engine.
//
// Per iteration, one command encoder records three compute passes and is
// submitted once:
//
//   repulsion   O(n^2)  tiled n-body, workgroup shared memory   -> disp
//   attraction  O(m)    CSR gather (or edge-parallel atomics)   -> disp (+=)
//   integrate   O(n)    move by min(|force|, temperature)        -> pos
//
// Node positions live in a GPU storage buffer for the whole session. The
// renderer reads the same buffer, so nothing is copied back to the CPU.
import { createShader } from './device.js';
import { createBufferWithData, readBuffer, u2f } from './buffers.js';
import commonWGSL from './shaders/common.wgsl?raw';
import repulsionWGSL from './shaders/repulsion.wgsl?raw';
import attractionWGSL from './shaders/attraction.wgsl?raw';
import attractionAtomicWGSL from './shaders/attraction_atomic.wgsl?raw';
import integrateWGSL from './shaders/integrate.wgsl?raw';
import boundsWGSL from './shaders/bounds.wgsl?raw';
import { Cooling } from '../layout/params.js';

const WORKGROUP = 64; // must match @workgroup_size in the layout shaders
const BOUNDS_WG = 256; // must match bounds.wgsl
const PARAMS_BYTES = 48; // must match struct Params in common.wgsl

export const EDGE_MODES = { csr: 0, atomic: 1 };

export class GPULayout {
  constructor(gpu, graph, csr, initialPositions, params) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.n = graph.n;
    this.m = graph.edges.length / 2;
    this.graph = graph;
    this.csr = csr;
    this.params = params;
    this.initialPositions = initialPositions;
    this.cooling = new Cooling(params);
    this.edgeMode = EDGE_MODES.csr;
    this.iteration = 0;
    this.lastBounds = null;

    // Fixed-point scale for the atomic variant: the worst-case per-node sum
    // (maxDegree * edgeCap * scale) must stay below 2^30 to fit in an i32.
    let maxDeg = 1;
    for (const d of csr.degree) if (d > maxDeg) maxDeg = d;
    this.edgeCap = 1e5;
    this.fixedScale = Math.min(65536, 2 ** 30 / (maxDeg * this.edgeCap));

    this.paramsData = new ArrayBuffer(PARAMS_BYTES);
    this.paramsU32 = new Uint32Array(this.paramsData);
    this.paramsF32 = new Float32Array(this.paramsData);
  }

  async init() {
    const { device } = this;
    const S = GPUBufferUsage;

    // ---- Buffers ---------------------------------------------------------
    this.paramsBuffer = device.createBuffer({ label: 'params', size: PARAMS_BYTES, usage: S.UNIFORM | S.COPY_DST });
    this.posBuffer = createBufferWithData(device, this.initialPositions, S.STORAGE | S.COPY_SRC | S.COPY_DST, 'positions');
    this.dispBuffer = device.createBuffer({ label: 'displacements', size: Math.max(16, this.n * 8), usage: S.STORAGE | S.COPY_SRC });
    this.offsetsBuffer = createBufferWithData(device, this.csr.offsets, S.STORAGE, 'csr-offsets');
    this.neighborsBuffer = createBufferWithData(device, this.csr.neighbors, S.STORAGE, 'csr-neighbors');
    this.edgesBuffer = createBufferWithData(device, this.graph.edges, S.STORAGE, 'edge-list');
    this.accumBuffer = device.createBuffer({ label: 'fixed-point-accum', size: Math.max(16, this.n * 8), usage: S.STORAGE | S.COPY_DST | S.COPY_SRC });

    // ---- One explicit bind group layout for every layout pass -------------
    const C = GPUShaderStage.COMPUTE;
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'layout-bgl',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'storage' } },
        { binding: 2, visibility: C, buffer: { type: 'storage' } },
        { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: C, buffer: { type: 'storage' } },
      ],
    });
    this.bindGroup = device.createBindGroup({
      label: 'layout-bg',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.posBuffer } },
        { binding: 2, resource: { buffer: this.dispBuffer } },
        { binding: 3, resource: { buffer: this.offsetsBuffer } },
        { binding: 4, resource: { buffer: this.neighborsBuffer } },
        { binding: 5, resource: { buffer: this.edgesBuffer } },
        { binding: 6, resource: { buffer: this.accumBuffer } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });

    const makePipeline = async (body, label) => {
      const module = await createShader(device, commonWGSL + '\n' + body, label);
      return device.createComputePipelineAsync({ label, layout: pipelineLayout, compute: { module, entryPoint: 'main' } });
    };
    [this.repulsionPipeline, this.attractionPipeline, this.attractionAtomicPipeline, this.integratePipeline] =
      await Promise.all([
        makePipeline(repulsionWGSL, 'repulsion'),
        makePipeline(attractionWGSL, 'attraction'),
        makePipeline(attractionAtomicWGSL, 'attraction-atomic'),
        makePipeline(integrateWGSL, 'integrate'),
      ]);

    // ---- Bounds reduction (for camera auto-fit) ----------------------------
    const boundsModule = await createShader(device, boundsWGSL, 'bounds');
    this.boundsPipeline = await device.createComputePipelineAsync({
      label: 'bounds',
      layout: 'auto',
      compute: { module: boundsModule, entryPoint: 'main' },
    });
    this.boundsCountBuffer = createBufferWithData(device, new Uint32Array([this.n, 0, 0, 0]), S.UNIFORM, 'bounds-count');
    this.boundsBuffer = device.createBuffer({ label: 'bounds', size: 16, usage: S.STORAGE | S.COPY_DST | S.COPY_SRC });
    this.boundsReadback = device.createBuffer({ label: 'bounds-readback', size: 16, usage: S.COPY_DST | S.MAP_READ });
    this.boundsBusy = false;
    this.boundsBindGroup = device.createBindGroup({
      layout: this.boundsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.boundsCountBuffer } },
        { binding: 1, resource: { buffer: this.posBuffer } },
        { binding: 2, resource: { buffer: this.boundsBuffer } },
      ],
    });
    this.boundsInit = new Uint32Array([0xffffffff, 0xffffffff, 0, 0]);
    return this;
  }

  get workgroups() {
    return Math.ceil(this.n / WORKGROUP);
  }

  /** Upload the per-iteration uniforms. queue.writeBuffer is ordered with submits. */
  writeParams(temperature) {
    const u = this.paramsU32;
    const f = this.paramsF32;
    u[0] = this.n;
    u[1] = this.m;
    f[2] = this.params.k;
    f[3] = temperature;
    f[4] = this.params.gravity;
    f[5] = this.params.bound;
    f[6] = this.fixedScale;
    f[7] = this.edgeCap;
    u[8] = this.edgeMode;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
  }

  /** Record one layout iteration (3 compute passes) into `encoder`. */
  encodeIteration(encoder, timer = null, timed = false, { integrate = true } = {}) {
    const tw = (name) => (timer ? timer.writes(name, timed) : undefined);
    const wg = this.workgroups;

    let pass = encoder.beginComputePass({ label: 'repulsion', timestampWrites: tw('repulsion') });
    pass.setPipeline(this.repulsionPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(wg); // ceil(n / 64) workgroups of 64 threads
    pass.end();

    if (this.edgeMode === EDGE_MODES.atomic) {
      encoder.clearBuffer(this.accumBuffer); // fixed-point sums start at 0
      pass = encoder.beginComputePass({ label: 'attraction-atomic', timestampWrites: tw('attraction') });
      pass.setPipeline(this.attractionAtomicPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(this.m / WORKGROUP))); // one thread per edge
      pass.end();
    } else {
      pass = encoder.beginComputePass({ label: 'attraction', timestampWrites: tw('attraction') });
      pass.setPipeline(this.attractionPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(wg); // one thread per node
      pass.end();
    }

    if (!integrate) return;
    pass = encoder.beginComputePass({ label: 'integrate', timestampWrites: tw('integrate') });
    pass.setPipeline(this.integratePipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  /** Run `count` iterations, one submit each (same as the interactive loop). */
  runIterations(count) {
    for (let i = 0; i < count; i++) {
      this.writeParams(this.cooling.next());
      const enc = this.device.createCommandEncoder({ label: 'layout-iteration' });
      this.encodeIteration(enc);
      this.device.queue.submit([enc.finish()]);
      this.iteration++;
    }
  }

  /** Record the bounds reduction; copy the 16-byte result if the readback buffer is free. */
  encodeBounds(encoder) {
    this.device.queue.writeBuffer(this.boundsBuffer, 0, this.boundsInit);
    const pass = encoder.beginComputePass({ label: 'bounds' });
    pass.setPipeline(this.boundsPipeline);
    pass.setBindGroup(0, this.boundsBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.n / BOUNDS_WG));
    pass.end();
    if (!this.boundsBusy) {
      encoder.copyBufferToBuffer(this.boundsBuffer, 0, this.boundsReadback, 0, 16);
      this.boundsBusy = true;
      this.boundsCopied = true;
    }
  }

  /** Call after the submit that contained encodeBounds(). */
  collectBounds() {
    if (!this.boundsCopied) return;
    this.boundsCopied = false;
    this.boundsReadback.mapAsync(GPUMapMode.READ).then(
      () => {
        const u = new Uint32Array(this.boundsReadback.getMappedRange());
        this.lastBounds = { minX: u2f(u[0]), minY: u2f(u[1]), maxX: u2f(u[2]), maxY: u2f(u[3]) };
        this.boundsReadback.unmap();
        this.boundsBusy = false;
      },
      () => {
        this.boundsBusy = false;
      },
    );
  }

  /**
   * Debug/verification: compute forces for the current positions WITHOUT
   * integrating, and read them back. Returns Float32Array of length 2n.
   */
  async debugForces(temperature = this.params.t0) {
    this.writeParams(temperature);
    const enc = this.device.createCommandEncoder({ label: 'debug-forces' });
    this.encodeIteration(enc, null, false, { integrate: false });
    this.device.queue.submit([enc.finish()]);
    const disp = await readBuffer(this.device, this.dispBuffer, this.n * 8, Float32Array);
    if (this.edgeMode === EDGE_MODES.atomic) {
      const acc = await readBuffer(this.device, this.accumBuffer, this.n * 8, Int32Array);
      for (let i = 0; i < disp.length; i++) disp[i] += acc[i] / this.fixedScale;
    }
    return disp;
  }

  async readPositions() {
    return readBuffer(this.device, this.posBuffer, this.n * 8, Float32Array);
  }

  /** One full iteration with an explicit temperature (for verification). */
  async stepWithTemperature(t) {
    this.writeParams(t);
    const enc = this.device.createCommandEncoder();
    this.encodeIteration(enc);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  get temperature() {
    return this.cooling.t;
  }

  reheat() {
    this.cooling.reheat();
  }

  destroy() {
    for (const b of [
      this.paramsBuffer, this.posBuffer, this.dispBuffer, this.offsetsBuffer, this.neighborsBuffer,
      this.edgesBuffer, this.accumBuffer, this.boundsCountBuffer, this.boundsBuffer, this.boundsReadback,
    ]) {
      b?.destroy();
    }
  }
}
