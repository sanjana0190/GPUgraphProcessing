// Two interchangeable engines with the same interface, so the UI can switch
// between them and the benchmark compares the same algorithm on CPU vs GPU.
import { CPULayout } from './cpu/layout.js';
import { Canvas2DRenderer } from './cpu/renderer2d.js';
import { GPULayout } from './gpu/layout.js';
import { GPUTimer } from './gpu/timing.js';

const ema = (prev, x, a = 0.1) => (prev === null ? x : prev + a * (x - prev));

export class CPUEngine {
  constructor(canvas) {
    this.kind = 'cpu';
    this.renderer = new Canvas2DRenderer(canvas);
    this.layoutMs = null;
    this.drawMs = null;
  }
  async load(scene) {
    this.layout = new CPULayout(scene.graph, scene.csr, scene.positions, scene.params);
    this.renderer.setGraph(scene.graph, scene.colors);
    this.layoutMs = null;
    this.drawMs = null;
  }
  frame({ iters, running, camera, style }) {
    if (running) {
      // Budget the main thread: at least one step, more only if they fit in ~30 ms.
      const t0 = performance.now();
      let done = 0;
      do {
        this.layout.step();
        done++;
      } while (done < iters && performance.now() - t0 < 30);
      this.layoutMs = ema(this.layoutMs, (performance.now() - t0) / done);
    }
    const t1 = performance.now();
    this.renderer.draw(this.layout.pos, camera, style);
    this.drawMs = ema(this.drawMs, performance.now() - t1);
  }
  bounds() {
    return this.layout.bounds();
  }
  stats() {
    return {
      layoutMs: this.layoutMs,
      renderMs: this.drawMs,
      temperature: this.layout.temperature,
      iteration: this.layout.iteration,
    };
  }
  reheat() {
    this.layout.reheat();
  }
  destroy() {}
}

export class GPUEngine {
  constructor(gpu, renderer) {
    this.kind = 'gpu';
    this.gpu = gpu;
    this.renderer = renderer;
    this.timer = new GPUTimer(gpu.device, gpu.hasTimestamps);
    this.frameCpuMs = null;
  }
  async load(scene) {
    this.layout?.destroy();
    this.layout = await new GPULayout(this.gpu, scene.graph, scene.csr, scene.positions, scene.params).init();
    this.renderer.setGraph(this.layout, scene.colors);
    this.timer.reset();
  }
  setEdgeMode(mode) {
    this.layout.edgeMode = mode;
  }
  frame({ iters, running, camera, style }) {
    const { device } = this.gpu;
    const t0 = performance.now();
    this.renderer.writeCamera(camera, style);
    const steps = running ? iters : 0;
    // Every iteration is its own submit so each one sees its own temperature
    // (queue.writeBuffer and submit execute in queue order).
    for (let i = 0; i < steps - 1; i++) {
      this.layout.writeParams(this.layout.cooling.next());
      const enc = device.createCommandEncoder({ label: 'layout' });
      this.layout.encodeIteration(enc);
      device.queue.submit([enc.finish()]);
      this.layout.iteration++;
    }
    // Last submit of the frame: one more iteration (timed) + bounds + render.
    const enc = device.createCommandEncoder({ label: 'frame' });
    if (steps > 0) {
      this.layout.writeParams(this.layout.cooling.next());
      this.layout.encodeIteration(enc, this.timer, true);
      this.layout.iteration++;
    }
    this.layout.encodeBounds(enc);
    this.renderer.encode(enc, this.timer.writes('render', steps > 0));
    const timed = steps > 0 && this.timer.resolve(enc);
    device.queue.submit([enc.finish()]);
    if (timed) this.timer.collect();
    this.layout.collectBounds();
    // CPU time spent recording + submitting (the GPU runs asynchronously).
    this.frameCpuMs = ema(this.frameCpuMs, performance.now() - t0);
  }
  bounds() {
    return this.layout.lastBounds;
  }
  stats() {
    const a = this.timer.avg;
    const has = this.timer.enabled && this.timer.samples > 0;
    return {
      layoutMs: has ? a.repulsion + a.attraction + a.integrate : null,
      renderMs: has ? a.render : null,
      passes: has ? { ...a } : null,
      encodeMs: this.frameCpuMs,
      temperature: this.layout.temperature,
      iteration: this.layout.iteration,
    };
  }
  reheat() {
    this.layout.reheat();
  }
  destroy() {
    this.layout?.destroy();
    this.layout = null;
  }
}
