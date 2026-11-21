// CPU reference implementation of Fruchterman-Reingold.
//
// This is the baseline the GPU is compared against, so it is written to be a
// *fair* CPU version, not a strawman:
//   - flat typed arrays, no per-node objects
//   - repulsion uses Newton's third law, so each pair is visited once (n^2 / 2)
//   - edge forces use the same CSR adjacency as the GPU
// It runs single-threaded on the main thread, like a typical JS layout.
import { Cooling } from '../layout/params.js';

export class CPULayout {
  constructor(graph, csr, initialPositions, params) {
    this.n = graph.n;
    this.m = graph.edges.length / 2;
    this.csr = csr;
    this.params = params;
    this.pos = Float32Array.from(initialPositions);
    this.disp = new Float64Array(this.n * 2);
    this.cooling = new Cooling(params);
    this.iteration = 0;
  }

  /** Net force on every node (repulsion + attraction), written into `out`. */
  computeForces(pos = this.pos, out = this.disp) {
    const { n } = this;
    const { k } = this.params;
    const k2 = k * k;
    const minD2 = 1e-4 * k2; // avoids division by zero for overlapping nodes
    out.fill(0);

    // Repulsion: O(n^2). This loop is where nearly all CPU time goes.
    for (let i = 0; i < n; i++) {
      const xi = pos[2 * i];
      const yi = pos[2 * i + 1];
      let ax = 0;
      let ay = 0;
      for (let j = i + 1; j < n; j++) {
        const dx = xi - pos[2 * j];
        const dy = yi - pos[2 * j + 1];
        let d2 = dx * dx + dy * dy;
        if (d2 < minD2) d2 = minD2;
        const f = k2 / d2; // (k^2 / d) along unit vector (dx, dy) / d
        const fx = dx * f;
        const fy = dy * f;
        ax += fx;
        ay += fy;
        out[2 * j] -= fx;
        out[2 * j + 1] -= fy;
      }
      out[2 * i] += ax;
      out[2 * i + 1] += ay;
    }

    // Attraction: O(m), each node walks its own CSR row (same as the GPU kernel).
    const { offsets, neighbors } = this.csr;
    for (let i = 0; i < n; i++) {
      const xi = pos[2 * i];
      const yi = pos[2 * i + 1];
      let ax = 0;
      let ay = 0;
      for (let e = offsets[i]; e < offsets[i + 1]; e++) {
        const j = neighbors[e];
        const dx = xi - pos[2 * j];
        const dy = yi - pos[2 * j + 1];
        const f = Math.sqrt(dx * dx + dy * dy) / k; // (d^2 / k) along unit vector
        ax -= dx * f;
        ay -= dy * f;
      }
      out[2 * i] += ax;
      out[2 * i + 1] += ay;
    }
    return out;
  }

  /** Move each node along its force, limited by the temperature. */
  integrate(temperature, pos = this.pos, disp = this.disp) {
    const { gravity, bound } = this.params;
    for (let i = 0; i < this.n; i++) {
      const px = pos[2 * i];
      const py = pos[2 * i + 1];
      const dx = disp[2 * i] - gravity * px;
      const dy = disp[2 * i + 1] - gravity * py;
      const len = Math.sqrt(dx * dx + dy * dy);
      let nx = px;
      let ny = py;
      if (len > 0) {
        const s = Math.min(len, temperature) / len;
        nx += dx * s;
        ny += dy * s;
      }
      pos[2 * i] = Math.max(-bound, Math.min(bound, nx));
      pos[2 * i + 1] = Math.max(-bound, Math.min(bound, ny));
    }
  }

  step() {
    const t = this.cooling.next();
    this.computeForces();
    this.integrate(t);
    this.iteration++;
    return t;
  }

  get temperature() {
    return this.cooling.t;
  }

  reheat() {
    this.cooling.reheat();
  }

  bounds() {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const p = this.pos;
    for (let i = 0; i < this.n; i++) {
      const x = p[2 * i], y = p[2 * i + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return { minX, minY, maxX, maxY };
  }
}
