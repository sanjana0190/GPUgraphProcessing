import { describe, it, expect } from 'vitest';
import { CPULayout } from '../src/cpu/layout.js';
import { buildCSR } from '../src/graph/csr.js';
import { grid, barabasiAlbert, randomPositions } from '../src/graph/generators.js';
import { layoutParams, Cooling } from '../src/layout/params.js';

function setup(g, seed = 7) {
  const params = layoutParams(g.n);
  const csr = buildCSR(g.n, g.edges);
  return new CPULayout(g, csr, randomPositions(g.n, params.initialSize, seed), params);
}

// Brute-force reference: no Newton's-third-law shortcut, no CSR.
function naiveForces(n, edges, pos, k) {
  const out = new Float64Array(2 * n);
  const k2 = k * k;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = pos[2 * i] - pos[2 * j], dy = pos[2 * i + 1] - pos[2 * j + 1];
      const d2 = Math.max(dx * dx + dy * dy, 1e-4 * k2);
      out[2 * i] += (dx * k2) / d2;
      out[2 * i + 1] += (dy * k2) / d2;
    }
  }
  for (let e = 0; e < edges.length; e += 2) {
    const u = edges[e], v = edges[e + 1];
    const dx = pos[2 * u] - pos[2 * v], dy = pos[2 * u + 1] - pos[2 * v + 1];
    const d = Math.hypot(dx, dy);
    out[2 * u] -= (dx * d) / k; out[2 * u + 1] -= (dy * d) / k;
    out[2 * v] += (dx * d) / k; out[2 * v + 1] += (dy * d) / k;
  }
  return out;
}

describe('CPU Fruchterman-Reingold', () => {
  it('optimized forces match a brute-force reference', () => {
    const g = barabasiAlbert(300, 2, 4);
    const lay = setup(g);
    const fast = lay.computeForces();
    const ref = naiveForces(g.n, g.edges, lay.pos, lay.params.k);
    let maxRef = 0, maxErr = 0;
    for (let i = 0; i < ref.length; i++) {
      maxRef = Math.max(maxRef, Math.abs(ref[i]));
      maxErr = Math.max(maxErr, Math.abs(ref[i] - fast[i]));
    }
    expect(maxErr / maxRef).toBeLessThan(1e-9);
  });

  it('never moves a node further than the temperature', () => {
    const g = barabasiAlbert(400, 2, 2);
    const lay = setup(g);
    const before = Float32Array.from(lay.pos);
    const t = lay.cooling.t;
    lay.step();
    for (let i = 0; i < g.n; i++) {
      const d = Math.hypot(lay.pos[2 * i] - before[2 * i], lay.pos[2 * i + 1] - before[2 * i + 1]);
      expect(d).toBeLessThanOrEqual(t * (1 + 1e-5) + 1e-3);
    }
  });

  it('unfolds a grid: edge lengths become uniform and near the ideal length', () => {
    const g = grid(144);
    const lay = setup(g);
    for (let i = 0; i < 600; i++) lay.step();
    const lens = [];
    for (let e = 0; e < g.edges.length; e += 2) {
      const u = g.edges[e], v = g.edges[e + 1];
      lens.push(Math.hypot(lay.pos[2 * u] - lay.pos[2 * v], lay.pos[2 * u + 1] - lay.pos[2 * v + 1]));
    }
    const mean = lens.reduce((a, b) => a + b, 0) / lens.length;
    const sd = Math.sqrt(lens.reduce((a, b) => a + (b - mean) ** 2, 0) / lens.length);
    expect(sd / mean).toBeLessThan(0.35); // fairly uniform
    expect(mean).toBeGreaterThan(lay.params.k * 0.5);
    expect(mean).toBeLessThan(lay.params.k * 6);
  });

  it('cooling decays geometrically to the floor', () => {
    const c = new Cooling({ t0: 100, cooling: 0.5, tMin: 10 });
    expect([c.next(), c.next(), c.next(), c.next(), c.next()]).toEqual([100, 50, 25, 12.5, 10]);
  });
});
