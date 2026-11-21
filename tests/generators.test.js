import { describe, it, expect } from 'vitest';
import { barabasiAlbert, erdosRenyi, grid, stochasticBlock } from '../src/graph/generators.js';
import { parseEdgeList } from '../src/graph/loaders.js';

function checkSimple(g) {
  const seen = new Set();
  for (let i = 0; i < g.edges.length; i += 2) {
    const u = g.edges[i], v = g.edges[i + 1];
    expect(u).not.toBe(v);
    expect(u).toBeLessThan(g.n);
    expect(v).toBeLessThan(g.n);
    const key = u < v ? `${u},${v}` : `${v},${u}`;
    expect(seen.has(key)).toBe(false);
    seen.add(key);
  }
}

describe('generators', () => {
  it('Erdos-Renyi has (at least) the requested edge count and is simple', () => {
    const g = erdosRenyi(1000, 4, 1);
    // 2000 random edges, plus one extra edge per node that would be isolated
    expect(g.edges.length / 2).toBeGreaterThanOrEqual(2000);
    expect(g.edges.length / 2).toBeLessThan(2100);
    checkSimple(g);
  });

  it('Barabasi-Albert is simple and has hubs', () => {
    const g = barabasiAlbert(2000, 2, 1);
    checkSimple(g);
    const deg = new Uint32Array(g.n);
    for (const v of g.edges) deg[v]++;
    expect(Math.max(...deg)).toBeGreaterThan(30); // preferential attachment creates hubs
  });

  it('grid has 2wh - w - h edges', () => {
    const g = grid(100); // 10 x 10
    expect(g.n).toBe(100);
    expect(g.edges.length / 2).toBe(180);
  });

  it('is deterministic for a given seed', () => {
    expect(Array.from(stochasticBlock(500, 5, 4, 0.3, 9).edges)).toEqual(
      Array.from(stochasticBlock(500, 5, 4, 0.3, 9).edges),
    );
  });

  it('parses SNAP edge lists, remapping ids and dropping duplicates', () => {
    const g = parseEdgeList('# comment\n10 20\n20 10\n20 30\n30 30\n');
    expect(g.n).toBe(3);
    expect(Array.from(g.edges)).toEqual([0, 1, 1, 2]);
  });
});

describe('no isolated nodes in synthetic graphs', () => {
  it.each([
    ['erdos-renyi', () => erdosRenyi(3000, 2, 5)],
    ['stochastic-block', () => stochasticBlock(3000, 8, 4, 0.2, 5)],
  ])('%s', (_, make) => {
    const g = make();
    const deg = new Uint32Array(g.n);
    for (const v of g.edges) deg[v]++;
    expect(deg.every((d) => d > 0)).toBe(true);
  });
});
