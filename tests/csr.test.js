import { describe, it, expect } from 'vitest';
import { buildCSR } from '../src/graph/csr.js';

describe('buildCSR', () => {
  it('stores every undirected edge in both rows', () => {
    // triangle 0-1-2 plus pendant 3 attached to 2
    const edges = Uint32Array.from([0, 1, 1, 2, 2, 0, 2, 3]);
    const { offsets, neighbors, degree } = buildCSR(4, edges);
    expect(Array.from(degree)).toEqual([2, 2, 3, 1]);
    expect(Array.from(offsets)).toEqual([0, 2, 4, 7, 8]);
    const row = (i) => Array.from(neighbors.slice(offsets[i], offsets[i + 1])).sort();
    expect(row(0)).toEqual([1, 2]);
    expect(row(1)).toEqual([0, 2]);
    expect(row(2)).toEqual([0, 1, 3]);
    expect(row(3)).toEqual([2]);
  });

  it('handles isolated nodes and empty graphs', () => {
    const { offsets, neighbors } = buildCSR(3, new Uint32Array(0));
    expect(Array.from(offsets)).toEqual([0, 0, 0, 0]);
    expect(neighbors.length).toBe(0);
  });
});
