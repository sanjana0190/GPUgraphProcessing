// Synthetic graph generators. Every generator returns:
//   { name, n, edges }  where edges is a flat Uint32Array [u0, v0, u1, v1, ...]
// Graphs are undirected, without self-loops or duplicate edges.
import { mulberry32 } from './rng.js';

function edgeKey(u, v) {
  return u < v ? u * 4294967296 + v : v * 4294967296 + u;
}

class EdgeSet {
  constructor() {
    this.seen = new Set();
    this.list = [];
  }
  add(u, v) {
    if (u === v) return false;
    const k = edgeKey(u, v);
    if (this.seen.has(k)) return false;
    this.seen.add(k);
    this.list.push(u, v);
    return true;
  }
  get size() {
    return this.list.length / 2;
  }
  toArray() {
    return Uint32Array.from(this.list);
  }
}

/**
 * Give every degree-0 node one edge. Isolated nodes have no spring holding
 * them, so repulsion pushes them into a far-away ring that wrecks the view.
 */
function connectIsolated(n, es, rand, pickPartner) {
  const deg = new Uint32Array(n);
  for (const v of es.list) deg[v]++;
  for (let i = 0; i < n; i++) {
    if (deg[i] > 0) continue;
    for (let tries = 0; tries < 20; tries++) {
      const j = pickPartner(i, rand);
      if (es.add(i, j)) {
        deg[i]++;
        deg[j]++;
        break;
      }
    }
  }
}

/** G(n, m) Erdos-Renyi: m uniformly random distinct edges. */
export function erdosRenyi(n, avgDegree = 3, seed = 1) {
  const rand = mulberry32(seed);
  const target = Math.min(Math.floor((n * avgDegree) / 2), (n * (n - 1)) / 2);
  const es = new EdgeSet();
  while (es.size < target) {
    es.add(Math.floor(rand() * n), Math.floor(rand() * n));
  }
  connectIsolated(n, es, rand, () => Math.floor(rand() * n));
  return { name: `erdos-renyi(n=${n}, deg=${avgDegree})`, n, edges: es.toArray() };
}

/** Barabasi-Albert preferential attachment: scale-free, hub-heavy, like real networks. */
export function barabasiAlbert(n, mPerNode = 2, seed = 1) {
  const rand = mulberry32(seed);
  const es = new EdgeSet();
  const m0 = Math.max(mPerNode + 1, 2);
  // Start from a small clique.
  for (let i = 0; i < m0; i++) for (let j = i + 1; j < m0; j++) es.add(i, j);
  // "repeated" holds each endpoint once per incident edge, so sampling
  // uniformly from it is sampling proportional to degree.
  const repeated = es.list.slice();
  for (let v = m0; v < n; v++) {
    let added = 0;
    let guard = 0;
    while (added < mPerNode && guard++ < 50 * mPerNode) {
      const u = repeated[Math.floor(rand() * repeated.length)];
      if (es.add(v, u)) {
        repeated.push(v, u);
        added++;
      }
    }
  }
  return { name: `barabasi-albert(n=${n}, m=${mPerNode})`, n, edges: es.toArray() };
}

/** 2D grid lattice. Good visual sanity check: a correct layout unfolds into a sheet. */
export function grid(n) {
  const w = Math.max(1, Math.round(Math.sqrt(n)));
  const h = Math.max(1, Math.ceil(n / w));
  const total = w * h;
  const es = new EdgeSet();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x + 1 < w) es.add(i, i + 1);
      if (y + 1 < h) es.add(i, i + w);
    }
  }
  return { name: `grid(${w}x${h})`, n: total, edges: es.toArray() };
}

/** Stochastic block model: k dense communities with sparse links between them. */
export function stochasticBlock(n, communities = 8, inDegree = 4, outDegree = 0.3, seed = 1) {
  const rand = mulberry32(seed);
  const es = new EdgeSet();
  const size = Math.ceil(n / communities);
  const block = (i) => Math.floor(i / size);
  const inTarget = Math.floor((n * inDegree) / 2);
  const outTarget = Math.floor((n * outDegree) / 2);
  let guard = 0;
  while (es.size < inTarget && guard++ < inTarget * 20) {
    const u = Math.floor(rand() * n);
    const b = block(u);
    const lo = b * size;
    const hi = Math.min(n, lo + size);
    es.add(u, lo + Math.floor(rand() * (hi - lo)));
  }
  let outAdded = 0;
  guard = 0;
  while (outAdded < outTarget && guard++ < outTarget * 20) {
    const u = Math.floor(rand() * n);
    const v = Math.floor(rand() * n);
    if (block(u) !== block(v) && es.add(u, v)) outAdded++;
  }
  connectIsolated(n, es, rand, (i) => {
    const lo = block(i) * size;
    return lo + Math.floor(rand() * (Math.min(n, lo + size) - lo));
  });
  return { name: `sbm(n=${n}, k=${communities})`, n, edges: es.toArray() };
}

export const GENERATORS = {
  'barabasi-albert': (n, seed) => barabasiAlbert(n, 2, seed),
  'stochastic-block': (n, seed) => stochasticBlock(n, Math.max(4, Math.round(n / 400)), 6, 0.15, seed),
  'erdos-renyi': (n, seed) => erdosRenyi(n, 3, seed),
  grid: (n) => grid(n),
};

/** Random initial positions inside a square of side `size` centered at the origin. */
export function randomPositions(n, size, seed = 7) {
  const rand = mulberry32(seed);
  const pos = new Float32Array(n * 2);
  for (let i = 0; i < n * 2; i++) pos[i] = (rand() - 0.5) * size;
  return pos;
}
