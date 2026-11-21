// Compressed Sparse Row adjacency.
//
// For node i, its neighbors are neighbors[offsets[i] .. offsets[i+1]).
// Every undirected edge (u, v) is stored twice (u->v and v->u), so each node
// can compute its own attractive force by reading only its own row.
// On the GPU this means one thread per node, each writing only its own output:
// no write conflicts and therefore no atomics needed.
export function buildCSR(n, edges) {
  const m = edges.length / 2;
  const degree = new Uint32Array(n);
  for (let e = 0; e < m; e++) {
    degree[edges[2 * e]]++;
    degree[edges[2 * e + 1]]++;
  }
  // Exclusive prefix sum of degrees -> row offsets.
  const offsets = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) offsets[i + 1] = offsets[i] + degree[i];

  const neighbors = new Uint32Array(2 * m);
  const cursor = offsets.slice(0, n);
  for (let e = 0; e < m; e++) {
    const u = edges[2 * e];
    const v = edges[2 * e + 1];
    neighbors[cursor[u]++] = v;
    neighbors[cursor[v]++] = u;
  }
  return { offsets, neighbors, degree };
}
