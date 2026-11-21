// PASS 1: repulsion, the O(n^2) part.
//
// One invocation per node i. Every node must look at every other node, so
// the naive version would read `pos` from global memory n times per thread.
// Instead we use the classic CUDA n-body tiling pattern:
//   1. each thread in the workgroup loads ONE position into shared memory
//   2. workgroupBarrier()   (CUDA: __syncthreads())
//   3. every thread reads all TILE positions from fast shared memory
//   4. workgroupBarrier() before the tile is overwritten
// Global memory traffic drops by a factor of TILE.

const TILE: u32 = 64u;

var<workgroup> tile: array<vec2<f32>, TILE>;   // CUDA: __shared__

@compute @workgroup_size(TILE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = params.n;
  let i = gid.x;
  let valid = i < n;

  // Threads past the end must NOT return early: every thread has to reach
  // every workgroupBarrier() (barriers require uniform control flow).
  let pi = pos[min(i, n - 1u)];

  let k2 = params.k * params.k;
  let minD2 = 1e-4 * k2;
  var acc = vec2<f32>(0.0, 0.0);

  let numTiles = (n + TILE - 1u) / TILE;
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let j = t * TILE + lid.x;
    tile[lid.x] = pos[min(j, n - 1u)];
    workgroupBarrier();

    let count = min(TILE, n - t * TILE);   // last tile may be partial
    for (var s = 0u; s < count; s = s + 1u) {
      let d = pi - tile[s];
      let d2 = max(dot(d, d), minD2);
      // k^2/d along the unit vector d/|d|  ==  d * k^2 / |d|^2  (no sqrt needed)
      // For s == i, d == 0 so the contribution is exactly 0: no branch needed.
      acc = acc + d * (k2 / d2);
    }
    workgroupBarrier();
  }

  if (valid) {
    disp[i] = acc;   // overwrite: this pass also "clears" last iteration's forces
  }
}
