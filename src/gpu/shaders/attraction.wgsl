// PASS 2 (default): attraction along edges, CSR "gather".
//
// Each node's thread walks its own row of the CSR adjacency and sums the
// spring force from every neighbor. Every thread writes only disp[i], its
// own slot, so there are no write conflicts and no atomics.
// The cost is that every edge is processed twice (once from each end).

@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> neighbors: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  let pi = pos[i];
  var acc = vec2<f32>(0.0, 0.0);
  let start = offsets[i];
  let end = offsets[i + 1u];
  for (var e = start; e < end; e = e + 1u) {
    let d = pi - pos[neighbors[e]];
    // d^2/k along the unit vector  ==  d * |d| / k
    acc = acc - d * (length(d) / params.k);
  }
  disp[i] = disp[i] + acc;
}
