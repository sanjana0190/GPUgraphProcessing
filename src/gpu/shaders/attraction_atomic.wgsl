// PASS 2 (alternative): attraction, edge-parallel "scatter" with atomics.
//
// One thread per EDGE. The thread computes the spring force once and adds it
// to both endpoints. Many edges share a node (hubs can have hundreds), so
// those adds race. WGSL atomics only exist for i32/u32, not f32, so we
// convert the force to FIXED POINT: round(force * fixedScale) as i32.
//
// Trade-off: fixedScale must be small enough that the largest possible sum
// (maxDegree * edgeCap * fixedScale) fits in an i32, which costs precision.
// The JS side picks fixedScale from the graph's max degree.

@group(0) @binding(5) var<storage, read> edges: array<u32>;
@group(0) @binding(6) var<storage, read_write> accum: array<atomic<i32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.m) {
    return;
  }
  let u = edges[2u * e];
  let v = edges[2u * e + 1u];
  let d = pos[u] - pos[v];
  var f = d * (length(d) / params.k);   // force on v; u gets -f
  f = clamp(f, vec2<f32>(-params.edgeCap), vec2<f32>(params.edgeCap));
  let q = vec2<i32>(round(f * params.fixedScale));

  atomicSub(&accum[2u * u], q.x);
  atomicSub(&accum[2u * u + 1u], q.y);
  atomicAdd(&accum[2u * v], q.x);
  atomicAdd(&accum[2u * v + 1u], q.y);
}
