// PASS 3: integrate. Move each node along its net force, limited by the
// current temperature, then clamp to the safety bound.
//
// WebGPU guarantees that storage writes from the previous pass are visible
// here: passes in one queue submission execute in order, and the
// implementation inserts the memory barriers for us.

@group(0) @binding(6) var<storage, read_write> accum: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  var p = pos[i];
  var d = disp[i];
  if (params.edgeMode == 1u) {
    // add the fixed-point edge forces accumulated by attraction_atomic.wgsl
    d = d + vec2<f32>(f32(accum[2u * i]), f32(accum[2u * i + 1u])) / params.fixedScale;
  }
  d = d - params.gravity * p;

  let len = length(d);
  if (len > 0.0) {
    p = p + d * (min(len, params.temperature) / len);
  }
  pos[i] = clamp(p, vec2<f32>(-params.bound), vec2<f32>(params.bound));
}
