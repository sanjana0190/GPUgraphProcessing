// Instanced node rendering: one quad (2 triangles, 6 vertices) per node.
// draw(6, n): vertex_index picks the corner, instance_index picks the node.
// The fragment shader turns the quad into an anti-aliased circle (SDF).

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
}

fn palette(t: f32) -> vec3<f32> {
  let lo = vec3<f32>(0.29, 0.56, 0.98);   // low degree: blue
  let hi = vec3<f32>(0.98, 0.55, 0.24);   // hubs: orange
  return mix(lo, hi, clamp(t, 0.0, 1.0));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  );
  let corner = corners[vi];
  let t = colorT[ii];
  let size = cam.pointSize * (1.0 + 1.5 * t);   // hubs are drawn a bit larger
  var out: VOut;
  out.clip = vec4<f32>(toClip(positions[ii]) + corner * size, 0.0, 1.0);
  out.uv = corner;
  out.color = palette(t);
  return out;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
  let r = length(v.uv);
  // Derivatives (fwidth) must run in uniform control flow, so take it
  // before the data-dependent discard.
  let edge = fwidth(r);
  if (r > 1.0) {
    discard;
  }
  let alpha = 1.0 - smoothstep(1.0 - edge, 1.0, r);
  return vec4<f32>(v.color, alpha);
}
