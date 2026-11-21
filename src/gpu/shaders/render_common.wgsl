// Shared by the node and edge render pipelines.
// The render passes read node positions DIRECTLY from the storage buffer the
// compute passes wrote. Positions never travel GPU -> CPU -> GPU.

struct Camera {
  center: vec2<f32>,     // world point at the middle of the screen
  scale: vec2<f32>,      // world units -> clip space (-1..1)
  pointSize: vec2<f32>,  // node radius in clip units (x and y differ by aspect)
  edgeAlpha: f32,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> edgeIdx: array<u32>;
@group(0) @binding(3) var<storage, read> colorT: array<f32>;

fn toClip(p: vec2<f32>) -> vec2<f32> {
  return (p - cam.center) * cam.scale;
}
