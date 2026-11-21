// Shared by every compute shader. Prepended to each module at build time.
//
// Uniform layout note: WGSL aligns vec3<f32> to 16 bytes, which silently
// shifts every following field if the JS side packs it as 12 bytes.
// This struct deliberately uses only 4-byte scalars, so the byte offsets are
// simply 0, 4, 8, ... and match the Float32Array/Uint32Array views in JS.
// Total size 48 bytes (a multiple of 16, as uniform buffers require).
struct Params {
  n: u32,             // node count
  m: u32,             // edge count
  k: f32,             // ideal edge length
  temperature: f32,   // max displacement this iteration
  gravity: f32,       // pull toward the origin
  bound: f32,         // safety clamp for positions
  fixedScale: f32,    // fixed-point scale for the atomic edge-force variant
  edgeCap: f32,       // per-edge force clamp so fixed-point sums cannot overflow i32
  edgeMode: u32,      // 0 = CSR gather (no atomics), 1 = edge-parallel atomics
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

// Bind group 0 for all layout passes. One explicit layout shared by every
// compute pipeline, so a single bind group serves all passes.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> disp: array<vec2<f32>>;
