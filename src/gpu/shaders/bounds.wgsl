// Bounding box of all node positions, used to auto-fit the camera.
// Reading back 16 bytes instead of the whole position buffer.
//
// Two classic GPU patterns in one kernel:
//   1. Tree reduction in workgroup shared memory: 256 values -> 1 in
//      log2(256) = 8 steps, with a barrier after each step.
//   2. Global combine with atomics. atomicMin/atomicMax only work on
//      integers, so floats are mapped to u32 with an ORDER-PRESERVING bit
//      trick: if a < b as floats then f2u(a) < f2u(b) as unsigned ints.

struct Bounds {
  minX: atomic<u32>,
  minY: atomic<u32>,
  maxX: atomic<u32>,
  maxY: atomic<u32>,
}

struct Count { n: u32, _a: u32, _b: u32, _c: u32 }

@group(0) @binding(0) var<uniform> count: Count;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bounds: Bounds;

const WG: u32 = 256u;
var<workgroup> smin: array<vec2<f32>, WG>;
var<workgroup> smax: array<vec2<f32>, WG>;

fn f2u(f: f32) -> u32 {
  let b = bitcast<u32>(f);
  // positive: flip the sign bit so it sorts above negatives
  // negative: flip all bits so larger magnitudes sort lower
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let l = lid.x;
  // Out-of-range threads re-read the last point: harmless for min/max.
  let p = points[min(gid.x, count.n - 1u)];
  smin[l] = p;
  smax[l] = p;
  workgroupBarrier();

  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (l < s) {
      smin[l] = min(smin[l], smin[l + s]);
      smax[l] = max(smax[l], smax[l + s]);
    }
    workgroupBarrier();
  }

  if (l == 0u) {
    atomicMin(&bounds.minX, f2u(smin[0].x));
    atomicMin(&bounds.minY, f2u(smin[0].y));
    atomicMax(&bounds.maxX, f2u(smax[0].x));
    atomicMax(&bounds.maxY, f2u(smax[0].y));
  }
}
