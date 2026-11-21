// Buffer helpers.
//
// Usage flags decide what a buffer can be used for:
//   STORAGE  - read/written by shaders as array<...>
//   UNIFORM  - small read-only constants (params, camera)
//   COPY_SRC - can be copied out of (e.g. to a staging buffer for readback)
//   COPY_DST - can be written by queue.writeBuffer / copied into
//   MAP_READ - can be mapped for CPU reading (only with COPY_DST)

/** Create a GPU buffer initialised with a typed array. */
export function createBufferWithData(device, data, usage, label) {
  // Storage bindings must be non-empty and sizes multiples of 4.
  const size = Math.max(16, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ label, size, usage, mappedAtCreation: true });
  const Ctor = data.constructor;
  new Ctor(buffer.getMappedRange(0, data.byteLength)).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Copy a GPU buffer back to the CPU. Used only for debugging/verification:
 * the render loop never does this.
 */
export async function readBuffer(device, src, byteLength, Ctor = Float32Array) {
  const staging = device.createBuffer({
    label: 'readback-staging',
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder({ label: 'readback' });
  enc.copyBufferToBuffer(src, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Ctor(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

/** Order-preserving u32 -> f32, the inverse of f2u() in bounds.wgsl. */
export function u2f(u) {
  const bits = u & 0x80000000 ? u & 0x7fffffff : ~u >>> 0;
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, bits >>> 0);
  return view.getFloat32(0);
}
