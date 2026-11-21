// GPU timing with timestamp queries.
//
// performance.now() around queue.submit() only measures how long the CPU
// takes to *record* commands; the GPU runs asynchronously. Timestamp queries
// are written by the GPU itself at the start and end of each pass, so they
// give the real per-pass GPU time.
//
// Note: Chrome rounds timestamps to 100 microseconds unless launched with
// --enable-webgpu-developer-features. Averaging over many frames smooths that.
export const PASSES = ['repulsion', 'attraction', 'integrate', 'render'];

export class GPUTimer {
  constructor(device, enabled) {
    this.device = device;
    this.enabled = enabled;
    this.avg = Object.fromEntries(PASSES.map((p) => [p, 0]));
    this.samples = 0;
    if (!enabled) return;
    const count = PASSES.length * 2;
    this.querySet = device.createQuerySet({ label: 'pass-timestamps', type: 'timestamp', count });
    this.resolveBuffer = device.createBuffer({
      label: 'timestamp-resolve',
      size: count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    // A small pool: while one readback buffer is being mapped, the next frame
    // can use another, so timing never stalls the pipeline.
    this.pool = Array.from({ length: 3 }, (_, i) =>
      device.createBuffer({
        label: `timestamp-readback-${i}`,
        size: count * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    );
    this.free = [...this.pool];
    this.pending = null;
  }

  /** timestampWrites for a pass descriptor, or undefined when not timing. */
  writes(passName, active) {
    if (!this.enabled || !active) return undefined;
    const i = PASSES.indexOf(passName);
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }

  /** Call after encoding timed passes. Returns false if no buffer was free. */
  resolve(encoder) {
    if (!this.enabled || this.free.length === 0) return false;
    const buf = this.free.pop();
    encoder.resolveQuerySet(this.querySet, 0, PASSES.length * 2, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, buf, 0, PASSES.length * 16);
    this.pending = buf;
    return true;
  }

  /** Call after queue.submit(). Reads the results asynchronously. */
  collect() {
    const buf = this.pending;
    if (!buf) return;
    this.pending = null;
    buf.mapAsync(GPUMapMode.READ).then(
      () => {
        const t = new BigInt64Array(buf.getMappedRange());
        const alpha = this.samples < 10 ? 1 / (this.samples + 1) : 0.1;
        PASSES.forEach((p, i) => {
          const ns = Number(t[2 * i + 1] - t[2 * i]);
          if (ns >= 0 && ns < 1e10) this.avg[p] += alpha * (ns / 1e6 - this.avg[p]);
        });
        this.samples++;
        buf.unmap();
        this.free.push(buf);
      },
      () => this.free.push(buf),
    );
  }

  reset() {
    for (const p of PASSES) this.avg[p] = 0;
    this.samples = 0;
  }
}
