// Node color value in [0, 1] from log-degree, so hubs stand out.
// Color by degree relative to the AVERAGE degree, on a log scale:
// nodes at or below average are blue, nodes with >= 8x the average (or the
// max degree, if hubs are bigger) are orange. A regular graph like a grid
// therefore stays blue, and only genuine hubs stand out.
export function degreeColorValues(degree) {
  const n = degree.length;
  let maxDeg = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += degree[i];
    if (degree[i] > maxDeg) maxDeg = degree[i];
  }
  const avg = Math.max(1, sum / Math.max(1, n));
  const denom = Math.log(Math.max(maxDeg / avg, 8));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.min(1, Math.max(0, Math.log(Math.max(degree[i], 1) / avg) / denom));
  }
  return out;
}

// Same palette as the WGSL node shader, for the Canvas 2D renderer.
export function paletteRGB(t) {
  const a = [0.29, 0.56, 0.98]; // blue, low degree
  const b = [0.98, 0.55, 0.24]; // orange, high degree
  const s = Math.min(1, Math.max(0, t));
  return a.map((v, i) => Math.round(255 * (v + (b[i] - v) * s)));
}
