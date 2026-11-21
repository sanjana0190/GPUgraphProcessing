// Layout constants shared by the CPU and GPU engines, so both run the exact
// same algorithm with the exact same numbers. That is what makes the
// benchmark and the correctness check apples-to-apples.
//
// Fruchterman-Reingold forces (d = distance between two nodes, k = ideal edge length):
//   repulsion  between every pair:      f_r(d) = k^2 / d      (pushes apart)
//   attraction along every edge:        f_a(d) = d^2 / k      (pulls together)
//   gravity toward the origin:          f_g(p) = -gravity * p (keeps components on screen)
// Each iteration a node moves along its net force, but by at most `temperature`.
// The temperature cools every iteration, so the layout settles.
export function layoutParams(n, overrides = {}) {
  const k = 10;
  const spread = k * Math.sqrt(Math.max(n, 1));
  const base = {
    k,
    gravity: 0.02, // weak: strong gravity squashes every graph into a disk
    initialSize: spread * 2, // side of the square that random initial positions fill
    t0: spread * 0.15, // initial max step per iteration
    cooling: 0.997, // t <- t * cooling each iteration (slow cooling = fewer folds)
    tMin: k * 0.05, // keep a small amount of motion so the layout can still adapt
    bound: 1e6, // safety clamp on positions
  };
  return { ...base, ...overrides };
}

/** Read numeric overrides such as ?gravity=0.1&cooling=0.995 from a query string. */
export function paramOverridesFromQuery(search) {
  const q = new URLSearchParams(search);
  const out = {};
  for (const key of ['gravity', 'cooling', 'tMin', 'k']) {
    if (q.has(key) && Number.isFinite(Number(q.get(key)))) out[key] = Number(q.get(key));
  }
  return out;
}

/** Temperature schedule ("simulated annealing" for the layout). */
export class Cooling {
  constructor(p) {
    this.p = p;
    this.t = p.t0;
  }
  /** Returns the temperature for this iteration, then cools. */
  next() {
    const t = this.t;
    this.t = Math.max(this.p.tMin, this.t * this.p.cooling);
    return t;
  }
  reheat() {
    this.t = Math.max(this.t, this.p.t0 * 0.5);
  }
}
