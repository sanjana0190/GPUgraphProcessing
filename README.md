# GPU-Accelerated Graph Processing Engine

Force-directed graph layout running on the GPU through **WebGPU compute shaders written in WGSL**, rendered by a **multi-pass WebGPU pipeline** that draws straight from the GPU buffers the simulation writes. A matching single-threaded JavaScript + Canvas 2D engine runs the identical algorithm as the baseline.

On an Apple M1, the GPU engine is **13x faster at 2,000 nodes and 38x to 50x faster from 10,000 to 50,000 nodes** per layout iteration.

## Run it

Requires Node 18+ and a browser with WebGPU: Chrome or Edge 113+, Safari 26+ (macOS 26), or Firefox 145+ on Apple Silicon.

```bash
npm install
npm run dev        # open http://localhost:5173 in Chrome
npm test           # unit tests (graph building, CPU layout correctness)
```

In the app you can switch between the GPU and CPU engines, pick a graph and size, watch per-pass GPU timings, and use two built-in tools:

- **Verify GPU vs CPU** runs the GPU force passes on a fixed graph, reads the forces back, and compares them with the CPU reference.
- **Run benchmark** times both engines from 500 to 50,000 nodes and prints a Markdown table.

The same tools run headless from the terminal against your installed Chrome:

```bash
node scripts/run-in-chrome.mjs verify
node scripts/run-in-chrome.mjs bench      # writes bench/results.md and bench/results.json
GRAPH=stochastic-block NODES=20000 node scripts/run-in-chrome.mjs shot out.png
```

## How it works

**Algorithm.** Fruchterman–Reingold. Every pair of nodes repels with force k²/d, every edge attracts with force d²/k, and weak gravity keeps components on screen. Each iteration a node moves along its net force by at most the current *temperature*, which cools geometrically so the layout settles. All constants live in [src/layout/params.js](src/layout/params.js) and are shared by both engines.

**Per iteration, one command encoder and one `queue.submit()`:**

| Pass | Kind | Work | Shader |
|---|---|---|---|
| 1. Repulsion | compute | O(n²), tiled through workgroup shared memory | [repulsion.wgsl](src/gpu/shaders/repulsion.wgsl) |
| 2. Attraction | compute | O(m), each thread walks its own CSR row | [attraction.wgsl](src/gpu/shaders/attraction.wgsl) |
| 3. Integrate | compute | O(n), temperature-limited move | [integrate.wgsl](src/gpu/shaders/integrate.wgsl) |
| 4. Render | render | edges as `line-list`, nodes as instanced quads, 4x MSAA | [edges.wgsl](src/gpu/shaders/edges.wgsl), [nodes.wgsl](src/gpu/shaders/nodes.wgsl) |

WebGPU orders passes within a submission and makes storage writes from one pass visible to the next, so no manual barriers are needed between passes.

**Key design decisions**

- **Tiled n-body repulsion.** Each 64-thread workgroup loads 64 positions into `var<workgroup>` memory, synchronizes with `workgroupBarrier()`, and every thread reads the whole tile from fast shared memory. This is the classic CUDA n-body pattern and cuts global memory reads by 64x.
- **CSR adjacency instead of atomics.** Every edge is stored in both endpoints' rows, so thread *i* only ever writes `disp[i]`. There are no write conflicts, and results are deterministic. WGSL has no floating-point atomics, so the alternative needs fixed-point integers.
- **The atomic alternative is implemented too.** A UI toggle switches to [attraction_atomic.wgsl](src/gpu/shaders/attraction_atomic.wgsl): one thread per edge, forces converted to fixed-point `i32` and scattered with `atomicAdd`. The scale is derived from the maximum degree so the worst-case sum cannot overflow, which trades precision for range.
- **No CPU readback.** Node positions live in one storage buffer for the whole session. The vertex shaders index into it directly, so positions never travel GPU→CPU→GPU. The only readbacks are a 16-byte bounding box and the timestamp queries.
- **Bounding box with a parallel reduction.** [bounds.wgsl](src/gpu/shaders/bounds.wgsl) reduces 256 points per workgroup in shared memory, then combines workgroups with `atomicMin`/`atomicMax` on floats mapped to order-preserving `u32` bits.
- **Uniform layout.** The `Params` struct in [common.wgsl](src/gpu/shaders/common.wgsl) uses only 4-byte scalars and is padded to 48 bytes, avoiding the `vec3` 16-byte alignment trap.
- **GPU timing with timestamp queries.** Each pass records begin and end timestamps on the GPU ([timing.js](src/gpu/timing.js)), read back through a small buffer pool so profiling never stalls the frame.

## Benchmark

Apple M1 (8-core GPU), Chrome 155 on macOS 26, Barabási–Albert graphs with 2 edges per new node. Metric: wall-clock milliseconds per full layout iteration (repulsion + attraction + integrate), rendering excluded on both sides.

| Nodes | Edges | CPU ms/iter | GPU ms/iter | Speedup |
|---:|---:|---:|---:|---:|
| 500 | 997 | 0.27 | 0.150 | 1.8x |
| 1,000 | 1,997 | 1.03 | 0.193 | 5.4x |
| 2,000 | 3,997 | 4.01 | 0.309 | 13.0x |
| 5,000 | 9,997 | 24.88 | 0.859 | 29.0x |
| 10,000 | 19,997 | 101.32 | 2.675 | 37.9x |
| 20,000 | 39,997 | 396.82 | 9.235 | 43.0x |
| 50,000 | 99,997 | 2,506.70 | 50.167 | 50.0x |

**Methodology** ([src/bench/benchmark.js](src/bench/benchmark.js)):

- Both engines get the same graph, seed, initial positions and constants.
- The CPU baseline is a fair one. It uses flat typed arrays, visits each pair once using Newton's third law, and warms up the JIT before timing. Only time inside the layout step is counted.
- The GPU number is wall-clock time and includes submission overhead. It waits for `queue.onSubmittedWorkDone()` every 10 iterations, so it reflects real completed GPU work, not just command recording.
- Small graphs gain little because a fixed per-submit cost dominates. The speedup grows with n because repulsion is O(n²) and is exactly the work that parallelizes.

**Correctness.** The verify tool compares GPU and CPU forces on a 2,000-node graph. Errors are normalized by the largest force:

| Check | Max error |
|---|---:|
| CSR gather forces | 1.1e-7 |
| Fixed-point atomic forces | 7.4e-8 |
| Positions after one full iteration, relative to step size | 4.5e-7 |

## Project layout

```
src/
  graph/       generators (BA, SBM, ER, grid), CSR builder, SNAP edge-list loader
  layout/      shared Fruchterman–Reingold constants and cooling schedule
  cpu/         CPU reference layout + Canvas 2D renderer
  gpu/         device setup, buffers, layout engine, renderer, timestamp timer
  gpu/shaders/ WGSL: repulsion, attraction (CSR + atomic), integrate, bounds, nodes, edges
  bench/       benchmark and GPU-vs-CPU verification
  engines.js   CPU and GPU engines behind one interface
cuda/          the same kernels in CUDA, for comparison (run on an NVIDIA GPU, e.g. Colab)
scripts/       headless Chrome runner for verify / bench / screenshots
tests/         Vitest unit tests
```

## Limits and next steps

- Repulsion is still O(n²) total work. Around 100,000 nodes the GPU drops to a few frames per second. The next step is Barnes–Hut (a GPU quadtree, as in GraphWaGu) or a uniform grid that only computes exact repulsion between nearby cells, bringing it to roughly O(n log n).
- Fruchterman–Reingold can settle into local minima, visible as folds in grid graphs. Multilevel coarsening would fix this.
- Chrome rounds timestamp queries to 100 µs unless started with `--enable-webgpu-developer-features`. The live per-pass numbers are averaged over many frames to smooth this.

Data: [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) (McAuley and Leskovec, 2012).
