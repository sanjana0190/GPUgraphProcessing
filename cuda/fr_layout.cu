// Fruchterman-Reingold layout in CUDA: the same three kernels as the WGSL
// version, written to show the 1:1 concept mapping between CUDA and WebGPU.
//
//   CUDA                         WGSL / WebGPU
//   ---------------------------  -------------------------------------------
//   thread block (blockDim)      workgroup (@workgroup_size)
//   grid of blocks               dispatchWorkgroups(ceil(n / 64))
//   threadIdx / blockIdx         local_invocation_id / workgroup_id
//   __shared__                   var<workgroup>
//   __syncthreads()              workgroupBarrier()
//   cudaMalloc / cudaMemcpy      device.createBuffer / queue.writeBuffer
//   kernel<<<grid, block>>>()    pass.setPipeline(); pass.dispatchWorkgroups()
//   cudaEvent timing             timestamp-query on each pass
//   atomicAdd(float*)            not available: WGSL needs i32/u32 atomics
//
// Not runnable on Apple Silicon (no NVIDIA GPU). To run it free on Google
// Colab: Runtime > Change runtime type > T4 GPU, then in a cell:
//   !nvcc -O3 -arch=sm_75 fr_layout.cu -o fr && ./fr 20000
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int TILE = 64;

struct Params { int n; float k, temperature, gravity, bound; };

// PASS 1: tiled O(n^2) repulsion (same as repulsion.wgsl)
__global__ void repulsion(const float2* __restrict__ pos, float2* __restrict__ disp, Params p) {
  __shared__ float2 tile[TILE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float2 pi = pos[min(i, p.n - 1)];
  float k2 = p.k * p.k, minD2 = 1e-4f * k2;
  float2 acc = make_float2(0.f, 0.f);
  for (int t = 0; t < (p.n + TILE - 1) / TILE; t++) {
    int j = t * TILE + threadIdx.x;
    tile[threadIdx.x] = pos[min(j, p.n - 1)];
    __syncthreads();
    int count = min(TILE, p.n - t * TILE);
    for (int s = 0; s < count; s++) {
      float dx = pi.x - tile[s].x, dy = pi.y - tile[s].y;
      float f = k2 / fmaxf(dx * dx + dy * dy, minD2);
      acc.x += dx * f;
      acc.y += dy * f;
    }
    __syncthreads();
  }
  if (i < p.n) disp[i] = acc;
}

// PASS 2: CSR gather attraction (same as attraction.wgsl)
__global__ void attraction(const float2* pos, float2* disp, const unsigned* offsets,
                           const unsigned* neighbors, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) return;
  float2 pi = pos[i], acc = make_float2(0.f, 0.f);
  for (unsigned e = offsets[i]; e < offsets[i + 1]; e++) {
    float2 pj = pos[neighbors[e]];
    float dx = pi.x - pj.x, dy = pi.y - pj.y;
    float f = sqrtf(dx * dx + dy * dy) / p.k;
    acc.x -= dx * f;
    acc.y -= dy * f;
  }
  disp[i].x += acc.x;
  disp[i].y += acc.y;
}

// PASS 3: integrate (same as integrate.wgsl)
__global__ void integrate(float2* pos, const float2* disp, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) return;
  float2 q = pos[i];
  float dx = disp[i].x - p.gravity * q.x, dy = disp[i].y - p.gravity * q.y;
  float len = sqrtf(dx * dx + dy * dy);
  if (len > 0.f) {
    float s = fminf(len, p.temperature) / len;
    q.x += dx * s;
    q.y += dy * s;
  }
  q.x = fminf(fmaxf(q.x, -p.bound), p.bound);
  q.y = fminf(fmaxf(q.y, -p.bound), p.bound);
  pos[i] = q;
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 10000;
  int iters = argc > 2 ? atoi(argv[2]) : 100;

  // Barabasi-Albert graph, m = 2 (same model as the web benchmark)
  std::mt19937 rng(1);
  std::vector<std::pair<unsigned, unsigned>> edges;
  std::set<std::pair<unsigned, unsigned>> seen;
  std::vector<unsigned> repeated;
  auto add = [&](unsigned u, unsigned v) {
    if (u == v) return false;
    auto key = std::minmax(u, v);
    if (!seen.insert(key).second) return false;
    edges.push_back({u, v});
    repeated.push_back(u);
    repeated.push_back(v);
    return true;
  };
  for (unsigned i = 0; i < 3; i++) for (unsigned j = i + 1; j < 3; j++) add(i, j);
  for (unsigned v = 3; v < (unsigned)n; v++)
    for (int added = 0, guard = 0; added < 2 && guard < 100; guard++)
      if (add(v, repeated[rng() % repeated.size()])) added++;

  // CSR
  std::vector<unsigned> offsets(n + 1, 0), neighbors(2 * edges.size());
  for (auto& e : edges) { offsets[e.first + 1]++; offsets[e.second + 1]++; }
  for (int i = 0; i < n; i++) offsets[i + 1] += offsets[i];
  std::vector<unsigned> cursor(offsets.begin(), offsets.end() - 1);
  for (auto& e : edges) { neighbors[cursor[e.first]++] = e.second; neighbors[cursor[e.second]++] = e.first; }

  float k = 10.f, spread = k * sqrtf((float)n);
  std::uniform_real_distribution<float> U(-spread, spread);
  std::vector<float2> pos(n);
  for (auto& q : pos) q = make_float2(U(rng), U(rng));

  // Device buffers (WebGPU: createBuffer + writeBuffer)
  float2 *dPos, *dDisp;
  unsigned *dOff, *dNb;
  CHECK(cudaMalloc(&dPos, n * sizeof(float2)));
  CHECK(cudaMalloc(&dDisp, n * sizeof(float2)));
  CHECK(cudaMalloc(&dOff, offsets.size() * sizeof(unsigned)));
  CHECK(cudaMalloc(&dNb, neighbors.size() * sizeof(unsigned)));
  CHECK(cudaMemcpy(dPos, pos.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dOff, offsets.data(), offsets.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dNb, neighbors.data(), neighbors.size() * sizeof(unsigned), cudaMemcpyHostToDevice));

  Params p{n, k, spread * 0.15f, 0.02f, 1e6f};
  int blocks = (n + TILE - 1) / TILE;  // WebGPU: dispatchWorkgroups(ceil(n / 64))
  auto step = [&]() {
    repulsion<<<blocks, TILE>>>(dPos, dDisp, p);
    attraction<<<blocks, TILE>>>(dPos, dDisp, dOff, dNb, p);
    integrate<<<blocks, TILE>>>(dPos, dDisp, p);
    p.temperature = fmaxf(k * 0.05f, p.temperature * 0.997f);
  };
  for (int i = 0; i < 10; i++) step();  // warm-up
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) step();
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaGetLastError());
  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("%s: n=%d m=%zu  %.3f ms / iteration (%d iterations)\n", prop.name, n, edges.size(), ms / iters, iters);

  cudaFree(dPos); cudaFree(dDisp); cudaFree(dOff); cudaFree(dNb);
  return 0;
}
