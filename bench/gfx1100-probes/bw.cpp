// streaming memory bandwidth probe: read-only reduce + copy.
// (a) checks the card reaches spec rates at all, (b) gives the bandwidth number the
// Q5_K GEMV verdict needs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CHK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(_e)); return 1; } } while (0)

__global__ void read_kernel(const float4 *__restrict__ in, float *out, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = {0, 0, 0, 0};
  for (; i < n4; i += stride) {
    float4 v = in[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  if (s == 1234.5678f) out[threadIdx.x] = s;   // never true; keeps the loads alive
}
__global__ void copy_kernel(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) out[i] = in[i];
}
int main(int argc, char **argv) {
  size_t mib = (argc > 1) ? (size_t)strtoul(argv[1], nullptr, 0) : 1024;
  size_t bytes = mib * 1024 * 1024, n4 = bytes / 16;
  size_t free_b = 0, tot_b = 0;
  CHK(hipMemGetInfo(&free_b, &tot_b));
  printf("VRAM free %.2f GiB / %.2f GiB, buffer %.0f MiB\n",
         free_b / 1073741824.0, tot_b / 1073741824.0, (double)mib);
  float4 *a = nullptr, *b = nullptr; float *o = nullptr;
  CHK(hipMalloc(&a, bytes)); CHK(hipMalloc(&b, bytes)); CHK(hipMalloc(&o, 4096));
  CHK(hipMemset(a, 1, bytes));
  int grid = 42 * 8, blk = 256;
  hipEvent_t ev0, ev1; CHK(hipEventCreate(&ev0)); CHK(hipEventCreate(&ev1));
  std::vector<double> rd, cp;
  for (int r = 0; r < 7; r++) {
    float t = 0;
    CHK(hipEventRecord(ev0)); read_kernel<<<grid, blk>>>(a, o, n4);
    CHK(hipEventRecord(ev1)); CHK(hipEventSynchronize(ev1));
    CHK(hipEventElapsedTime(&t, ev0, ev1));
    if (r && t > 0) rd.push_back(bytes / (t * 1e-3) / 1e9);
    CHK(hipEventRecord(ev0)); copy_kernel<<<grid, blk>>>(a, b, n4);
    CHK(hipEventRecord(ev1)); CHK(hipEventSynchronize(ev1));
    CHK(hipEventElapsedTime(&t, ev0, ev1));
    if (r && t > 0) cp.push_back(2.0 * bytes / (t * 1e-3) / 1e9);
    if (r == 0) printf("first (warmup) pass: read %.3f ms, copy %.3f ms\n", t, t);
  }
  if (rd.empty()) { printf("no valid timings\n"); return 2; }
  std::sort(rd.begin(), rd.end()); std::sort(cp.begin(), cp.end());
  printf("read-only  (%.0f MiB stream): %.1f GB/s  [%.1f..%.1f] over %zu runs\n",
         (double)mib, rd[rd.size() / 2], rd.front(), rd.back(), rd.size());
  printf("copy r+w   (%.0f MiB in+out) : %.1f GB/s  [%.1f..%.1f]\n",
         (double)mib, cp[cp.size() / 2], cp.front(), cp.back());
  printf("spec RX 7900 XT            : 800 GB/s (20 Gbps x 320-bit)\n");
  return 0;
}
