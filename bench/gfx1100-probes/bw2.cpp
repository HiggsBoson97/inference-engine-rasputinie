// bw2.cpp -- parameterised streaming-bandwidth probe for gfx1100 (RX 7900 XT).
//
// Answers: (1) is 619 GB/s a kernel artefact or the wall?  (2) does load width
// per lane matter, or only bytes-in-flight?  (3) how many waves/SIMD does a
// stream need?  (4) does the memory clock ramp (DPM) explain a low first pass?
// (5) does a non-temporal store beat write-allocate on a copy?
//
// usage: ./bw2 <MiB> [reps]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

typedef unsigned v4u __attribute__((ext_vector_type(4)));

#define CHK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(_e)); exit(1); } } while (0)

// ---------------------------------------------------------------- read kernels
// V = element type per lane (4B / 8B / 16B). N = independent loads in flight.
template <class V, int N>
__global__ void readN(const V *__restrict__ in, size_t n, unsigned *sink) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  size_t step = stride * N;
  // main loop: N loads issued back-to-back, all independent
  for (; i + (N - 1) * stride < n; i += step) {
    V v[N];
#pragma unroll
    for (int k = 0; k < N; k++) v[k] = in[i + k * stride];
#pragma unroll
    for (int k = 0; k < N; k++) {
      const unsigned *p = reinterpret_cast<const unsigned *>(&v[k]);
#pragma unroll
      for (int j = 0; j < (int)(sizeof(V) / 4); j++) acc ^= p[j];
    }
  }
  for (; i < n; i += stride) {
    V v = in[i];
    const unsigned *p = reinterpret_cast<const unsigned *>(&v);
    for (int j = 0; j < (int)(sizeof(V) / 4); j++) acc ^= p[j];
  }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

template <class V, int N>
__global__ void readN_rep(const V *__restrict__ in, size_t n, unsigned *sink, int rp) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (int p = 0; p < rp; p++) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (N - 1) * stride < n; i += stride * N) {
      V v[N];
#pragma unroll
      for (int k = 0; k < N; k++) v[k] = in[i + k * stride];
#pragma unroll
      for (int k = 0; k < N; k++) {
        const unsigned *q = reinterpret_cast<const unsigned *>(&v[k]);
#pragma unroll
        for (int j = 0; j < (int)(sizeof(V) / 4); j++) acc ^= q[j];
      }
    }
  }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

// non-temporal (streaming) variant: slc=1 dlc=1 on the load
template <int N>
__global__ void readN_nt(const v4u *__restrict__ in, size_t n, unsigned *sink) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  size_t step = stride * N;
  for (; i + (N - 1) * stride < n; i += step) {
    v4u v[N];
#pragma unroll
    for (int k = 0; k < N; k++) v[k] = __builtin_nontemporal_load(in + i + k * stride);
#pragma unroll
    for (int k = 0; k < N; k++) acc ^= v[k][0] ^ v[k][1] ^ v[k][2] ^ v[k][3];
  }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

// slab variant: block b owns a contiguous slab; inside, whole waves march
// linearly. Different address-to-MCD distribution from grid-stride.
template <class V, int N>
__global__ void readN_slab(const V *__restrict__ in, size_t n, unsigned *sink) {
  size_t per_block = (n + gridDim.x - 1) / gridDim.x;
  size_t begin = (size_t)blockIdx.x * per_block;
  size_t end = begin + per_block; if (end > n) end = n;
  size_t stride = blockDim.x;
  size_t i = begin + threadIdx.x;
  unsigned acc = 0;
  for (; i + (N - 1) * stride < end; i += stride * N) {
    V v[N];
#pragma unroll
    for (int k = 0; k < N; k++) v[k] = in[i + k * stride];
#pragma unroll
    for (int k = 0; k < N; k++) {
      const unsigned *p = reinterpret_cast<const unsigned *>(&v[k]);
#pragma unroll
      for (int j = 0; j < (int)(sizeof(V) / 4); j++) acc ^= p[j];
    }
  }
  for (; i < end; i += stride) {
    V v = in[i];
    const unsigned *p = reinterpret_cast<const unsigned *>(&v);
    for (int j = 0; j < (int)(sizeof(V) / 4); j++) acc ^= p[j];
  }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

// ------------------------------------------------------------------ copy kernels
__global__ void copy4(const uint4 *__restrict__ in, uint4 *__restrict__ out, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    out[i] = in[i];
}
__global__ void copy4_nt(const v4u *__restrict__ in, v4u *__restrict__ out, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    __builtin_nontemporal_store(in[i], out + i);
}

// --------------------------------------------------------------------- harness
static uint4 *g_in; static uint4 *g_out; static unsigned *g_sink;

struct Res { double gbs; };

template <class F>
static void run(const char *name, size_t bytes, int grid, int blk, int reps, F launch) {
  hipEvent_t e0, e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
  CHK(hipDeviceSynchronize());
  std::vector<double> v;
  for (int r = 0; r < reps; r++) {
    float t = 0;
    CHK(hipEventRecord(e0));
    launch(grid, blk);
    CHK(hipEventRecord(e1));
    CHK(hipEventSynchronize(e1));
    CHK(hipEventElapsedTime(&t, e0, e1));
    v.push_back(bytes / (t * 1e-3) / 1e9);
  }
  std::vector<double> s = v; std::sort(s.begin(), s.end());
  double med = s[s.size() / 2];
  printf("%-34s grid=%-6d blk=%-5d  med %7.1f GB/s   min %7.1f  max %7.1f  first %6.1f\n",
         name, grid, blk, med, s.front(), s.back(), v[0]);
  hipEventDestroy(e0); hipEventDestroy(e1);
}

int main(int argc, char **argv) {
  size_t mib = (argc > 1) ? strtoull(argv[1], nullptr, 0) : 2048;
  int reps = (argc > 2) ? atoi(argv[2]) : 7;
  size_t bytes = mib << 20;
  long off_mib = getenv("OFF_MIB") ? atol(getenv("OFF_MIB")) : 0;   // shift the buffer start
  size_t n16 = bytes / 16;

  hipDeviceProp_t p; CHK(hipGetDeviceProperties(&p, 0));
  printf("device      : %s\n", p.name);
  printf("pciBusID    : %d   multiProcessorCount %d   clock %d MHz  memclock %d MHz\n",
         p.pciBusID, p.multiProcessorCount, p.clockRate / 1000, p.memoryClockRate / 1000);
  printf("buffer      : %zu MiB   (spec 800 GB/s)\n\n", mib);

  CHK(hipMalloc(&g_in, bytes + (off_mib << 20)));
  CHK(hipMalloc(&g_out, bytes + (off_mib << 20)));
  if (off_mib) { g_in = (uint4*)((char*)g_in + (off_mib << 20)); g_out = (uint4*)((char*)g_out + (off_mib << 20)); }
  CHK(hipMalloc(&g_sink, 8192));
  CHK(hipMemset(g_in, 1, bytes));
  CHK(hipMemset(g_out, 0, bytes));

  const int SM = p.multiProcessorCount;   // 42 on this card (128-lane units)

  printf("--- load width per lane, 8 independent loads in flight, grid-stride\n");
  int g = SM * 8, b = 256;
  run("read u32   (4B/lane) x8", bytes, g, b, reps, [&](int G,int B){ readN<unsigned,8><<<G,B>>>(reinterpret_cast<unsigned*>(g_in), n16*4, g_sink); });
  run("read uint2 (8B/lane) x8", bytes, g, b, reps, [&](int G,int B){ readN<uint2,8><<<G,B>>>(reinterpret_cast<uint2*>(g_in), n16*2, g_sink); });
  run("read uint4 (16B/lane) x8", bytes, g, b, reps, [&](int G,int B){ readN<uint4,8><<<G,B>>>(g_in, n16, g_sink); });
  run("read uint4 NT (slc=dlc=1) x8", bytes, g, b, reps, [&](int G,int B){ readN_nt<8><<<G,B>>>(reinterpret_cast<const v4u*>(g_in), n16, g_sink); });

  if (getenv("RP")) {
    int rp = atoi(getenv("RP"));
    for (int m : {1, 2, 4, 8}) {
      char nm[80]; snprintf(nm, sizeof nm, "read uint4 x8 in-kernel x%d (same footprint)", rp);
      run(nm, bytes * rp, SM * 8, 256, reps, [&](int G,int B){
        readN_rep<uint4,8><<<G,B>>>(g_in, n16, g_sink, rp); });
      (void)m; break;
    }
    printf("\n");
  }

  printf("\n--- loads in flight (ILP), 16B/lane\n");
  run("read uint4 x1", bytes, g, b, reps, [&](int G,int B){ readN<uint4,1><<<G,B>>>(g_in, n16, g_sink); });
  run("read uint4 x2", bytes, g, b, reps, [&](int G,int B){ readN<uint4,2><<<G,B>>>(g_in, n16, g_sink); });
  run("read uint4 x4", bytes, g, b, reps, [&](int G,int B){ readN<uint4,4><<<G,B>>>(g_in, n16, g_sink); });
  run("read uint4 x8", bytes, g, b, reps, [&](int G,int B){ readN<uint4,8><<<G,B>>>(g_in, n16, g_sink); });
  run("read uint4 x16", bytes, g, b, reps, [&](int G,int B){ readN<uint4,16><<<G,B>>>(g_in, n16, g_sink); });

  printf("\n--- occupancy sweep, 16B/lane x4, grid-stride (blocks = SM*N)\n");
  for (int m : {1, 2, 4, 8, 16, 32}) {
    char nm[64]; snprintf(nm, sizeof nm, "read uint4 x4 grid=SM*%d", m);
    run(nm, bytes, SM * m, 256, reps, [&](int G,int B){ readN<uint4,4><<<G,B>>>(g_in, n16, g_sink); });
  }
  printf("\n--- block size sweep, 16B/lane x4, grid = lanes*8/blk\n");
  for (int bb : {64, 128, 256, 512, 1024}) {
    int gg = (SM * 8 * 256) / bb;
    char nm[64]; snprintf(nm, sizeof nm, "read uint4 x4 blk=%d", bb);
    run(nm, bytes, gg, bb, reps, [&](int G,int B){ readN<uint4,4><<<G,B>>>(g_in, n16, g_sink); });
  }

  printf("\n--- address distribution: slab-per-block vs grid-stride\n");
  run("read uint4 x4 slab", bytes, g, b, reps, [&](int G,int B){ readN_slab<uint4,4><<<G,B>>>(g_in, n16, g_sink); });
  run("read uint4 x4 slab grid=SM*32", bytes, SM*32, b, reps, [&](int G,int B){ readN_slab<uint4,4><<<G,B>>>(g_in, n16, g_sink); });

  printf("\n--- copy (r+w counted)\n");
  run("copy uint4", bytes, g, b, reps, [&](int G,int B){ copy4<<<G,B>>>(g_in, g_out, n16); });
  run("copy uint4 NT store", bytes, g, b, reps, [&](int G,int B){ copy4_nt<<<G,B>>>(reinterpret_cast<const v4u*>(g_in), reinterpret_cast<v4u*>(g_out), n16); });
  run("copy uint4 grid=SM*32", bytes, SM*32, b, reps, [&](int G,int B){ copy4<<<G,B>>>(g_in, g_out, n16); });

  printf("\n--- DPM ramp check: 40 consecutive passes, best config\n");
  {
    hipEvent_t e0,e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
    for (int r = 0; r < 40; r++) {
      float t = 0; CHK(hipEventRecord(e0));
      readN<uint4,4><<<SM*8,256>>>(g_in, n16, g_sink);
      CHK(hipEventRecord(e1)); CHK(hipEventSynchronize(e1));
      CHK(hipEventElapsedTime(&t, e0, e1));
      if (r % 4 == 0 || r == 39) printf("   pass %2d: %7.1f GB/s\n", r, bytes/(t*1e-3)/1e9);
    }
  }
  return 0;
}
