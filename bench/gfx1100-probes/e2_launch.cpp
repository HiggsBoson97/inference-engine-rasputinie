// launch.cpp -- is the per-launch bandwidth penalty on gfx1100 real, and what
// does a realistic ~400-launch decode step actually achieve?
//
// Modes
//   sweep  <GiB> <reps>      N launches of (GiB/N) at constant total bytes
//   null   <N>               per-launch cost of empty / tiny kernels
//   orig   <reps>            the original 8 x 1 GiB observation + controls
//   hot    <GiB> <N> <cond> <reps>   cond = cold | spin | gaps | spin_nosync
//   decode <manifest> <grouping> <steps> <reps>
//                            grouping = tensor | layer | layer8
//
// All timing is hipEvent around a launch group (and optionally per launch),
// warm-up discarded, median reported.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <unistd.h>

#define CHK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(_e)); exit(1); } } while (0)

// ---------------------------------------------------------------- kernels
// 8 independent 16 B loads per lane, grid-stride over one contiguous range.
__global__ void read_range(const uint4 *__restrict__ in, size_t n4, unsigned *sink) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (; i + 7 * stride < n4; i += 8 * stride) {
    uint4 v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = in[i + k * stride];
#pragma unroll
    for (int k = 0; k < 8; k++) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
  }
  for (; i < n4; i += stride) { uint4 v = in[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

__global__ void read_range_rep(const uint4 *__restrict__ in, size_t n4, unsigned *sink, int rp) {
  unsigned acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < rp; p++) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 7 * stride < n4; i += 8 * stride) {
      uint4 v[8];
#pragma unroll
      for (int k = 0; k < 8; k++) v[k] = in[i + k * stride];
#pragma unroll
      for (int k = 0; k < 8; k++) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }
    for (; i < n4; i += stride) { uint4 v = in[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

// ILP variants for the diag sweep
template <int N>
__global__ void read_ilp(const uint4 *__restrict__ in, size_t n4, unsigned *sink) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (; i + (N - 1) * stride < n4; i += (size_t)N * stride) {
    uint4 v[N];
#pragma unroll
    for (int k = 0; k < N; k++) v[k] = in[i + k * stride];
#pragma unroll
    for (int k = 0; k < N; k++) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
  }
  for (; i < n4; i += stride) { uint4 v = in[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
}

__global__ void empty_k() {}
__global__ void touch1(const uint4 *__restrict__ in, unsigned *sink) {
  if (blockIdx.x == 0 && threadIdx.x == 0) sink[0] = in[0].x;
}
__global__ void spin_k(volatile float *p, long iters) {
  float x = 1.0f;
  for (long i = 0; i < iters; i++) x = x * 1.0000001f + (float)(threadIdx.x & 1);
  if (x == -1.0f) *p = x;
}

// ---------------------------------------------------------------- helpers
static int SM = 42, BLK = 256;
static unsigned *g_sink;
static uint4 *g_base;
static size_t g_arena_bytes;

struct Slice { size_t off; size_t bytes; };

static int grid_for(size_t n4) {
  static int gm = getenv("GRIDMULT") ? atoi(getenv("GRIDMULT")) : 0;
  if (gm) {                       // fixed multiple of SM, capped by work
    size_t need = (n4 + BLK - 1) / BLK;
    size_t want = (size_t)SM * gm;
    if (want > need) want = need;
    if (want < 1) want = 1;
    return (int)want;
  }
  size_t need = (n4 + (size_t)BLK * 8 - 1) / ((size_t)BLK * 8);
  if (need < 1) need = 1;
  if (need > (size_t)SM * 8) need = (size_t)SM * 8;
  return (int)need;
}

static void launch_slice(const Slice &s) {
  size_t n4 = s.bytes >> 4;
  int g = grid_for(n4);
  read_range<<<g, BLK>>>((const uint4 *)((const char *)g_base + s.off), n4, g_sink);
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0;
}

static double now_ms() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// hipEvent pool
static const int MAXEV = 4096;
static hipEvent_t g_ev[MAXEV];
static void ev_init() { for (int i = 0; i < MAXEV; i++) CHK(hipEventCreate(&g_ev[i])); }

// Time one group of launches. Returns total ms; fills per-launch ms if wanted.
static double time_group(const std::vector<Slice> &sl, bool per_launch,
                         std::vector<double> *out, int gap_us) {
  int n = (int)sl.size();
  double total = 0;
  if (per_launch) {
    if (2 * n + 1 > MAXEV) { printf("too many launches for event pool\n"); exit(1); }
    CHK(hipEventRecord(g_ev[0]));
    for (int i = 0; i < n; i++) {
      if (gap_us) { double t0 = now_ms(); while (now_ms() - t0 < gap_us / 1000.0) {} }
      CHK(hipEventRecord(g_ev[2 * i + 1]));
      launch_slice(sl[i]);
      CHK(hipEventRecord(g_ev[2 * i + 2]));
    }
    CHK(hipEventSynchronize(g_ev[2 * n]));
    if (out) out->clear();
    for (int i = 0; i < n; i++) {
      float t = 0; CHK(hipEventElapsedTime(&t, g_ev[2 * i + 1], g_ev[2 * i + 2]));
      if (out) out->push_back(t);
    }
    float t = 0; CHK(hipEventElapsedTime(&t, g_ev[0], g_ev[2 * n]));
    total = t;
  } else {
    double h0 = now_ms();
    CHK(hipEventRecord(g_ev[0]));
    for (int i = 0; i < n; i++) {
      if (gap_us) { double t0 = now_ms(); while (now_ms() - t0 < gap_us / 1000.0) {} }
      launch_slice(sl[i]);
    }
    CHK(hipEventRecord(g_ev[1]));
    double h1 = now_ms();
    CHK(hipEventSynchronize(g_ev[1]));
    float t = 0; CHK(hipEventElapsedTime(&t, g_ev[0], g_ev[1]));
    total = t;
    (void)h0; (void)h1;
  }
  return total;
}

static size_t sum_bytes(const std::vector<Slice> &s) {
  size_t b = 0; for (auto &x : s) b += x.bytes; return b;
}

// ---------------------------------------------------------------- modes
static void mode_sweep(double gib, int reps) {
  size_t total = (size_t)(gib * (double)(1ULL << 30));
  int Ns[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  printf("# sweep: total %.3f GiB, reps %d, grid<=SM*8, blk %d\n", gib, reps, BLK);
  printf("# N  slice_MiB  med_GB/s  min  max  us_per_launch  med_ms\n");
  for (int ni = 0; ni < 9; ni++) {
    int N = Ns[ni];
    size_t slice = total / N;
    slice &= ~(size_t)255;                      // 256 B align
    std::vector<Slice> sl(N);
    std::vector<double> gbs, pl;
    // warm-up
    for (int i = 0; i < N; i++) sl[i] = { (size_t)i * slice, slice };
    time_group(sl, false, nullptr, 0);
    double pl_ms = 0;
    for (int r = 0; r < reps + 1; r++) {
      // rotate the window by one slice per rep so no slice is re-read
      size_t start = (size_t)(r % N) * slice;
      for (int i = 0; i < N; i++) {
        size_t off = start + (size_t)i * slice;
        if (off + slice > g_arena_bytes) off = (size_t)i * slice;
        sl[i] = { off, slice };
      }
      std::vector<double> per;
      double ms = time_group(sl, r == 0, &per, 0);
      if (r == 0) {  // discard: measured with per-launch events (costs a little)
        if (!per.empty()) pl_ms = median(per) * 1e3;
        continue;
      }
      gbs.push_back((double)sum_bytes(sl) / (ms * 1e-3) / 1e9);
    }
    std::vector<double> s = gbs; std::sort(s.begin(), s.end());
    double med = median(gbs);
    double ms_med = (double)sum_bytes(sl) / (med * 1e9) * 1e3;
    printf("%3d  %8.2f  %8.1f  %6.1f  %6.1f  %12.1f  %7.3f\n",
           N, slice / 1048576.0, med, s.front(), s.back(), pl_ms, ms_med);
    fflush(stdout);
  }
}

static void mode_null(int N) {
  printf("# null: %d launches\n", N);
  // (a) empty kernel, 1 block
  std::vector<double> t;
  for (int r = 0; r < 6; r++) {
    CHK(hipEventRecord(g_ev[0]));
    for (int i = 0; i < N; i++) empty_k<<<1, 32>>>();
    CHK(hipEventRecord(g_ev[1]));
    CHK(hipEventSynchronize(g_ev[1]));
    float ms = 0; CHK(hipEventElapsedTime(&ms, g_ev[0], g_ev[1]));
    if (r) t.push_back(ms * 1e3 / N);
  }
  printf("NULL empty_k  grid=1x32     us_per_launch=%.2f\n", median(t));

  // (b) empty kernel with the full streaming grid
  t.clear();
  for (int r = 0; r < 6; r++) {
    CHK(hipEventRecord(g_ev[0]));
    for (int i = 0; i < N; i++) empty_k<<<SM * 8, BLK>>>();
    CHK(hipEventRecord(g_ev[1]));
    CHK(hipEventSynchronize(g_ev[1]));
    float ms = 0; CHK(hipEventElapsedTime(&ms, g_ev[0], g_ev[1]));
    if (r) t.push_back(ms * 1e3 / N);
  }
  printf("NULL empty_k  grid=SM*8x%d  us_per_launch=%.2f\n", BLK, median(t));

  // (c) full grid touching one 16 B word: dispatch + drain of a real grid
  t.clear();
  for (int r = 0; r < 6; r++) {
    CHK(hipEventRecord(g_ev[0]));
    for (int i = 0; i < N; i++) touch1<<<SM * 8, BLK>>>(g_base, g_sink);
    CHK(hipEventRecord(g_ev[1]));
    CHK(hipEventSynchronize(g_ev[1]));
    float ms = 0; CHK(hipEventElapsedTime(&ms, g_ev[0], g_ev[1]));
    if (r) t.push_back(ms * 1e3 / N);
  }
  printf("NULL touch1   grid=SM*8x%d  us_per_launch=%.2f\n", BLK, median(t));

  // (d) host-side API cost only (no sync), async queue depth
  double h0 = now_ms();
  for (int i = 0; i < N; i++) empty_k<<<SM * 8, BLK>>>();
  double h1 = now_ms();
  CHK(hipDeviceSynchronize());
  printf("NULL host_api grid=SM*8     us_per_launch=%.2f\n", (h1 - h0) * 1e3 / N);
}

static void mode_orig(int reps) {
  // the original observation: 8 back-to-back 1 GiB launches
  size_t gib = (size_t)1 << 30;
  std::vector<Slice> sl8(8), sl1(1), sl4(4), sl2(2);
  for (int i = 0; i < 8; i++) sl8[i] = { (size_t)i * gib, gib };
  sl1[0] = { 0, 8 * gib };
  for (int i = 0; i < 4; i++) sl4[i] = { (size_t)i * 2 * gib, 2 * gib };
  for (int i = 0; i < 2; i++) sl2[i] = { (size_t)i * 4 * gib, 4 * gib };
  struct C { const char *n; std::vector<Slice> *s; };
  C cs[] = { {"8 x 1 GiB", &sl8}, {"2 x 4 GiB", &sl2}, {"4 x 2 GiB", &sl4}, {"1 x 8 GiB", &sl1} };
  printf("# orig: same 8 GiB, different launch counts\n");
  for (auto &c : cs) {
    time_group(*c.s, false, nullptr, 0);
    std::vector<double> v, per;
    for (int r = 0; r < reps; r++) {
      double ms = time_group(*c.s, false, nullptr, 0);
      v.push_back((double)sum_bytes(*c.s) / (ms * 1e-3) / 1e9);
    }
    // per-launch distribution for the 8 x 1 GiB case
    std::vector<double> p;
    time_group(*c.s, true, &p, 0);
    std::vector<double> ps = p; std::sort(ps.begin(), ps.end());
    printf("ORIG %-10s med_GB/s=%7.1f  min=%7.1f max=%7.1f  per_launch_ms med=%.3f min=%.3f max=%.3f\n",
           c.n, median(v), *std::min_element(v.begin(), v.end()),
           *std::max_element(v.begin(), v.end()),
           median(ps), ps.front(), ps.back());
    fflush(stdout);
  }
}

static void spin_for(int ms) {
  // calibrate iteration count once, then run ~ms
  float *p; CHK(hipMalloc(&p, 4));
  long it = 200000;
  CHK(hipEventRecord(g_ev[0]));
  spin_k<<<SM * 8, BLK>>>(p, it);
  CHK(hipEventRecord(g_ev[1])); CHK(hipEventSynchronize(g_ev[1]));
  float t = 0; CHK(hipEventElapsedTime(&t, g_ev[0], g_ev[1]));
  long need = (long)((double)it * (double)ms / (t > 0.01 ? t : 0.01));
  spin_k<<<SM * 8, BLK>>>(p, need);
  CHK(hipDeviceSynchronize());
  hipFree(p);
}

static void mode_hot(double gib, int N, const char *cond, int reps) {
  size_t total = (size_t)(gib * (double)(1ULL << 30));
  size_t slice = (total / N) & ~(size_t)255;
  std::vector<Slice> sl(N);
  for (int i = 0; i < N; i++) sl[i] = { (size_t)i * slice, slice };
  std::string c(cond);
  int gap = (c == "gaps") ? (getenv("GAP_US") ? atoi(getenv("GAP_US")) : 1000) : 0;
  printf("# hot: N=%d slice=%.2f MiB cond=%s\n", N, slice / 1048576.0, cond);
  time_group(sl, false, nullptr, 0);
  std::vector<double> v, per;
  for (int r = 0; r < reps; r++) {
    if (c == "cold") { usleep(3000 * 1000); }          // let the GPU go idle/suspend
    if (c == "spin" || c == "spin_nosync") { spin_for(2000); }
    if (c == "spin_sync") { spin_for(2000); usleep(200 * 1000); }
    double ms = time_group(sl, false, nullptr, gap);
    v.push_back((double)sum_bytes(sl) / (ms * 1e-3) / 1e9);
  }
  // per-launch detail (no gaps) for diagnosis
  time_group(sl, true, &per, 0);
  std::vector<double> ps = per; std::sort(ps.begin(), ps.end());
  printf("HOT N=%d cond=%-10s med_GB/s=%7.1f  min=%7.1f max=%7.1f  per_launch_ms med=%.3f p10=%.3f max=%.3f\n",
         N, cond, median(v), *std::min_element(v.begin(), v.end()),
         *std::max_element(v.begin(), v.end()),
         median(ps), ps[ps.size() / 10], ps.back());
  fflush(stdout);
}

static std::vector<Slice> load_manifest(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("cannot open %s\n", path); exit(1); }
  std::vector<Slice> v;
  unsigned long long off, nb;
  while (fscanf(f, "%llu %llu", &off, &nb) == 2) v.push_back({ (size_t)off, (size_t)nb });
  fclose(f);
  return v;
}

static void mode_decode(const char *path, const char *grouping, int steps, int reps) {
  std::vector<Slice> seq = load_manifest(path);
  size_t bytes = sum_bytes(seq);
  printf("# decode: %s  launches=%zu  bytes=%.4f GB  steps=%d reps=%d\n",
         grouping, seq.size(), bytes / 1e9, steps, reps);
  // warm-up
  {
    std::vector<double> per;
    for (int s = 0; s < steps; s++) time_group(seq, false, nullptr, 0);
  }
  std::vector<double> step_gbs;
  for (int r = 0; r < reps; r++) {
    double ms = 0;
    for (int s = 0; s < steps; s++) ms += time_group(seq, false, nullptr, 0);
    double per_step_ms = ms / steps;
    step_gbs.push_back(bytes / (per_step_ms * 1e-3) / 1e9);
  }
  // per-launch detail for one step
  std::vector<double> per;
  time_group(seq, true, &per, 0);
  std::vector<double> ps = per; std::sort(ps.begin(), ps.end());
  double first = per.empty() ? 0 : per[0];
  double rest = 0; for (size_t i = 1; i < per.size(); i++) rest += per[i];
  if (per.size() > 1) rest /= (per.size() - 1);
  double med = median(step_gbs);
  std::vector<double> s = step_gbs; std::sort(s.begin(), s.end());
  printf("DECODE %-7s launches=%3zu bytes=%.4fGB steps=%d med_GB/s=%7.1f min=%7.1f max=%7.1f "
         "per_step_ms=%.3f toks=%.1f\n",
         grouping, seq.size(), bytes / 1e9, steps, med, s.front(), s.back(),
         bytes / (med * 1e9) * 1e3, med * 1e9 / 11.76e9);
  printf("DECODE_LAUNCH %-7s n=%zu med_ms=%.4f min_ms=%.4f max_ms=%.4f first_ms=%.4f rest_ms=%.4f\n",
         grouping, per.size(), median(ps), ps.front(), ps.back(), first, rest);
  fflush(stdout);
}

// ---------------------------------------------------------------------
// single-launch bandwidth vs kernel duration: isolates "does the achieved
// rate depend on how long the kernel runs?" from "how many launches?"
static void mode_size(int max_mib, int reps) {
  printf("# size: ONE launch, varying bytes. rate vs kernel duration.\n");
  printf("# MiB  med_GB/s  min  max  med_ms  us_per_launch\n");
  for (int mib = 8; mib <= max_mib; mib *= 2) {
    size_t bytes = (size_t)mib << 20;
    std::vector<Slice> sl(1);
    std::vector<double> v, per;
    sl[0] = { 0, bytes };
    time_group(sl, false, nullptr, 0);
    for (int r = 0; r < reps; r++) {
      size_t off = (size_t)((r * 7919 * 4096) % (g_arena_bytes - bytes));
      off &= ~(size_t)255;
      sl[0] = { off, bytes };
      double ms = time_group(sl, false, nullptr, 0);
      v.push_back((double)bytes / (ms * 1e-3) / 1e9);
    }
    double med = median(v);
    std::vector<double> s = v; std::sort(s.begin(), s.end());
    printf("%5d  %8.1f  %6.1f  %6.1f  %7.3f  %13.1f\n",
           mib, med, s.front(), s.back(), bytes / (med * 1e9) * 1e3, bytes / (med * 1e9) * 1e6);
    fflush(stdout);
  }
}

// N launches of a FIXED slice size: separates launch count from slice size
static void mode_sizesub(int slice_mib, int N, int reps) {
  size_t slice = ((size_t)slice_mib << 20) & ~(size_t)255;
  if ((size_t)N * slice > g_arena_bytes) {
    printf("REFUSE: %d x %d MiB > arena\n", N, slice_mib); return;
  }
  std::vector<Slice> sl(N);
  printf("# sizesub: %d launches of %.1f MiB (total %.2f GiB)\n",
         N, slice_mib, N * (double)slice / 1e9);
  for (int i = 0; i < N; i++) sl[i] = { (size_t)i * slice, slice };
  time_group(sl, false, nullptr, 0);
  std::vector<double> v, per;
  for (int r = 0; r < reps; r++) {
    size_t start = (size_t)(r % N) * slice;
    for (int i = 0; i < N; i++) {
      size_t off = start + (size_t)i * slice;
      if (off + slice > g_arena_bytes) off = (size_t)i * slice;
      sl[i] = { off, slice };
    }
    double ms = time_group(sl, false, nullptr, 0);
    v.push_back((double)sum_bytes(sl) / (ms * 1e-3) / 1e9);
  }
  time_group(sl, true, &per, 0);
  std::vector<double> ps = per; std::sort(ps.begin(), ps.end());
  printf("SIZESUB slice=%4d MiB N=%3d med_GB/s=%7.1f min=%7.1f max=%7.1f  "
         "per_launch_ms med=%.4f min=%.4f max=%.4f\n",
         slice_mib, N, median(v), *std::min_element(v.begin(), v.end()),
         *std::max_element(v.begin(), v.end()), median(ps), ps.front(), ps.back());
  fflush(stdout);
}

// N launches all reading the SAME slice: if the per-launch spread collapses,
// the spread is address-dependent, not time-dependent.
static void mode_same(int slice_mib, int N, int reps) {
  size_t slice = ((size_t)slice_mib << 20) & ~(size_t)255;
  std::vector<Slice> sl(N);
  for (int i = 0; i < N; i++) sl[i] = { 0, slice };
  printf("# same: %d launches, all reading the same %.1f MiB slice\n", N, slice_mib);
  time_group(sl, false, nullptr, 0);
  std::vector<double> v, per;
  for (int r = 0; r < reps; r++) {
    double ms = time_group(sl, false, nullptr, 0);
    v.push_back((double)sum_bytes(sl) / (ms * 1e-3) / 1e9);
  }
  time_group(sl, true, &per, 0);
  std::vector<double> ps = per; std::sort(ps.begin(), ps.end());
  printf("SAME slice=%4d MiB N=%3d med_GB/s=%7.1f  per_launch_ms med=%.4f min=%.4f max=%.4f "
         "first=%.4f\n",
         slice_mib, N, median(v), median(ps), ps.front(), ps.back(), per.empty() ? 0 : per[0]);
  fflush(stdout);
}

// one launch, S bytes, k in-kernel passes over the same S bytes
static void mode_rp(int slice_mib, int passes, int reps) {
  size_t slice = ((size_t)slice_mib << 20) & ~(size_t)255;
  size_t n4 = slice >> 4;
  int g = grid_for(n4);
  printf("# rp: one launch, %.1f MiB x %d passes = %.2f GiB read\n",
         (double)slice_mib, passes, (double)slice * passes / 1e9);
  std::vector<double> v;
  read_range_rep<<<g, BLK>>>((const uint4 *)g_base, n4, g_sink, passes);
  CHK(hipDeviceSynchronize());
  for (int r = 0; r < reps; r++) {
    CHK(hipEventRecord(g_ev[0]));
    read_range_rep<<<g, BLK>>>((const uint4 *)g_base, n4, g_sink, passes);
    CHK(hipEventRecord(g_ev[1]));
    CHK(hipEventSynchronize(g_ev[1]));
    float ms = 0; CHK(hipEventElapsedTime(&ms, g_ev[0], g_ev[1]));
    v.push_back((double)(slice * passes) / (ms * 1e-3) / 1e9);
  }
  std::vector<double> s = v; std::sort(s.begin(), s.end());
  printf("RP slice=%4d MiB passes=%2d grid=%3d med_GB/s=%7.1f min=%7.1f max=%7.1f "
         "kernel_ms=%.3f\n",
         slice_mib, passes, g, median(v), s.front(), s.back(),
         (double)(slice * passes) / (median(v) * 1e9) * 1e3);
  fflush(stdout);
}

// arbitrary list of single-launch sizes, to locate the 1 GiB dip precisely
static void mode_sizelist(const char *csv, int reps) {
  printf("# sizelist: ONE launch per point\n");
  char buf[512]; snprintf(buf, sizeof buf, "%s", csv);
  char *save = nullptr;
  for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(nullptr, ",", &save)) {
    int mib = atoi(tok);
    size_t bytes = (size_t)mib << 20;
    if (bytes + 65536 > g_arena_bytes) { printf("%5d  (too big)\n", mib); continue; }
    std::vector<Slice> sl(1);
    std::vector<double> v;
    sl[0] = { 0, bytes };
    time_group(sl, false, nullptr, 0);
    for (int r = 0; r < reps; r++) {
      size_t off = ((size_t)((r * 7919 * 4096) % (g_arena_bytes - bytes))) & ~(size_t)255;
      sl[0] = { off, bytes };
      double ms = time_group(sl, false, nullptr, 0);
      v.push_back((double)bytes / (ms * 1e-3) / 1e9);
    }
    std::vector<double> s = v; std::sort(s.begin(), s.end());
    printf("SIZELIST mib=%5d med_GB/s=%7.1f min=%7.1f max=%7.1f ms=%.3f\n",
           mib, median(v), s.front(), s.back(), bytes / (median(v) * 1e9) * 1e3);
    fflush(stdout);
  }
}

// grid-size / ILP sensitivity at one size: is the dip an access-pattern artefact?
static void mode_diag(int mib, int reps) {
  size_t bytes = (size_t)mib << 20;
  size_t n4 = bytes >> 4;
  printf("# diag: one launch, %d MiB, grid x ILP sweep\n", mib);
  int gms[] = {1, 2, 4, 8, 16, 32};
  for (int gi = 0; gi < 6; gi++) {
    int g = SM * gms[gi];
    if ((size_t)g * BLK > n4) continue;
    for (int ilp : {4, 8, 16}) {
      std::vector<double> v;
      for (int r = 0; r < reps + 1; r++) {
        CHK(hipEventRecord(g_ev[0]));
        if (ilp == 4) read_ilp<4><<<g, BLK>>>((const uint4 *)g_base, n4, g_sink);
        else if (ilp == 8) read_ilp<8><<<g, BLK>>>((const uint4 *)g_base, n4, g_sink);
        else read_ilp<16><<<g, BLK>>>((const uint4 *)g_base, n4, g_sink);
        CHK(hipEventRecord(g_ev[1]));
        CHK(hipEventSynchronize(g_ev[1]));
        float ms = 0; CHK(hipEventElapsedTime(&ms, g_ev[0], g_ev[1]));
        if (r) v.push_back((double)bytes / (ms * 1e-3) / 1e9);
      }
      std::vector<double> s = v; std::sort(s.begin(), s.end());
      printf("DIAG mib=%4d grid=SM*%2d ilp=%2d med_GB/s=%7.1f min=%7.1f max=%7.1f\n",
             mib, gms[gi], ilp, median(v), s.front(), s.back());
      fflush(stdout);
    }
  }
}


// sustained loop over a manifest for <seconds>: lets the host sample clocks
// while launches are actually in flight
static void mode_loop(const char *path, double seconds) {
  std::vector<Slice> seq = load_manifest(path);
  size_t bytes = sum_bytes(seq);
  printf("# loop: %s launches=%zu bytes=%.4f GB for %.1f s\n",
         path, seq.size(), bytes / 1e9, seconds);
  time_group(seq, false, nullptr, 0);
  double t0 = now_ms();
  int passes = 0;
  std::vector<double> per_pass;
  while (now_ms() - t0 < seconds * 1e3) {
    double ms = time_group(seq, false, nullptr, 0);
    per_pass.push_back(ms);
    passes++;
  }
  double wall = (now_ms() - t0) / 1e3;
  double gbs = (double)passes * bytes / wall / 1e9;
  std::vector<double> s = per_pass; std::sort(s.begin(), s.end());
  printf("LOOP %-14s passes=%d wall=%.2fs med_GB/s=%7.1f min=%7.1f max=%7.1f "
         "med_pass_ms=%.3f\n",
         path, passes, wall, gbs,
         (double)bytes / (s.back() * 1e-3) / 1e9,
         (double)bytes / (s.front() * 1e-3) / 1e9, median(s));
  fflush(stdout);
}


// per-launch recovery profile: after an idle period / spin kernel, print the
// GB/s of each successive launch so the warm-up shape is visible
static void mode_wake(int mib, int N, const char *cond) {
  size_t slice = ((size_t)mib << 20) & ~(size_t)255;
  if ((size_t)N * slice > g_arena_bytes) {
    printf("REFUSE: %d x %d MiB = %.2f GiB > arena %.2f GiB\n", N, mib,
           (double)N * slice / 1e9, (double)g_arena_bytes / 1e9);
    return;
  }
  std::vector<Slice> sl(N);
  for (int i = 0; i < N; i++) sl[i] = { (size_t)i * slice, slice };
  std::string c(cond);
  printf("# wake: %d x %d MiB after cond=%s (per-launch GB/s)\n", N, mib, cond);
  // make sure the code path is resident, then let the GPU settle
  time_group(sl, false, nullptr, 0);
  if (c != "nosleep") usleep(3000 * 1000);
  if (c == "spin") spin_for(2000);
  if (c == "spin200") spin_for(200);
  if (c == "spin_mem") {   // spin that also streams memory, no launch gap
    for (int i = 0; i < 20; i++) time_group(sl, false, nullptr, 0);
  }
  std::vector<double> per;
  double ms = time_group(sl, true, &per, 0);
  printf("WAKE cond=%-6s mib=%d group_ms=%.3f group_GB/s=%.1f\n",
         cond, mib, ms, (double)sum_bytes(sl) / (ms * 1e-3) / 1e9);
  for (int i = 0; i < (int)per.size(); i++)
    printf("  launch %2d  %8.4f ms  %7.1f GB/s\n", i, per[i], (double)slice / (per[i] * 1e-3) / 1e9);
  fflush(stdout);
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
  if (argc < 2) { printf("usage: see header\n"); return 1; }
  std::string mode = argv[1];

  int dev = 0;
  hipDeviceProp_t p; CHK(hipGetDeviceProperties(&p, dev));
  SM = p.multiProcessorCount;
  printf("# device %s  pci %d  CUs(SM units) %d  sclk %d MHz  mclk %d MHz\n",
         p.name, p.pciBusID, SM, p.clockRate / 1000, p.memoryClockRate / 1000);

  size_t arena;
  if (mode == "decode" || mode == "loop") arena = (size_t)(11.85e9);
  else if (mode == "orig") arena = (size_t)8.6e9;
  else if (mode == "sweep" || mode == "hot") arena = (size_t)(4.2e9) + (size_t)(1ULL << 30);
  else if (mode == "wake" || mode == "sizesub") arena = (size_t)(9.6e9);
  else if (mode == "size" || mode == "sizelist" || mode == "diag") arena = (size_t)(9.0e9);
  else arena = (size_t)(1ULL << 30);

  CHK(hipMalloc(&g_base, arena));
  g_arena_bytes = arena;
  CHK(hipMalloc(&g_sink, 8192));
  CHK(hipMemset(g_base, 1, arena));
  CHK(hipDeviceSynchronize());
  ev_init();

  if (mode == "sweep") {
    double gib = (argc > 2) ? atof(argv[2]) : 4.0;
    int reps = (argc > 3) ? atoi(argv[3]) : 5;
    mode_sweep(gib, reps);
  } else if (mode == "null") {
    mode_null((argc > 2) ? atoi(argv[2]) : 1000);
  } else if (mode == "orig") {
    mode_orig((argc > 2) ? atoi(argv[2]) : 7);
  } else if (mode == "hot") {
    double gib = (argc > 2) ? atof(argv[2]) : 4.0;
    int N = (argc > 3) ? atoi(argv[3]) : 8;
    const char *cond = (argc > 4) ? argv[4] : "cold";
    int reps = (argc > 5) ? atoi(argv[5]) : 6;
    mode_hot(gib, N, cond, reps);
  } else if (mode == "decode") {
    const char *path = (argc > 2) ? argv[2] : "seq_tensor.txt";
    const char *grp = (argc > 3) ? argv[3] : "tensor";
    int steps = (argc > 4) ? atoi(argv[4]) : 1;
    int reps = (argc > 5) ? atoi(argv[5]) : 5;
    mode_decode(path, grp, steps, reps);
  } else if (mode == "size") {
    mode_size((argc > 2) ? atoi(argv[2]) : 8192, (argc > 3) ? atoi(argv[3]) : 7);
  } else if (mode == "sizesub") {
    mode_sizesub((argc > 2) ? atoi(argv[2]) : 512, (argc > 3) ? atoi(argv[3]) : 8,
                 (argc > 4) ? atoi(argv[4]) : 5);
  } else if (mode == "same") {
    mode_same((argc > 2) ? atoi(argv[2]) : 1024, (argc > 3) ? atoi(argv[3]) : 8,
              (argc > 4) ? atoi(argv[4]) : 5);
  } else if (mode == "rp") {
    mode_rp((argc > 2) ? atoi(argv[2]) : 1024, (argc > 3) ? atoi(argv[3]) : 4,
            (argc > 4) ? atoi(argv[4]) : 7);
  } else if (mode == "wake") {
    mode_wake((argc > 2) ? atoi(argv[2]) : 512, (argc > 3) ? atoi(argv[3]) : 12,
              (argc > 4) ? argv[4] : "cold");
  } else if (mode == "loop") {
    mode_loop((argc > 2) ? argv[2] : "seq_tensor.txt", (argc > 3) ? atof(argv[3]) : 3.0);
  } else if (mode == "sizelist") {
    mode_sizelist((argc > 2) ? argv[2] : "512,1024,2048", (argc > 3) ? atoi(argv[3]) : 5);
  } else if (mode == "diag") {
    mode_diag((argc > 2) ? atoi(argv[2]) : 1024, (argc > 3) ? atoi(argv[3]) : 5);
  } else { printf("unknown mode\n"); return 1; }
  CHK(hipDeviceSynchronize());
  return 0;
}
