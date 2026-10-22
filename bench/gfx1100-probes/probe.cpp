// gfx1100 (RDNA3, RX 7900 XT) dot-product throughput probe
//   measures: v_dot4_i32_i8, v_dot8_i32_i4, v_dot2_f32_f16, v_fma_f32
//   plus a per-intrinsic correctness check against a scalar C reference.
//
// Rationale for inline asm: __builtin_amdgcn_sdot4 / __builtin_amdgcn_sudot8 do NOT
// compile for gfx1100 on this toolchain (ROCm 7.16 / clang 24): the front end gates
// them on target features ("dot1-insts") that gfx1100 no longer advertises, while the
// MC layer assembles both mnemonics for gfx1100 without complaint. See RESULTS.md.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>

#define BLKMAX 1024
static int BLK = 256;
#define NACC  16
#define REPS  5

// ---------------------------------------------------------------- intrinsics
template <int OPID>
__device__ __forceinline__ int apply(int acc, int x, int k) {
  if constexpr (OPID == 0) {              // 4 x int8 MAC -> int32
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "v"(k));
    return acc;
  } else if constexpr (OPID == 1) {       // 8 x int4 MAC -> int32
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "v"(k));
    return acc;
  } else if constexpr (OPID == 2) {       // 2 x f16 MAC -> f32
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "v"(k));
    return acc;
  } else {                                // 1 x f32 MAC (scalar, fused)
    asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "v"(k));
    return acc;
  }
}

// ---------------------------------------------------------------- throughput
// One dependency chain per accumulator; the varying operand is the loop counter
// (one v_mov from an SGPR per iteration, i.e. 1 extra VALU op per NACC dots).
template <int OPID>
__global__ void bench_kernel(unsigned iters, int *sink,
                                                    unsigned seed) {
  int acc[NACC];
  int ks[NACC];
  unsigned t = seed + (unsigned)threadIdx.x * 0x9e3779b9u + (unsigned)blockIdx.x;
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    acc[i] = (int)(t * (unsigned)(i + 1) | 1u);
    // keep f16 operands finite/benign: 1.0h, 0.5h, ...
    ks[i]  = (OPID == 2) ? (int)(0x38003800u + (unsigned)i)
                         : (int)(0x7f7f7f7fu ^ (unsigned)(i * 7 + 3));
  }
  unsigned i = 0;
  if constexpr (OPID >= 4) {
    while (i < iters) {
#pragma unroll
      for (int k = 0; k < NACC; k++)
        acc[k] = apply<OPID - 4>(acc[k], ks[k], ks[(k + 1) % NACC]);
      ++i;
    }
  } else {
    while (i < iters) {
      int x = (int)i;
#pragma unroll
      for (int k = 0; k < NACC; k++) acc[k] = apply<OPID>(acc[k], x, ks[k]);
      ++i;
    }
  }
  int s = 0;
#pragma unroll
  for (int k = 0; k < NACC; k++) s ^= acc[k];
  sink[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;   // keeps everything alive
}

// ---------------------------------------------------------------- correctness
// int ops: dot4 / dot8. floated ops: fma (f32 x f32 + f32), dot2f16 (2xf16 + f32).
__global__ void check_int_kernel(int *out, const uint32_t *a, const uint32_t *b,
                                 const uint32_t *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int r4, r8;
  asm("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r4) : "v"((int)a[i]), "v"((int)b[i]), "v"((int)c[i]));
  asm("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(r8) : "v"((int)a[i]), "v"((int)b[i]), "v"((int)c[i]));
  out[2 * i + 0] = r4;
  out[2 * i + 1] = r8;
}
__global__ void check_flt_kernel(uint32_t *out, const uint32_t *a, const uint32_t *b,
                                 const uint32_t *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned rf, rd;
  asm("v_fma_f32 %0, %1, %2, %3" : "=v"(rf) : "v"((int)a[i]), "v"((int)b[i]), "v"((int)c[i]));
  asm("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(rd) : "v"((int)a[i]), "v"((int)b[i]), "v"((int)c[i]));
  out[2 * i + 0] = rf;
  out[2 * i + 1] = rd;
}

static inline int s8(uint32_t v) { return (int)((int8_t)(v & 0xff)); }
static inline int s4(uint32_t v) { int t = (int)(v & 0xf); return (t & 8) ? t - 16 : t; }
static inline float as_f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t as_u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline int ulp_diff(float x, float y) {
  if (x == y) return 0;                 // covers +0.0 vs -0.0
  uint32_t ux = as_u(x), uy = as_u(y);
  if ((ux >> 31) != (uy >> 31)) return (ux == uy) ? 0 : 0x7fffffff;
  return std::abs((int)(ux & 0x7fffffffu) - (int)(uy & 0x7fffffffu));
}
static uint32_t pk_h2(float lo, float hi) {
  _Float16 l = (_Float16)lo, h = (_Float16)hi;
  uint32_t a = 0, b = 0;              // must be zeroed: memcpy fills only 2 bytes
  memcpy(&a, &l, 2); memcpy(&b, &h, 2);
  return (a & 0xffffu) | (b << 16);
}

static bool run_correctness(bool *dot2_exact) {
  hipError_t e;
  // ---- integer ops
  std::vector<uint32_t> A, B, C;
  const uint32_t edge[][3] = {
      {0x80808080u, 0x80808080u, 0u},           {0x7f7f7f7fu, 0x7f7f7f7fu, 0u},
      {0x80808080u, 0x7f7f7f7fu, 0u},           {0x00000000u, 0x80808080u, 0u},
      {0x7f7f7f7fu, 0x00000000u, 0x12345678u},  {0xffffffffu, 0xffffffffu, 0u},
      {0x80807f7fu, 0x7f7f8080u, 0xdeadbeefu},  {0x00ff0000u, 0xff00ffffu, 0x7fffffffu},
      {0x7f7f7f7fu, 0x7f7f7f7fu, 0x7fffffffu},  {0x80808080u, 0x80808080u, 0x80000000u},
  };
  for (auto &x : edge) { A.push_back(x[0]); B.push_back(x[1]); C.push_back(x[2]); }
  uint32_t s = 12345u;
  for (int i = 0; i < 8192; i++) {
    s = s * 1664525u + 1013904223u; A.push_back(s);
    s = s * 1664525u + 1013904223u; B.push_back(s);
    s = s * 1664525u + 1013904223u; C.push_back(s);
  }
  int n = (int)A.size();
  int *d_oi; uint32_t *d_a, *d_b, *d_c;
  e = hipMalloc(&d_oi, 2 * n * sizeof(int)); 
  e = hipMalloc(&d_a, n * 4); e = hipMalloc(&d_b, n * 4); e = hipMalloc(&d_c, n * 4);
  e = hipMemcpy(d_a, A.data(), n * 4, hipMemcpyHostToDevice);
  e = hipMemcpy(d_b, B.data(), n * 4, hipMemcpyHostToDevice);
  e = hipMemcpy(d_c, C.data(), n * 4, hipMemcpyHostToDevice);
  check_int_kernel<<<(n + 255) / 256, 256>>>(d_oi, d_a, d_b, d_c, n);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("int check kernel failed: %s\n", hipGetErrorString(e)); return false; }
  std::vector<int> hi_(2 * n);
  e = hipMemcpy(hi_.data(), d_oi, 2 * n * sizeof(int), hipMemcpyDeviceToHost);

  int bad4 = 0, bad8 = 0;
  for (int i = 0; i < n; i++) {
    uint32_t r4 = C[i], r8 = C[i];
    for (int j = 0; j < 4; j++) r4 += (uint32_t)(s8(A[i] >> (8 * j)) * s8(B[i] >> (8 * j)));
    for (int j = 0; j < 8; j++) r8 += (uint32_t)(s4(A[i] >> (4 * j)) * s4(B[i] >> (4 * j)));
    if ((int)r4 != hi_[2 * i + 0]) { if (bad4 < 3) printf("  dot4 mismatch: a=%08x b=%08x acc=%08x got %d want %d\n", A[i], B[i], C[i], hi_[2*i], (int)r4); bad4++; }
    if ((int)r8 != hi_[2 * i + 1]) { if (bad8 < 3) printf("  dot8 mismatch: a=%08x b=%08x acc=%08x got %d want %d\n", A[i], B[i], C[i], hi_[2*i+1], (int)r8); bad8++; }
  }
  printf("int check: %d cases, incl. edge bytes -128/127/0, acc overflow wrap\n", n);
  printf("  v_dot4_i32_i8  : %s  (%d/%d mismatches)\n", bad4 ? "FAIL" : "PASS (bit-exact)", bad4, n);
  printf("  v_dot8_i32_i4  : %s  (%d/%d mismatches)\n", bad8 ? "FAIL" : "PASS (bit-exact)", bad8, n);
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_oi);

  // ---- float ops: fma on f32; dot2 on half2 (integer-valued, exact in fp16)
  std::vector<uint32_t> fa, fb, fc;
  float fe[][3] = {{0.f, 0.f, 0.f}, {-128.f, 127.f, 1.f}, {127.f, -128.f, -7.f},
                   {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.5f}, {1024.f, 0.001f, -3.5f}};
  for (auto &x : fe) { fa.push_back(as_u(x[0])); fb.push_back(as_u(x[1])); fc.push_back(as_u(x[2])); }
  int nf = (int)fa.size();
  for (int i = 0; i < 8192; i++) {
    s = s * 1664525u + 1013904223u; float x = (float)((int)(s % 20001) - 10000) / 100.0f;
    s = s * 1664525u + 1013904223u; float y = (float)((int)(s % 20001) - 10000) / 100.0f;
    s = s * 1664525u + 1013904223u; float z = (float)((int)(s % 20001) - 10000) / 100.0f;
    fa.push_back(as_u(x)); fb.push_back(as_u(y)); fc.push_back(as_u(z)); nf++;
  }
  // half2 data: integers in [-16,15] -> every product is an integer <= 256,
  // exactly representable, so an "exact" reference is meaningful for the sum.
  std::vector<uint32_t> ha(fa.begin(), fa.begin() + nf), hb(fb.begin(), fb.begin() + nf),
      hc(fc.begin(), fc.begin() + nf);
  for (int i = 0; i < nf; i++) {
    s = s * 1664525u + 1013904223u; float p = (float)((int)(s % 32) - 16), q = (float)((int)((s >> 8) % 32) - 16);
    s = s * 1664525u + 1013904223u; float r = (float)((int)(s % 32) - 16), w = (float)((int)((s >> 8) % 32) - 16);
    ha[i] = pk_h2(p, q); hb[i] = pk_h2(r, w);
    s = s * 1664525u + 1013904223u;
    hc[i] = as_u((float)((int)(s % 9) - 4));
  }
  uint32_t *d_fa, *d_fb, *d_fc, *d_fo;
  e = hipMalloc(&d_fa, nf * 4); e = hipMalloc(&d_fb, nf * 4); e = hipMalloc(&d_fc, nf * 4);
  e = hipMalloc(&d_fo, 2 * nf * 4);
  e = hipMemcpy(d_fa, fa.data(), nf * 4, hipMemcpyHostToDevice);
  e = hipMemcpy(d_fb, fb.data(), nf * 4, hipMemcpyHostToDevice);
  e = hipMemcpy(d_fc, fc.data(), nf * 4, hipMemcpyHostToDevice);
  check_flt_kernel<<<(nf + 255) / 256, 256>>>(d_fo, d_fa, d_fb, d_fc, nf);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("float check kernel failed: %s\n", hipGetErrorString(e)); return false; }
  std::vector<uint32_t> hfo(2 * nf);
  e = hipMemcpy(hfo.data(), d_fo, 2 * nf * 4, hipMemcpyDeviceToHost);
  std::vector<uint32_t> hfo2(2 * nf);
  {
    uint32_t *d_ha, *d_hb, *d_hc, *d_ho2;
    e = hipMalloc(&d_ha, nf * 4); e = hipMalloc(&d_hb, nf * 4); e = hipMalloc(&d_hc, nf * 4);
    e = hipMalloc(&d_ho2, 2 * nf * 4);
    e = hipMemcpy(d_ha, ha.data(), nf * 4, hipMemcpyHostToDevice);
    e = hipMemcpy(d_hb, hb.data(), nf * 4, hipMemcpyHostToDevice);
    e = hipMemcpy(d_hc, hc.data(), nf * 4, hipMemcpyHostToDevice);
    check_flt_kernel<<<(nf + 255) / 256, 256>>>(d_ho2, d_ha, d_hb, d_hc, nf);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("half check kernel failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpy(hfo2.data(), d_ho2, 2 * nf * 4, hipMemcpyDeviceToHost);
    hipFree(d_ha); hipFree(d_hb); hipFree(d_hc); hipFree(d_ho2);
  }

  int badf = 0; double maxulp_f = 0;
  for (int i = 0; i < nf; i++) {
    float want = fmaf(as_f(fa[i]), as_f(fb[i]), as_f(fc[i]));   // correctly-rounded fma
    float got = as_f(hfo[2 * i + 0]);
    int d = ulp_diff(got, want); if (d) badf++;
    maxulp_f = std::max(maxulp_f, (double)d);
  }
  // dot2f16: exact reference (integer arithmetic, exact in f32)
  int bad2 = 0, n2 = 0, n2_exact = 0; double maxerr_2 = 0, maxtol_2 = 0;
  for (int i = 0; i < nf; i++) {
    _Float16 a0, a1, b0, b1;
    uint32_t x = ha[i], y = hb[i];
    memcpy(&a0, &x, 2); memcpy(&a1, ((char *)&x) + 2, 2);
    memcpy(&b0, &y, 2); memcpy(&b1, ((char *)&y) + 2, 2);
    if (!std::isfinite((float)a0) || !std::isfinite((float)a1) ||
        !std::isfinite((float)b0) || !std::isfinite((float)b1)) continue;
    double p0 = (double)(float)a0 * (double)(float)b0, p1 = (double)(float)a1 * (double)(float)b1;
    float want = (float)((double)as_f(hc[i]) + p0 + p1);
    float got = as_f(hfo2[2 * i + 1]);
    // error bound: this instruction is NOT bit-exact (last-bit noise), so bound it
    double mag = fabs((double)as_f(hc[i])) + fabs(p0) + fabs(p1);
    double tol = 8.0 * 1.1920929e-7 * (mag > 1.0 ? mag : 1.0);
    double err = fabs((double)got - (double)want);
    if (got == want) n2_exact++;
    if (err > tol) { if (bad2 < 3) printf("  dot2 out of bound: got %.9g want %.9g err %.3g tol %.3g\n", got, want, err, tol); bad2++; }
    maxerr_2 = std::max(maxerr_2, err); maxtol_2 = std::max(maxtol_2, tol); n2++;
  }
  *dot2_exact = (bad2 == 0);
  printf("float check: %d f32 cases (incl. -128/127/0), %d half2 cases (integer-valued)\n", nf, n2);
  printf("  v_fma_f32      : %s  (%d/%d mismatches vs correctly-rounded fmaf, max %g ULP)\n",
         badf ? "FAIL" : "PASS (bit-exact)", badf, nf, maxulp_f);
  printf("  v_dot2_f32_f16 : %s  (bit-exact on %d/%d cases; %d outside the 8-eps bound; "
         "worst abs err %.3g, max bound %.3g)\n",
         bad2 ? "FAIL" : "PASS (within 8-eps bound, NOT bit-exact)", n2_exact, n2, bad2,
         maxerr_2, maxtol_2);
  hipFree(d_fa); hipFree(d_fb); hipFree(d_fc); hipFree(d_fo);
  return !(bad4 || bad8 || badf || bad2);
}

// ---------------------------------------------------------------- clock
static double read_sclk_mhz() {
  FILE *p = popen("amd-smi metric --clock 2>/dev/null | awk '/GFX_0/{f=1} f&&/CLK:/{print $2; exit}'", "r");
  if (!p) return -1;
  double v = -1; if (fscanf(p, "%lf", &v) != 1) v = -1; pclose(p);
  return v;
}

static std::mutex g_mu;
static std::vector<double> g_clk;
static std::atomic<bool> g_stop{false};
static void clk_sampler() {
  while (!g_stop.load()) {
    double v = read_sclk_mhz();
    if (v > 0) { std::lock_guard<std::mutex> l(g_mu); g_clk.push_back(v); }
    usleep(120000);
  }
}
static double clk_load_median(int *nsamp, double *lo, double *hi) {
  std::vector<double> v;
  { std::lock_guard<std::mutex> l(g_mu); v = g_clk; }
  std::vector<double> busy;                      // drop idle-DPM samples (< 800 MHz)
  for (double x : v) if (x > 800.0) busy.push_back(x);
  *nsamp = (int)busy.size();
  if (busy.empty()) { *lo = *hi = 0; return 0; }
  std::sort(busy.begin(), busy.end());
  *lo = busy.front(); *hi = busy.back();
  return busy[busy.size() / 2];
}

struct Result {
  const char *name, *mnemonic;
  int macs_per_instr;
  double median_ms, min_ms, max_ms, gops, instr_per_s, clk_mhz;
};

template <int OPID>
static void bench_one(const char *name, const char *mnem, int macs, unsigned iters,
                      int grid, double clock_mhz, std::vector<Result> &out) {
  int *d_sink;
  hipMalloc(&d_sink, (size_t)grid * BLK * sizeof(int));
  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  std::vector<double> ms;
  for (int r = 0; r < REPS + 3; r++) {      // 3 untimed warmup launches
    hipEventRecord(e0);
    hipEventRecord(e0);
    bench_kernel<OPID><<<grid, BLK>>>(iters, d_sink, 0x1234u + r);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float t = 0; hipEventElapsedTime(&t, e0, e1);
    if (r >= 3) ms.push_back((double)t);
  }
  std::sort(ms.begin(), ms.end());
  double med = ms[ms.size() / 2];
  double lanes = (double)grid * BLK;
  double total_ops = (double)iters * NACC * lanes;           // MAC-equivalents
  double total_instr = (double)iters * (NACC + (OPID >= 4 ? 0 : 1)) * lanes;  // +1 v_mov
  double gops = total_ops / (med * 1e-3) / 1e9;
  double ips = total_instr / (med * 1e-3) / 1e9;
  out.push_back({name, mnem, macs, med, ms.front(), ms.back(), gops, ips, clock_mhz});
  printf("  %-16s %-9s mac/ins=%d  med=%.4f ms [%.4f..%.4f]  %.1f Gmac/s  %.1f Ginstr/s\n",
         name, mnem, macs, med, ms.front(), ms.back(), gops, ips);
  hipFree(d_sink);
  hipEventDestroy(e0); hipEventDestroy(e1);
}

int main(int argc, char **argv) {
  unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], nullptr, 0) : (1u << 24);
  if (argc > 2) BLK = (int)strtoul(argv[2], nullptr, 0);
  int bpc = (argc > 3) ? (int)strtoul(argv[3], nullptr, 0) : 1;
  int dev = 0;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { printf("no device\n"); return 1; }
  // HIP reports 42 here: on RDNA3 hipDeviceProp_t.multiProcessorCount is the WGP
  // count, and a WGP is 2 CUs. Ground truth from amd-smi/rocminfo is 84 CUs.
  // hipDeviceProp reports 42 on this card; amd-smi/rocminfo report 84 "CUs" and
  // 84 x 64 lanes == 42 x 128 lanes == 5376 stream processors, which is the RX 7900 XT
  // spec number. So the machine has 5376 fp32 lanes total, NOT 84 x 128 = 10752.
  int ncu = prop.multiProcessorCount;          // 42 units of 128 lanes
  const int LANES_PER_CU = 128;
  double total_lanes = (double)ncu * LANES_PER_CU;   // 5376
  int grid = (bpc < 0) ? -bpc : 2 * ncu * bpc;  // 84 blocks at bpc=1
  printf("device : %s\n", prop.name);
  printf("CU     : %d units x %d lanes = %.0f fp32 lanes (7900 XT spec: 5376 SP)\n",
         ncu, LANES_PER_CU, total_lanes);
  printf("clock  : max level %d MHz\n", prop.clockRate / 1000);
  printf("iters  : %u per lane (~2^%.2f), %d blocks x %d threads = %d waves/SIMD, %d accumulators\n",
         iters, log2((double)iters), grid, BLK, (long)grid * BLK / 32 / (ncu * 4), NACC);

  bool dot2_exact = false;
  bool ok = run_correctness(&dot2_exact);
  if (!ok) { printf("ABORT: correctness failed, no throughput reported\n"); return 2; }

  std::thread sampler(clk_sampler);
  { // ~2 s warmup at full occupancy so the DPM state is stable before timing
    int *d_sink; hipMalloc(&d_sink, (size_t)grid * BLK * sizeof(int));
    for (int w = 0; w < 6; w++) bench_kernel<3><<<grid, BLK>>>(iters, d_sink, 7u);
    hipDeviceSynchronize(); hipFree(d_sink);
  }
  double clk0 = 0;
  printf("throughput (median of %d timed runs):\n", REPS);

  std::vector<Result> res;
  bench_one<0>("v_dot4_i32_i8", "sdot4", 4, iters, grid, clk0, res);
  bench_one<1>("v_dot8_i32_i4", "sudot8", 8, iters, grid, clk0, res);
  bench_one<2>("v_dot2_f32_f16", "fdot2", 2, iters, grid, clk0, res);
  bench_one<3>("v_fma_f32", "fma", 1, iters, grid, clk0, res);
  printf("  --- pure loop bodies (no v_mov of the loop counter; 16 dots + scalar branch) ---\n");
  bench_one<4>("v_dot4_i32_i8", "sdot4*pure", 4, iters, grid, clk0, res);
  bench_one<5>("v_dot8_i32_i4", "sudot8*pure", 8, iters, grid, clk0, res);
  bench_one<6>("v_dot2_f32_f16", "fdot2*pure", 2, iters, grid, clk0, res);
  bench_one<7>("v_fma_f32", "fma*pure", 1, iters, grid, clk0, res);
  g_stop = true; sampler.join();
  int nsamp = 0; double clo = 0, chi = 0;
  double clk_load = clk_load_median(&nsamp, &clo, &chi);
  printf("sclk under load : median %.0f MHz [%.0f..%.0f] over %d samples\n",
         clk_load, clo, chi, nsamp);
  if (clk_load <= 0) clk_load = (double)prop.clockRate / 1000;
  for (auto &r : res) r.clk_mhz = clk_load;

  FILE *j = fopen("results.json", "w");
  fprintf(j, "{\n  \"device\": \"%s\",\n  \"cu\": %d,\n  \"hip_multiProcessorCount\": %d,\n"
             "  \"blocks_per_cu\": %d,\n  \"threads_per_block\": %d,\n"
             "  \"clock_max_level_mhz\": %d,\n  \"iters_per_lane\": %u,\n"
             "  \"accumulators\": %d,\n  \"reps\": %d,\n"
             "  \"sclk_under_load_mhz\": %.0f,\n  \"sclk_range_mhz\": [%.0f, %.0f],\n"
             "  \"dot2_f32_f16_bit_exact\": %s,\n  \"results\": [\n",
          prop.name, ncu, (int)total_lanes, bpc, BLK, prop.clockRate / 1000, iters,
          NACC, REPS, clk_load, clo, chi, dot2_exact ? "true" : "false");
  for (size_t i = 0; i < res.size(); i++) {
    auto &r = res[i];
    double clk = r.clk_mhz > 0 ? r.clk_mhz : (double)prop.clockRate / 1000;
    double ipc = r.instr_per_s * 1e9 / (total_lanes * clk * 1e6);
    double macpc = r.gops * 1e9 / (total_lanes * clk * 1e6);
    fprintf(j, "    {\"instr\": \"%s\", \"mnemonic\": \"%s\", \"macs_per_instr\": %d, "
               "\"median_ms\": %.6f, \"min_ms\": %.6f, \"max_ms\": %.6f, "
               "\"gmac_per_s\": %.3f, \"ginstr_per_s\": %.3f, "
               "\"instr_per_clk_per_cu\": %.3f, \"mac_per_clk_per_cu\": %.2f, "
               "\"pct_of_128_lane_valu\": %.1f}%s\n",
            r.name, r.mnemonic, r.macs_per_instr, r.median_ms, r.min_ms, r.max_ms,
            r.gops, r.instr_per_s, ipc, macpc, 100.0 * ipc,
            i + 1 == res.size() ? "" : ",");
  }
  fprintf(j, "  ]\n}\n");
  fclose(j);

  printf("\n%-16s %-9s %8s %10s %16s %12s %14s\n", "instr", "mnemonic", "mac/ins",
         "Gmac/s", "laneop/clk/lane", "mac/clk/CU", "% of peak lane-ops");
  for (auto &r : res) {
    double clk = r.clk_mhz > 0 ? r.clk_mhz : (double)prop.clockRate / 1000;
    double ipc = r.instr_per_s * 1e9 / (total_lanes * clk * 1e6);
    double macpc = r.gops * 1e9 / (total_lanes * clk * 1e6);
    printf("%-16s %-9s %8d %10.1f %16.3f %12.2f %13.1f%%\n", r.name, r.mnemonic,
           r.macs_per_instr, r.gops, ipc, macpc, 100.0 * ipc);
  }
  printf("\nwrote results.json\n");
  return 0;
}
