// gemv.cpp -- fused 4-bit dequant + matvec (M=1) on gfx1100, bandwidth accounting.
//
// Answers: what fraction of the measured read wall does a fused dequant-GEMV
// actually reach, and which layout choice moves it?
//   * soa  : quants 16B/32w contiguous, scales in a separate plane  (4.06 bpw)
//   * aos  : GGUF Q4_0 layout, 18B per 32w, scale inline            (4.50 bpw)
//   * read : same addressing, no dequant (floor)
//   * RB   : rows per warp (x reuse + weight-load ILP)
//
// usage: ./gemv <rows> <cols> <read|soa|aos> <RB> <reps>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <hip/hip_fp16.h>

#define CHK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(_e)); exit(1); } } while (0)

__device__ __forceinline__ int dp4a(int acc, int a, int b) {
  asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
  return acc;
}
__device__ __forceinline__ void warpwalk(int &lane, int &warp, int &nwarps) {
  lane = threadIdx.x & 31;
  warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  nwarps = (gridDim.x * blockDim.x) >> 5;
}

// ---------------------------------------------------------------- kernels
template <int RB>
__global__ void gemv_soa(const uint4 *__restrict__ w, const __half *__restrict__ s,
                         const uint4 *__restrict__ xq, float *__restrict__ y,
                         int rows, int nblk, int rp) {
  int lane, warp, nwarps; warpwalk(lane, warp, nwarps);
  for (int p = 0; p < rp; p++)
  for (int r0 = warp * RB; r0 < rows; r0 += nwarps * RB) {
    float acc[RB];
#pragma unroll
    for (int m = 0; m < RB; m++) acc[m] = 0.f;
    for (int b = lane; b < nblk; b += 32) {
      const uint4 xa = xq[2 * b], xb = xq[2 * b + 1];
      const int u0 = xa.x, u1 = xa.y, u2 = xa.z, u3 = xa.w;
      const int u4 = xb.x, u5 = xb.y, u6 = xb.z, u7 = xb.w;
#pragma unroll
      for (int m = 0; m < RB; m++) {
        const uint4 q = w[(size_t)(r0 + m) * nblk + b];
        int t = 0;
        t = dp4a(t, q.x & 0x0F0F0F0F, u0); t = dp4a(t, (q.x >> 4) & 0x0F0F0F0F, u4);
        t = dp4a(t, q.y & 0x0F0F0F0F, u1); t = dp4a(t, (q.y >> 4) & 0x0F0F0F0F, u5);
        t = dp4a(t, q.z & 0x0F0F0F0F, u2); t = dp4a(t, (q.z >> 4) & 0x0F0F0F0F, u6);
        t = dp4a(t, q.w & 0x0F0F0F0F, u3); t = dp4a(t, (q.w >> 4) & 0x0F0F0F0F, u7);
        acc[m] += (float)t * __half2float(s[(size_t)(r0 + m) * nblk + b]);
      }
    }
#pragma unroll
    for (int m = 0; m < RB; m++) {
      float v = acc[m];
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) y[r0 + m] = v;
    }
  }
}

template <int RB>
__global__ void gemv_seq(const uint4 *__restrict__ w, const __half *__restrict__ s,
                         const uint4 *__restrict__ xq, float *__restrict__ y,
                         int rows, int nblk, int rp) {
  int lane, warp, nwarps; warpwalk(lane, warp, nwarps);
  for (int p = 0; p < rp; p++)
  for (int r0 = warp * RB; r0 < rows; r0 += nwarps * RB) {
#pragma unroll
    for (int m = 0; m < RB; m++) {
      const int row = r0 + m;
      if (row >= rows) break;
      float acc = 0.f;
      for (int b = lane; b < nblk; b += 32) {
        const uint4 xa = xq[2 * b], xb = xq[2 * b + 1];
        const uint4 q = w[(size_t)row * nblk + b];
        int t = 0;
        t = dp4a(t, q.x & 0x0F0F0F0F, xa.x); t = dp4a(t, (q.x >> 4) & 0x0F0F0F0F, xb.x);
        t = dp4a(t, q.y & 0x0F0F0F0F, xa.y); t = dp4a(t, (q.y >> 4) & 0x0F0F0F0F, xb.y);
        t = dp4a(t, q.z & 0x0F0F0F0F, xa.z); t = dp4a(t, (q.z >> 4) & 0x0F0F0F0F, xb.z);
        t = dp4a(t, q.w & 0x0F0F0F0F, xa.w); t = dp4a(t, (q.w >> 4) & 0x0F0F0F0F, xb.w);
        acc += (float)t * __half2float(s[(size_t)row * nblk + b]);
      }
      float v = acc;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) y[row] = v;
    }
  }
}

// diagnostic: MODE 0 = full, 1 = scale from a register (no scale stream),
//             2 = x from registers (no x loads), 3 = quants only + dot, no scale/x
template <int MODE>
__global__ void gemv_diag(const uint4 *__restrict__ w, const __half *__restrict__ s,
                          const uint4 *__restrict__ xq, float *__restrict__ y,
                          int rows, int nblk, int rp) {
  int lane, warp, nwarps; warpwalk(lane, warp, nwarps);
  const uint4 cx = xq[lane & 1];
  for (int p = 0; p < rp; p++)
  for (int r0 = warp; r0 < rows; r0 += nwarps) {
    float acc = 0.f;
    for (int b = lane; b < nblk; b += 32) {
      const uint4 q = w[(size_t)r0 * nblk + b];
      const uint4 xa = (MODE == 2 || MODE == 3) ? cx : xq[2 * b];
      const uint4 xb = (MODE == 2 || MODE == 3) ? cx : xq[2 * b + 1];
      int t = 0;
      t = dp4a(t, q.x & 0x0F0F0F0F, xa.x); t = dp4a(t, (q.x >> 4) & 0x0F0F0F0F, xb.x);
      t = dp4a(t, q.y & 0x0F0F0F0F, xa.y); t = dp4a(t, (q.y >> 4) & 0x0F0F0F0F, xb.y);
      t = dp4a(t, q.z & 0x0F0F0F0F, xa.z); t = dp4a(t, (q.z >> 4) & 0x0F0F0F0F, xb.z);
      t = dp4a(t, q.w & 0x0F0F0F0F, xa.w); t = dp4a(t, (q.w >> 4) & 0x0F0F0F0F, xb.w);
        float sc;
      if (MODE == 6) {                         // batched scale: 1 x 16B per 8 blocks, shuffled
        uint4 sv = make_uint4(0,0,0,0);
        const int lead = lane & ~7;
        if ((lane & 7) == 0)
          sv = *reinterpret_cast<const uint4 *>(s + (size_t)r0 * nblk + (b - (lane & 7)));
        const unsigned hw = (unsigned)(lane & 7) >> 1;
        unsigned dw = (hw == 0) ? sv.x : (hw == 1) ? sv.y : (hw == 2) ? sv.z : sv.w;
        dw = __shfl(dw, lead);
        sc = __half2float(((lane & 1) == 0) ? __ushort_as_half((unsigned short)(dw & 0xffffu))
                                            : __ushort_as_half((unsigned short)(dw >> 16)));
      }
      else if (MODE == 1 || MODE == 2 || MODE == 3) sc = 0.0125f;
      else if (MODE == 4)                      // one scale per 8 blocks = 256 weights
        sc = __half2float(__ldg(&s[(size_t)r0 * nblk + (b & ~7)]));
      else if (MODE == 5)                      // one scale per 2 blocks (64 weights)
        sc = __half2float(__ldg(&s[(size_t)r0 * nblk + (b & ~1)]));
      else sc = __half2float(s[(size_t)r0 * nblk + b]);
      acc += (float)t * sc;
    }
    float v = acc;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    if (lane == 0) y[r0] = v;
  }
}

template <int RB>
__global__ void read_soa(const uint4 *__restrict__ w, const uint4 *__restrict__ xq,
                         float *__restrict__ y, int rows, int nblk, int rp) {
  int lane, warp, nwarps; warpwalk(lane, warp, nwarps);
  unsigned acc = 0;
  for (int p = 0; p < rp; p++)
  for (int r0 = warp * RB; r0 < rows; r0 += nwarps * RB) {
    for (int b = lane; b < nblk; b += 32) {
#pragma unroll
      for (int m = 0; m < RB; m++) {
        const uint4 q = w[(size_t)(r0 + m) * nblk + b];
        acc ^= q.x ^ q.y ^ q.z ^ q.w;
      }
    }
  }
  if (acc == 0xdeadbeefu) y[threadIdx.x] = (float)acc;
}

// GGUF Q4_0 exactly: 18 B per 32 weights, fp16 scale inline before the quants
template <int RB>
__global__ void gemv_aos(const unsigned char *__restrict__ w,
                         const uint4 *__restrict__ xq, float *__restrict__ y,
                         int rows, int nblk, int rp) {
  int lane, warp, nwarps; warpwalk(lane, warp, nwarps);
  for (int p = 0; p < rp; p++)
  {
  const size_t pitch = (size_t)nblk * 18;
  for (int r0 = warp * RB; r0 < rows; r0 += nwarps * RB) {
    float acc[RB];
#pragma unroll
    for (int m = 0; m < RB; m++) acc[m] = 0.f;
    for (int b = lane; b < nblk; b += 32) {
      const uint4 xa = xq[2 * b], xb = xq[2 * b + 1];
      const int u0 = xa.x, u1 = xa.y, u2 = xa.z, u3 = xa.w;
      const int u4 = xb.x, u5 = xb.y, u6 = xb.z, u7 = xb.w;
#pragma unroll
      for (int m = 0; m < RB; m++) {
        const unsigned char *p = w + (size_t)(r0 + m) * pitch + (size_t)b * 18;
        const uint4 q = *reinterpret_cast<const uint4 *>(p + 2);
        const __half sc = *reinterpret_cast<const __half *>(p);
        int t = 0;
        t = dp4a(t, q.x & 0x0F0F0F0F, u0); t = dp4a(t, (q.x >> 4) & 0x0F0F0F0F, u4);
        t = dp4a(t, q.y & 0x0F0F0F0F, u1); t = dp4a(t, (q.y >> 4) & 0x0F0F0F0F, u5);
        t = dp4a(t, q.z & 0x0F0F0F0F, u2); t = dp4a(t, (q.z >> 4) & 0x0F0F0F0F, u6);
        t = dp4a(t, q.w & 0x0F0F0F0F, u3); t = dp4a(t, (q.w >> 4) & 0x0F0F0F0F, u7);
        acc[m] += (float)t * __half2float(sc);
      }
    }
#pragma unroll
    for (int m = 0; m < RB; m++) {
      float v = acc[m];
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) y[r0 + m] = v;
    }
  }
  }
}

// --------------------------------------------------------------------- main
static unsigned rng = 12345;
static unsigned nextu() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(int argc, char **argv) {
  int rows = (argc > 1) ? atoi(argv[1]) : 17408;
  int cols = (argc > 2) ? atoi(argv[2]) : 5120;
  std::string kern = (argc > 3) ? argv[3] : "soa";
  int RB = (argc > 4) ? atoi(argv[4]) : 4;
  int reps = (argc > 5) ? atoi(argv[5]) : 7;
  int gm = (argc > 6) ? atoi(argv[6]) : 8;   // blocks per 128-lane unit
  int rp = (argc > 7) ? atoi(argv[7]) : 1;   // in-kernel repeats
  int L  = (argc > 8) ? atoi(argv[8]) : 1;   // launches per timed rep
  int nblk = cols / 32;
  if (cols % 1024) { printf("cols must be a multiple of 1024\n"); return 1; }

  hipDeviceProp_t p; CHK(hipGetDeviceProperties(&p, 0));
  int SM = p.multiProcessorCount;

  size_t qbytes = (size_t)rows * nblk * 16;
  size_t sbytes = (size_t)rows * nblk * 2;
  size_t abytes = (size_t)rows * nblk * 18;
  double bpw = (kern == "aos") ? 4.5 : (kern == "read" ? 4.0 : 4.0625);
  size_t wbytes = (kern == "aos") ? abytes : (kern == "read" ? qbytes : qbytes + sbytes);
  if (kern == "d3") wbytes = qbytes;

  uint4 *wq = nullptr; __half *ws = nullptr; unsigned char *wa = nullptr;
  uint4 *xq = nullptr; float *y = nullptr;
  CHK(hipMalloc(&wq, qbytes)); CHK(hipMalloc(&ws, sbytes)); CHK(hipMalloc(&wa, abytes));
  CHK(hipMalloc(&xq, (size_t)nblk * 32)); CHK(hipMalloc(&y, (size_t)rows * 4));

  {
    size_t nquads = (size_t)rows * nblk;
    std::vector<unsigned> hp(nquads * 4);
    for (auto &v : hp) v = nextu();
    CHK(hipMemcpy(wq, hp.data(), qbytes, hipMemcpyHostToDevice));
    std::vector<__half> hs(nquads);
    for (size_t i = 0; i < hs.size(); i++) hs[i] = __float2half(0.01f + (i % 7) * 0.001f);
    CHK(hipMemcpy(ws, hs.data(), sbytes, hipMemcpyHostToDevice));
    if (kern == "aos") {
      std::vector<unsigned char> ha(abytes);
      for (size_t r = 0; r < (size_t)rows; r++)
        for (int b = 0; b < nblk; b++) {
          unsigned char *dst = ha.data() + r * nblk * 18 + (size_t)b * 18;
          *reinterpret_cast<__half *>(dst) = __float2half(0.01f + (b % 7) * 0.001f);
          memcpy(dst + 2, hp.data() + (r * nblk + b) * 4, 16);
        }
      CHK(hipMemcpy(wa, ha.data(), abytes, hipMemcpyHostToDevice));
    }
    std::vector<int> hx(nblk * 8);
    for (auto &v : hx) v = (int)nextu();
    CHK(hipMemcpy(xq, hx.data(), (size_t)nblk * 32, hipMemcpyHostToDevice));
  }

  if (getenv("VERIFY")) {
    // run the fused kernel once and compare against a host reference on the same bytes
    hipEvent_t ve0, ve1; CHK(hipEventCreate(&ve0)); CHK(hipEventCreate(&ve1));
    CHK(hipEventRecord(ve0));
    if (kern == "soa") gemv_soa<1><<<SM * 8, 256>>>(wq, ws, xq, y, rows, nblk, 1);
    else if (kern == "aos") gemv_aos<1><<<SM * 8, 256>>>(wa, xq, y, rows, nblk, 1);
    CHK(hipEventRecord(ve1)); CHK(hipEventSynchronize(ve1));
    std::vector<float> hy(rows);
    CHK(hipMemcpy(hy.data(), y, (size_t)rows * 4, hipMemcpyDeviceToHost));
    // rebuild the same inputs on the host
    rng = 12345;
    std::vector<unsigned> hq((size_t)rows * nblk * 4);
    for (auto &v : hq) v = nextu();
    std::vector<float> hsc((size_t)rows * nblk);
    for (size_t i = 0; i < hsc.size(); i++) hsc[i] = __half2float(__float2half(0.01f + (i % 7) * 0.001f));
    std::vector<int> hxi(nblk * 8);
    for (auto &v : hxi) v = (int)nextu();
    const signed char *xi = reinterpret_cast<const signed char *>(hxi.data());   // 32*8 int8
    double worst = 0; int bad = 0;
    for (int r = 0; r < rows; r++) {
      double ref = 0;
      for (int b = 0; b < nblk; b++) {
        const unsigned *q = hq.data() + ((size_t)r * nblk + b) * 4;
        const signed char *xb = xi + b * 32;
        long long t = 0;
        for (int k = 0; k < 4; k++) {
          for (int j = 0; j < 4; j++) {
            t += (long long)((q[k] >> (8 * j)) & 0xF) * xb[4 * k + j];
            t += (long long)((q[k] >> (8 * j + 4)) & 0xF) * xb[16 + 4 * k + j];
          }
        }
        ref += (double)t * hsc[(size_t)r * nblk + b];
      }
      double err = fabs(hy[r] - ref) / (fabs(ref) + 1e-6);
      if (err > worst) worst = err;
      if (err > 2e-3) bad++;
    }
    printf("VERIFY: rows=%d worst rel err %.3e, %%rows>2e-3 = %.3f%%\n",
           rows, worst, 100.0 * bad / rows);
    {
      unsigned dw = 0; __half ds = __float2half(0.f); unsigned dx = 0;
      CHK(hipMemcpy(&dw, wq, 4, hipMemcpyDeviceToHost));
      CHK(hipMemcpy(&ds, ws, 2, hipMemcpyDeviceToHost));
      CHK(hipMemcpy(&dx, xq, 4, hipMemcpyDeviceToHost));
      printf("  data: dev q0=%08x s0=%.6f x0=%08x | host q0=%08x s0=%.6f x0=%08x\n",
             dw, __half2float(ds), dx, hq[0], hsc[0], (unsigned)hxi[0]);
    }
    // dump row 0 detail
    {
      int r = 0; double ref = 0;
      for (int b = 0; b < nblk; b++) {
        const unsigned *q = hq.data() + ((size_t)r * nblk + b) * 4;
        const signed char *xb = xi + b * 32;
        long long t = 0;
        for (int k = 0; k < 4; k++)
          for (int j = 0; j < 4; j++) {
            t += (long long)((q[k] >> (8 * j)) & 0xF) * xb[4 * k + j];
            t += (long long)((q[k] >> (8 * j + 4)) & 0xF) * xb[16 + 4 * k + j];
          }
        if (b < 2) printf("  b=%d t=%lld sc=%.6f\n", b, t, hsc[b]);
        ref += (double)t * hsc[(size_t)r * nblk + b];
      }
      printf("  row0: gpu %.6f  ref %.6f  ratio %.4f\n", hy[0], ref, hy[0] / ref);
    }
    return 0;
  }

  const int blk = 256, grid = SM * gm;
  printf("gemv: rows=%d cols=%d nblk=%d kern=%s RB=%d rp=%d L=%d | weights %.3f GiB, %.3f bpw | grid %d x %d\n",
         rows, cols, nblk, kern.c_str(), RB, rp, L, wbytes / 1073741824.0, bpw, grid, blk);

  hipEvent_t e0, e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
  std::vector<double> gb;
  for (int r = 0; r < reps; r++) {
    float t = 0; CHK(hipEventRecord(e0));
    for (int li = 0; li < L; li++) {
    if (kern == "soa") {
      if (RB == 1) gemv_soa<1><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
      else if (RB == 2) gemv_soa<2><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
      else if (RB == 4) gemv_soa<4><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
      else gemv_soa<8><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d0") { gemv_diag<0><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d1") { gemv_diag<1><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d2") { gemv_diag<2><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d3") { gemv_diag<3><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d4") { gemv_diag<4><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d5") { gemv_diag<5><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "d6") { gemv_diag<6><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "seq") {
      if (RB == 1) gemv_seq<1><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
      else if (RB == 2) gemv_seq<2><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
      else if (RB == 4) gemv_seq<4><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
      else gemv_seq<8><<<grid, blk>>>(wq, ws, xq, y, rows, nblk, rp);
    } else if (kern == "aos") {
      if (RB == 1) gemv_aos<1><<<grid, blk>>>(wa, xq, y, rows, nblk, rp);
      else if (RB == 2) gemv_aos<2><<<grid, blk>>>(wa, xq, y, rows, nblk, rp);
      else if (RB == 4) gemv_aos<4><<<grid, blk>>>(wa, xq, y, rows, nblk, rp);
      else gemv_aos<8><<<grid, blk>>>(wa, xq, y, rows, nblk, rp);
    } else {
      if (RB == 1) read_soa<1><<<grid, blk>>>(wq, xq, y, rows, nblk, rp);
      else if (RB == 2) read_soa<2><<<grid, blk>>>(wq, xq, y, rows, nblk, rp);
      else if (RB == 4) read_soa<4><<<grid, blk>>>(wq, xq, y, rows, nblk, rp);
      else read_soa<8><<<grid, blk>>>(wq, xq, y, rows, nblk, rp);
    }
    }
    CHK(hipEventRecord(e1)); CHK(hipEventSynchronize(e1));
    CHK(hipEventElapsedTime(&t, e0, e1));
    gb.push_back((double)wbytes * rp * L / (t * 1e-3) / 1e9);
    if (getenv("RAWMS")) printf("    raw %.3f ms\n", t);
  }
  printf("  reps:");
  for (double d : gb) printf(" %.0f", d);
  printf("\n");
  std::sort(gb.begin(), gb.end());
  printf("  -> median %7.1f GB/s  (min %.1f  max %.1f)  %.2f ms/pass\n",
         gb[gb.size() / 2], gb.front(), gb.back(),
         wbytes / (gb[gb.size() / 2] * 1e9) * 1e3);
  return 0;
}
