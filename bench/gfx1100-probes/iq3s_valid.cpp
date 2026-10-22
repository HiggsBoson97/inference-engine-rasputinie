// iq3s_valid.cpp -- IQ3_S codebook-addressing probe with a hard validity gate.
//
// Layout semantics come only from the documented field sizes
// (docs/research/2026-10-22/03c-gguf-quant-layouts.md) plus arithmetic on them:
//   110 B = qs 64 + qh 8 + signs 32 + scales 4 + d 2   over 256 weights
//   => 8 sub-blocks of 32 weights; 64 indices of 9 bits (64 B qs + 8 B qh);
//      1 sign bit per weight; one 4-bit multiplier per 32-weight group;
//      one fp16 super-scale per 256 weights.
// The field ORDER inside the block is decided by the data (see find_scales.py:
// d@0 is positive and finite for 100% of blocks, every other offset is a coin flip).
//
// WHY THE GATE EXISTS: a kernel that reads less than it counts reports more than
// the card can deliver. Every kernel here therefore accumulates a data-dependent
// checksum (order-independent sum, so it does not depend on the thread mapping),
// writes it to a global array, and the host compares it against the same sum
// replayed from the GGUF bytes. Any skipped or hoisted load changes the sum and
// the run is discarded. A GB/s figure above the measured read wall is also an
// automatic INVALID.
//
// usage: ./iq3s_valid <manifest> <gguf> <NV> <rp> [lay]
// env: GRIDMUL, BLK, REPL (replicate the footprint N times), DBG

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

#define CHK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  printf("FAIL %s: %s @%d\n", #x, hipGetErrorString(_e), __LINE__); exit(1); } } while (0)

#define QK   256
#define BLK  110
#define GW   32
#define GPR  8
#define WALL 776.0

template <int LAY> struct Off {
  static constexpr int qs = LAY ? 0   : 2;
  static constexpr int qh = LAY ? 64  : 66;
  static constexpr int sg = LAY ? 72  : 74;
  static constexpr int sc = LAY ? 104 : 106;
  static constexpr int d  = LAY ? 108 : 0;
};

__device__ unsigned int g_cbook[512];

__device__ __forceinline__ int dp4a(int acc, int a, int b) {
  asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
  return acc;
}
__device__ __forceinline__ unsigned cneg(unsigned x, unsigned b) {
  unsigned m = (((b & 0xFu) * 0x10204080u) & 0x80808080u) >> 7;
  m *= 0xFFu;
  return (x ^ m) + (m & 0x01010101u);
}
__device__ __forceinline__ void load8(const unsigned char *__restrict__ p, unsigned char *__restrict__ o) {
  const unsigned short *h = reinterpret_cast<const unsigned short *>(p);
  o[0] = (unsigned char)(h[0] & 0xFF); o[1] = (unsigned char)(h[0] >> 8);
  o[2] = (unsigned char)(h[1] & 0xFF); o[3] = (unsigned char)(h[1] >> 8);
  o[4] = (unsigned char)(h[2] & 0xFF); o[5] = (unsigned char)(h[2] >> 8);
  o[6] = (unsigned char)(h[3] & 0xFF); o[7] = (unsigned char)(h[3] >> 8);
}
__device__ __forceinline__ void wwalk(int &lane, int &warp, int &nwarp, size_t &gtid) {
  gtid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  lane  = threadIdx.x & 31;
  warp  = (int)(gtid >> 5);
  nwarp = (int)(((size_t)gridDim.x * blockDim.x) >> 5);
}

// -------------------------------------------------- flat aligned stream of the bytes
__global__ void k_flat(const uint4 *__restrict__ W, unsigned long long *__restrict__ out,
                       size_t nquad, int rp, size_t rot) {
  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long sum = 0;
  for (int p = 0; p < rp; ++p)
    for (size_t k = gtid; k < nquad; k += stride) {
      size_t kk = k + (size_t)p * rot;        // rotate: no address repeats across passes
      if (kk >= nquad) kk -= nquad;
      uint4 q = W[kk];
      sum += (unsigned long long)q.x + (unsigned long long)q.y
           + (unsigned long long)q.z + (unsigned long long)q.w;
    }
  out[gtid] = sum;
}

// ---------------------------------------- (1) reader-only, real block addressing
template <int LAY>
__global__ void k_read(const unsigned char *__restrict__ W, unsigned long long *__restrict__ out,
                       int rows, int nblk, int rp, int rotr) {
  int lane, warp, nwarp; size_t gtid; wwalk(lane, warp, nwarp, gtid);
  const int G = nblk * GPR;
  const size_t pitch = (size_t)nblk * BLK;
  unsigned long long sum = 0;
  for (int p = 0; p < rp; ++p) {
    for (int rw = warp; rw < rows; rw += nwarp) {
      int r = rw + p * rotr; if (r >= rows) r -= rows;
      const unsigned char *row = W + (size_t)r * pitch;
      for (int b = lane; b < G; b += 32) {
        const unsigned char *pb = row + (size_t)(b >> 3) * BLK;
        const int sb = b & 7;
        unsigned char idx[8];
        load8(pb + Off<LAY>::qs + 8 * sb, idx);
        const unsigned short *sg = reinterpret_cast<const unsigned short *>(pb + Off<LAY>::sg + 4 * sb);
        const unsigned qh = (unsigned)pb[Off<LAY>::qh + sb];
        const unsigned sc = (unsigned)pb[Off<LAY>::sc + (sb >> 1)];
        const unsigned dw = (unsigned)reinterpret_cast<const unsigned short *>(pb + Off<LAY>::d)[0];
        const unsigned q0 = (unsigned)idx[0] | ((unsigned)idx[1] << 8) | ((unsigned)idx[2] << 16) | ((unsigned)idx[3] << 24);
        const unsigned q1 = (unsigned)idx[4] | ((unsigned)idx[5] << 8) | ((unsigned)idx[6] << 16) | ((unsigned)idx[7] << 24);
        sum += (unsigned long long)q0 * 0x9E3779B97F4A7C15ull
             + (unsigned long long)q1 * 0xC2B2AE3D27D4EB4Full
             + (unsigned long long)qh * 0x165667B19E3779F9ull
             + (unsigned long long)((unsigned)sg[0] ^ (unsigned)sg[1]) * 0x27D4EB2F165667C5ull
             + (unsigned long long)sc * 0x100000001B3ull
             + (unsigned long long)dw;
      }
    }
  }
  out[gtid] = sum;
}

// ---------------------------------------- (2) real dequant + dot
// MODE 0 = real, 1 = no codebook load (diagnostic), 2 = signs off (diagnostic)
template <int LAY, int MODE>
__global__ void k_full(const unsigned char *__restrict__ W, const int *__restrict__ x32,
                       float *__restrict__ y, unsigned long long *__restrict__ out,
                       int rows, int nblk, int rp, int rotr) {
  int lane, warp, nwarp; size_t gtid; wwalk(lane, warp, nwarp, gtid);
  const int G = nblk * GPR;
  const size_t pitch = (size_t)nblk * BLK;
  unsigned long long sum = 0;
  for (int p = 0; p < rp; ++p) {
    for (int rw = warp; rw < rows; rw += nwarp) {
      int r = rw + p * rotr; if (r >= rows) r -= rows;
      const unsigned char *row = W + (size_t)r * pitch;
      float acc = 0.f;
      for (int b = lane; b < G; b += 32) {
        const unsigned char *pb = row + (size_t)(b >> 3) * BLK;
        const int sb = b & 7;
        unsigned char qb[8];
        load8(pb + Off<LAY>::qs + 8 * sb, qb);
        const unsigned qh = (unsigned)pb[Off<LAY>::qh + sb];
        const unsigned short *sgs = reinterpret_cast<const unsigned short *>(pb + Off<LAY>::sg + 4 * sb);
        const unsigned sg = (unsigned)sgs[0] | ((unsigned)sgs[1] << 16);
        const unsigned sc = (unsigned)pb[Off<LAY>::sc + (sb >> 1)];
        const float d = __half2float(reinterpret_cast<const __half *>(pb + Off<LAY>::d)[0]);
        const float s = d * (1.f + 2.f * (float)((sb & 1) ? (sc >> 4) : (sc & 15)));
        const int *xa = x32 + (size_t)b * 8;
        int t = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const int i0 = qb[2 * l]     | ((qh << (8 - 2 * l)) & 0x100);
          const int i1 = qb[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100);
          unsigned g0, g1;
          if (MODE == 1) {
            g0 = (((unsigned)i0 * 2654435761u) & 0x0F0F0F0Fu) | 0x01010101u;
            g1 = (((unsigned)i1 * 2654435761u) & 0x0F0F0F0Fu) | 0x01010101u;
          } else {
            g0 = g_cbook[i0]; g1 = g_cbook[i1];
          }
          if (MODE != 2) {
            g0 = cneg(g0, (sg >> (8 * l)) & 0xF);
            g1 = cneg(g1, (sg >> (8 * l + 4)) & 0xF);
          }
          t = dp4a(t, (int)g0, xa[2 * l]);
          t = dp4a(t, (int)g1, xa[2 * l + 1]);
        }
        acc += (float)t * s;
        sum += (unsigned long long)(unsigned)t;
      }
      float v = acc;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) y[r] = v;
    }
  }
  out[gtid] = sum;
}

// ---------------------------------------- (3) lossless expanded layout
// SCALEMODE 0 = exact (fp16 d + original 4-bit multipliers), 1 = fp16 per 32 weights,
//            2 = exact scale with signs forced off (diagnostic)
template <int SCALEMODE>
__global__ void k_exp(const uint4 *__restrict__ M, const unsigned *__restrict__ S,
                      const unsigned char *__restrict__ SC, const __half *__restrict__ SC16,
                      const int *__restrict__ x32, float *__restrict__ y,
                      unsigned long long *__restrict__ out,
                      int rows, int G, int nblk, int rp, int rotr) {
  int lane, warp, nwarp; size_t gtid; wwalk(lane, warp, nwarp, gtid);
  const int rowsc = nblk * 6;
  unsigned long long sum = 0;
  for (int p = 0; p < rp; ++p) {
    for (int rw = warp; rw < rows; rw += nwarp) {
      int r = rw + p * rotr; if (r >= rows) r -= rows;
      float acc = 0.f;
      const size_t base = (size_t)r * G;
      for (int b = lane; b < G; b += 32) {
        const uint4 q = M[base + b];
        const unsigned sgn = (SCALEMODE == 2) ? 0u : S[base + b];
        float s; unsigned scraw = 0;
        if (SCALEMODE == 1) {
          s = __half2float(SC16[base + b]);
          scraw = (unsigned)__half_as_ushort(SC16[base + b]);
        } else {
          const unsigned char *sp = SC + (size_t)r * rowsc + (size_t)(b >> 3) * 6;
          const float d = __half2float(reinterpret_cast<const __half *>(sp)[0]);
          const unsigned sc = sp[2 + ((b & 7) >> 1)];
          s = d * (1.f + 2.f * (float)((b & 1) ? (sc >> 4) : (sc & 15)));
          scraw = (unsigned)sp[0] | ((unsigned)sp[1] << 8) | ((unsigned)sc << 16);
        }
        if (SCALEMODE == 3) {   // load-only floor: consume the bytes, no dequant, no dot
          sum += (unsigned long long)q.x + (unsigned long long)q.y
               + (unsigned long long)q.z + (unsigned long long)q.w
               + (unsigned long long)sgn + (unsigned long long)scraw;
          continue;
        }
        const int *xa = x32 + (size_t)b * 8;
        int t = 0;
        t = dp4a(t, (int)cneg(q.x        & 0x0F0F0F0Fu, (sgn >>  0) & 0xF), xa[0]);
        t = dp4a(t, (int)cneg((q.x >> 4) & 0x0F0F0F0Fu, (sgn >>  4) & 0xF), xa[1]);
        t = dp4a(t, (int)cneg(q.y        & 0x0F0F0F0Fu, (sgn >>  8) & 0xF), xa[2]);
        t = dp4a(t, (int)cneg((q.y >> 4) & 0x0F0F0F0Fu, (sgn >> 12) & 0xF), xa[3]);
        t = dp4a(t, (int)cneg(q.z        & 0x0F0F0F0Fu, (sgn >> 16) & 0xF), xa[4]);
        t = dp4a(t, (int)cneg((q.z >> 4) & 0x0F0F0F0Fu, (sgn >> 20) & 0xF), xa[5]);
        t = dp4a(t, (int)cneg(q.w        & 0x0F0F0F0Fu, (sgn >> 24) & 0xF), xa[6]);
        t = dp4a(t, (int)cneg((q.w >> 4) & 0x0F0F0F0Fu, (sgn >> 28) & 0xF), xa[7]);
        acc += (float)t * s;
        sum += (unsigned long long)(unsigned)t;
      }
      float v = acc;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) y[r] = v;
    }
  }
  out[gtid] = sum;
}

// ---------------------------------------------------------------- host helpers
static double half2d(uint16_t h) {
  unsigned s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  if (e == 0) return (s ? -1 : 1) * ldexp((double)m, -24);
  if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
  return (s ? -1 : 1) * ldexp((double)(1024 + m), (int)e - 25);
}
static void build_cbook(unsigned *tab) {
  unsigned s = 0x9E3779B9u;
  for (int i = 0; i < 512; ++i) {
    unsigned v = 0;
    for (int j = 0; j < 4; ++j) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      v |= (1u + 2u * (s % 8u)) << (8 * j);
    }
    tab[i] = v;
  }
}
static inline int idx_of(const unsigned char *q8, unsigned qh, int l) {
  return q8[l] | ((l & 1) ? (int)((qh << (7 - 2 * (l / 2))) & 0x100)
                          : (int)((qh << (8 - 2 * (l / 2))) & 0x100));
}
struct Tens { char name[128]; long long off, nbytes; int rows, nblk; };

int main(int argc, char **argv) {
  if (argc < 5) { printf("usage: %s <manifest> <gguf> <NV> <rp> [lay]\n", argv[0]); return 1; }
  const char *mf = argv[1], *gp = argv[2];
  const int NV = atoi(argv[3]), rp = atoi(argv[4]);
  const int LAY = (argc > 5) ? atoi(argv[5]) : 0;

  FILE *fm = fopen(mf, "r");
  if (!fm) { printf("no manifest\n"); return 1; }
  std::vector<Tens> T; char line[256];
  while (fgets(line, sizeof line, fm)) {
    if (line[0] == '#') continue;
    Tens t; memset(&t, 0, sizeof t);
    if (sscanf(line, "%127s %lld %lld %d %d", t.name, &t.off, &t.nbytes, &t.rows, &t.nblk) != 5) continue;
    T.push_back(t);
  }
  fclose(fm);
  long long base_total = 0; int base_rows = 0;
  for (auto &t : T) {
    long long expect = (long long)t.nblk * BLK * t.rows;      // stride x nblocks
    if (expect != t.nbytes) { printf("LAYOUT FAIL %s\n", t.name); return 2; }
    if (t.nblk != T[0].nblk) { printf("mixed shapes\n"); return 2; }
    base_total += t.nbytes; base_rows += t.rows;
  }
  const int nblk = T[0].nblk, cols = nblk * QK, G = nblk * GPR;
  const int REPL = getenv("REPL") ? atoi(getenv("REPL")) : 1;
  const int rows = base_rows * REPL;
  const long long total = base_total * REPL;
  printf("manifest: %zu tensors x%d repl, %d rows x %d cols, nblk=%d, footprint %lld B = %.4f GiB\n",
         T.size(), REPL, rows, cols, nblk, total, (double)total / 1073741824.0);

  int fd = open(gp, O_RDONLY);
  if (fd < 0) { printf("cannot open gguf\n"); return 1; }
  std::vector<unsigned char> hraw((size_t)base_total);
  {
    size_t o = 0;
    for (auto &t : T) {
      if (pread(fd, hraw.data() + o, (size_t)t.nbytes, t.off) != (ssize_t)t.nbytes) { printf("short read\n"); return 1; }
      o += (size_t)t.nbytes;
    }
  }
  close(fd);

  unsigned hcb[512]; build_cbook(hcb);
  std::vector<int8_t> x8(cols);
  for (int i = 0; i < cols; ++i) x8[i] = (int8_t)(((i * 37 + 11) % 199) - 99);

  const int od = LAY ? 108 : 0, oqs = LAY ? 0 : 2, oqh = LAY ? 64 : 66,
            osg = LAY ? 72 : 74, osc = LAY ? 104 : 106;

  // ---- host replay: the exact sums the kernels must produce, for one pass
  unsigned long long exp_flat = 0, exp_read = 0, exp_dot = 0, exp_dot_nosign = 0, exp_dot_nogather = 0;
  unsigned long long exp_expread = 0;
  {
    const size_t nquad = (size_t)base_total / 16;
    for (size_t k = 0; k < nquad; ++k) {
      const uint4 q = reinterpret_cast<const uint4 *>(hraw.data())[k];
      exp_flat += (unsigned long long)q.x + (unsigned long long)q.y
                + (unsigned long long)q.z + (unsigned long long)q.w;
    }
    for (int r = 0; r < base_rows; ++r) {
      const unsigned char *row = hraw.data() + (size_t)r * (size_t)nblk * BLK;
      for (int b = 0; b < nblk; ++b) {
        const unsigned char *pb = row + (size_t)b * BLK;
        const unsigned dw = (unsigned)*reinterpret_cast<const unsigned short *>(pb + od);
        for (int sb = 0; sb < GPR; ++sb) {
          const unsigned char *q = pb + oqs + 8 * sb;
          const unsigned qh = pb[oqh + sb];
          const unsigned char *sg = pb + osg + 4 * sb;
          // reader hash (same formula as k_read)
          const unsigned q0 = (unsigned)q[0] | ((unsigned)q[1] << 8) | ((unsigned)q[2] << 16) | ((unsigned)q[3] << 24);
          const unsigned q1 = (unsigned)q[4] | ((unsigned)q[5] << 8) | ((unsigned)q[6] << 16) | ((unsigned)q[7] << 24);
          const unsigned sgw = (unsigned)sg[0] | ((unsigned)sg[1] << 8) | ((unsigned)sg[2] << 16) | ((unsigned)sg[3] << 24);
          exp_read += (unsigned long long)q0 * 0x9E3779B97F4A7C15ull
                    + (unsigned long long)q1 * 0xC2B2AE3D27D4EB4Full
                    + (unsigned long long)qh * 0x165667B19E3779F9ull
                    + (unsigned long long)((sgw & 0xFFFF) ^ (sgw >> 16)) * 0x27D4EB2F165667C5ull
                    + (unsigned long long)pb[osc + (sb >> 1)] * 0x100000001B3ull
                    + (unsigned long long)dw;
          // the expanded plane words for this group (same packing as the builder)
          unsigned magw[4];
          for (int l = 0; l < 4; ++l) {
            const unsigned w0 = hcb[idx_of(q, qh, 2 * l)], w1 = hcb[idx_of(q, qh, 2 * l + 1)];
            unsigned v = 0;
            for (int j = 0; j < 4; ++j)
              v |= (((w0 >> (8 * j)) & 0xFu) << (8 * j))
                 | ((((w1 >> (8 * j)) & 0xFu) << 4) << (8 * j));
            magw[l] = v;
          }
          // integer group dot (same value as the dp4a chain)
          const int8_t *ag = x8.data() + (size_t)(b * GPR + sb) * GW;
          int t = 0, tn = 0, tg = 0;
          for (int l = 0; l < 4; ++l) {
            const unsigned v0 = hcb[idx_of(q, qh, 2 * l)], v1 = hcb[idx_of(q, qh, 2 * l + 1)];
            const unsigned m0 = (sg[l]) & 0xF, m1 = (sg[l] >> 4) & 0xF;
            const int i0 = idx_of(q, qh, 2 * l), i1 = idx_of(q, qh, 2 * l + 1);
            const unsigned ng0 = (((unsigned)i0 * 2654435761u) & 0x0F0F0F0Fu) | 0x01010101u;
            const unsigned ng1 = (((unsigned)i1 * 2654435761u) & 0x0F0F0F0Fu) | 0x01010101u;
            int acc_s = 0, acc_n = 0, acc_g = 0;
            for (int j = 0; j < 4; ++j) {
              const int bs = (((m0 >> j) & 1) ? -1 : 1), bh = (((m1 >> j) & 1) ? -1 : 1);
              const int ls = (int)((v0 >> (8 * j)) & 0xF) * bs;
              const int hs = (int)((v1 >> (8 * j)) & 0xF) * bh;
              acc_s += ls * ag[8 * l + j] + hs * ag[8 * l + 4 + j];
              acc_n += (int)((v0 >> (8 * j)) & 0xF) * ag[8 * l + j] + (int)((v1 >> (8 * j)) & 0xF) * ag[8 * l + 4 + j];
              const int gl = (int)((ng0 >> (8 * j)) & 0xF) * bs;
              const int gh = (int)((ng1 >> (8 * j)) & 0xF) * bh;
              acc_g += gl * ag[8 * l + j] + gh * ag[8 * l + 4 + j];
            }
            t += acc_s; tn += acc_n; tg += acc_g;
          }
          // load-only expanded floor: same words the expanded planes hold
          exp_expread += (unsigned long long)(unsigned)magw[0] + (unsigned long long)(unsigned)magw[1]
                       + (unsigned long long)(unsigned)magw[2] + (unsigned long long)(unsigned)magw[3]
                       + (unsigned long long)sgw
                       + (unsigned long long)(dw | ((unsigned)pb[osc + (sb >> 1)] << 16));
          exp_dot += (unsigned long long)(unsigned)t;
          exp_dot_nosign += (unsigned long long)(unsigned)tn;
          exp_dot_nogather += (unsigned long long)(unsigned)tg;
        }
      }
    }
  }
  printf("gate: expected sums (1 pass) flat %llu, read %llu, dot %llu, dot_nosign %llu, dot_nogather %llu\n",
         exp_flat, exp_read, exp_dot, exp_dot_nosign, exp_dot_nogather);

  hipDeviceProp_t prop; CHK(hipGetDeviceProperties(&prop, 0));
  const int SM = prop.multiProcessorCount;
  {
    char busid[32] = {0};
    if (hipDeviceGetPCIBusId(busid, sizeof busid, 0) != hipSuccess) busid[0] = 0;
    printf("device: %s, %d CU, PCI %s (HIP_VISIBLE_DEVICES=%s)\n", prop.name, SM, busid,
           getenv("HIP_VISIBLE_DEVICES") ? getenv("HIP_VISIBLE_DEVICES") : "unset");
  }

  const int blk = getenv("BLK") ? atoi(getenv("BLK")) : 256;
  const int gm  = getenv("GRIDMUL") ? atoi(getenv("GRIDMUL")) : 8;
  const int grid = SM * gm;
  const size_t nthreads = (size_t)grid * blk;
  printf("config: grid=%d x %d = %zu threads (%zu warps), rp=%d, rows/warp=%.1f\n",
         grid, blk, nthreads, nthreads / 32, rp, (double)rows / (double)(nthreads / 32));

  unsigned char *dW = nullptr; int *dx = nullptr; float *dy = nullptr;
  unsigned long long *dout = nullptr;
  CHK(hipMalloc(&dW, (size_t)total));
  for (int i = 0; i < REPL; ++i)
    CHK(hipMemcpy(dW + (size_t)i * base_total, hraw.data(), (size_t)base_total, hipMemcpyHostToDevice));
  CHK(hipMalloc(&dx, (size_t)cols));
  CHK(hipMemcpy(dx, x8.data(), (size_t)cols, hipMemcpyHostToDevice));
  CHK(hipMalloc(&dy, (size_t)rows * 4));
  CHK(hipMalloc(&dout, nthreads * 8));
  CHK(hipMemcpyToSymbol(HIP_SYMBOL(g_cbook), hcb, sizeof hcb));

  auto ref_row = [&](int r) -> double {
    const unsigned char *p = hraw.data() + (size_t)(r % base_rows) * (size_t)nblk * BLK;
    double acc = 0.0;
    for (int b = 0; b < nblk; ++b) {
      const unsigned char *pb = p + (size_t)b * BLK;
      const double d = half2d(*reinterpret_cast<const uint16_t *>(pb + od));
      for (int sb = 0; sb < GPR; ++sb) {
        const unsigned char *q = pb + oqs + 8 * sb;
        const unsigned qh = pb[oqh + sb];
        const unsigned char *sg = pb + osg + 4 * sb;
        const unsigned sc = pb[osc + (sb >> 1)];
        const int n = (sb & 1) ? (int)(sc >> 4) : (int)(sc & 15);
        const double s = d * (1.0 + 2.0 * (double)n);
        const int8_t *ag = x8.data() + (size_t)(b * GPR + sb) * GW;
        for (int l = 0; l < 4; ++l) {
          const unsigned v0 = hcb[idx_of(q, qh, 2 * l)], v1 = hcb[idx_of(q, qh, 2 * l + 1)];
          const unsigned m0 = (sg[l]) & 0xF, m1 = (sg[l] >> 4) & 0xF;
          for (int j = 0; j < 4; ++j) {
            const double w0 = (double)((v0 >> (8 * j)) & 0xF) * (((m0 >> j) & 1) ? -1.0 : 1.0);
            const double w1 = (double)((v1 >> (8 * j)) & 0xF) * (((m1 >> j) & 1) ? -1.0 : 1.0);
            acc += s * (w0 * (double)ag[8 * l + j] + w1 * (double)ag[8 * l + 4 + j]);
          }
        }
      }
    }
    return acc;
  };

  // ---- expansion
  const size_t magbytes  = (size_t)rows * G * 16;
  const size_t sgnbytes  = (size_t)rows * G * 4;
  const size_t scbytes   = (size_t)rows * nblk * 6;
  const size_t sc16bytes = (size_t)rows * G * 2;
  std::vector<unsigned char> hmags(magbytes), hsgn(sgnbytes), hsc(scbytes);
  std::vector<uint16_t> hsc16((size_t)rows * G);
  for (int r = 0; r < rows; ++r) {
    const unsigned char *prow = hraw.data() + (size_t)(r % base_rows) * (size_t)nblk * BLK;
    uint4 *mp = reinterpret_cast<uint4 *>(hmags.data()) + (size_t)r * G;
    unsigned *sp = reinterpret_cast<unsigned *>(hsgn.data()) + (size_t)r * G;
    unsigned char *scp = hsc.data() + (size_t)r * nblk * 6;
    uint16_t *s16 = hsc16.data() + (size_t)r * G;
    for (int b = 0; b < nblk; ++b) {
      const unsigned char *pb = prow + (size_t)b * BLK;
      memcpy(scp + (size_t)b * 6, pb + od, 2);
      memcpy(scp + (size_t)b * 6 + 2, pb + osc, 4);
      const double d = half2d(*reinterpret_cast<const uint16_t *>(pb + od));
      for (int sb = 0; sb < GPR; ++sb) {
        const unsigned char *q = pb + oqs + 8 * sb;
        const unsigned qh = pb[oqh + sb];
        unsigned char mag[GW];
        for (int l = 0; l < 4; ++l) {
          const unsigned v0 = hcb[idx_of(q, qh, 2 * l)], v1 = hcb[idx_of(q, qh, 2 * l + 1)];
          for (int j = 0; j < 4; ++j) {
            mag[8 * l + j]     = (unsigned char)((v0 >> (8 * j)) & 0xF);
            mag[8 * l + 4 + j] = (unsigned char)((v1 >> (8 * j)) & 0xF);
          }
        }
        uint4 w; unsigned *wc = reinterpret_cast<unsigned *>(&w);
        for (int k = 0; k < 4; ++k) {
          unsigned v = 0;
          for (int j = 0; j < 4; ++j)
            v |= (((unsigned)mag[8 * k + j]     & 0xFu) << (8 * j))
               | (((unsigned)mag[8 * k + 4 + j] & 0xFu) << 4 << (8 * j));
          wc[k] = v;
        }
        const int g = b * GPR + sb;
        mp[g] = w;
        sp[g] = (unsigned)pb[osg + 4 * sb] | ((unsigned)pb[osg + 4 * sb + 1] << 8)
              | ((unsigned)pb[osg + 4 * sb + 2] << 16) | ((unsigned)pb[osg + 4 * sb + 3] << 24);
        const unsigned sc = pb[osc + (sb >> 1)];
        const int n = (sb & 1) ? (int)(sc >> 4) : (int)(sc & 15);
        s16[g] = __half_as_ushort(__float2half((float)(d * (1.0 + 2.0 * (double)n))));
      }
    }
  }
  const size_t expb = magbytes + sgnbytes + scbytes, exp16b = magbytes + sgnbytes + sc16bytes;
  printf("expansion: %lld B -> %zu B = %.4f bpw, %.4fx of IQ3_S  (exp16: %zu B, %.4f bpw, %.4fx)\n",
         total, expb, 8.0 * (double)expb / ((double)rows * cols), (double)expb / (double)total,
         exp16b, 8.0 * (double)exp16b / ((double)rows * cols), (double)exp16b / (double)total);

  uint4 *dM = nullptr; unsigned *dS = nullptr; unsigned char *dSC = nullptr; __half *dS16 = nullptr;
  CHK(hipMalloc(&dM, magbytes)); CHK(hipMemcpy(dM, hmags.data(), magbytes, hipMemcpyHostToDevice));
  CHK(hipMalloc(&dS, sgnbytes)); CHK(hipMemcpy(dS, hsgn.data(), sgnbytes, hipMemcpyHostToDevice));
  CHK(hipMalloc(&dSC, scbytes)); CHK(hipMemcpy(dSC, hsc.data(), scbytes, hipMemcpyHostToDevice));
  CHK(hipMalloc(&dS16, sc16bytes));
  CHK(hipMemcpy(dS16, hsc16.data(), sc16bytes, hipMemcpyHostToDevice));

  // ---- correctness vs fp64 reference
  {
    std::vector<float> yf(NV), ye(NV), y16(NV);
    float *dyn = nullptr; CHK(hipMalloc(&dyn, (size_t)NV * 4));
    k_full<0, 0><<<grid, blk>>>(dW, dx, dyn, dout, NV, nblk, 1, 0);
    CHK(hipMemcpy(yf.data(), dyn, (size_t)NV * 4, hipMemcpyDeviceToHost));
    k_exp<0><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dyn, dout, NV, G, nblk, 1, 0);
    CHK(hipMemcpy(ye.data(), dyn, (size_t)NV * 4, hipMemcpyDeviceToHost));
    k_exp<1><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dyn, dout, NV, G, nblk, 1, 0);
    CHK(hipMemcpy(y16.data(), dyn, (size_t)NV * 4, hipMemcpyDeviceToHost));
    double wf = 0, we = 0, w16 = 0, wc = 0; int badf = 0, bade = 0;
    for (int r = 0; r < NV; ++r) {
      const double ref = ref_row(r);
      const double ef = fabs((double)yf[r] - ref) / (fabs(ref) + 1e-30);
      const double ee = fabs((double)ye[r] - ref) / (fabs(ref) + 1e-30);
      const double e16 = fabs((double)y16[r] - ref) / (fabs(ref) + 1e-30);
      const double cx = fabs((double)ye[r] - (double)yf[r]) / (fabs((double)yf[r]) + 1e-30);
      if (std::isfinite(ef) && ef > wf) wf = ef;
      if (std::isfinite(ee) && ee > we) we = ee;
      if (std::isfinite(e16) && e16 > w16) w16 = e16;
      if (std::isfinite(cx) && cx > wc) wc = cx;
      if (ef > 2e-3) badf++;
      if (ee > 2e-3) bade++;
    }
    printf("VERIFY %d rows: full vs fp64 %.3e | exp vs fp64 %.3e | exp16 vs fp64 %.3e | exp vs full %.3e"
           " | rows>2e-3 full %d exp %d\n", NV, wf, we, w16, wc, badf, bade);
    CHK(hipFree(dyn));
  }

  const int rotr_rows = rows / (rp > 0 ? rp : 1) + 1;
  const size_t rotr_flat = ((size_t)total / 16) / (rp > 0 ? rp : 1) + 1;


  struct Row { const char *name; double bytes, gbs; bool sum_ok, speed_ok, gated; int rp_used; };
  std::vector<Row> vrows;

  auto bench = [&](const char *name, double bytes, int which, unsigned long long expect, int rp_used) {
    hipEvent_t e0, e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
    std::vector<double> gb;
    unsigned long long got = 0;
    bool finite_ok = true;
    for (int rep = 0; rep < 5; ++rep) {
      float ms = 0; CHK(hipEventRecord(e0));
      switch (which) {
        case 0: k_flat<<<grid, blk>>>(reinterpret_cast<uint4 *>(dW), dout, (size_t)total / 16, rp_used, rotr_flat); break;
        case 9: k_flat<<<grid, blk>>>(reinterpret_cast<uint4 *>(dW), dout, (size_t)total / 16, 1, 0); break;
        case 1: k_read<0><<<grid, blk>>>(dW, dout, rows, nblk, rp_used, rotr_rows); break;
        case 2: k_full<0, 0><<<grid, blk>>>(dW, dx, dy, dout, rows, nblk, rp_used, rotr_rows); break;
        case 3: k_full<0, 1><<<grid, blk>>>(dW, dx, dy, dout, rows, nblk, rp_used, rotr_rows); break;
        case 4: k_exp<0><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dy, dout, rows, G, nblk, rp_used, rotr_rows); break;
        case 5: k_exp<1><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dy, dout, rows, G, nblk, rp_used, rotr_rows); break;
        case 6: k_full<0, 2><<<grid, blk>>>(dW, dx, dy, dout, rows, nblk, rp_used, rotr_rows); break;
        case 7: k_exp<2><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dy, dout, rows, G, nblk, rp_used, rotr_rows); break;
        case 8: k_exp<3><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dy, dout, rows, G, nblk, rp_used, rotr_rows); break;
        default: k_exp<0><<<grid, blk>>>(dM, dS, dSC, dS16, dx, dy, dout, rows, G, nblk, rp_used, rotr_rows); break;
      }
      CHK(hipGetLastError());
      CHK(hipEventRecord(e1)); CHK(hipEventSynchronize(e1));
      CHK(hipEventElapsedTime(&ms, e0, e1));
      if (rep == 0) {
        std::vector<unsigned long long> h(nthreads);
        CHK(hipMemcpy(h.data(), dout, nthreads * 8, hipMemcpyDeviceToHost));
        unsigned long long ssum = 0;
        for (auto v : h) ssum += v;
        got = ssum;
        if (rows > 0) {
          std::vector<float> yy(std::min(rows, 8));
          CHK(hipMemcpy(yy.data(), dy, yy.size() * 4, hipMemcpyDeviceToHost));
          for (auto v : yy) if (!std::isfinite(v)) finite_ok = false;
        }
      } else {
        gb.push_back(bytes * rp_used / (ms * 1e-3) / 1e9);
      }
    }
    std::sort(gb.begin(), gb.end());
    const double med = gb[gb.size() / 2];
    const bool gated = (expect != 0);
    const bool sum_ok = !gated ||
        (got == (unsigned long long)rp_used * (unsigned long long)REPL * expect);
    const bool speed_ok = med <= WALL * 1.02;
    Row r{name, bytes, med, sum_ok, speed_ok, gated, rp_used};
    vrows.push_back(r);
    CHK(hipEventDestroy(e0)); CHK(hipEventDestroy(e1));
    return r;
  };

  bench("wall_flat", (double)total, 0, exp_flat, rp);
  bench("read", (double)total, 1, exp_read, rp);
  bench("exp_read", (double)expb, 8, exp_expread, rp);
  bench("full", (double)total, 2, exp_dot, rp);
  bench("nogather", (double)total, 3, exp_dot_nogather, rp);
  bench("full_nosgn", (double)total, 6, exp_dot_nosign, rp);
  bench("exp", (double)expb, 4, exp_dot, rp);
  bench("exp16", (double)exp16b, 5, 0, rp);
  bench("wall_1pass", (double)total, 9, exp_flat, 1);
  bench("exp_nosgn", (double)expb, 7, exp_dot_nosign, rp);

  // ---- per-row validity table (the only place numbers are reported from)
  auto get = [&](const char *n) -> const Row * {
    for (auto &r : vrows) if (!strcmp(r.name, n)) return &r;
    return nullptr;
  };
  auto ven = [&](const Row &r) { return (r.sum_ok && r.speed_ok) ? "VALID" : "INVALID"; };
  printf("\n%-12s %-9s %-8s %10s %8s %6s\n", "mode", "checksum", "verdict", "GB/s", "%wall", "rp");
  for (auto &r : vrows)
    printf("%-12s %-9s %-8s %10.1f %7.1f%% %6d  %s\n", r.name,
           r.gated ? (r.sum_ok ? "match" : "MISMATCH") : "ungated", ven(r), r.gbs,
           100.0 * r.gbs / WALL, r.rp_used, (r.sum_ok && r.speed_ok) ? "" : "<-- DISCARDED");

  int nvalid = 0;
  for (auto &r : vrows) if (r.sum_ok && r.speed_ok) nvalid++;
  printf("\nvalid rows: %d of %zu\n", nvalid, vrows.size());

  auto ok = [&](const char *n) { const Row *r = get(n); return r && r->sum_ok && r->speed_ok; };
  auto bw = [&](const char *n) { const Row *r = get(n); return r ? r->gbs : 0.0; };

  printf("\n-- ablations, each computed only from rows that passed the gate --\n");
  if (ok("full") && ok("nogather"))
    printf("ABL gather      (full - nogather)   = %+6.1f GB/s  (%+5.1f%% of the read wall)\n",
           bw("full") - bw("nogather"), 100.0 * (bw("full") - bw("nogather")) / WALL);
  else printf("ABL gather      UNAVAILABLE (a row failed the gate)\n");
  if (ok("full") && ok("full_nosgn"))
    printf("ABL sign        (full_nosgn - full) = %+6.1f GB/s  (%+5.1f%% of the read wall)\n",
           bw("full_nosgn") - bw("full"), 100.0 * (bw("full_nosgn") - bw("full")) / WALL);
  else printf("ABL sign        UNAVAILABLE\n");
  if (ok("read") && ok("full_nosgn"))
    printf("ABL index+dot   (read - full_nosgn) = %+6.1f GB/s  (%+5.1f%% of the read wall)\n",
           bw("read") - bw("full_nosgn"), 100.0 * (bw("read") - bw("full_nosgn")) / WALL);
  else printf("ABL index+dot   UNAVAILABLE\n");
  if (ok("exp") && ok("exp_nosgn"))
    printf("ABL sign (exp)  (exp_nosgn - exp)   = %+6.1f GB/s  (%+5.1f%% of the read wall)\n",
           bw("exp_nosgn") - bw("exp"), 100.0 * (bw("exp_nosgn") - bw("exp")) / WALL);
  else printf("ABL sign (exp)  UNAVAILABLE\n");
  if (ok("full") && ok("read"))
    printf("ABL dequant+dot (read - full)       = %+6.1f GB/s  (%+5.1f%% of the read wall)\n",
           bw("read") - bw("full"), 100.0 * (bw("read") - bw("full")) / WALL);

  // ---- file-level consequence, from measured rows only
  const double IQ_GiB = 3.393, REST_GiB = 7.560;        // measured type histogram, this file
  const double REST_BW = 752.0;                          // 4-bit dequant-GEMV class (WALL doc)
  auto tok = [&](double bw_iq, double ratio) {
    const double ti = IQ_GiB * ratio * 1073741824.0 / (bw_iq * 1e9);
    const double tr = REST_GiB * 1073741824.0 / (REST_BW * 1e9);
    return 1.0 / (ti + tr);
  };
  const double r_ratio = (double)expb / (double)total;
  printf("\n-- whole-file estimate (IQ3_S %.3f GiB measured, rest %.3f GiB at %.0f GB/s) --\n",
         IQ_GiB, REST_GiB, REST_BW);
  if (ok("full"))     printf("EST serve IQ3_S as-is            : %.1f tok/s\n", tok(bw("full"), 1.0));
  if (ok("full_nosgn")) printf("EST signs at fixed cost          : %.1f tok/s\n", tok(bw("full_nosgn"), 1.0));
  if (ok("exp"))      printf("EST lossless expansion (%.4fx B)  : %.1f tok/s\n", r_ratio, tok(bw("exp"), r_ratio));
  if (ok("exp_nosgn")) printf("EST 4.1875 bpw ideal (%.4fx B, lossy): %.1f tok/s\n",
                              134.0 / 110.0, tok(bw("exp_nosgn"), 134.0 / 110.0));
  if (ok("full") && ok("exp")) {
    const double ratio = r_ratio;
    printf("BREAKEVEN expansion needs %.1f GB/s (%.2fx of measured full), read wall %.0f -> headroom %+.1f%%\n",
           bw("full") * ratio, ratio, WALL, 100.0 * (WALL / (bw("full") * ratio) - 1.0));
  }
  return 0;
}
