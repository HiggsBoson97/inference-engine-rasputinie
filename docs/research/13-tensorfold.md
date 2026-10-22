# TensorFold — 7a90efacb351ce6442da2823d6df1e9237a90f29 | Apache-2.0 | Zig (2860 files), Metal MSL (303), CUDA C++ (136), HIP C++ (46)

## What it is

A shipped engine (v1.1.2) for Apple Silicon and NVIDIA — with an **opt-in HIP backend that already targets gfx1100 and already serves Qwen3.8-27B**. Three hand-written kernel families behind one scheduler, no Python in decode (`ZIG-PREVIEW.md:3`): ~6,100 lines of HIP C++, a Zig host layer, committed tuning tables for gfx1030 and gfx1100.

## The "fold": the name is a brand, not an algorithm

There is no folding transform in the hot path. The word appears three ways: a **LoRA fold at merge time** (`core/affine4_fold.zig:1` — a delta folded into MLX 4-bit codes on each group's grid, rounding error carried as in GPTQ; `:56`, reached by `tensorfold merge`, `CHANGELOG.md:218`); **"leading dimensions folded"** in layout helpers (`core/quant/mlx.zig:202`); and **"the fold gradient"**, a UI gradient (`app/README.md:5`).

The idea the engine is built around is **lanes**: speculative rows (drafts, trees, context copies) verified in one forward with a byte-identical guarantee (`ZIG-PREVIEW.md:11`). The name does not describe it.

## Architecture in one text diagram

```
CLI/server (Zig)                      ← OpenAI API, prompt cache, cluster/tp
  └─ core/lanes ── be.Backend vtable (34 entries, 8 required)   core/lanes/backend.zig:82
        ├─ families/qwen3_5/backend.zig:54          (Metal)
        ├─ families/qwen3_5/cuda_native.zig:302     (CUDA)
        └─ backend/hip/engine/hip_lanes.zig:156     (HIP → gfx1100)
              └─ hip/ops/Ops  typed launches, arena, prefill|window   hip/ops/ops.zig:24
                    └─ core/registry.zig  cheapest entry that fits
                          └─ hip/tuning/gfx1100.zon   build-time costs
                                └─ kernels/hip/{tiles,quant,ops,attention,decode}
```

## Kernel design (b): what batch-1 decode actually does

**Quant.** On HIP, MLX affine only: `w = code*scale + bias` per group of K, `BITS ∈ {2,4,6,8}`, tables bf16/fp16/f32 (`quant/mlx_decoder.hpp:23`). No GGUF k-quants in `src/hip/quant/`.

**Layout is what makes GEMV work.** Weights are **column-major** — one output column's whole K row of codes contiguous at `words + col*(k*BITS/32)` (`mlx_decoder.hpp:35`, `tiles/stream.hpp:80`). A lane streams one column top to bottom: no transpose, no staging, no LDS.

**The decode tile** (`stream.hpp`): "1 to 8 rows: a lane owns a 32-code chunk of a column; no LDS, lanes fold by shuffle" (`:3`). Constraint `CB columns × R rows ≤ 16` accumulators (`:18`); K splits across a wave's lanes and partials fold with `__shfl_xor` over the lane group only (`:193`). At `R == 1` loader and dot loop are software-pipelined (`:168-178`), and each column starts its K-sweep at a different round, `rot = (col0>>3) % rounds` (`:161`), so a wave's columns don't all hit one DRAM channel together.

**Arithmetic.** 4-bit codes become fp16 pairs by OR-ing `0x6400` into each half and subtracting — two ALU ops, exact below 256 (`mlx_decoder.hpp:67,175`). The product is `v_dot2c_f32_f16_dpp` with `quad_perm`: **one dot2 instruction computes four quads' partial dots** by rotating x within the quad (`common/dot2.hpp:36`). bf16 dot2 has no DPP form, so x moves with `mov_dpp` first (`:63`).

**The GEMV/GEMM boundary is a table, not a kernel heuristic.** On gfx1100 the stream tile takes ≤16 rows (`rows_matrix 15`), WMMA ≥16 (`common/wmma.hpp:25` — `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`, no rocWMMA header), plain GEMM ≥9, K-parallel tiles cover 2–32-row prompts (`hip/tuning/gfx1100.zon:14-25`, `tiles/gemm_kp.hpp:3`). `arch.hpp:21` sets `kWmma = true` for gfx11 — consistent with wave32 WMMA and no wgmma.

## Techniques worth stealing

| Technique | Where (file:line) | What it buys on gfx1100 | Effort | Verdict |
|---|---|---|---|---|
| Column-major weights + `stream` GEMV tile (lane owns a 32-code column chunk, no LDS, shuffle-only reduce) | `kernels/hip/tiles/stream.hpp:3,18,80,193` | The decode ceiling: 592 GiB/s needs each byte read once, unstaged | M | **TRANSFERS** — already compiles for gfx1100 |
| `v_dot2c_f32_f16_dpp` quad_perm, 4 partial dots/instruction | `common/dot2.hpp:33-43` | Keeps 4-bit GEMV compute-light at 1 row | S | **TRANSFERS** |
| Code→fp16 by `\|0x6400` + `hsub2`; bf16 high-half trick | `quant/mlx_decoder.hpp:67,175` | 2 ALU ops per pair, no lookup | S | **TRANSFERS** |
| Double-buffered fetch/work at R==1 | `tiles/stream.hpp:168-178` | Hides load latency, no extra registers | S | **TRANSFERS-WITH-WORK** — retune for 592 vs 448 GiB/s |
| Per-column staggered K-sweep | `tiles/stream.hpp:161` | Spreads DRAM channel pressure; free | S | **TRANSFERS** |
| Deterministic cost-table registry (build-time `.zon`, cheapest-that-fits) | `core/registry.zig:1`, `core/tuning.zig:1`, `hip/tuning/gfx1100.zon:14` | Ranks pick identical kernels — precondition for tp | S | **TRANSFERS** |
| Bump arena reset per forward + **counting arena** dry run | `hip/arena.zig:1,21,31,47` | No allocation per round; stable addresses make replay legal; peak measured | S | **TRANSFERS** |
| HIP graph capture per round shape | `hip/graph.zig:12-20` | Dozens of launches/token → one replay | M | **TRANSFERS** (done there) |
| `be.Backend` vtable seam | `core/lanes/backend.zig:82` | One scheduler, three kernel families; 8 required entries, the rest opt-in | S | **TRANSFERS** as a shape |
| MLX affine 2/4/6/8-bit decoders | `quant/mlx_decoder.hpp:18` | — | M | **DOES-NOT-TRANSFER** as-is: Q4_K_XL needs a new `Dec` policy |

## How much is Zig-specific (c)

Almost none of the kernel work. `stream.hpp`, `dot2.hpp`, `mlx_decoder.hpp`, `gemm_kp.hpp` are plain C++ compiled by hipcc — no Zig, no Metal. Zig lives in arg marshalling (`hip/launches/`, `hip/args.zig`), registry plumbing, and the scheduler: **the kernels are a copy, the host layer is a rewrite.** Apache-2.0 bounds it — carry `LICENSE` + `NOTICE`; kernel files have no per-file SPDX header, so attribution is repo-level. Worth it because the decisions I would otherwise discover empirically are already encoded and gated on `gfx1100`.

## Metal vs CUDA (d): one trait, three bodies

The abstraction is real and worth copying: `be.Backend` is `{ ptr, vtable }` — 34 entries, 8 of them required (`core/lanes/backend.zig:82`) — and there are 30 non-test implementations of it: Metal, CUDA, HIP, XPU (Level Zero), a cluster leader, plus fakes and recorders the lane core's own tests run against. Kernel *choice* is portable too — `core/registry.zig` is backend-agnostic and reads per-GPU tables at build time (`core/tuning.zig:1`).

The bodies are not: 303 MSL files with `simdgroup`/`simdgroup_multiply_accumulate` (151 call sites) versus 136 CUDA `.cu` versus the HIP tree. **One seam, three implementations.** Copy the seam; ignore the MSL and CUDA bodies.

## Benchmarks (e) and whether the method holds up

Method (`server/benchmark_suite.zig:11,126,143`): own loopback server, prompt reuse off, one warm-up per cell, five repeats, median published **only if every declared repeat succeeded**. Receipts carry hardware identity, checkpoint SHA and per-sample timings.

Receipts I could read — gfx1151, Radeon 8060S, 40 CU, unified memory, **not** gfx1100 and not a 7900 XT: Qwen3.8-27B MLX 4-bit, 20 samples, **12.87 tok/s** median decode; Qwen3.6-35B-A3B **78.5 tok/s** drafting, **68.8** serial. gfx1100 figures exist only as prose (`CHANGELOG.md:343`, Radeon PRO W7800 — 60 CU, 448 GB/s): 35B-A3B 176.5 tok/s code / 140.8 chat; `:345` 372.6 tok/s on two W7800s vs 283.7 on one.

Scaling 12.87 from ~256 GB/s to 592 GB/s gives ≈30 tok/s for 27B — near the 35 tok/s ceiling the Q4_K_XL arithmetic predicts, so decode is plausibly bandwidth-bound. Weaknesses: 14-token prompts and 256 output tokens barely exercise prefill, the gfx1100 figures are unattested self-reports, and ROCm board publication is disabled (`CHANGELOG.md:97`).

## Top 3 to port first

**1. Column-major layout + the `stream` GEMV tile.** 27B decode is 16.35 GiB of reads per token against 592 GiB/s; nothing else matters next to issuing those reads once, contiguously, unstaged. ~235 lines that already compile for gfx1100. First step: port `mlx_decoder.hpp` + `stream.hpp` + `dot2.hpp` into one HIP TU, feed it one projection (n=5120, k=5120, 4-bit g=64) with a seeded pattern, check against float64 *before* timing — the repo's own rule (`CHANGELOG.md:341`).

**2. The dot2-DPP + code-decode pair.** With no tensor cores and fp8 needing widening, instruction density is the only compute lever left in a GEMV: `quad_perm` gives 4 partial dots per `v_dot2c`, the `0x6400` trick 2 ALU ops per code pair. First step: microbenchmark one wave against constant codes at 1 row; if throughput is the limiter, this is the fix.

**3. Registry + arena pair.** `arena.zig:21`'s counting arena *measures* peak scratch by dry-running the forward — "measure before and after" as machinery — and the cheapest-that-fits table stops two ranks on a no-P2P host-bounce split from choosing different kernels. First step: build both with three cost entries (stream/wmma/gemm) before a second kernel exists.

## What does NOT transfer and why

- **Metal `simdgroup`/`simdgroup_matrix`** (303 MSL files, 151 matmul sites, e.g. `kernels/metal/core/tile_qmv.metal:1`): no AMD equivalent and none needed — `common/wmma.hpp:25` does the job with a gfx11 builtin.
- **wgmma / TMA / clusters / DSMEM**: absent from the HIP path by construction; the design never assumed them, which is why it ports. **CUDA graphs**: fine — `hip/graph.zig:12`.
- **cuBLASLt / CK / NCCL / Triton AOT** (`cuda/{cublaslt,ck,nccl,triton}.zig`): AMD counterparts exist (hipBLASLt, CK, RCCL at `hip/comm/rccl.zig`) — a swap, not a port.
- **EXL3/EXLR8 trellis, NVFP4, MXFP8** (`cuda/{exl3,nvfp4,mxfp8}.zig`): NVIDIA-format kernels with no HIP decoder.
- **MLX affine formats**: we need Q4_K_XL; the `Dec` interface takes a new decoder, shipped ones are affine-only.
- **Metal residency sets, ANE, ICBs, the 1637-file TS app**: Apple-only. **DFlash2 drafter on HIP**: refused (`CHANGELOG.md:99`).

## Claims I could not verify

- Any gfx1100 number for **Qwen3.8-27B**: the 27B receipt is gfx1151; W7800 figures are prose with no receipts. README speed tables (277 tok/s, M5 Max) likewise — I ran nothing.
- That *every* launch path consults `registry.select`; and whether MSL `tile_qmv.metal` and HIP `stream.hpp` really emit identical bytes (called twins, not checked).
- Byte-identity across row counts: asserted at `CHANGELOG.md:341`, checked by `hip-kernel-test`, which I did not run.
- Whether HIP ships in release archives: `CHANGELOG.md:340` says nothing HIP is in them, while `build/hip.zig:122` defines a `native-hip` install step.

## Source map

- `core/affine4_fold.zig:1,56` · `core/quant/mlx.zig:202` · `app/README.md:5` — the three "folds"
- `ZIG-PREVIEW.md:3,11` — native engine, lanes exactness rule
- `core/lanes/backend.zig:82` + impls `families/qwen3_5/backend.zig:54`, `qwen3_5/cuda_native.zig:302`, `backend/hip/engine/hip_lanes.zig:156`
- `core/registry.zig:1` · `core/tuning.zig:1,25` · `hip/tuning/gfx1100.zon:14-25`
- `hip/ops/ops.zig:24` · `hip/ops/project.zig:34` — launch surface, format dispatch
- `kernels/hip/tiles/stream.hpp:3,18,59,80,96,161,168,193` — decode GEMV tile
- `kernels/hip/common/dot2.hpp:31,36,57,63` · `common/wmma.hpp:25` · `common/arch.hpp:21`
- `kernels/hip/quant/mlx_decoder.hpp:23,35,67,121,144,175` — affine decoder, column stride, bit tricks
- `kernels/hip/tiles/gemm_kp.hpp:3` · `decode/decode.hip:34,105` — K-parallel GEMM; router, merged tail+norm
- `hip/arena.zig:1,21,31,47` · `hip/admission.zig:50` · `hip/graph.zig:12-20` · `hip/comm/rccl.zig`
- `cuda/{ck,exl3,nvfp4,mxfp8,triton}.zig` — NVIDIA-only territory
- `server/benchmark_suite.zig:11,126,143` · `docs/receipts/rocm-host-identity/gpu-run/27b-auto.json`
- `CHANGELOG.md:97,99,218,340,341,343,345,352`
