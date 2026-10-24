# 00 — Measured facts

gfx1100 (`AMD Radeon RX 7900 XT`), PCI `07:00.0`, host `rasputin`. Every figure is measured
2026-10-22 and attributed to its source. Re-run the probe if card, driver or ROCm changes.

## Wall

| Quantity | Measured | Source |
|---|---|---|
| **Streaming read, one card** | **776 GB/s** (max 777.4, verified twice) | `bench/gfx1100-probes/WALL-2026-10-22.md` |
| Copy, read+write counted | 618 GB/s (non-temporal store 658) | same |
| Cross-GPU copy / host↔GPU | 14.2 GiB/s | same; `bench/00_hardware_probe.py` |
| CPU DRAM copy | 46.4 GiB/s | `bench/00_hardware_probe.py` |
| NVMe read, 1 MB / 4 MB / 4 streams | 3.90 / 4.44 / 5.59 GiB/s | same |

**776 GB/s is the wall; every ceiling in this tree divides by it.** Two independent runs,
more than one load in flight per lane. A 1.6 ms kernel with a serial accumulator and no
unrolled loads reports ~619 GB/s — a kernel artefact, retired, and nothing here derives
from it. Load width per lane is worth 0 % (4, 8 and 16 B all land
at 763–767), occupancy little once ILP is present, slab-per-block addressing costs 7 %.
Kernel *duration* is not the variable: a 0.36 ms launch reaches 770 GB/s
(`E2-LAUNCH-2026-10-22.md`).

Cross-GPU copy equals host-bounce speed: **no peer DMA**, despite `can_device_access_peer`
returning true. Two-card work is two host-fed streams, not a fabric.

Compute: BF16 GEMM at 2048x5120x17408 runs **4.23 ms = 86.4 TFLOP/s**, the same shape at M=1
does **472 GB/s (61 % of the wall)**. The FLOPs exist, so batch-1 is a bandwidth path and 472
is a kernel-quality number, not a wall. FP8 E4M3 matmul fails under torch
(`normal_kernel_cuda` not implemented for Float8); FP8 is excluded
(`docs/adr/0004-exclude-fp8.md`).

## Instructions

`bench/gfx1100-probes/probe.cpp`, `results.json`. sclk 2802 MHz sustained, median of 5 runs,
2^24 iterations per lane; peak 5376 lanes x 2.802 GHz = 15.06 T lane-ops/s.

| Instruction | MAC/instr | Ginstr/s | GMAC/s | Verdict |
|---|---|---|---|---|
| `v_dot2_f32_f16` | 2 | 12120 | 24239 | full |
| `v_fma_f32` | 1 | 12215 | 12215 | full |
| `v_dot4_i32_i8` | 4 | 7228 | 28913 | half |
| `v_dot8_i32_i4` | 8 | 3837 | 30692 | quarter |

- **5376 lanes (84 CU x 64), not 10752** — any estimate on 84 x 128 is 2x too optimistic.
- **`__builtin_amdgcn_sdot4` does not compile for gfx1100.** It needs `dot1-insts`, which the
  subtarget does not advertise even though `llvm-mc` assembles the instruction, so
  `v_dot4_i32_i8` is reachable **through inline asm only**; `sdot2`/`sudot8` fail too.
- **`v_pk_fma_f32` does not exist on gfx11** — the baseline is scalar `v_fma_f32`.
- **A quantised GEMV at the wall costs ~1.8 T MAC/s against 24 T available**: single-digit
  issue occupancy, so the work is layout, coalescing, reading each byte once.

## The file

`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` / `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`, parsed by
`bench/02_gguf_probe.py` and `bench/03_tensor_inventory.py`: **11,771,546,784 B (10.963 GiB),
851 tensors, 26.896 B params, 3.498 bits per weight, 11 quant types**, GGUF v3, `arch=qwen35`,
52 metadata keys. Metadata region plus every tensor's bytes equals the file size exactly,
with no gap and no overlap.

MTP sibling `...-mtp.gguf`: 12,120,016,960 B (+15 tensors, +0.325 GiB). Layer 64 is
full-attention plus four `nextn.*` tensors, **every matrix Q6_K**, and it **shares
`output.weight`** — no second LM head. All 851 base tensors identical in type, shape, size.

Bytes by type: IQ3_S 31.0 %, IQ4_XS 25.7 %, IQ3_XXS 17.6 %, Q4_K 13.4 %, IQ2_S 6.8 % — those
five cover 94.5 % — then Q2_K 2.0 %, IQ2_XS 1.9 %, IQ2_XXS 0.8 %, BF16 0.4 %, IQ1_M 0.2 %,
F32 0.1 %. **Exactly one tensor is IQ1_M**: `blk.13.ffn_gate`, type id 29, 1.75 bpw, 19.5 MB,
so dispatch is **per tensor, never per file**; `general.file_type` is only an average.

Bytes by role (`docs/plan/TENSOR-INVENTORY.md` §4): MLP 62.90 %, GDN projections 20.99 %,
embed + head 9.54 %, full attention 6.07 %, GDN state ops 0.47 %, norms 0.02 % —
**10.57 GiB read per token**, everything but `token_embd`.

## Per-sequence state

KV is 64 KiB/token bf16 (16 layers x 4 kv heads x 256 x 2 x 2 B), 32 KiB at q8: 2 / 8 / 16 GiB
at 32K / 128K / 256K context. GatedDeltaNet state is 151 MB FP32 + 8 MB conv = **159 MB per
sequence, fixed and not shareable**. Only 16 of 64 layers keep KV, so context is cheap and
the 159 MB state is the real concurrency cost.

## Kernel results

Real IQ3_S tensors, read-only, checksum-gated (order-independent sum replayed from the same
bytes; above the wall is invalid). `bench/gfx1100-probes/E1-IQ3S-2026-10-22.md`.

| Mode | GB/s | % of wall |
|---|---|---|
| Read-only, real block addressing | **717.7** | 92.5% |
| Dequant + dot (the real kernel) | **516.4** | 66.5% |
| Sign decode removed | 608.2 | 78.4% |
| Codebook gather removed | 493.5 | 63.6% |
| Lossless 4-bit expansion, 1.509× bytes | 618.1 | 79.7 % (needs 100.4 % to break even) |

The codebook gather is **free** (removing it lowers throughput), the sign bit costs **~12 % of
the wall**, and the lossless expansion is **rejected**.

## Dispatch and fusion

A 401-launch decode step of 11.2955 GB. `bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md`.

| Quantity | Measured |
|---|---|
| Marginal launch cost | **2.5–2.9 us** — same as an empty kernel |
| Footprint cliff | **~800 MiB per launch** → ~230 us, clock-independent, cause unproven |
| Per-tensor step | **712.3 GB/s, 60.6 tok/s** |
| 2–4 layer fusion | **774.2 GB/s, 65.8 tok/s** (99.7 % of wall) |
| 8-layer fusion | 691.4 GB/s — worse than per-tensor |

There is no per-launch bandwidth penalty in steady state; the deficit is `401 x 3.3 us =
1.31 ms`. **Fuse 2–4 layers, never 8** — 8-layer launches cross the footprint cliff.

## Clocks and toolchain

1249 MHz is the top memory step, idle 96 MHz; after an idle GPU the first one to three launches
carry a **~3 ms wake-up transient** (2.13 ms → 0.73 ms with clocks pinned) that
`power_dpm_force_performance_level=high` deletes. At idle clocks you measure the ramp, not the kernel.

ROCm 10.1 ships as `amdrocm-*` under `/opt/rocm/core-10.1`; hipcc 7.16, clang 24. Required:
`amdrocm-runtime-dev10.1`, `amdrocm-core-dev10.1`, `amdrocm-blas-dev10.1` — without them there
is no `hip/hip_runtime.h` and no `hip-lang-config.cmake`, and every HIP build fails; `core-dev`
also supplies `thrust/` and `hipcub/`, without which torch's own headers do not compile. torch
2.14.1+rocm7.2 (HIP 7.2.53211) skew is fine for `load_inline --offload-arch=gfx1100`; sglang
0.5.9 imports and `sgl_kernel` 0.4.10 builds for gfx1100.

## What shapes the engine

1. **Decode is one 10.57 GiB read** — 14.6 ms ≈ 68 tok/s at the wall, 60.6 per-tensor, 65.8
   fused. Two levers only: fewer bytes, or more bytes per second.
2. **The kernel gate is 0.6 x 776 = 466 GB/s**; the real dequant-dot kernel clears it at 516.4.
3. **Fuse 2–4 layers, never 8** — 8-layer launches (~1.3 GB) cross the footprint cliff.
4. **Per-tensor quant dispatch is mandatory** — eleven types, one of them on a single tensor.
5. **Two cards are two memory spaces joined by a slow link** — cross-GPU copy is 14.2 GiB/s,
   host-bounce speed. Any topology has to earn its PCIe cost; TP=2 versus two replicas is
   **open** (`docs/plan/REQUIREMENTS.md` A4), with conflicting measurements.
