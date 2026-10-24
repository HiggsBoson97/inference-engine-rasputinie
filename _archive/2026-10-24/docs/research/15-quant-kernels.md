# Quantised GEMV/GEMM on gfx1100 — tensorfold `7a90efa`, exllamav3 `da68d57`, FreeToken `546c5fe`

Base: 2× RX 7900 XT (gfx1100, 84 CU, wave32, 64 KiB LDS/CU), 19.98 GiB, 592 GiB/s. 27B byte counts come from
parsing the real GGUF (`unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`: 17,559,178,144 B, 866 tensors,
50 KV, `architecture=qwen35`, `block_count=65`).

## Excluded sources

llama.cpp is excluded by owner decision: its CUDA quantisation kernels and anything derived from them may not be
read, cited, or used as a baseline. We therefore own a GGUF reader and a dequant+GEMV family written from the
format specification, with no upstream C reference for that kernel class. Two files that look usable are not:
FreeToken vendors a copy of that excluded tree under `python/freetoken/kernel/csrc/ggml/`, and TensorFold's
`zig/kernels/cuda/ggml.cu` self-attributes its tables to the same origin. Neither is cited here.

## The kernel budget

One token reads 16.353 GiB → **27.6 ms → 36.2 tok/s** ceiling at 592 GiB/s; rocBLAS at 410 GiB/s gives 25.1
tok/s, so GEMV headroom is ~1.17×, not 1.44×.

| group (by tensor span) | GiB | % | us @592 | calls |
|---|---|---|---|---|
| ffn_down 17408×5120 | 3.474 | 21.2 | 5868 | 65 |
| ffn_up 5120×17408 | 3.355 | 20.5 | 5667 | 65 |
| ffn_gate 5120×17408 | 3.263 | 20.0 | 5512 | 65 |
| attn_qkv 5120×10240 | 1.456 | 8.9 | 2460 | 48 |
| ssm_out 6144×5120 | 1.025 | 6.3 | 1731 | 48 |
| output 5120×248320 | 0.971 | 5.9 | 1641 | 1 |
| attn_gate 5120×6144 | 0.909 | 5.6 | 1536 | 48 |
| attn_q 5120×12288 | 0.630 | 3.9 | 1065 | 17 |
| attn_output 6144×5120 | 0.373 | 2.3 | 630 | 17 |
| attn_k/attn_v 5120×1024 | 0.147 | 0.9 | 249 | 34 |
| embedding gather (not GEMV) | 0.666 | 4.1 | 1125 | 1 |

The three FFN GEMVs are **61.7%** of every token, in two shapes: K=5120/N=17408 and its transpose; one
1×5120×17408 call at Q5_K is 58.4 MiB = 96 us at peak. Batch sizes: **1** (decode, ~70% of wall time), **2–8**
(MTP verify, still GEMV), **≥64** (prefill, where a WMMA GEMM wins). The Flash-Next expert shape 1×2560×640 at
IQ3_S is 688 KiB = 1.11 us, below launch overhead — a batching problem, not a kernel problem.

## Fused dequant + dot anatomy

Every type is a 256-weight superblock (IQ4_NL: 32) whose dequant is *integer code → affine map → fp32*, so the
dot consumes integers and scales apply after accumulation (`third_party/gguf/quants.py`):

- `Q4_K` (`:476-523`), `Q5_K` (`:526-550`): `w = d·sc[j]·q − dmin·m[j]`, 8 sub-blocks of 32, scales and mins
  packed into 12 six-bit fields (`:480-502`). **Two accumulators**, because `−dmin·m` multiplies a constant by
  `Σx`. Q5_K adds a 32-byte 5th-bit plane. `Q6_K` (`:553-573`): symmetric 6-bit, `q−32` — one accumulator plus
  `Σx` to fold the −32.
- `IQ4_XS` (`:1352-1378`): 4-bit **codes into a 16-entry non-uniform LUT**, one signed scale per 32, no min →
  one accumulator. `IQ3_S` (`:1058-1100`): 3-bit indices into a 512×4 grid (2 KiB) — not affine, a gather per
  weight.

Block sizes (`constants.py:6222-6235`) give the bpw in the table below. Activations are Q8_1, 36 B per 32
values: one fp16 `d`, **two fp16 per-16 sums**, 32 int8 — the sums are why affine types are cheap, since `Σx` is
computed once per activation group rather than per weight.

The gfx1100 machine form is exllamav3's RDNA GEMV (`rocm/quant/exl3_gemv_kernel_rdna.cuh:1-13`): one warp per
16-wide output tile, 16 active lanes, B staged quantised through LDS, dequantised in registers, unswizzled to
row-major, consumed by `V_DOT2_F32_F16`. Two findings transfer as-is: A is read from LDS as a **broadcast** so
all 16 lanes see the same k (`:176-197`), and splitting K across lane halves "breaks this broadcast and regresses
kernel time" (`:170-175`); LDS columns pad to 18 halves against bank conflicts (`:27-30`).

| NVIDIA-only construct | where | gfx1100 |
|---|---|---|
| `mma.sync.m16n8k16` | `ptx.cuh:73-125` | WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`; other fragment layout → rewrite, not shim (TRANSFERS-WITH-WORK) |
| `ldmatrix.x4` | `ptx.cuh:334-343` | none; `ds_read_b128` + manual swizzle (TRANSFERS-WITH-WORK) |
| `cp.async.cg` | `ptx.cuh:261-332` | none at this width; plain loads, overlap lost (TRANSFERS-WITH-WORK) |
| `__dp4a` | `codebook.cuh:94` | `v_dot4_i32_i8` via `__builtin_amdgcn_sudot4`, gfx11+ (TRANSFERS) |
| `__shfl_xor_sync` | `hadamard_inner.cuh:28-82` | `__shfl_xor` → `ds_bpermute` (TRANSFERS) |
| `grid.sync()` | `exl3_gemv_kernel.cuh:24` | no grid barrier; device counter + resident blocks (TRANSFERS-WITH-WORK) |
| `lop3.b32`, `__hfma2`, `__ldcs` | `codebook.cuh:10,51` | `v_lop3_32`, native half2, `__ldcs` (TRANSFERS) |

## Quant types actually present in the 27B UD-Q4_K_XL file

The name misleads: no Q4_K in most of the MLP, no IQ2/IQ3_XXS at all. Measured histogram, type ids via
`constants.py:6007-6022`:

| type | tensors | GiB | % | bpw | per 32 weights in a matvec |
|---|---|---|---|---|---|
| Q5_K | 191 | 7.382 | 45.1 | 5.5 | 16 B qs + 4 B qh + 2 B scale, 8 dots, 2 acc |
| IQ4_XS | 70 | 2.910 | 17.8 | 4.25 | 16 B qs, 8 LUT gathers, 1 acc |
| Q4_K | 69 | 2.850 | 17.4 | 4.5 | 16 B qs + 2 B scale, 8 dots, 2 acc |
| Q6_K | 56 | 2.666 | 16.3 | 6.5625 | 16 B ql + 8 B qh, 8 dots, 2 acc |
| IQ4_NL | 24 | 0.261 | 1.6 | 4.5 | 32-wide blocks, LUT, 1 acc |
| Q8_0 | 110 | 0.122 | 0.7 | 8.5 | 8 dots |
| Q3_K | 3 | 0.107 | 0.7 | 3.4375 | 3-bit + 6-bit scales, 2 acc |
| IQ3_S | 1 | 0.036 | 0.2 | 3.4375 | 2 KiB grid gather, ALU-hostile |
| F32 | 360 | 0.020 | 0.1 | — | norms, `ssm_a`, `ssm_dt.bias` |

The k-quant trio is **79%** of the file, and cost is not proportional to bpw: Q5_K and Q4_K cost the same
instruction count, IQ4_XS is cheaper in ALU than Q4_K, Q6_K is 19% more bytes for the same count. Choose type per
tensor on bytes alone — except IQ3_S, which goes last.

## EXL3 trellis

Not a GGUF type and not a k-quant re-derivation. The unit is a **256-weight tile that is a tail-biting ring of
16-bit sliding windows**: weight `t` decodes from the 16 bits ending at its code field, index wrapped mod
`bits*256/32` (`exl3_dq.cuh:18-29`), so neighbours share `16−K` bits and "8 weights per 16-bit word" holds only
at 2.0 bpw. bpw = K exactly: 2.0 → 64 B/tile, 2.5 → 80 B, 3.0 → 96 B, 3.5 → 112 B (`exl3_dq.cuh:25`;
half-steps are mul1-only, `bits_k.cuh:17-18`). Scaling is rank-1, per input channel (`suh`) and per output
channel (`svh`) (`quantize.py:1850-1863`), and `exl3_dq.cuh` applies no scale at all — hence 3.0 bpw being
exactly 3.000 bits. The codebook is **procedural, three ALU ops** per index (`codebook.cuh:66-100`): a
65536-entry fp16 LUT would be 128 KiB against 64 KiB of LDS (`arch.cuh:33-37`), and only the encoder builds it
(`quantize.cu:74`). Decode is `exl3_gemv_kernel` (`exl3_gemv.cu:112-122`, gated on `size_m≤8`, `size_k%128==0`);
prefill is `exl3_gemm_kernel` with a 3-stage `cp.async` pipeline and `ldmatrix` (`exl3_gemm_inner.cuh:254-917`).
A Hadamard-128 is fused into both ends of every call (`exl3_gemv_kernel.cuh:218-234`) at O(m·k)+O(m·n) against
an O(m·k·n) GEMM, but with two grid barriers — the barriers, not the arithmetic, are the price.

**Port effort: high; it is a rewrite, not a hipify** — "This is NOT a port … It has no RDNA equivalent"
(`exl3_gemv_kernel_rdna.cuh:7-9`), "the fragment layouts do not correspond … could not be done by shimming"
(`exl3_gemm_inner_rdna.cuh:14-30`). An RDNA path already exists in-tree (~5,300 lines under
`rocm/quant/exl3_gemv*_rdna.*`): re-derive from it. Two gfx1100 risks: it is labelled "RDNA 3.5" throughout
(`:4`) though `v_dot2_f32_f16` is a gfx11 instruction, and the gfx1100 WMMA table is a 14-shape stub ("library
only, untuned here") against 532 tuned shapes for gfx1151 (`wmma_gemm_table.cuh:55-56,592-593`). Verdict: decode
GEMV TRANSFERS-WITH-WORK; the prefill GEMM DOES-NOT-TRANSFER as written.

## GSQ-RCO

**It is not a format.** GSQ-RCO is a per-tensor mixed-precision *recipe*: "RCO" is Riemannian Constrained
Optimization (assign one of K quant options to each of N groups under a cost budget, arXiv:2605.00649), "GSQ" a
learned-grid quantiser with group scales (arXiv:2604.18556). The released files carry unmodified GGUF tensor
types and no transform metadata: the card states they are standard GGUF that run unmodified in stock loaders
(read on huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF, not tested here), `tensor-allocation/*.rco-allocation.txt`
lists per-tensor assignments, and every header type id comes from the GGUF type table
(`constants.py:6007-6022`). Kernel implication: **nothing new to implement.** Support the IQ/Q_K types, dispatch
**per tensor from the type field**, and ignore `general.file_type` — a file labelled IQ3_S still holds Q4_K and
IQ4_XS. TRANSFERS: same kernel family, wider type mix.

## Recommended kernel #1

**`gemv_k_q8_1` — Q5_K × int8-activation GEMV, N-major, batch 1.** Shapes K=5120/N=17408 (58.4 MiB, 96 us at
peak) and K=17408/N=5120; one code path for Q5_K/Q4_K/Q6_K via a template. That is 79% of the token read, and
the only kernel the 27B needs to approach its 36 tok/s ceiling.

Design, wave/block level (wave32, no tensor cores):

- Workgroup = 4 waves = 128 threads owning **4 output rows, one row per wave**; launch `ceil(N/4)` workgroups.
  No K-split across workgroups → no atomics, no second pass.
- A row of K=5120 is 20 superblocks = 3520 B. Inside a wave, 32 lanes = 4 groups of 8; **lane `j` of group `g`
  owns sub-block `j` (32 weights) of superblock `4·step + g`**, so a wave covers 1024 weights per step and 5
  steps retire the row.
- Loads per lane per step: one `uint4` at `qs+48+16j` (8 lanes = one contiguous 128-B line) and one `uint` at
  `qh+4j` (one 32-B line). 20 B/lane, coalesced. `d`,`dmin` broadcast from LDS; each lane unpacks its own 6-bit
  `sc[j]`,`m[j]`.
- Inner: 8 × `v_dot4_i32_i8(q_word, act_word, acc)` against 8 int32 activation words in VGPRs → `Σq·x`; a second
  int accumulator over the precomputed per-16 `Σx` supplies the `−dmin·m` term (`quants.py:517-523`); two fp32
  FMAs close each 32 weights. Epilogue: 3 `ds_bpermute` steps within each 8-lane group, then 2 across the groups;
  ~40 VGPRs, so LDS sets occupancy. The `x → q8_1` pre-pass (160 blocks of 36 B for K=5120) is a separate ~5 us
  kernel run once per layer.

**Expected: 480 GiB/s target (81% of 592), floor 450 → 114–130 us per 1×5120×17408 and ~29–31 ms per token.**
Stated so it can be falsified: 592 GiB/s means 925 G weights/s; the scheme spends ~20 lane-instructions per 32
weights (2 loads, 8 unpack, 8 dots, 2 FMA) = 578 G instr/s against ~800 G issue slots/s on 84 CU at 2.39 GHz —
72% issue, before the rate of `v_dot4_i32_i8` is known. If that op is quarter-rate on gfx11 the kernel is
issue-bound near 500 GiB/s, and the fix is fewer dots per weight, not more parallelism.

**Oracle (torch/numpy only):** dequantise with `quants.py:dequantize` into fp64, then
`y_ref = W_deq.astype(np.float64) @ x.astype(np.float64)`. First test: `test_gemv_q5k_int_exact` — K=5120, N=8, x
= random ints in [−127,127], reference the same integer path in `np.int64`; must match **exactly** (int32, zero
tolerance), which isolates bit-plane unpacking from scaling and catches every `qh`/`sc` off-by-one. Second:
`test_gemv_q5k_f32` — x ~ N(0,1): `max|Δ| ≤ 2e-3·max|y|` for the int8-activation path, plus a 1e-5 variant
against an fp32-x reference kernel. Both pass on K ∈ {256, 2560, 5120}, N ∈ {1, 4, 640} before any benchmark.

## Measured: the microbenchmark is done, and it overturns backlog #1

`bench/gfx1100-probes/RESULTS.md` measured the instructions on card 0.

- `__builtin_amdgcn_sdot4` **does not compile** on gfx1100 (needs `dot1-insts`,
  which the subtarget does not advertise). `v_dot4_i32_i8` is inline-asm only.
- `v_dot4_i32_i8` is **half rate** (2.08 slots/instr), not quarter, and delivers
  28.9 TMAC/s — 2.4x scalar FMA.
- At 619 GB/s measured read bandwidth, a Q5_K GEMV needs about **6.6% of the
  issue slots**. It is bandwidth-bound with ~15x margin.

Consequence for this document: the "20 lane-instructions per 32 weights = 72%
issue occupancy" estimate below is wrong in two ways — it counts lane
instructions against wave slots (32x) and it assumes 10752 lanes instead of
5376. Do not optimise dot count. Use the simplest correct arithmetic (a
dequantise-to-fp16 path on full-rate `v_dot2_f32_f16` is ample) and spend the
effort on one-pass coalesced reads.

## Ranked backlog

1. `v_dot4_i32_i8` / `v_dot2_f32_f16` throughput microbenchmark — decides the whole GEMV budget (30 min).
2. `gemv_k_q8_1` for Q5_K, then template to Q4_K/Q6_K (79% of bytes).
3. Activation `q8_1` quantiser fused with RMSNorm — on the critical path of every GEMV.
4. IQ4_XS GEMV (17.8%): same skeleton, 16-entry LUT in LDS, one accumulator.
5. Fused SwiGLU gate+up GEMV: one x-read for two tensors, one launch less.
6. Output projection 5120×248320 (5.9%, one 1.6 ms call) — same kernel; consider split-K.
7. Q3_K + IQ3_S (0.9%) — 27B completeness and Flash-Next IQ3_S.
8. Grouped expert GEMV for 1×2560×640 — only as one batched dispatch.
9. WMMA prefill GEMM at batch ≥64, then EXL3 (re-derive the in-tree RDNA GEMV), then GSQ-RCO (dispatch only).

## Claims I could not verify

- `v_dot4_i32_i8` and `v_dot2_f32_f16` issue rates on gfx1100; the 480 vs 500 GiB/s fork rests on it.
- 592 GiB/s is from the brief, not re-measured; 84 CU for the 7900 XT is a spec value.
- Whether exllamav3's RDNA GEMV runs correctly on gfx1100 rather than gfx1151 — labelled RDNA 3.5 throughout.
- The toolchain behind the 27B per-tensor assignment (the imatrix is named in the header, not shipped), and the
  GSQ-RCO card's load-compatibility claim (card read, not tested; no GGUF writer exists in GSQ's main branch).

## Source map

- `development/_research/tensorfold` @ `7a90efacb351ce6442da2823d6df1e9237a90f29` —
  `zig/kernels/metal/ops/qmv.metal`, `zig/kernels/cuda/drama3_staged_gemv.cu`.
- `development/_research/exllamav3` @ `da68d57de61781e5b74ad68698ac4caf36408d7a` — `exllamav3_ext/quant/`
  (`exl3_dq.cuh`, `codebook.cuh`, `bits_k.cuh`, `hadamard_inner.cuh`, `exl3_gemv_kernel.cuh`,
  `exl3_gemm_inner.cuh`), `exllamav3_ext/rocm/quant/`.
- `development/_research/FreeToken` @ `546c5fe2e4468e2f3f8a574c64021a47016c35f7` —
  `python/freetoken/third_party/gguf/{constants.py,quants.py}`; `development/_research/strata` @
  `011c9e600a93bbc9fd451c563501b5831c32bbca` — `metal/src/gguf_lite.cpp`, `metal/tests/gemv_parity.cpp`.
- GGUF v3 header layout: the public GGUF specification (v2/v3 chapters), verified by parsing the file directly
  (header = 10,945,379 B).
- GSQ-RCO: huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (card + `tensor-allocation/`), arXiv:2604.18556
  (GSQ), arXiv:2605.00649 (RCO), github.com/IST-DASLab/{GSQ,RCO}.
