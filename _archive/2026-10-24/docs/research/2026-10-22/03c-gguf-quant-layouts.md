# GGUF block layouts vs. dequant+GEMV cost on RDNA3 (gfx1100)

Tags: **[D]** documented in source/spec I read · **[M]** measured elsewhere (cited) · **[C]** claimed/derived.

## Bottom line

- **Q4_0's layout is the most GEMV-friendly on this chip**: 18 B per 32 weights, one 8-byte load per lane per 16 weights, 4 `dp4a` per 16 weights, a single f16 scale applied *after* the group dot, no min term, no table. **[D]**
- **IQ4_XS is the only competitive i-quant** (4.25 bpw) and only because its 16-entry table is synthesised in registers via `v_perm_b32`, never gathered. IQ2/IQ3 do 4–8 dependent scalar loads into 256/512-entry global tables per 32 weights. **[D]**
- Achieved bandwidth **falls as bpw falls** at fixed model on AMD (487 GB/s at Q8_0 → 202 GB/s at Q2_K, W7900). Below ~5 bpw decode is issue-bound, not DRAM-bound. **[C]** For you: 11.76 GB/token at 619 GB/s = 19.0 ms = **52.6 tok/s ceiling**; at Q2_K's efficiency (~260 GB/s) you'd cap out near 22 tok/s whatever the format size.

## 1. Byte layouts — all **[D]**, `ggml/src/ggml-common.h`

| type | block | bytes | bpw | scale/min overhead | per-lane access (bs=1) |
|---|---|---|---|---|---|
| Q4_0 | 32 w | 18 (`d`+16) | 4.5 | 2 B/32 w = 11 % of bytes | 8 B contiguous / 16 w |
| Q4_K | 256 w, 8×32 | 144 (`dm`4+`scales`12+`qs`128) | 4.5 | 16 B/256 w = 11 % | 2× 4 B, 16 B apart / 16 w |
| Q5_K | 256 w | 176 (+`qh`32) | 5.5 | 9 % | 2× 4 B + qh bit |
| Q6_K | 256 w, 16×16 | 210 (`ql`128+`qh`64+`scales`16+`d`2) | 6.5625 | 8.6 % | 2× 4 B ql + qh |
| Q8_0 | 32 w | 34 | 8.5 | 5.9 % | 32 B contiguous |
| IQ4_XS | 256 w, 8×32 | 136 (`d`2+`scales_h`2+`scales_l`4+`qs`128) | 4.25 | 6 B/256 = 4.4 %; 6-bit scale, offset +32 | 16 B contiguous / 32 w |
| IQ3_S | 256 w | 110 (qs 64+qh 8+signs 32+scales 4+d 2) | 3.4375 | 5.5 % + 32 B signs | 8 B qs + 2 B signs + 8 grid loads |
| IQ3_XXS | 256 w | 98 | 3.0625 | 2 B only | 8 grid loads, signs in grid |
| IQ2_S | 256 w | 82 | 2.5625 | 10 B (12 %) | 8 B qs + 1 B qh + 8 grid loads |

Q4_K's 12-byte field packs 8× 6-bit scales **and** 8× 6-bit mins (SWAR unpack in `vec_dot_q4_K_q8_1`); Q6_K is 16 int8 sub-scales × one f16 super-scale. "4.5 bpw" in file terms = Q4_0 exactly, or Q4_K_S (4.58)/IQ4_XS (4.25) as averages; "3.5 bpw" ≈ Q3_K_S/IQ3_S (3.44); "3 bpw" ≈ IQ3_XXS (3.06) **[M]** (#22821).

## 2. Inner-loop cost per 32 weights (counted from `vecdotq.cuh`; **[C]**)

| type | dp4a | int ALU | table / LDS |
|---|---|---|---|
| Q4_0 | 8 | ~8 | none — scale hoisted (`d4*(sumi*ds8 - 8*vdr/QI*ds8.y)`) |
| Q4_K | 16 | ~24 | none — 2× work: extra `dp4a` vs `0x01010101` gives Σq8 for the min term; 6-bit unpack per 32 w |
| Q6_K | 16 | ~30 | none — 2-level scale |
| IQ4_XS | 8 | ~24 perms | 16-B table in 4 registers, 6× `__builtin_amdgcn_perm` per 8 w |
| IQ3_S | 8 | ~40 | **8 scalar loads** into 512×u32 grid + 8 `__vcmpne4` + 8 `__vsub4` |
| IQ2_S | 8 | ~45 | **4 × 8-B loads** into 1024×u64 grid, signs in separate 32 B |

- **Scales hoist per sub-block** for Q4_0/IQ4_XS/Q8_0 (one multiply after a 32-weight dot) but **not** for Q4_K/Q5_K/Q6_K, where scale×min enters inside the accumulation.
- **No LDS in this path at all.** i-quant grids are `static const __device__` (`ggml-common.h:538`) → plain global loads through L1/read-only. So "LDS bank conflicts" isn't the mechanism; the cost is 4–8 *serialised dependent* table loads per 32 weights, each on the critical path of the next `dp4a`.

## 3. Empirical evidence on AMD

**gfx1201 (RDNA4), llama-2-7B, `llama-bench -p2048 -n128`, PR #22821** **[M]**; eff. BW **[C]** = t/s × 6.74 Gp × bpw/8:

| type | bpw | tg128 | pp2048 | eff. GB/s |
|---|---:|---:|---:|---:|
| Q4_1 | 5.00 | 100.6 | 4021 | 424 |
| Q4_0 | 4.50 | 107.8 | 2224 | 409 |
| IQ4_NL | 4.50 | 106.8 | 2364 | 405 |
| IQ4_XS | 4.25 | **111.4** | 2367 | 399 |
| Q4_K_M | 4.85 | 96.2 | 3739 | 393 |
| Q3_K_S | 3.44 | 92.8 | **3699** | 269 |
| IQ3_S | 3.44 | 81.6 | 2378 | 236 |
| IQ3_XXS | 3.06 | 83.4 | 2514 | 215 |
| IQ2_S | 2.56 | 85.3 | 3142 | 184 |
| Q2_K_S | 2.625 | 118.8 | 1765 | 260 |
| IQ2_XXS | 2.06 | 89.0 | 2464 | 155 |
| IQ1_S | 1.56 | 137.1 | 3706 | 180 |

**gfx1100 (W7900, 864 GB/s peak), 7B, PR #20831** **[M]**: Q4_0 98.7 t/s (374 GB/s, 43 %), Q2_K 92.4 (202), Q3_K_S 88.9 (257), Q4_K_S 84.8 (327), Q5_K_S 77.1 (357), Q6_K 76.3 (422), Q8_0 68.0 (487, 56 %). **No IQ type was ever swept on gfx1100** in any table I found.

So the CUDA result **does reproduce on AMD**: IQ2_XXS delivers 2.6× less bandwidth per byte than Q4_0, IQ3_S 1.7×. Prefill differs — IQ4_XS prefills as fast as Q3_K_XL on 27B (≈470–500 vs 475–480 t/s, gfx1101, #28828) because MMQ amortises the LUT over 128 columns. Two smoking guns that the loss is instruction issue, not memory: branch-free SWAR `__vsub4`/`__vcmpne4` gave **+23 % decode on IQ2_S at identical bpw** (#28616); Q4_K gains **+11–34 %** from *cutting* warps 8→1 on gfx1100 (#23528) — impossible for a bandwidth-bound kernel. 27B/gfx1100 decode data is thin: #28003 reports Q4_K GEMV 68.9→62.6 µs per kernel on a 7900 XTX with Qwen3.8-27B-Q4_K_M **[M]**.

## 4. llama.cpp MMVQ design **[D]** (`mmvq.cu`, `vecdotq.cuh`, `common.cuh`)

- **One warp per output row** at bs=1 (`rows_per_block=1` on RDNA); each lane owns a strided K-slice: `blocks_per_iter = vdr·warp_size/qi` = 16× 32-weight blocks for Q4_0 (512 w), 2 super-blocks for Q4_K, 4 for IQ4_XS/IQ2_S. Per-lane `float tmp`, then a 5-step `__shfl_xor_sync` tree.
- **Zero shared memory** ("Warp-level reduction only - no shared memory needed"). The weight row streams from global; the *activation* is quantised once to `block_q8_1` (int8 + f16 `d`, `d·Σq`) and reused from registers/L1 by all warps. All math is `v_dot4_i32_i8` — `__builtin_amdgcn_sudot4` on gfx11, `sdot4` on gfx9/RDNA2 (`common.cuh:724`).
- **Per-arch nwarps**: RDNA3 uses 8 for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL, 2 for Q6_K, **1 for everything else incl. Q4_K, Q2_K, Q5_K, all IQ2/IQ3**; comments: "Q2_K/Q5_K/IQ4_XS regress in full quant sweeps", "complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register pressure and lookup table contention at higher thread counts."
- **wave32/wave64**: `WARP_SIZE` is a compile-time 32, physical warp is 64 only for `__GFX9__/__GFX8__` (`common.cuh:390`); gfx1100 is native wave32 (LLVM AMDGPUUsage), so no wave64 tax. Real AMD warts: host/device nwarps disagreement → NaN from unwritten smem (#25620, #28339); iGPU carve-out for nwarps=8 (#21068).

## 5. Competing layouts: layout or instruction mix?

- **Marlin** repacks at load into `mma.m16n8k32` fragment order with permuted scales and an *implicit* zero point (no runtime subtraction), async global→smem, fp16 accumulate. **[D]** arXiv 2408.11743.
- **Machete** (vLLM #7174) generalises that for Hopper `wgmma`, motivated explicitly by "Marlin v1 uses a *specific weight storage layout* specialised for mma". **[D]**
- **GPTQ-kernel / AWQ-gemv** dequantise on the fly to fp16, no repack. gfx11 has full-rate (packed `v_pk_fma_f16` double-rate) fp16, which is why llama.cpp's `mmvdq` dequant-to-float matvec (#26301) gains +4–10 % on RDNA3.5 with **no layout change** — evidence that instruction mix alone buys speed here. **[M]**
- **EXL2/EXL3** pre-shuffle at load as standard practice: exllamav2 stores `q_perm`/`q_invperm` per linear (`exllamav2/linear.py`), plus `q_scale_max`; EXL3 uses a trellis codebook. **[D]**
- **QuIP#** E8 lattice codebooks are the parent of the 8-element GGUF i-quant grids (`iq2s_grid` 1024×u64, `iq3s_grid` 512×u32) **[D]**; **T-MAC** (arXiv 2407.00088) is the LUT-mpGEMM formulation, ported toward GGUF in #10181 (CPU-oriented).
- **Published pre-shuffle-vs-on-the-fly analysis at bs=1 on AMD: none found.** Marlin argues repacking is a *precondition* for tensor cores, not itself a speedup. **[C]**

## 6. LUT/i-quant work specific to AMD GPUs

**No published work exists** — no ROCm blog, AMD AI blog post or paper on LUT/i-quant GEMV for RDNA. All AMD evidence is PR-level: #22821, #28616, #30517, #28828, #23528/#21068/#28339. **Missing:** any per-quant decode table on 7900 XT/XTX including IQ2/IQ3, and any RDNA3 IQ4_XS-vs-Q4_0 decode comparison at fixed model.

## Implications for your engine

1. **Copy Q4_0's shape, not its name**: 32-weight group, one f16 scale, one contiguous 8–16 B chunk per lane, scale applied after the group dot. Wrap in a 256-weight super-block only for the scale *field* (IQ4_XS-style 6-bit) to cut bytes without adding a min term.
2. **No gathers.** If you want 3.0–3.5 bpw, only a register-resident perm-LUT (IQ4 shape) stays issue-efficient; 256-entry grids are L1-latency-bound at 1 warp/row.
3. **Pre-shuffle at load** (Marlin/EXL2 both do): store per-lane 16 B chunks and scales in a separate array, so the loop is `LDG.128 → 4 dp4a → 1 ffma`.
4. **Don't go below ~4.5 bpw** unless you also cut instructions per weight: a 4.5 bpw layout already reaches ~409 GB/s of the 619 GB/s you measured (66 %), and that gap is ALU, not bytes.

## Sources

[1] https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h · [2] .../ggml/src/ggml-cuda/vecdotq.cuh · [3] .../ggml/src/ggml-cuda/mmvq.cu · [4] .../ggml/src/ggml-cuda/common.cuh · [5] https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes · [6] PR #19478 · [7] PR #20831 (W7900+R9700 sweeps) · [8] PR #22821 (full per-quant sweep) · [9] PR #23528 · [10] PR #21068 · [11] PR #28616 · [12] issue #28828 · [13] PR #26301 · [14] PR #30517 · [15] PR #28003 · [16] PR #10181 · [17] PR #28339 — all `https://github.com/ggml-org/llama.cpp/pull/<n>` or `/issues/<n>`
[18] Marlin https://arxiv.org/abs/2408.11743 · [19] QuIP# https://arxiv.org/abs/2402.04396 · [20] GPTQ https://arxiv.org/abs/2210.17323 · [21] AWQ https://arxiv.org/abs/2306.00978 · [22] T-MAC https://arxiv.org/abs/2407.00088 · [23] Machete https://github.com/vllm-project/vllm/pull/7174 · [24] https://github.com/turboderp-org/exllamav2 · [25] gfx1100 wave32 https://llvm.org/docs/AMDGPUUsage.html
