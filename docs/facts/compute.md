# facts — compute

The instruction layer: what the card issues, what it costs, and the one model that was refuted.
`_archive/2026-10-24/…` paths are pre-migration locations.

| fact | value | instrument | shape | date | verified-by |
|---|---|---|---|---|---|
| CUs / SIMD32 / wavefront | 84 CU = **168 SIMD32**, wavefront 32 | `rocminfo` | gfx1100 | 2026-10-22 | `bench/gfx1100-probes/probe.cpp` + `results.json` |
| Sustained sclk | **2802 MHz** (range 2435–2929), median of 836 in-run samples | `amd-smi metric --clock` inside the benchmark | 16-chain loop | 2026-10-22 | same |
| Peak | **15.06 T lane-ops/s** = 470.7 G wave-instr/s | 5376 lanes × 2.802 GHz | — | 2026-10-22 | same |
| Harness efficiency at 16 chains | 81 % (12.2 T lane-slots/s) | `probe.cpp`, median of 5 | 16 independent accumulators | 2026-10-22 | same |
| `v_dot4_i32_i8` | **half rate** — 2.08 slots/instr; cannot dual-issue | `probe.cpp` vs FMA baseline | 2^24 iters/lane | 2026-10-22 | same |
| `v_dot2_f32_f16` | full rate — 1.24 slots/instr | same | — | 2026-10-22 | same |
| `v_fma_f32` | full rate — 1.23 slots/instr | same | — | 2026-10-22 | same |
| Packed fp32 FMA | **does not exist** on gfx11 (`llvm-mc` rejects `v_pk_fma_f32`) | `llvm-mc` | — | 2026-10-22 | same |
| `__builtin_amdgcn_sdot4` | **does not compile** — needs `dot1-insts`, which the subtarget does not advertise | `hipcc` | gfx1100 | 2026-10-22 | same |
| `__dp4a` | **undeclared** for gfx1100 — not feature-gated, absent | `hipcc` two-line probe | gfx1100 | 2026-10-23 | `docs/cleanup/MANIFEST-2026-10-24.md` §4 |
| I-cache | 128 B lines, 32 KB per WGP, shared by both CUs | ISA survey | — | 2026-10-22 | `_archive/2026-10-24/docs/research/2026-10-23/s3-amd-sinks.md` |
| Type-sorted dispatch | interleaving quant types in one CU costs **−10 %**; sorting costs 0 | E2 | — | 2026-10-22 | `bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md` |
| Pages | **4 KB only**; no HIP/ROCm API or env var gives 2 MB | `amdgpu` behaviour | — | 2026-10-22 | `s3-amd-sinks.md` |
| Async copy | **no `cp.async`** on gfx11; MUBUF has no LDS bit | ISA | — | 2026-10-22 | `s3-amd-sinks.md` |
| VGPRs | 1536 per SIMD32, allocation granularity 24 → full occupancy at ≤96 regs | ISA | — | 2026-10-22 | `s3-amd-sinks.md` |
| IQ3_S body | **193 instruction words per 32 weights per lane** | `llvm-objdump -d --mcpu=gfx1100`, delimited by the single backward branch | `k_full<0,0>`, `[0x60BC,0x65B8]` | 2026-10-23 | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-2.md` |
| Body composition | 8 `dp4a`, 48 sign, 34 index, 31 `s_delay_alu`, 27 loop control, 21 global-mem, 9 address, 8 scale, 5 field loads, 2 checksum | same | residue 0 | 2026-10-23 | same |
| Verify width scaling | 193 → **421 words** at k=8 (**+32 per vector**); ~102 words are shared dequant | same | `k_fullk<LAY,K>` | 2026-10-23 | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-9.md` |
| Issue occupancy at 521 GB/s | **48.6 %** of the slot rate | 193 words + measured throughput | 1.0700 GiB | 2026-10-23 | `H-2.md` + `H-1.md` |
| Words needed to saturate issue at 521 GB/s | 338 per 32 weights — the body has 193 (57 %) | same | — | 2026-10-23 | same |

## The refuted model, kept because the record of a wrong belief is evidence

`docs/plan/DESIGN-instruction-budget.md` claimed decode was **instruction-issue-bound** with
`n ≈ 1.57` against a budget of `n ≤ 1.04`. **It was void, twice over:**

1. **Unit error of exactly the wavefront width.** The slot rate counts *wave* instructions; one wave
   instruction advances 32 lanes × 4 weights = 128 weights. The formula `4·slots·(bpw/8)/n` treats a
   slot as buying 4. The correct ceiling at `n = 1.57` is **16.5 TB/s**, not 515 GB/s.
2. **`n` was solved for, not measured:** `809.08/516.4 = 1.5668` and `809.08/776 = 1.0426`. Both of
   the note's constants are reproductions of two measurements to four figures.

**Refuted by:** freeing memory returns **+23.7 %** (521.3 → 645.1 GB/s), which an issue-saturated
kernel cannot do. The kernel sits at a corner: memory costs 16.0 % of the wall, instructions 16.9 %.
See `docs/plan/DESIGN-instruction-budget.md` (corrected) and `facts/memory.md`.

## The two-bound model that replaced it

| | GB/s | % of 776 |
|---|---:|---:|
| achieved, VRAM | 521.3 | 67.2 |
| achieved, memory free | 645.1 | 83.1 |
| wall | 776 | 100 |

Memory headroom **+123.8 GB/s** (16.0 % of the wall); instruction headroom **+130.9 GB/s** (16.9 %).
Neither is saturated; both are close. **No single-resource story survives on this kernel.**
