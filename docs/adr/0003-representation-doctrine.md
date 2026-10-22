# 0003 — Representation doctrine: bf16 compute, GGUF weights, no FP8

Status: **accepted**, 2026-10-22. Hardware-forced, not a preference.

## Context

gfx1100 (RDNA3) has **WMMA** matrix instructions for bf16/fp16 at wave32, and
integer dot instructions (`v_dot4_i32_i8`, half rate, inline asm only). It has
**no FP8 matrix instruction of any kind**. Any FP8 weight must be expanded to
bf16 before a matrix op happens.

Decode on this machine is bandwidth-bound: measured 619 GB/s per card, and the
27B reads every weight once per step. So the weight format decides the ceiling,
and the compute format decides nothing that is not already free — the
instruction-throughput probe showed the GEMV needs ~6% of the issue slots.

Measured consequences:

| Representation | Bytes per card at TP=2 | Floor per step | Observed |
|---|---|---|---|
| FP8 (e4m3) | 14.52 GB | 23.5 ms | **857 ms** — 91.8% in the emulated fp8 GEMM at 17.7 GB/s |
| IQ3_S GGUF (3.44 bpw) | 5.9 GB | 9.5 ms | not yet measured |
| bf16 | 27.8 GB | does not fit | — |

A bf16 hipBLASLt GEMM of the same shape reaches **472 GB/s** on the same card,
which is the proof that the memory system is not the problem and that the FP8
path is emulation overhead on a format this chip cannot accelerate.

## Decision

1. **Weights: GGUF k/i-quants**, primary target `Qwen3.8-27B-GSQ-RCO-IQ3_S`
   (3.50 bpw, 10.96 GiB, task-lossless against BF16 on the publisher's suite).
   The kernel family covers IQ3_S, IQ4_XS, IQ3_XXS, Q4_K and IQ2_S, which are
   94.5% of that file's bytes, plus a slow path for the single IQ1_M tensor.
2. **Compute: bf16 WMMA**, with `v_dot4_i32_i8` allowed only where a measurement
   beats the bf16 path. Dequantise in registers, never through a materialised
   bf16 copy of the whole tensor.
3. **FP8 is a functional reference only.** It is the one 27B checkpoint sglang
   can load today, so it serves as a smoke test of TP, serving and the API. It
   is never a performance target and no native kernel is written for it.
4. **KV cache: q8**, not fp8. It is a bandwidth and capacity decision, and q8
   needs no unavailable instruction.

## Consequences

- The FP8 tile configs committed on 2026-10-22 remain, marked reference-only.
  They make the reference checkpoint usable; they are not the path.
- Inventory row 1 changes from "fp8 GEMM" to "GGUF dequant + bf16 WMMA GEMV".
- The performance ladder is re-based on IQ3_S bytes: 52.6 tok/s ceiling on one
  card, 105 with TP=2, before speculation. Speculation multiplies whichever
  floor applies.
- The quality gate uses the GSQ-RCO publisher's numbers (AIME25 100.00,
  LCB v6 85.71, GPQA-D 89.39 at 3.50 bpw) as the external reference, not our own
  guess at what 3.5 bits should cost.
