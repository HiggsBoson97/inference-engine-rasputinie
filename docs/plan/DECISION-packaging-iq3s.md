# Decision — serve the published IQ3_S file as-is

**Status:** decided 2026-10-22. Verified independently by the orchestrator
(separate build, separate run: 56.1 tok/s as-is, 51.1 expanded, breakeven
780.8 GB/s against a 776 wall, 9 of 10 ablation rows passing the checksum gate).

## The question

Should the engine convert `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` once into a
gather-free layout, trading bytes for load independence? The hypothesis came from
third-party RDNA data reporting i-quants at ~236 GB/s effective against Q4_0's
409, and from this project's own research concluding that i-quant loss is
*instruction issue, not memory*.

## What was measured

On the real file, at a 1.07 GiB footprint, with every row gated on a data-
dependent checksum so a kernel that skips bytes cannot report a number:

| variant | GB/s | % of read wall |
|---|---|---|
| read, no quant work | 734–752 | 95–97 % |
| **IQ3_S dequant+dot (as published)** | **~514** | **~66 %** |
| IQ3_S with sign decode removed (ablation, numerically wrong) | ~594 | ~77 % |
| IQ3_S with the gather removed (ablation) | ~495 | ~64 % |
| lossless expansion (1.509×| bytes) | ~632 | ~82 % |

Ablations, as a share of the read wall: **sign decode 8.6–15.6 %**,
**index+dot 15–16 %**, **gather ≈ 0–3.5 % (indistinguishable from noise)**.

## The verdict

**Expansion loses.** It moves 1.509× the bytes, so it needs **~781 GB/s** to
break even against a **776 GB/s** wall — it would have to run at 100.6 % of the
memory wall to tie. It runs at 82 %. Whole-file estimate: **51 tok/s expanded
against 56 as-is**. Even the ideal 1.22× expansion would only reach a tie.

**Serve the file as published.** No converter, no prepared artifact, no
bit-exactness machinery, no extra disk.

## What this corrects in the record

1. **The gather is not the problem.** The plan was preparing to attack the
   codebook indirection. Measured, it costs nothing; removing it makes the kernel
   *slower*. The research's "instruction issue, not memory" conclusion was right,
   but the instruction was misidentified.
2. **Sign decode is the removable cost** — and it is exactly what the reported
   "+23 % from branch-free SWAR at identical bit-width" was fixing. Independent
   arrival at the same bottleneck from a different direction.
3. **The 236 GB/s figure does not transfer** to this card or this access pattern.
   IQ3_S on gfx1100 runs at ~66 % of the wall, not ~30 %.

## Consequences for the plan

- The loader reads GGUF directly. O1b's "one internal layout" still applies to
  kernels, but there is no offline conversion step.
- The first kernel slice has a measured target: **sign decode**, worth
  **+6.4 % at file level** (56.1 → 59.7 tok/s), plus index+dot as the second
  target.
- The headline single-card estimate is **~56 tok/s**, and it is *optimistic*: it
  assumes the non-IQ3_S types (69 % of the bytes) run at the clean layout's
  752 GB/s. IQ3_XXS and IQ2_S have more gathers than IQ3_S and will not. The
  blended step is what must be measured, not estimated.
