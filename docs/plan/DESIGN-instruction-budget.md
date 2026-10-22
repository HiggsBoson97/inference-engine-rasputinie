# Design note — IQ3_S is instruction-bound, not bandwidth-bound

**Status:** derived 2026-10-23, from four independent research lanes. The claim is
checkable in one afternoon with hardware counters, and **it reorders the whole
kernel plan.** Until it is confirmed or refuted, no memory-side optimisation
counts.

## The model

gfx1100 issues **2 VALU instructions per cycle per CU**, 84 CUs at 2.802 GHz
measured sclk:

```
slots = 84 × 2 × 2.802e9 = 470.7 G instruction/s
```

Let `n` = VALU instructions per 4 weights (one `dp4a` consumes 4 weights).
Then the ceiling is `4 · slots · (bpw/8) / n`.

| bits/weight | bytes/weight | n=1 | n=1.25 | n=1.5 | n=1.57 | measured |
|---|---|---|---|---|---|---|
| 3.00 | 0.375 | 706 | 565 | 471 | 450 | — |
| **3.44 (IQ3_S)** | 0.430 | 809 | 647 | 539 | **515** | **516.4** |
| **4.00 (clean)** | 0.500 | 941 | **753** | 628 | 600 | **752.5** |
| 5.50 | 0.688 | 1295 | 1036 | 863 | 825 | — |

Solving each measurement for `n`:

| format | n needed to reach 776 GB/s | n implied by measurement | verdict |
|---|---|---|---|
| 4.00 bpw | 1.21 | **1.25** | **at the wall** — memory-bound |
| 3.44 bpw | 1.04 | **1.57** | **50 % over the instruction budget** |

**Quantisation removed bytes; it did not remove instructions.** The per-weight
instruction cost is roughly fixed, so below about 4 bits/weight the instruction
budget becomes the binding constraint and the format gets *no faster* — and
eventually slower — as it gets smaller. IQ3_S sits in that regime **by design**,
because its quality is why we chose it.

This single equation explains every anomaly the ablation study produced:

- Removing the codebook gather made the kernel **slower** (−3.0 %): the gather
  is not the cost, so deleting work from a non-binding resource buys nothing.
- The sign path costs 11.8 % of the wall: signs are instructions.
- The "clean 4-bit layout hits 752" result: at 4 bpw the same `n` is *under*
  budget, so memory binds and the wall is reached.

## Three consequences that change the plan

1. **Memory-side work on IQ3_S is wasted until `n` falls.** Layout permutations,
   prefetch, non-temporal hints, MCD balancing, LDS staging — none of them can
   move a kernel that is already issuing more instructions than it has slots.
2. **The lever is instruction count.** Candidates, in order of expected value:
   folding the sign plane into a 512-entry table built at load (the `__vsub4` /
   `__vcmpne4` path costs 5–6 ops each on HIP); amortising scales over 256
   weights; and forming the codebook index with fewer ops. Target `n ≤ 1.04`.
3. **Speculation is the only way back to memory-bound.** `k` accepted tokens
   divide the per-token instruction cost by `k`: at 2.5 tokens per weight sweep
   the effective `n` is 0.63, under budget, and the wall becomes the limit
   again. This is the mechanism behind TensorFold's measured depth rule
   (`draft_depth.zig:55-95`), and it is worth more than every micro-optimisation
   combined.

## The five ISA facts behind it, all gfx1100-specific

1. **`v_dot4_i32_i8` cannot dual-issue.** RDNA3's VOPD opcode set is 17
   FP32-ish operations with three integer ops available only in the Y slot. The
   dual-issue idea is dead for our hot instruction — and a "half rate / full
   rate" table can never tell you that.
2. **4 KB pages only.** `amdgpu` maps 4 KB PTEs; no HIP or ROCm API or env var
   produces 2 MB pages. The page-size theory for the ~800 MiB anomaly is
   arithmetically excluded (3.15 M pages/launch would be 73 ps/page).
3. **No `cp.async` on gfx11; MUBUF has no LDS bit.** LDS staging is not slow, it
   is unavailable. Our measurement said 0 %; the ISA says why.
4. **1536 VGPRs per SIMD32 at allocation granularity 24** — full occupancy means
   ≤96 registers. The occupancy arithmetic in `research/2026-10-22/03b` is wrong.
5. **I-cache is 128 B lines, 32 KB per WGP, shared by both CUs.** Sorting
   dispatch by quant type costs 0 %; interleaving different types in one CU
   costs **−10 %**. Our per-`(role, type)` dispatch must dispatch type-sorted.

## What would falsify this

`rocprofv3` on the real kernel: if issue-slot occupancy is not high at
516 GB/s, the model is wrong and memory work returns to the table. One counter
readout decides it. **Run that before anything else.**

## Related

- `docs/plan/RECORD.md` — the measured rates this note derives from.
- `docs/03-native-kernel-inventory.md` — row 1a is the sign path; this note
  explains why it is the whole problem rather than one item.
- `docs/research/2026-10-23/s3-amd-sinks.md` — the sink survey this rests on.
- `docs/research/2026-10-23/s2-tensorfold-freetoken.md` — the depth rule and the
  LUT-instead-of-SWAR observation.
