# Design note — the two batch regimes

**Status:** proposed, 2026-10-22. Not yet ratified by the plan gate.
**Depends on:** the corrected 776 GB/s wall (`bench/gfx1100-probes/WALL-2026-10-22.md`).

## The asymmetry that creates two kernels

At batch 1, one token requires reading **every weight exactly once**. The weight
is used in a single multiply-add, so the step is a matrix-vector product and the
card waits on HBM: arithmetic intensity is ~2 MAC per weight byte and the
measured compute headroom is ~13×.

At batch N, the same single read of the weights serves **N tokens**. Arithmetic
intensity rises N-fold, and past a threshold the card waits on FLOPs instead.

Those are different kernels. A GEMV tuned for batch 1 is a bad GEMM and vice
versa. The question is where the boundary is, and whether both are worth
building.

## Where the boundary is

Compute can absorb 0.499 GB/token at the measured 86.4 TFLOP/s. A sequence's KV
read grows with context, so the crossover depends almost entirely on context:

| context | KV per token | crossover batch | regime above it |
|---|---|---|---|
| 1K | 0.067 GB | 27 | compute |
| 2K | 0.134 GB | 32 | compute |
| 4K | 0.268 GB | **51** | compute |
| 8K | 0.537 GB | **never** | bandwidth |
| 32K | 2.147 GB | **never** | bandwidth |
| 128K | 8.590 GB | **never** | bandwidth |

Both crossover figures are **upper bounds**: 86.4 TFLOP/s was measured on a
2048-row GEMM, and a 51-row GEMM will not reach it. So compute binds *sooner*
than the table says, never later.

## The decision

1. **The GEMV path is the engine.** It serves every context at low and moderate
   batch, which is every operating point this engine will actually see. It is
   built first and it is what the packaging, the dequant and the layout work is
   for.
2. **The GEMM path is not built yet.** Its entire justification is the corner
   below 8K context and above ~30-50 concurrent sequences. It enters the backlog
   when a measurement shows that corner being visited — not before.
3. **Both paths share one weight layout.** This is the reason to fix the layout
   (O1b) before either kernel exists: two kernels over two layouts is two of
   everything, forever.

## How the engine knows which regime it is in

One counter, no model of the workload: **achieved GB/s against the 776 wall.**

- Near the wall → bandwidth-bound. Add work (batch, speculation) and the record
  improves.
- Falling away from the wall → something else binds: compute, launch overhead,
  or contention. That is the signal to change shape, not to add batch.

This is why fraction-of-wall is the recorded metric (O12) rather than tok/s: it
is the only number that tells you *why* you are slow.

## What this means for the record

Records are per-shape, because a single number hides the regime. Two shapes are
the minimum for the ledger:

- `decode/1/4096` — batch 1 at 4K: weights 11.76 GB + KV 0.27 GB = 12.03 GB,
  wall ceiling **64.5 tok/s**.
- `decode/8/32768` — the long-context operating point, KV-dominated.

A slice that improves one and regresses the other must say so.

## Open question this note does not settle

Whether a GEMM path over a **quantised** weight layout can reach the wall at all.
The measured 752 GB/s was a GEMV; a quantised GEMM re-reads the scale plane per
column block and the arithmetic stops being free. That is a measurement, not a
design opinion, and it only becomes urgent if the short-context corner matters.

---

## Dispatch shape — decided by measurement, 2026-10-22

E2 built a launch sequence whose sizes follow the real file (401 launches,
11.2955 GB) and measured the dispatch pattern directly
(`bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md`):

| grouping | launches/step | GB/s | tok/s | % of wall |
|---|---|---|---|---|
| per-tensor | 401 | 712.3 | 60.6 | 91.7 % |
| 2 tensors | 201 | 725.9 | 61.7 | 93.6 % |
| **1 layer** | **65** | **760.9** | **64.7** | **98.1 %** |
| **2 layers** | **33** | **774.2** | **65.8** | **99.7 %** |
| **4 layers** | **17** | **773.8** | **65.8** | **99.7 %** |
| 8 layers | 9 | 691.4 | 58.8 | 89.1 % |

**Decision: fuse 2–4 consecutive layers per launch.** Never 8: a 1.35 GB launch
crosses the ~800 MiB footprint cliff, pays ~199 µs, and lands *below*
per-tensor dispatch.

Two facts make this cheap to adopt:

- **The marginal launch cost is 2.5–2.9 µs** — the same as launching an empty
  kernel. There is no mysterious per-launch penalty.
- **Per-tensor dispatch is viable** at 60.6 tok/s and needs no warm-up protocol.
  The 8 % fusion buys is exactly the launch cost; nothing about the kernels
  improves.

Which means the dispatch question is **closed at low cost**, and the whole
remaining performance problem is the one E1 found: the dequant+dot arithmetic
(66.6 % of wall on IQ3_S-type tensors). Fusion cannot fix that.

A caveat that must not be lost: these are **read-stream** numbers — the upper
bound of the dispatch pattern, not a GEMV. Real decode will be lower, and by how
much is E1's arithmetic, not E2's.
