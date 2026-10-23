# Record ledger

The goal is not a target number (O12). It is to **break the standing record,
every time**, and to be able to say why. This file is that record.

## How to read it

- **Metric:** achieved GB/s as a fraction of the measured wall. Bounded,
  comparable across shapes, not gameable by changing the workload.
- **Wall:** 776 GB/s (streaming read, kernels longer than ~3 ms). See
  `bench/gfx1100-probes/WALL-2026-10-22.md`.
- **Status:** `measured` means a real run on the card reproduced by the
  orchestrator. `estimated` means arithmetic from other measurements — it may
  not enter a plan claim.

## Standing records

| shape | best achieved | % of wall | status | source |
|---|---|---|---|---|
| `read/wall 6 GiB, 4 passes` | **776.4 GB/s** | 97.1 % of the 800 spec | measured | `bench/gfx1100-probes/WALL-2026-10-22.md` |
| `read/wall 768 MiB, 1 launch` | 773.5 GB/s | 99.7 % | measured | E2 diagnostic, 2026-10-22 |
| `gemv/clean-4bit 1.04 GiB, 8 passes` | **752.5 GB/s** | 97.0 % | measured | `WALL-2026-10-22.md` |
| `gemv/iq3s read-only 1.07 GiB` | 717.7 GB/s | 92.5 % | measured | E1 final, 30 real tensors |
| `gemv/iq3s 1.07 GiB, as published` | **516.4 GB/s** | 66.6 % | measured | E1 final, 10/10 rows gated |
| `gemv/iq3s 1.07 GiB, 2026-10-23 re-run` | **522.7 GB/s** | 67.4 % | measured | `run-artifacts/hypotheses/H-1.md` §5 |
| `gemv/iq3s 38.3 MB, L2-resident, rp=64` | **645.1 GB/s** | **83.1 %** | measured — **the kernel's memory-free ceiling** | `H-1.md` §5, median of 3 |
| `gemv/iq3s 38.3 MB, L2-resident, best` | 679.6 GB/s | 87.6 % | measured | `H-1.md` §5 |
| `gemv/iq3s sign-decode removed` (ablation) | 608.2 GB/s | 78.4 % | measured | E1 — numerically wrong, diagnostic only |
| `gemv/iq3s expanded 1.51× (signs kept)` | 618.1 GB/s | 79.7 % | measured | E1 — rejected |
| `gemv/iq3s expanded, signs dropped` | 784.5 GB/s | 101.1 % | measured, **above the wall — not trusted** | E1 |
| `gemv/iq4xs read / dot, 1.1024 GiB` | **637.1 / 502.1 GB/s** | 82.1 / 64.7 % | measured, gated, fp64 err 2.2e-06 | `H-4.md` |
| `gemv/q3xxs read, 1.2393 GiB` | 643.6 GB/s | 82.9 % | measured (dot **not gated** — fp64 err 1.4, checksum mismatch) | `H-4.md` |
| `gemv/iq2s read, 1.1378 GiB` | 656.6 GB/s | 84.6 % | measured (dot **not gated** — documented layout does not close) | `H-4.md` |
| `gemv/q4k read / dot, 1.3321 GiB` | **659.6 / 452.2 GB/s** | 85.0 / 58.3 % | measured, gated; dot is **kernel**-limited (344 words/16 w) | `H-4.md` |
| `exp` vs `full` correctness | **0.000e+00 bit-identical** | — | measured | E1 losslessness proof |
| `decode/1/4096 per-tensor, 401 launches` | **712.3 GB/s → 60.6 tok/s** | 91.7 % | measured (read-stream upper bound) | `E2-LAUNCH-2026-10-22.md` §4 |
| `decode/1/4096 2-layer fusion, 33 launches` | **774.2 GB/s → 65.8 tok/s** | **99.7 %** | measured (read-stream upper bound) | `E2` §5 |
| `decode/1/4096 4-layer fusion, 17 launches` | 773.8 GB/s → 65.8 tok/s | 99.7 % | measured | `E2` §5 |
| `decode/1/4096 8-layer fusion` | 691.4 GB/s → 58.8 tok/s | 89.1 % | measured — **worse than per-tensor** | `E2` §5, 1.35 GB launches cross the ~800 MiB cliff |
| `decode/1/4096 IQ3_S, whole file` | **47.1 tok/s** | ~73 % blended | **estimated** — only 2 of 4 types measured | `H-4.md`, 21.18 ms/token. Replaces the 56.1 estimate, which assumed 752 GB/s for all 69 % non-IQ3_S bytes and overstated by 19 %. Still an upper bound: IQ3_XXS + IQ2_S (24.5 %) ride on 752 |
| `decode/1/4096 IQ3_S, signs fixed` | 59.7 tok/s | — | **estimated, on the superseded 56.1 base** | E1 — do not cite without re-deriving |
| `decode/1/4096 IQ3_S, expanded` | 51.1 tok/s | — | **estimated, on the superseded 56.1 base** | E1 — rejected by `DECISION-packaging-iq3s.md` |

## Records that are not yet measured

These are shapes the engine will be judged on, and none has a number yet. They are listed
so that "no record" is visible rather than implied.

**They are records to keep, not targets to hit.** The owner has not named a session count or
a context length, so no specific shape is "the operating point" and none may be treated as
one. What governs the plan is the **surface**, not a point: the load-bearing variable is
resident tokens `B × C`, and the boundaries are derived in
[`COST-LEDGER.md`](COST-LEDGER.md).

- `decode/1/<C>` — the single stream, over a context ladder (4K … 256K).
- `decode/<B>/<C>` — the concurrent case, recorded as a family rather than a fixed pair.
- `prefill/<N>` — prefill is compute-bound above 30 tokens, so this is a
  different regime and a different record.
- `mtp/1/4096` — speculative decode, tokens per weight read.

## The rule

A slice that does not beat the standing record for the shape it touches must say
why in its report. A regression with an explanation is a finding; a regression
without one is a defect.
