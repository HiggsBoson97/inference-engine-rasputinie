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
| `gemv/iq3s sign-decode removed` (ablation) | 608.2 GB/s | 78.4 % | measured | E1 — numerically wrong, diagnostic only |
| `gemv/iq3s expanded 1.51× (signs kept)` | 618.1 GB/s | 79.7 % | measured | E1 — rejected |
| `gemv/iq3s expanded, signs dropped` | 784.5 GB/s | 101.1 % | measured, **above the wall — not trusted** | E1 |
| `exp` vs `full` correctness | **0.000e+00 bit-identical** | — | measured | E1 losslessness proof |
| `decode/1/4096 IQ3_S, whole file` | **56.1 tok/s** | ~85 % blended | **estimated** | E1 whole-file model — optimistic, assumes non-IQ3_S types at 752 |
| `decode/1/4096 IQ3_S, signs fixed` | 59.7 tok/s | — | **estimated** | E1 |
| `decode/1/4096 IQ3_S, expanded` | 51.1 tok/s | — | **estimated** | E1 — rejected by `DECISION-packaging-iq3s.md` |

## Records that are not yet measured

These are the shapes the engine will actually be judged on, and none has a
number yet. They are listed so that "no record" is visible rather than implied.

- `decode/1/32768` — the long-context single stream.
- `decode/8/32768` — the multi-session operating point, KV-dominated.
- `prefill/4096` — prefill is compute-bound above 30 tokens, so this is a
  different regime and a different record.
- `mtp/1/4096` — speculative decode, tokens per weight read.

## The rule

A slice that does not beat the standing record for the shape it touches must say
why in its report. A regression with an explanation is a finding; a regression
without one is a defect.
