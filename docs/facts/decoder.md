# facts — decode: rate, verification, drafters

What one token costs, what a verification step costs at width k, and what each drafter adds.
`_archive/2026-10-24/…` paths are pre-migration locations.

## The decode rate

| fact | value | instrument | shape | date | verified-by |
|---|---|---|---|---|---|
| Whole-file decode | **47.1 tok/s** (21.18 ms/token) | Σ per-type measured rates; the 24.5 % ungated rides on an assumption | batch 1, 4K | 2026-10-23 | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-4.md` |
| Previous estimate | 56.0 tok/s — **19 % too high**, because it assumed 752 GB/s for all 69 % non-IQ3_S bytes | same, before A2 | — | 2026-10-22 | superseded; `docs/plan/RECORD.md` marks it |
| Speed of light | 10.57 GiB ÷ 776 GB/s = **14.6 ms/token = 68.4 tok/s** | ARITHMETIC | batch 1 | — | `facts/memory.md` |
| Per-tensor dispatch (401 launches) | 712.3 GB/s → **60.6 tok/s** | E2, read-stream upper bound | 4K | 2026-10-22 | `bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md` §4 |
| 2-layer fusion (33 launches) | **774.2 GB/s → 65.8 tok/s** (99.7 % of wall) | same | 4K | 2026-10-22 | same §5 |
| 4-layer fusion (17 launches) | 773.8 GB/s → 65.8 tok/s | same | 4K | 2026-10-22 | same §5 |
| 8-layer fusion | 691.4 GB/s → 58.8 tok/s — **worse than per-tensor** | same | 1.35 GB launches | 2026-10-22 | same §5 |
| Launch overhead | 17–33 launches × 5–30 µs | E2 | per step | 2026-10-22 | same |

**Fuse 2–4 layers per launch, never 8.** `assumed:` the ~800 MiB "footprint cliff" is the clock ramp
(800 MiB ÷ 750 GB/s = 1.12 ms, i.e. a fixed ~230 µs stops being amortised there). Unverified — the
clock-pinned re-run is the check.

## The verify curve — measured, and it changes what limits depth

| fact | value | instrument | shape | date | verified-by |
|---|---|---|---|---|---|
| Step time vs verify width | **`ms(k) = 1.764 + 0.588·k`** | `k_fullk<LAY,K>`, K accumulators over the same block read | 1.0700 GiB IQ3_S, `grid=336×256`, `rp=4` | 2026-10-23 | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-9.md` |
| Ratio | `ms(8)/ms(1) = 2.75`; per-token 2.35 → 0.81 ms (2.9×) | same | — | 2026-10-23 | same |
| **Marginal cost of the k-th token** | **flat at 0.51–0.72 ms (median 0.567), no knee** | same | — | 2026-10-23 | same |
| Why it is flat | ~102 of the 193 body words are dequant that all k vectors share; words/token 193 → 52.6 | ISA census | k=1…8 | 2026-10-23 | same + `facts/compute.md` |
| Gate | fp64 max rel err **2.94e-05** against a 2e-03 tolerance declared first; checksum match; 20/20 rows | same | every k | 2026-10-23 | same |
| Whole-file marginal | **≈5.40 ms/token**, so the budget per *drafted* token is `p × 5.40 ≈ 3.87 ms` at p = 0.72 | ARITHMETIC from the curve and the 21.18 ms step | — | 2026-10-23 | same |

**Consequence, and it inverts the intuition:** there is no depth at which verification stops paying.
Depth is limited by **acceptance**, and the real cost is the **drafter**.

## The drafters

| fact | value | verified-by |
|---|---|---|
| MTP head cost | **1.064 GB per draft token** (0.325 GiB Q6_K layer + **0.666 GiB LM-head re-read**) → 7.445 GB for 7 tokens = **65.6 % of a weight sweep** | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-5.md` |
| DFlash2, BF16 | 3.849 GB, 1.924e9 params, no embedding, no LM head → 4.564 GB per 7-token block (40.2 % of a sweep) | same |
| DFlash2, 4.5 bpw | 1.798 GB per block (**15.8 %**), and **flat in block size** — one forward per block | same |
| Effective rate at published acceptance (5.02 @ k=7) | MTP **139** · DFlash2 BF16 **164** · DFlash2 4.5 bpw **199** tok/s | same |
| Share of the 3.87 ms draft budget | MTP **53 %** · DFlash2 BF16 32 % · DFlash2 4.5 bpw **13 %** | ARITHMETIC |
| Break-even acceptance | A = 1.16–1.66 for all four configurations | same |
| DFLASH in the vendored tree | **first-class**: 9 files ~7.5 k LOC, `SpeculativeAlgorithm.DFLASH` at `spec_info.py:39`, Triton kernels; forces `num_steps=1`, `eagle_topk=1` | same |
| DFlash2 architecture | 5 layers, hidden 5120, inter 17408, head_dim 128, 32/8 heads, **non-causal**, window 2048, block 8, 2-tap conv, selector rank 256 top-16, target layers `[5,19,33,47,61]` | same |
| DFlash2 needs from the target | hidden states at those 5 layers + the LM head as a readable matrix. The *vendored loader* rejects a quantised head (would force a 2.54 GB BF16 copy) — **a loader artefact, not a requirement** | same |
| Licences | code **MIT** (`z-lab/dflash`), weights **Apache-2.0**; vendored engine Apache-2.0 | same |

`assumed:` that acceptance transfers from the published H200/BF16 numbers to this 3.44-bpw target.
The drafter conditions on the target's hidden states, which come from quantised weights. **The check:**
teacher-forced replay on our own prompts — acceptance at 3.0 / 4.25 / 5.5 / 8.0 bpw of drafter.
