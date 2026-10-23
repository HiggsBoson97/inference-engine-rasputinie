# Probe brief — H-9: the verify-width cost curve ms(k), k = 1…8

```
STATION: probe (queue item 1, `COST-LEDGER.md`; feeds A1b/H-3 and the drafter decision)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin`, card PCI 07:00.0 (`HIP_VISIBLE_DEVICES=1`) ; scratch `~/a9/`
TOOLSET: ssh, hipcc, llvm-objdump, amd-smi, python3, sudo -n
BUDGET: 100 tool calls; at ~90 % write out; unfinished items report as failed(budget)
ROUND: 1
CONTRACT: none — Discovery probe
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-9.md
```

## GOAL

Measure **how long one decode step takes as a function of the verification width `k`**, for
`k = 1…8`, on the real IQ3_S dequant-GEMV, one card, fixed footprint.

This curve is what makes draft depth a decision instead of a guess. Speculative decoding
proposes `k` tokens and verifies them in one pass over the weights: the weight bytes are read
**once** and used `k` times, so `ms(k)` must be far sub-linear. The value of speculating at
depth `k` is `acceptance(k) / ms(k)`, and neither term is known for this engine.

## CONTEXT

- Harness: `~/e1/iq3s_valid.cpp`, built as `~/e1/iq3s_valid`. Its `k_full` computes one dot
  per row from one activation vector `x`; a proper verification step needs **`k` activation
  vectors and `k` accumulators per lane, over the same block read**. That is the only change
  you need: do not alter the block read, the addressing, or the checksum.
- The measured single-width baseline: `full` = 521.3 GB/s at 1.0700 GiB, `GRIDMUL=8`
  (`H-1.md`). `k_full`'s body is 193 instruction words per 32 weights per lane
  (`H-2.md`): 8 `dp4a`, 48 sign, 34 index, 31 `s_delay_alu`, 27 loop control, 21 global-mem.
  At `k > 1` the block read amortises and the VALU count per token falls, so the kernel walks
  from memory-bound toward issue-bound — `H-1` measured a memory-free ceiling of 645 GB/s,
  which is the shape of the prediction.
- The target model's decode reads 10.57 GiB per step at `k = 1`. At `k = 8` the same 10.57 GiB
  serves 8 tokens.

## RULES

1. **`HIP_VISIBLE_DEVICES=1`** — PCI `07:00.0`, headless. HIP 0 / PCI `03:00.0` is the display
   card; a load there wedged gnome-shell once.
2. **llama.cpp is excluded** — no clone, no build, no source read, no derived code. Name it in
   your compliance line.
3. **Every `k` carries its own fp64 CPU reference**, computed in the same binary, with the
   tolerance stated **before** the run. A curve whose `k = 8` point is wrong is worse than no
   curve.
4. **Every `k` carries the data-dependent checksum gate** (host replay from the same bytes).
5. **Fixed footprint 1.0700 GiB** for every `k`, `GRIDMUL=8 BLK=256` (`grid=336×256`), clocks
   pinned `high` and restored `auto`. Check the printed config line every run.
6. **Each timed region ≥ 3 ms.** At `k = 8` the step gets faster in per-token terms but the
   kernel may get longer; verify the duration and raise `rp` if needed.
7. Kill by PID, never `pkill -f`. `~/e1/` is read-only; copy into `~/a9/`.

## WHAT TO COLLECT

For each `k` from 1 to 8, at the same footprint:

- **ms per step**, median of 3 runs, with the spread, and the GB/s equivalent (bytes read ×
  passes ÷ time) so it can be compared to `H-1`'s 521.3.
- **marginal ms of the k-th token** = `ms(k) − ms(k−1)`. This is the number that decides depth:
  speculate while the marginal cost is below the value of the expected accepted token.
- **issued instructions per token** from the emitted ISA: `llvm-objdump -d --mcpu=gfx1100`,
  delimit the body by the single backward branch as `H-2` did, and report words per 32 weights
  per lane for `k = 1, 2, 4, 8` at least. The body should grow with `k`; how much it grows is
  the compute cost of verification.
- **the fp64 error and the checksum verdict** for every `k`, and the tolerance you set first.

## HOW TO VERIFY YOU DID IT RIGHT

- `ms(1)` reproduces `H-1`'s `full` within ±3 % at the same footprint; if it does not, the
  harness change broke the baseline and nothing above it counts.
- `ms(k)` is monotonic and sub-linear: if `ms(8) ≥ 8 × ms(1)`, the kernel is not amortising the
  weight read and the result is a bug, not a finding.
- The GB/s equivalent **rises** with `k` (more weights' worth of work per byte read) and must
  not exceed the memory-free ceiling of 645 GB/s by much; a number far above 776 at a 1.07 GiB
  footprint is an automatic INVALID.
- Every `k`'s checksum reads `match` and its fp64 error is inside the stated tolerance.
- Report the marginal cost explicitly; a curve reported only as totals does not answer the
  question.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-9.md`, then print it:

```
# H-9 — verify-width cost curve ms(k), k = 1..8
## Question and the decision it feeds
## Configuration (card, clocks, footprint, grid, rp per k, durations)
## Correctness (per k: fp64 error, tolerance stated before, checksum verdict)
## The curve (table: k, ms/step, spread, GB/s equivalent, marginal ms, words/32w)
## Reading it (where the curve stops being linear, and what that implies for depth)
## Verdict (the k the arithmetic favours at acceptance 3 / 4 / 5, with the numbers)
## Compliance (llama.cpp excluded; repo unmodified; clocks restored; display card untouched)
## Receipts (absolute paths under ~/a9/)
```

Effort: **high**. One kernel change, eight points, each gated twice. The value is the marginal
column and the word-count growth — those are what turn "speculate at depth 5" from a copied
default into a derived number.
