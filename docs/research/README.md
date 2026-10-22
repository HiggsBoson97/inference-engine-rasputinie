# Research index

Two generations of documents. **Read `00-facts.md` for numbers; this directory
for reasoning.**

## Current — `2026-10-22/`

Seven lanes written against this project, measurement-tagged, with an
`INDEX.md` that carries the correction: every absolute GB/s and tok/s figure in
them was computed against a 619 GB/s wall that was later shown to be a kernel
artefact. **The real wall is 776 GB/s**, so their absolute numbers are ~25 % low
while every relative comparison and mechanism finding stands.

Start with `2026-10-22/03-bandwidth.md` for the wall itself.

## Referenced — `14` to `20`

Written earlier in the same project, still useful for facts about the engine and
the card. Where their numbers are superseded, this is the list:

| file | status |
|---|---|
| `14-sglang-gfx1100.md` | engine anatomy — current |
| `15-quant-kernels.md` | quant kernel design — current, read the measured addendum first |
| `16-sglang-gfx1100-build.md` | the build recipe — current, carries its own correction banner |
| `17-dflash2.md` | the drafter artifact — current, a later option |
| `18-tp-roccl.md` | TP microbenchmarks — **its throughput conclusion is contradicted** by `2026-10-22/05-serving-knobs.md`; the measurements stand, the recommendation does not |
| `19-aiter-gfx1100.md` | aiter verdict — current |
| `20-decode-profile-27b.md` | the FP8 decode profile — its numbers are the evidence for ADR 0004; **its "optimise FP8 first" recommendation is superseded** by ADR 0004 |

## Removed

The four donor-porting reports and the composition synthesis were removed with
ADR 0002's ratification: the composition approach they examined is closed, and
the decision is the record. Their text survives in git history.
