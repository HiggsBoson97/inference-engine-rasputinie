# The cost ledger — what a token costs, as a function of shape

**No shape is a target.** The owner has not named a session count or a context length, and
this file must not invent one. What it does instead is express every byte on the critical
path as a function of `B` (concurrent sequences) and `C` (context length), so the *strategy*
is chosen for the whole surface rather than tuned to a point on it.

A strategy optimised for one `(B, C)` is a strategy that leaves performance on the table
everywhere else. That is the failure this document exists to prevent.

## The terms, and how they scale

| term | per token | scales with | provenance |
|---|---|---|---|
| **weight read** | `10.57 GiB / B` | 1/B — read once per step, used by every sequence | measured (`H-1`: 521–645 GB/s, 10.57 GiB/file) |
| **KV read** | `C × KV/position` | C, independent of B | derived from geometry: 16 full-attention layers × 4 KV heads × 256 dim × K,V. **64 KiB/position at bf16, 34 KiB at q8_0** (the f16 block scale costs 2 KiB) — `docs/research/2026-10-22/04-kv-cache.md` |
| **GDN recurrent state** | `0.318 GB` | independent of B and C; per sequence, so it multiplies by B per step | derived: 159 MB per sequence, FP32, read+write per token — `REQUIREMENTS.md` R5, `00-facts.md` |
| **drafter** | MTP: **1.064 GB per draft token** (0.325 GiB Q6_K layer + 0.666 GiB LM-head re-read) — 7.445 GB for 7 tokens, **65.6 % of a sweep**. DFlash2: **0.257 GB per token**, flat in block size (one forward per block) — 1.798 GB at 4.5 bpw, **15.8 % of a sweep** | per step, so MTP scales with draft depth and DFlash2 does not | measured as a cost model in `run-artifacts/hypotheses/H-5.md`; the earlier "MTP ≈ 0.348 GB per pass" was wrong by 3× because the head re-reads the LM head |
| **all-reduce / handoff** | `128 × RTT` (TP) or `64 × RTT` (PP) | per step, independent of B | RCCL floor measured at **~19.5 µs**, 20.65 µs amortised at 10 KiB, not tunable by NCCL env (`docs/research/18-tp-roccl.md`) |
| **launch overhead** | 17–33 launches × 5–30 µs | per step | measured, E2 |
| **verification increment** | `ms(k) = 1.764 + 0.588·k` on a 1.0700 GiB IQ3_S slice; marginal **0.567 ms per extra token**, flat, no knee | per step; the first token pays the weight read, each further one pays only compute | measured, `H-9`. Whole-file: marginal ≈ **5.40 ms/token**, so the budget per *drafted* token is `p × 5.40` ≈ **3.87 ms** at p = 0.72 |

## What the verify curve changes — the drafter is the bottleneck, not the verifier

`H-9` measured the verification increment directly: `ms(k) = 1.764 + 0.588·k` on a 1.0700 GiB
slice of real IQ3_S, and the marginal cost of the k-th token is **flat at 0.51–0.72 ms with no
knee**. Every extra token costs the same small increment, because ~102 of the 193 instruction
words in the body are dequant work that all `k` activation vectors share — words per token fall
from 193 to 52.6.

Scaled to the whole file, the marginal verification cost is ≈ **5.40 ms per token**, so the
budget a drafter has to beat per *drafted* token is `p × 5.40` ≈ **3.87 ms** at p = 0.72:

| drafter | ms per draft token | share of budget | verdict |
|---|---:|---:|---|
| MTP head, 7 tokens | 2.04 | 53 % | pays, expensively |
| DFlash2 BF16 | 1.25 | 32 % | pays |
| DFlash2 4.5 bpw | **0.49** | **13 %** | pays comfortably |

**So depth is not limited by verification.** There is no depth at which the verifier stops
paying; depth is limited by acceptance, and the cost is in the drafter. That inverts the earlier
intuition (and the reason I had queued the curve first). What remains unmeasured is the
**acceptance curve** `A(k)` — A3's teacher-forced replay — which is now the only unknown
standing between these numbers and a chosen depth.

## The two boundaries — the whole shape story

Two crossovers divide the `(B, C)` plane. Neither is a target; both are properties of the
model, the file and the card.

**1. KV overtakes the weights at a fixed number of *resident tokens*.**

```
B × C  ≈  10.57 GiB ÷ (KV bytes per position)
       ≈  173,000 resident tokens at bf16 KV
       ≈  326,000 resident tokens at q8 KV
```

Because it is a product, `8 × 32K` and `1 × 256K` are the same point for this purpose — and
so is any other factorisation. Below the boundary the weight read leads; above it the
attention path leads. This is the repo's own figure (`04-kv-cache.md` §1.2), restated
without a shape attached to it.

**2. The GDN state overtakes the weights at ~36 concurrent sequences**, at any context.
It is a fixed per-sequence cost, so it only becomes the leading term when concurrency is
high.

Everything else — the drafter at 3–17 %, launch overhead at low single digits — is a
second-order term at every point on the plane, and is priced in the table above so no one
has to re-derive it.

## What the ledger says to work on, in order

1. **The weight path** — dominant for every shape below the first boundary, which is
   *most* of the plane, and the only term that helps every `(B, C)`. This is where the
   project has been, and the ledger says it was not wrong to start here.
2. **The attention/KV path** — dominant beyond `B × C ≈ 173K`, and the largest *unmeasured*
   term at present: there is no kernel, no rate, and no design. Whether it is worth
   building now depends on whether any plausible shape crosses the boundary — at B = 1
   that needs 173K context; at B = 8, 21.6K. **The owner's intended shapes decide this, not
   the ledger**, and that is a question for the owner, not an assumption to bake in.
3. **The GDN state** — only above ~36 concurrent sequences. Worth modelling now, worth
   building for later.

## Ranked measurement queue (shape-independent)

**Measure in the engine, not in scratch harnesses.** A rate from a hand-cut manifest of 30
concatenated tensors is a proxy: it does not exercise the layer sequence, the dispatch, the
GDN recurrence, the KV path or the second card. Scratch harnesses are for questions the engine
cannot yet answer; once the engine runs, every term below is measured in situ.

| # | measurement | why it ranks here |
|---|---|---|
| **0** | **the engine baseline** — does the vendored sglang serve this GGUF on this card, which dequant path does it take on ROCm, and at what tok/s | every later number is measured against it, and the only engine figure on record (857 ms/token) is a different checkpoint in an excluded format. `H-10` |
| 1 | verify-width cost curve `ms(k)`, k = 1…8 | speculation's value is `acceptance(k)/step_ms(k)`; `step_ms(k)` is not linear, so `k` cannot be chosen without it. TensorFold measures exactly this curve (`cuda_cost_adapter.zig`) |
| 2 | the attention path: paged decode-attention rate at q8 and k8v4, several contexts | the largest unmeasured term; decides whether boundary 1 becomes reachable in practice |
| 3 | two-card round trip, p99 over 1000 iterations, with RCCL's 19.5 µs floor as the baseline to beat | decides TP vs PP vs DP; a custom mapped-flag protocol may beat RCCL's floor |
| 4 | prefill at a few prompt lengths | `2 × 26.9e9 × tokens` FLOPs; 4K is ~1.6–3.2 s to first token on one/two cards. Deferred by `S1` and unbounded by any measurement |
| 5 | per-type dequant rates for the other 10 types | `A2`, in flight |
| 6 | drafter acceptance vs drafter bit-width | quantisation cannot corrupt output (the target verifies), so acceptance is the only cost — measure 3.0/4.25/5.5/8.0 bpw |

## What this forbids

- **Naming a shape and calling it the operating point.** The load-bearing variable is
  `B × C`, and the plan must be argued over its range.
- **Optimising a term that is not leading anywhere in the plane.** At present nothing
  qualifies; the two leading terms are both real.
- **Quoting a tok/s figure without the `(B, C)` it was measured at.** A tok/s number
  without its shape is not a record, it is a slogan.

## Open inputs

The ledger is arithmetic over inputs, and each input is tagged:

- **measured:** weight rate (`H-1`), RCCL all-reduce floor (`18-tp-roccl`), launch overhead (`E2`), wall (776).
- **derived:** KV bytes/position, GDN state traffic, drafter bytes.
- **assumed:** that decode reads each weight once per step; that the state is read+written per token; that the drafter's pass count is 1 per block (DFlash2) or 1 per token (MTP).

An input that moves changes the boundaries, so the boundaries are only as good as their
provenance — which is why the queue above exists.
