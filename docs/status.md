# status

As of **2026-10-24**. This file is written as part of landing a unit, never as a unit of its
own. It carries the frontier and the last landed state; the reasoning lives in `docs/decisions/`
and the numbers in `docs/facts/`.

## Last landed — unit 0.1, the recipe diff — `DONE`

The GGUF's per-tensor types match the publisher's RCO allocation **exactly**.

```
proof:    python3 bench/04_recipe_diff.py <gguf> bench/manifests/rco-iq3s-allocation.txt
          → "matched 851/851  only-GGUF 0  only-alloc 0  mismatched 0" / "RESULT: PASS" / exit 0
expect:   exit 0, and the allocation's own header tally equals its body
seam:     the GGUF tensor table vs the publisher's text allocation — both primary artifacts
control:  --mutate blk.0.attn_qkv.weight=Q4_K → "mismatched 1" / "RESULT: FAIL" / exit 1,
          and the header cross-check reports Q4_K 39→40, IQ4_XS 96→95
state:    DONE
```

Two controls that failed to be controls, kept because they are the point: `blk.0.attn_q.weight`
is not a tensor name in this model (GDN layers use `attn_qkv`), and mutating a tensor to the type
it already has proves nothing.

**What it settles:** our inventory is the publisher's recipe, not a guess. The 72 `(role, type)`
pairs and the per-type byte shares in `facts/quant.md` are now externally corroborated.

## Frontier — unit 0.2, the i-quant grid tables (not started)

- **start:** 0.1 landed. The grids are the last correctness prerequisite; **84 % of the file's
  bytes are i-quant types whose dequant is undefined without them**.
- **goal:** one block per i-quant type dequantises **bit-exactly** against the publisher's own
  reference, with the tables taken from `IST-DASLab/GSQ` (Apache-2.0) — never llama.cpp (`D-02`).
- **proof:** `<run the GSQ reference dequant on one block> · <expected bytes> · <seam: GGUF block
  bytes vs the reference output> · <control: feed a mismatched type id, must fail>`.
- **state:** none yet — not started.

Then, in the order the plan declares: **1.1 remove aiter** → **1.2 one generic Triton GGUF GEMV**
(correct, not fast) → **1.3 wire it on HIP** → **1.4 the engine baseline**. Nothing downstream is
measurable until 1.4 exists.

## Open, each with the check that closes it

| open | closes with |
|---|---|
| Is a Triton GEMV as fast as the 521 GB/s hand-written kernel? | the first per-type Triton kernel, measured against `facts/memory.md`'s 521.3 |
| Does acceptance transfer from the published BF16 target to this 3.44-bpw one? | teacher-forced replay at 3.0 / 4.25 / 5.5 / 8.0 bpw of drafter |
| Which topology — TP, PP or DP? | a mapped-publish flag round trip, p99 ≥ 1000 iterations, against RCCL's 19.5 µs |
| What does the attention path cost? | a paged decode-attention kernel over a context ladder, q8 and k8v4 |
| Is the ~800 MiB launch cliff the clock ramp? | a clock-pinned re-run of the E2 fusion ladder |
| What is prefill TTFT? | `2 × 26.9e9 × tokens` FLOPs against measured achievable FLOPS |
