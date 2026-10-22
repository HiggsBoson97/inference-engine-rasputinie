# 0001 — Engine composition: thin engine, borrowed parts

Status: **superseded by `0002-fork-sglang.md`**, 2026-10-22. Kept for the record:
the composition it proposed was rejected by the owner, who wants the engine
built inside a forked sglang. The exclusion it records still binds, and it is
the reason the GGUF reader and the dequant kernels are ours.

## Context

Six porting reports (`docs/research/`) measured what each candidate engine
actually does on gfx1100. The brief assumed sglang was the base to customise.
The evidence says otherwise:

- **sglang** implements both target models completely, including the
  gated-deltanet layer, the LLLF hybrid wiring, the ngram/PLE layers, the
  hyper-connections and both MTP heads. It has **no RDNA3 architecture helper**
  (only gfx942/gfx950/gfx1250), its AOT kernel build hard-exits on gfx1100, and
  the default attention backend on HIP is `aiter`, whose non-CK path is opt-in
  and off. At M=1 with 512 experts its Triton MoE pads 10 real rows to ~160,
  about 6% arithmetic efficiency. Forking means ~10 patch sites plus the same
  kernels, inside a CDNA-shaped tree.
- **Strata** compiles the same `.cu` sources with hipcc behind a shim, validates
  gfx1100, and already does three-tier residency and hipGraph-per-layer.
- **FreeToken** is the only engine that models the 151 MB per-sequence recurrent
  state as a cacheable currency alongside KV pages.
- **TensorFold** ships hand-written HIP GEMV kernels for gfx1100 that already
  serve the 27B.
- **DeepInfer** is SM100/Ascend only; its kernels do not exist for us.

## Decision

Build a thin engine and borrow by licence. No fork of any candidate.

| Part | Source | Licence |
|---|---|---|
| Model reference and API contract | sglang | Apache-2.0 |
| Tiering: router look-ahead, expert arena, ngram reader, hipGraph per layer | Strata | MIT |
| Hybrid radix cache: KV + GDN state + SWA as tree currencies | FreeToken | Apache-2.0 |
| 27B GEMV kernels and tuning tables | TensorFold | Apache-2.0 |
| MoE psum layout | DeepInfer | Apache-2.0 |

Host language: **Python for scheduling and model code, HIP C++ for kernels, no
Triton in the hot path**. Graph capture covers the decode step, so Python costs
one call per token against a ~28 ms token.

## Alternatives considered

- **Fork sglang.** Rejected: the arch layer has no RDNA3 case, the kernel build
  refuses the card, and the model code can be read without inheriting the tree.
- **Fork Strata as the base.** Strongest alternative. Rejected as the *base*
  because it has no confirmed dense `qwen3_5` implementation and its host layer
  is C++20 with a slower iteration loop, but it stays the source for tiering and
  graph capture.
- **Build from scratch.** Rejected: it would rebuild four measured things.
- **C++ host end to end.** Kept as a fallback. Rejected for now because the
  kernel work is identical either way and the measurement loop closes faster in
  Python.

## Consequences

- The GGUF reader and every dequant kernel are ours; an ADR already records the
  exclusion that makes this mandatory.
- Per-tensor dispatch is required: files labelled by one type hold several, and
  `general.file_type` is a label, not a promise (measured: 9 types in the 27B
  test file, four covering 96.7% of the bytes).
- Attribution work is real: TensorFold's kernel files carry no per-file SPDX
  headers, so `LICENSE` and `NOTICE` travel with the borrowed sources.
- The plan's phase order changes: TP=2 and MTP move ahead of kernel work, and
  the first kernel is a Q5_K GEMV, not a "Q4" one.
