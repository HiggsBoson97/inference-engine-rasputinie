# 0004 — FP8 is excluded as a weight format

Status: **accepted**, 2026-10-22. Owner decision. Amends
[ADR 0003](0003-representation-doctrine.md) clause 3.

## Context

ADR 0003 demoted FP8 to "a functional reference only", which was read as *keep
the checkpoint, do not optimise it*. The measurement says something stronger.

gfx1100 has no FP8 matrix instruction. Every FP8 multiply-add is therefore
emulated. The one run of the 27B FP8 checkpoint made the cost explicit:

| Measurement | Value |
|---|---|
| Decode | **857 ms/token = 1.17 tok/s** |
| Effective weight bandwidth | 14.52 GB / 0.857 s = **16.9 GB/s** |
| As a fraction of the measured wall | **2.7% of 619 GB/s** |
| Context dependence | none — 857 ms at 319 tokens and at 4,334 tokens |

Decode cost was independent of context length, so it was not attention, not KV,
and not the linear-attention state. It was the per-step weight path running
**36× below the memory wall**. That is what emulated arithmetic looks like.

## Decision

**FP8 is excluded.** No FP8 checkpoint is supported, loaded, kept as a
reference, or written a kernel for.

Weights are **quantised** (GGUF block formats, or a quantised safetensors
container). Compute is **bf16**. That is the only path.

## Consequences

- The 28.8 GiB FP8 checkpoint stops being the project's only working real-model
  reference. The reference becomes a bf16 torch implementation driven by the
  same quantised weights — see the correctness-measurement work, which shows a
  27B reference must be run layer-by-layer or offloaded on 40 GB.
- The 1.17 tok/s figure is retained as **evidence for this exclusion**, not as
  a baseline to beat. It is not a performance target and must never be quoted
  as one.
- "safetensors support" means a **quantised** safetensors container. A bf16
  safetensors checkpoint is also out: 27B in bf16 is 51.77 GiB against 40 GB of
  VRAM, so it cannot be resident at any concurrency.
- Any future proposal that reintroduces FP8 must bring a kernel that runs inside
  a supported instruction — there is no such instruction on this chip, so the
  proposal would have to change the hardware, not the software.
