# 0001 — llama.cpp is excluded

Status: accepted, 2026-10-22. Owner decision, not a technical preference.

## Context

We need three things that llama.cpp appears to provide: a GGUF weight reader, a
family of dequant-plus-multiply kernels for k-quants and i-quants, and an
existing engine to benchmark against. It is the obvious reference for all three,
and a first pass used it that way.

## Decision

llama.cpp is excluded from this project completely:

1. It is not a baseline. No number in this repository comes from it.
2. It is not a reference implementation. No kernel, loader, layout or heuristic
   is copied from it, and its source is not read while writing ours.
3. It is not installed on `rasputin`. The clone and the HIP build created during
   the first session were deleted.

## Consequences

- **The GGUF reader is ours.** Written against the GGUF v2/v3 format
  specification. FreeToken, Strata and TensorFold contain independent GGUF
  readers in Python, C++ and Zig; those are the permitted source material.
- **The dequant and GEMV kernels are ours.** Written against the tensor-type
  block layouts in the specification. EXL3 layouts come from exllamav3; GSQ-RCO
  layouts come from ISTA-DASLab's material.
- **The correctness oracle is a torch / Hugging Face reference forward**, not
  another engine's output. Every kernel is checked against a CPU or torch
  implementation of its own arithmetic before any speed number is recorded.
- **Baselines are sglang** (FP8, TP=1 and TP=2) and, if it builds on gfx1100,
  Strata. The memory-bandwidth ceiling in `docs/00-facts.md` is the target that
  replaces "beat the other engine".
- **Cost:** a native GGUF path is now a build item of several weeks, not a thin
  interop layer. That cost is accepted and must not be hidden in estimates.

## Enforcement

The rig's gate set carries a forbidden-dependency check: a slice fails if
`llama.cpp`, `llama-bench`, `ggml`, or a vendored copy appears in the tree or in
a build manifest. The check exists because the shortcut is tempting, not because
the rule is unclear.
