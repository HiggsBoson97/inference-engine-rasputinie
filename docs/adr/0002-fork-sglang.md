# 0002 — Fork sglang and rebuild its gfx1100 path

Status: **accepted**, 2026-10-22, owner decision. Supersedes 0001.

## Decision

1. `engine/` in this repository is **our fork of sglang**, vendored at commit
   `d0a7565` (`engine/UPSTREAM.md`). We own that tree. Optimisation work happens
   inside it, in place.
2. The other engines are **sources of strategy**, not parts to assemble. When a
   technique from Strata, FreeToken, TensorFold or DeepInfer is worth having, we
   port it and implement it for gfx1100 in our tree. We do not bolt their code
   together and call it an engine.
3. Upstream sglang is a **reading reference, not a rebase target.** The fork will
   diverge; that is the point.

## Why this supersedes 0001

0001 proposed assembling a thin engine from four donor projects. That was wrong
on two axes. It optimised for "least work" when the goal is "most performance on
this exact machine", and it assumed sglang could not be the base because its
ROCm layer targets CDNA. But sglang already implements both of our model
families completely — the gated-deltanet layer, the LLLF hybrid wiring, the MTP
head — and it already carries the scheduler, the radix cache and the graph
capture infrastructure. The missing part is the gfx1100 layer and the kernels.
That is a rewrite *inside* the tree, which is exactly the work the owner asked
for, and it is a much better starting position than an empty directory.

## What we rewrite, and what we inherit

| Layer | Disposition |
|---|---|
| HTTP/API server, tokenizer, request lifecycle | **Inherit.** No performance payoff in rewriting it |
| Model code (`models/qwen3_5.py`, MTP) | **Inherit, then modify** where the gfx1100 path needs it |
| Architecture gating (`utils/common.py`, backend defaults) | **Rewrite.** No RDNA3 helper exists; the HIP default is a CK backend that gfx1100 cannot run |
| Kernel build (`sgl_kernel` AOT, `setup_rocm.py`) | **Rewrite.** It hard-exits unless gfx942/950/1250 |
| Quantised matmul / GEMV | **Replace.** Native GGUF loader and our own HIP kernels |
| Attention | **Replace** the CK/aiter path with a wave32 gfx1100 path; keep Triton as the correctness fallback |
| Scheduler, radix cache, continuous batching | **Inherit, then extend** for the hybrid recurrent state and q8 KV |
| Graph capture | **Inherit** (it already maps to hipGraph), then verify on GDN + MTP |

## Consequences

- The first enabling change is small and blocking: admit `gfx1100` to the AOT
  kernel build, and add an `is_rdna3_supported()` helper beside the gfx942/950
  and gfx1250 ones so the ~71 gfx95 call sites can be audited instead of being
  silently false.
- The default attention backend on HIP must stop being `aiter` on RDNA3.
- The GGUF reader and every dequant kernel are ours. `0001`'s exclusion of the
  obvious reference still binds, and the consequence still holds: this is weeks
  of work, stated up front.
- Licensing: sglang is Apache-2.0 and its `LICENSE` travels with the tree. New
  files we author carry our own headers.
- Scope is **Qwen3.8-27B only**. Flash-Next work is out.

> **Ratified by the owner, 2026-10-22.** This clause was previously an assertion
> with no owner quote behind it (`docs/plan/REQUIREMENTS-DRAFT.md` C2 flagged the
> gap). The owner has now stated the scope directly: *"this is the model this
> inference engine should 100% support, the one and only model, at IQ3_S with
> MTP."* The narrowing stands, ratified. The MTP head is in scope from the first
> milestone (O11); DFlash2 is a later option.
