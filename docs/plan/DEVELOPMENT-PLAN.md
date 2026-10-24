# Development plan — every row is a unit

A unit is a **start**, a **goal**, and a **proof**. It lands when the proof passes *and its control
fails*. No unit is "research only" or "code only"; the research is whatever the proof needs.

**Phase 0 order, declared once so no session re-litigates it:**
**0.1 (the recipe diff — cheapest, validates the inventory) → 0.2 (the grids — largest fan-out,
gates every kernel's correctness) → 1.1 (remove aiter).**

**The stance.** The engine is ours. sglang is a scaffold we inherited, not a product we serve.
Where its assumptions are CDNA assumptions we cut them out; where software is missing for hardware
this card *has*, we write it; where hardware is missing, we never emulate it (`AGENTS.md` 1–4, 9–10).

Proofs follow `check.md`: `command · expected output · seam · negative control`. The commands below
are the shape; the exact invocation is written when the unit starts.

## Phase 0 — correctness prerequisites (no GPU)

| unit | start | goal | proof |
|---|---|---|---|
| **0.1** ✅ | the inventory + the publisher's `rco-allocation.txt` | prove our per-tensor recipe is the publisher's | `bench/04_recipe_diff.py <gguf> <alloc>` → `matched 851/851`, exit 0 · control `--mutate blk.0.attn_qkv.weight=Q4_K` → 1 mismatch, exit 1. **DONE 2026-10-24** |
| **0.2** | 0.1 landed | one block per i-quant type dequantises **bit-exactly** against the publisher's reference; tables from `IST-DASLab/GSQ` (Apache-2.0), never llama.cpp | run the GSQ reference on one block · expected byte string · seam: GGUF block bytes vs reference output · control: a wrong type id must fail |

**0.2 gates 7 of the 11 types = 84 % of the bytes.** Until it lands, every i-quant rate in
`facts/quant.md` is a rate over correct *accounting*, not over the model's true weights — the
`assumed:` markers there are this unit's open items.

## Phase 1 — a running engine (strictly sequential)

| unit | start | goal | proof |
|---|---|---|---|
| **1.1** | 0.1, 0.2 | remove `aiter` from the reachable path and force `--attention-backend triton` — delete, not guard (`D-07`) | `python -c "import sglang.srt.configs.model_config"` exits 0 · seam: module import + resolved backend · control: re-add the module-scope import, must fail |
| **1.2** | 1.1 | **one** Triton GGUF GEMV covering all 11 types — correct, not fast | per type: fp64 reference within a tolerance declared first · seam: the linear-layer op · control: a transposed index must fail |
| **1.3** | 1.2 | wire it on HIP: the `gguf.py` HIP branch + an honest ROCm gate | weights load; logits on a fixed prompt match a reference · control: a wrong gate constant must fail to load |
| **1.4** | 1.3 | **the baseline**: the engine serves the file | 32 coherent greedy tokens, quoted; TTFT, decode tok/s, VRAM, and `(B, C)` printed · control: a sabotaged kernel must produce incoherent text |

**Phase 1 is strictly sequential.** Nothing downstream is measurable until 1.4 exists.

## Phase 2 — the weight path, type by type (independent; ordered by byte share)

Each replaces the generic kernel from 1.2 for one type and is measured against 1.4.

| unit | type | share | goal |
|---|---|---:|---|
| 2.1 | IQ4_XS | 25.7 % | a natively-shaped kernel — the closest to the measured 752 GB/s clean-4-bit reference |
| 2.2 | IQ3_XXS | 17.6 % | the type whose CUDA gate failed at fp64 error 1.43 — the gate *is* the research |
| 2.3 | Q4_K | 13.5 % | beat the 344 words/16 w the CUDA version needed |
| 2.4 | IQ2_S | 6.8 % | settle the layout that does not close, or record it unresolved |
| 2.5 | tail | 5.4 % | Q2_K, IQ2_XS, IQ2_XXS, IQ1_M, BF16, F32 — cheap, and the model needs them |

Proof for each: fp64 within tolerance + checksum, then tok/s against 1.4, with the instruction
count per 32 weights so the gain is **explained**, not merely observed.

## Phase 3 — the two large terms

| unit | start | goal | proof |
|---|---|---|---|
| **3.1** | 1.4 | strip to the runtime closure — 5,363 dead files / 1.48 M LOC measured | same tok/s ±1 %; reachable files fall from 1,070 · control: delete a file the closure needs, must fail |
| **3.2** | 1.4 | the attention path in situ: q8 and k8v4 | GB/s and ms/step over a context ladder · control: a wrong scale must fail the reference |
| **3.3** | 1.4 | speculation: verify path first, then MTP, then DFlash2 (`D-08`) | tok/s at the depth acceptance supports; acceptance by teacher forcing · control: an always-accept drafter must corrupt the output |

## Phase 4 — the machine, not the card

| unit | start | goal | proof |
|---|---|---|---|
| **4.1** | — | a mapped-publish flag round trip | p99 over ≥1000 iterations vs RCCL's 19.5 µs floor; single-card loopback as the floor · control: a stale-flag ack must be rejected |
| **4.2** | 4.1 | two cards in the engine, whichever topology the round trip argues (`D-09`) | two-card tok/s vs 1.4, at B=1 and B=8 |
| **4.3** | 1.4 | prefill TTFT | a 4K prompt's TTFT; if > 1 s it is a first-class problem |

## What this plan refuses

- **Measuring scratch kernels.** They may inform a unit; they cannot be its proof once 1.4 exists.
- **Emulating absent hardware.** No MFMA shim, no FP8 path, no CDNA fallback.
- **Guarding what should be deleted.** aiter, the CUDA-only methods we cannot use, the dead set.
- **A number without its shape.** Every tok/s carries `(B, C, prompt length)`.

## Ordering, at a glance

```
0.1 recipe ✅─┐
0.2 grids ───┴─> 1.2 generic GEMV ─> 1.3 wire ─> 1.4 BASELINE ─┬─> 2.x per-type  (parallel)
 1.1 aiter ─────────────────────────┘                          ├─> 3.1 strip
                                                               ├─> 3.2 attention
                                                               ├─> 3.3 speculation
                                                               └─> 4.1 round trip ─> 4.2 two cards
                                                                   4.3 prefill (independent)
```
