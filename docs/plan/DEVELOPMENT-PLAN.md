# Development plan — mould the engine to this hardware

**Stance.** The engine is ours. sglang is a scaffold we inherited, not a product we serve. Where
its assumptions are CDNA assumptions, we cut them out; where software is missing for hardware
this card *has*, we write it; where hardware is missing, we never emulate it.

This replaces "adapt sglang until it works". Each slice below is **one research question and the
implementation that answers it**, gated on evidence. No slice is "research only" or "code only".

## The four rules that order this plan

1. **Absent hardware is not a software gap.** gfx1100 has no matrix cores and no FP8. Never
   shim, emulate, or approximate them. Absent *software* for hardware it does have — notably the
   GGUF dequant-GEMV, which exists only as CUDA source that cannot compile — is ours to write.
2. **Delete, do not guard.** `aiter` is for gfx942/950. A `try/except` around its import keeps a
   dead dependency alive and hides the next failure. Remove it from the reachable path.
3. **One attributable delta per slice.** Every slice ends with a number measured against the
   previous slice's number, in the engine, with its shape printed. A slice that cannot state its
   delta has not finished.
4. **The gate is not optional.** Every kernel is checked against an fp64 CPU reference with the
   tolerance declared *before* the run, and against a data-dependent checksum. Wrong-but-fast is
   the failure mode this project has already produced once (`H-4`, IQ3_XXS at 581 GB/s).

## Phase 0 — correctness prerequisites (no GPU)

| slice | research | implementation | gate |
|---|---|---|---|
| **0.1 The recipe** | Fetch `tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S.rco-allocation.txt`; read how the publisher names types. | Diff it, tensor by tensor, against `bench/manifests/iq3s.json`. | 72 `(role, type)` pairs match exactly, or the difference is explained per tensor |
| **0.2 The grids** | Read `IST-DASLab/GSQ` (Apache-2.0) for the i-quant grid tables; confirm each type's grid and its indexing. | A table module in the repo, one per i-quant type, with provenance. | One block per i-quant type dequantises **bit-exactly** against the publisher's own reference |

These two are prerequisite to every kernel. **A7 is not a research nicety — 84 % of the file's
bytes are i-quant types whose dequant is undefined without them.**

## Phase 1 — a running engine (the gate to everything else)

| slice | research | implementation | gate |
|---|---|---|---|
| **1.1 Remove aiter** | Where the HIP default selects it; every module-scope import under `_is_hip`; `cuda_supported`/`rocm_supported` allow-lists. | Delete the dependency from the reachable path and force `--attention-backend triton`. Not a guard. | `import sglang.srt.configs.model_config` succeeds; `ModelConfig(<gguf>)` constructs; the resolved backend is `triton` |
| **1.2 A generic Triton GGUF GEMV** | The 11 types' block layouts, already documented in `docs/research/2026-10-22/03c-*`; the `ggml_mul_mat_vec_a8` contract sglang's linear layer expects. | **One** Triton kernel, parameterised by type, correct but not fast. | Per type: fp64 reference within a pre-declared tolerance, checksum match. Throughput recorded but **not** optimised |
| **1.3 Wire it on HIP** | `layers/quantization/gguf.py`'s missing HIP branch; the `ModelConfig` ROCm gate. | The HIP import branch pointing at 1.2; the gate lifted honestly. | Weights load; logits on a fixed prompt match a reference within tolerance |
| **1.4 The baseline** | — | Serve the model, greedy, fixed prompt, 32 tokens. | **Coherent text, quoted.** TTFT, decode tok/s, VRAM, and the shape printed. This is the "before" |

**Phase 1 is strictly sequential.** Nothing else can be measured until 1.4 exists.

## Phase 2 — the weight path, type by type

Each slice is independent of the others and ordered by **share of file bytes**. Each replaces
the generic kernel from 1.2 for one type, and each is measured against 1.4.

| slice | type | bytes | the research question |
|---|---|---:|---|
| **2.1** | IQ4_XS | 25.7 % | a 16-entry grid and a 4-bit nibble path — the closest shape to the measured 752 GB/s clean-4-bit reference |
| **2.2** | IQ3_XXS | 17.6 % | signs folded into the grid; the type whose fp64 gate failed as CUDA (`H-4`) — the gate is the research |
| **2.3** | Q4_K | 13.5 % | 6-bit sc/min unpack was instruction-limited at 344 words/16 w; is Triton's version better or worse? |
| **2.4** | IQ2_S | 6.8 % | the documented layout does not close — settle it from the file, or record it as unresolved |
| **2.5** | the tail | 5.4 % | Q2_K, IQ2_XS, IQ2_XXS, IQ1_M, BF16, F32 — cheap, but the model will not run without them |

Gate per slice: fp64 within tolerance, checksum match, then **tok/s against 1.4**, and the
instruction count per 32 weights (`H-2`'s method) so the gain is explained, not just observed.

## Phase 3 — the two large terms

| slice | research | implementation | gate |
|---|---|---|---|
| **3.1 Strip the engine** | What the runtime actually imports (dynamic `sys.modules`, not static AST). | Delete the dead set — 5,363 files / 1.48 M LOC measured. | Same tok/s ±1 %; the reachable count falls from 1,070 files |
| **3.2 The attention path** | The paged KV contract sglang's layer expects; q8 and k8v4 layouts. | Measure, then tune or replace the decode-attention kernel. | GB/s and ms/step at a context ladder; the resident-token boundary is where it starts to matter |
| **3.3 Speculation** | `ms(k)` is already measured and flat; the drafter budget is 3.87 ms per drafted token. | The verify path first (block-shaped, from day one), then the MTP head, then DFlash2. | tok/s at the depth the acceptance curve supports; acceptance measured by teacher forcing |

## Phase 4 — the machine, not the card

| slice | research | implementation | gate |
|---|---|---|---|
| **4.1 The round trip** | The mapped-publish protocol; RCCL's 19.5 µs floor is the number to beat. | A 10 KiB flag round trip, p99 over ≥1000 iterations, with a single-card loopback as the floor. | A latency number that decides TP vs PP vs DP |
| **4.2 Two cards in the engine** | What sglang's TP/PP plumbing already does, and what a layer split needs. | Whichever the round trip argues for. | Two-card tok/s against the one-card baseline, at B=1 and B=8 |
| **4.3 Prefill** | The compute-bound regime, no MFMA; a Triton GEMM path. | A prefill kernel, or the honest finding that Triton's GEMM is the path. | TTFT for a 4K prompt; if it exceeds 1 s it is a first-class problem |

## What this plan refuses

- **Measuring scratch kernels.** Every number from `~/e1`, `~/a2`, `~/a9` is a proxy. They may
  inform a slice; they cannot be its gate once the engine runs (1.4).
- **Emulating absent hardware.** No MFMA shim, no FP8 path, no CDNA fallback.
- **Guarding what should be deleted.** aiter, the CUDA-only quantisation methods we cannot use,
  the 5,363 dead files.
- **A number without its shape.** Every tok/s carries `(B, C, prompt length)`.

## Ordering, at a glance

```
0.1 recipe ─┐
0.2 grids ──┴─> 1.2 generic GEMV ─> 1.3 wire ─> 1.4 BASELINE ─┬─> 2.x per-type  (parallel)
 1.1 aiter ─────────────────────────┘                          ├─> 3.1 strip
                                                               ├─> 3.2 attention
                                                               ├─> 3.3 speculation
                                                               └─> 4.1 round trip ─> 4.2 two cards
                                                                   4.3 prefill (independent)
```
