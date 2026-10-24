# facts — the file and its quantisation

What the model is, what is in it, and what each type actually runs at.
`_archive/2026-10-24/…` paths are pre-migration locations.

## Geometry — the numbers every kernel is shaped by

| fact | value | instrument | date | verified-by |
|---|---|---|---|---|
| File | `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`, **11,771,546,784 B** (10.963 GiB), GGUF v3, `arch=qwen35` | `bench/02_gguf_probe.py` | 2026-10-22 | `_archive/2026-10-24/docs/00-facts.md` §The file |
| Parse reconciles | metadata + Σ tensor bytes = file size, **0 B** | same | 2026-10-22 | same |
| Tensors / params / bpw | **851** / **26,895,998,464** (26.896 B) / **3.498** | `bench/03_tensor_inventory.py` | 2026-10-22 | `docs/plan/TENSOR-INVENTORY.md` |
| Layers | **64** = 48 GatedDeltaNet + 16 full attention | same | 2026-10-22 | same |
| hidden / intermediate / vocab | 5120 / 17408 / 248320 | same, cross-checked against TensorFold's `qwen35` geometry | 2026-10-23 | same + `_archive/2026-10-24/docs/research/2026-10-23/s5-donor-transfer.md` |
| Attention heads | q 24, kv **4**, v 48, head_dim 128, context 262144 | same | 2026-10-22 | `docs/plan/TENSOR-INVENTORY.md` |
| Per-step read | **10.57 GiB** | Σ tensor bytes | 2026-10-22 | `_archive/2026-10-24/docs/00-facts.md` |
| MTP sibling | `…-IQ3_S-mtp.gguf`, **12,120,016,960 B**, 866 tensors; layer 64 is full-attention with four `nextn.*`; every matrix there is **Q6_K**; shares `output.weight` (no second head) | `bench/02_gguf_probe.py` | 2026-10-22 | `_archive/2026-10-24/docs/01-models.md` §The MTP sibling |

## The 11 types — this is a mixed-precision file, not an "IQ3_S model"

| type | id | GiB | share | bpw | block B/256 | dot rate measured |
|---|---:|---:|---:|---:|---:|---:|
| IQ3_S | 21 | 3.393 | **31.0 %** | 3.4375 | 110 | **521.3** ✓ |
| IQ4_XS | 23 | 2.814 | 25.7 % | 4.25 | 136 | **502.1** ✓ |
| IQ3_XXS | 18 | 1.933 | 17.6 % | 3.0625 | 98 | **INCONCLUSIVE** |
| Q4_K | 12 | 1.474 | 13.5 % | 4.50 | 144 | **452.2** ✓ |
| IQ2_S | 22 | 0.745 | 6.8 % | 2.5625 | 82 | **INCONCLUSIVE** |
| Q2_K | 10 | 0.224 | 2.05 % | 2.625 | 84 | not measured |
| IQ2_XS | 17 | 0.206 | 1.88 % | 2.3125 | 74 | not measured |
| IQ2_XXS | 16 | 0.092 | 0.84 % | 2.0625 | 66 | not measured |
| BF16 | 30 | 0.044 | 0.40 % | 16.0 | — | n/a |
| IQ1_M | 29 | 0.018 | 0.17 % | 1.75 | 56 | not measured |
| F32 | 0 | 0.010 | 0.09 % | 32.0 | — | n/a |

**IQ3_S is 31 % of the bytes; 69 % is something else.** Four types carry 64 %.

## Load floors — the layout is innocent

Each type's block-strided read against an aligned `uint4` stream of the *same bytes*:

| type | load floor GB/s | % of wall | its own aligned control | penalty |
|---|---:|---:|---:|---:|
| IQ4_XS | 637.1 | 82.1 | 657.4 | −3.1 % |
| IQ3_XXS | 643.6 | 82.9 | 668.4 | −3.7 % |
| IQ2_S | 656.6 | 84.6 | 659.4 | −0.4 % |
| Q4_K | 659.6 | 85.0 | 676.6 | −2.5 % |

**No quant format pays a coalescing penalty on this card.** The 110 B / 98 B / 82 B strides are free.

## The two that could not be gated — `INCONCLUSIVE`, not measured

| type | what happened | why it is not a rate |
|---|---|---|
| IQ3_XXS | fp64 worst rel err **1.43** on 32/32 rows and the checksum **mismatched**; the discarded rows ran **581–592 GB/s** | wrong-but-fast: above IQ3_S's gated 521. The block accounting is exact (2 d + 64 index + 32 sign), so the kernel is wrong, not the layout |
| IQ2_S | the documented layout **does not close**: d 2 + qs 64 + qh 8 + scales 8 = 82, leaving no room for the documented 32 B sign plane; no fp16 scale signature anywhere in bytes 0–80 | the layout is unresolved, so any kernel is guessing |

Both keep their **load floors** (above) and nothing else. Closing them is unit `0.2`.

## GSQ-RCO is a method, not a container format

The publisher's card: *"The resulting files are standard GGUF and run unmodified in llama.cpp,
Ollama, and LM Studio."*

- **GSQ** (Gumbel-Softmax Quantization) learns per-coordinate grid assignments and per-group scales.
- **RCO** (Riemannian Constrained Optimization) picks one of K quant types per tensor under a size budget.

So there is no custom block layout. What is file-specific is the **recipe**, shipped as data:
`tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S.rco-allocation.txt` on the HF repo. Licence: Apache-2.0.

## Grids — a correctness prerequisite, and the one fact that contradicts an older record

The i-quant types (all except Q4_K, Q2_K, BF16, F32) dequantise through **grid tables**, and
**84 % of the file's bytes** are those types. The tables are the standard GGML ones (implied by
"standard GGUF"), and the legitimate source is `github.com/IST-DASLab/GSQ` (Apache-2.0) — never
llama.cpp (`D-01`).

**The contradiction, resolved.** `docs/plan/RECORD.md` calls IQ3_S "measured, gated" while
`docs/plan/REQUIREMENTS.md` A7 says the codebook was a placeholder. Both are true of different
things, and this is a category error, not a disagreement:

- **measured:** the rate, 521.3 GB/s, gated by a self-consistent checksum replayed from the same bytes;
- **assumed:** the *values*. `H-1`'s harness used a placeholder 512-entry table, so the dequantised
  weights are plausible in distribution (mean |w| 0.0145, rms 0.0172) but are **not guaranteed to be
  the model's weights**.

`assumed:` every i-quant rate in this file is a rate over *correct block accounting*, not over the
model's true weights. **The check that closes it:** unit `0.2` — the publisher's GSQ grid tables,
verified bit-exactly on one block per type.
