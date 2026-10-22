# 01 — The target model

Qwen3.8-27B in the `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S container. Architecture
fields from the Hugging Face `config.json`; every size and count from the file itself via
`bench/02_gguf_probe.py` and `bench/03_tensor_inventory.py`. Measured 2026-10-22.

Scope: this model only (`docs/adr/0002-fork-sglang.md`). FP8 is excluded
(`docs/adr/0004-exclude-fp8.md`).

## Architecture

`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, Apache-2.0.

| Field | Value |
|---|---|
| Layers | 64: **48 GatedDeltaNet + 16 full attention**, pattern `LLLF` x16, full at 3, 7, …, 63 |
| Hidden | 5120 |
| MLP | 17408 intermediate, SiLU, no MoE |
| Full attention | 24 q heads / **4 KV heads** / head_dim 256, `attn_output_gate: true`, gate `swish` |
| Rotation | `partial_rotary_factor 0.25`, theta 10M, `mrope_interleaved`, `mrope_section [11,11,10]` |
| GatedDeltaNet | 16 key heads x 128, 48 value heads x 128, `conv1d` kernel 4 |
| Embeddings | vocab 248320, **untied**: `token_embd` IQ2_S, `output` Q4_K |
| Extra | 27-block visual tower; the MTP sibling adds one full-attention layer |

## Memory

| Item | Size |
|---|---|
| File | **11,771,546,784 B = 10.963 GiB** |
| Tensors / params / bits per weight | 851 / 26.896 B / 3.498 |
| Quant types | 11 — IQ3_S 31.0%, IQ4_XS 25.7%, IQ3_XXS 17.6%, Q4_K 13.4%, IQ2_S 6.8%, tail ≤ 2.0% |
| Read per token | **10.57 GiB** — all but `token_embd` |
| KV, bf16 / q8 per token | 64 KiB / 32 KiB |
| KV at 32K / 128K / 256K | 2 / 8 / 16 GiB |
| GatedDeltaNet state per sequence | 151 MB FP32 + 8 MB conv = **159 MB, fixed, not shareable** |

The KV cache is small because only 16 of 64 layers keep one. The 48 linear layers' recurrent
state is the bigger per-sequence cost, and it cannot be shared between prefixes.

Byte budget: MLP 62.90 %, GDN projections 20.99 %, embedding + head 9.54 %, full attention
6.07 %, GDN state ops 0.47 %, norms 0.02 % (`docs/plan/TENSOR-INVENTORY.md` §4).

## Decode ceiling

One card, everything resident. Wall 776 GB/s (`bench/gfx1100-probes/WALL-2026-10-22.md`).

| Setup | Bytes per token | Measured |
|---|---|---|
| Wall, weights only | 10.57 GiB | 14.6 ms ≈ 68 tok/s |
| Per-tensor dispatch, 401 launches | 11.2955 GB | 712.3 GB/s, **60.6 tok/s** |
| 2–4 layers per launch | 11.2955 GB | 774.2 GB/s, **65.8 tok/s** (99.7 % of wall) |
| 8 layers per launch | — | 691.4 GB/s — worse than per-tensor |

`bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md`. The per-tensor deficit is launch overhead
(401 x 3.3 us = 1.31 ms); fusion to 2–4 layers recovers it, 8 layers crosses the ~800 MiB
per-launch footprint cliff.

Two cards: cross-GPU copy is 14.2 GiB/s — host-bounce speed, so an all-reduce is a real
cost, not a fabric shortcut. Whether to pay it is **open** (`docs/plan/REQUIREMENTS.md` A4):
two independent replicas cost zero PCIe and give two independent streams, while TP=2 halves
the per-card weight read but pays 2.64–3.8 ms/token in all-reduce. The measurements
conflict — `2026-10-22/05-serving-knobs.md` has TP=2 ahead, `18-tp-roccl.md` has it behind
at concurrency 32 — so no topology is recorded as settled.

## The MTP sibling

`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`: 12,120,016,960 B, 866 tensors. Layer 64 is
full-attention and carries four `nextn.*` tensors, every matrix in it is Q6_K, and it
**shares `output.weight`** — no second LM head. All 851 base tensors are identical in type,
shape and size, so one loader reads both files.
