# 01 — The two target models

Facts come from the Hugging Face `config.json` and `model.safetensors.index.json`
files, fetched 2026-10-22. They are not guesses.

> **Scope note, 2026-10-22:** Qwen3.8-Flash-Next is now **out of scope**. The
> engine targets Qwen3.8-27B only (`docs/02-plan.md`, `docs/adr/0002`). The
> Flash-Next section stays as measured reference material for the day the
> question returns; nothing in the plan depends on it.

## Qwen3.8-27B

`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, Apache-2.0.

| Field | Value |
|---|---|
| Layers | 64: 48 `linear_attention`, 16 `full_attention`, pattern `LLLF` x 16 |
| Hidden | 5120 |
| Full attention | 24 q heads, **4 KV heads**, head_dim 256, `attn_output_gate: true`, gate `swish` |
| Rotation | `partial_rotary_factor: 0.25`, theta 10M, `mrope_interleaved`, `mrope_section [11,11,10]` |
| Linear attention | gated deltanet: 16 key heads x 128, 48 value heads x 128, `conv1d` kernel 4 |
| MLP | 17408 intermediate, SiLU, no MoE |
| Vocab | 248320, untied embeddings |
| Extra | 27-block visual tower, 1 MTP layer |
| Params | 27.78B |

### Memory

| Item | Size |
|---|---|
| Weights, BF16 | 51.77 GiB |
| Weights, Q8_0 | 27.05 GiB |
| Weights, Q6_K | 20.47 GiB |
| Weights, Q5_K_M | 18.41 GiB |
| Weights, Q4_K_XL | 16.35 GiB |
| KV per token | 64 KiB (16 layers x 4 heads x 256 x 2 x 2 B) |
| KV at 32K / 128K / 256K context | 2 GiB / 8 GiB / 16 GiB |
| Recurrent state per sequence | 151 MB in FP32 (48 x 48 x 128 x 128 x 4 B) plus 8 MB conv state |

The KV cache is small because only 16 of 64 layers keep one. The recurrent
state of the 48 linear layers is the bigger per-user cost: 159 MB per sequence,
fixed, and it cannot be shared between different prefixes.

### Decode ceilings

Weight bytes divided by 592 GiB/s, before overheads.

| Setup | Bytes read per token | Ceiling |
|---|---|---|
| 1 card, Q4_K_XL | 16.35 GiB | 35 token/s |
| 2 cards, TP=2, Q4_K_XL | 8.18 GiB per card | 70 token/s |
| 2 cards, TP=2, Q8_0 | 13.5 GiB per card | 43 token/s |
| 2 cards, DP=2, Q4_K_XL (one copy per card) | 16.35 GiB per card | 35 token/s per stream, 70 total |

Add about 86 us per token for TP=2 all-reduce. Subtract 30% for real-world
efficiency. The MTP layer should recover 1.3x to 1.8x at batch 1.

## Qwen3.8-Flash-Next

`Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp`, 180B total.

| Field | Value |
|---|---|
| Layers | 48: 36 `linear_attention`, 12 `full_attention` |
| Hidden | 2560 |
| Full attention | 24 q heads, **2 KV heads**, head_dim 256, gate `sigmoid` |
| Sparse attention | indexer: 4 heads, 1 KV head, dim 128, budget 2048, compress ratio 4 |
| MoE | **512 experts, top-10**, moe_intermediate 640, shared expert 640 |
| Hyper-connections | `hc_count 4`, `hc_lowrank 320`, one set per attention and per MLP |
| PLE / ngram | per-layer embedding in layer 1; **ngram table ~52B params** |
| MTP | 1 hybrid layer |
| KV per token | 24 KiB (12 layers x 2 heads x 256 x 2 x 2 B) |
| KV at 128K context | 3 GiB |
| Recurrent state per sequence | 113 MB in FP32 |

### Where the 180B lives

Measured from the shard sizes, BF16 total 335.28 GiB:

| Part | GiB | Share | Params |
|---|---|---|---|
| Routed experts | 236.0 | 70.4% | ~127B |
| ngram embedding table | 97.1 | 29.0% | ~52B |
| Everything else | 2.2 | 0.6% | ~1B |

Two facts follow, and they decide the whole Flash-Next design:

1. **29% of the model is one sparse lookup table in one layer.** It is read by
   index, a few rows per token. It must live on NVMe behind a small hot cache.
   Streaming it would be a design error.
2. **70% of the model is experts, and only 10 of 512 fire per token.** Expert
   paging with a hot-set cache is the lever. The routing skew is what makes it
   work.

### Decode ceilings

Active bytes per token at ~3.5 bits per weight: 10 experts x 3 matrices x
2560 x 640 x 48 layers = 2.36B parameters, about 1.03 GB per token, plus the
ngram lookup.

| Where the active experts live | Time per token | Ceiling |
|---|---|---|
| All in VRAM | 1.03 GB / 1183 GiB/s = 0.9 ms | compute-bound, over 300 token/s |
| 60% VRAM, 40% RAM | 0.6 ms + 8.9 ms | about 95 token/s |
| All in RAM | 1.03 GB / 46.4 GiB/s = 22 ms | about 45 token/s |
| 10% of traffic from NVMe | plus 19 ms | about 25 token/s |

The target for Flash-Next is 40 to 60 token/s with the ngram table on NVMe and
the hot experts pinned in VRAM. That would be a 180B model at usable speed on a
desktop.

## Quant availability

Both models have GGUF quants from multiple sources. `ISTA-DASLab` publishes
GSQ-RCO quants; `unsloth` publishes k-quants and i-quants; MTP weights ship as
separate GGUF files.

| Model | Quant | Size |
|---|---|---|
| 27B | Q4_K_XL | 16.35 GiB |
| 27B | Q8_0 | 27.05 GiB |
| Flash-Next | IQ2_XS | 63.35 GiB |
| Flash-Next | IQ3_XXS | 70.63 GiB |
| Flash-Next | IQ3_S | 77.88 GiB |
