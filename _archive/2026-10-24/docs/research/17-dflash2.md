# DFlash2 — sources and mechanics

## What it is

DFlash2 is a **block-diffusion drafter**, not an autoregressive draft model (EAGLE-3/Medusa) and not self-speculation. Sources: the ICML 2026 paper *DFlash: Block Diffusion for Flash Speculative Decoding* (Chen, Liang, Liu, arXiv:2602.06036), the Z Lab project page, and the release note "DFlash 2: Keep Drafting Parallel" (Inco AI, Aug 2026). DFlash 2 has no paper of its own; it is v2 of that drafter, documented on the `z-lab/Qwen3.8-27B-DFlash2` card and the Inco blog.

1. The **target**'s hidden states at layers 5, 19, 33, 47, 61 of 64 are fused by `fc` into one context vector per token (`config.json` → `target_layer_ids`; `models/dflash.py:687-700`).
2. That context is injected into the **KV of every draft layer**, not just layer 0 as in EAGLE-3 (paper ablation: τ 3.5→4.2 on GSM8K at fixed depth).
3. The drafter gets an 8-slot block: slot 0 is the last verified token, slots 1-7 are `mask_token_id` 248070, as the kernel writes it (`kernels/ops/speculative/dflash.py:180-181`). One **non-causal** forward denoises all 8 positions (`is_causal: false`, all layers `sliding_attention`, window 2048); intra-block coherence comes from two-tap grouped convolutions (kernel 2, group 16).
4. Per position the **target's LM head** returns the top 16 candidates (`models/dflash.py:1232-1240`).
5. A **candidate selector** with learned predecessor/successor codebooks [vocab, 256] scores the K×K transitions across slots (`models/dflash.py:1103-1160`) and walks **one** coherent path, emitting 7 draft tokens plus a per-slot `q`.
6. One target verify pass accepts the longest prefix where `candidates[:,1:] == target_predict[:,:-1]`, then appends the target's next token as bonus (`kernels/ops/speculative/dflash.py:348-367`): up to 8 tokens per forward, linearly verified, not a tree (`dflash_info.py:39`).

**Versus MTP.** Qwen3.8's MTP is one extra decoder layer plus `nextn.eh_proj` (424,699,392 params / 849.4 MB bf16) predicting **one** token per step autoregressively: each step is a separate small forward plus a full LM-head read, so cost grows with depth and acceptance saturates (ROCm blog Figure 2: MTP plateaus near 245 tok/s past budget 8, DFlash reaches 396 at block 16). DFlash2's draft cost is **flat** — one drafter read plus one batched LM-head read for all 8 positions.

That is the source of "4× vs 2×". Acceptance here: DFlash2 3.35-4.14 / 3.72-4.33 / 3.83-4.36 across W4A16 / W4A4 / W4A4-W8A8 vs MTP 2.60-2.97 / 2.60-2.92 / 2.65-2.87 (card, accept-length bullet).

## The drafter artifact

From `FP8/DFlash2-FP8/{config.json,manifest.json}` plus an HTTP range read of the safetensors header:

- **Architecture** `DFlash2DraftModel`, `model_type: qwen3`, 5 layers, hidden 5120, intermediate 17408, head_dim **128**, 32 q / 8 kv (GQA 4 — *not* the target's 24/4 @ 256), `is_causal: false`, window 2048, vocab 248320, `block_size: 8`, `selector_rank: 256`, `selector_top_k: 16`, `conv_kernel_size: 2`, `conv_group_size: 16`, `mask_token_id: 248070`, `num_target_layers: 64`.
- **Size**: 101 tensors, 2,407,379,992 bytes. 20 FP8 E4M3 tensors, 128×128 block scales (o/gate/up/down × 5 layers = 1.44 GB), 61 BF16 (0.97 GB). ≈1.92B params.
- **No embedding, no LM head** — neither tensor exists. Stated at `models/dflash.py:1-3` and `:1240-1242` ("The draft has no head of its own; the worker points this at the target's"). The **tokenizer is the target's**; the DFlash2 directory ships none.
- **Bytes**: `fc.weight [5120, 25600]` 262 MB (5×5120 fused — direct evidence of five source layers); codebooks `[248320, 256]` 127 MB each; per layer `kernel_projection [1280, 5120]` 13.1 MB × 2.
- **Context**: five aux hidden states (≈410 KB per verify step) plus its own 20 KiB/token KV (`dflash_utils.py:128-136`).

## How a server runs it (sglang paths)

No build required — DFlash/DFlash2 is first-class in the vendored tree: enum `SpeculativeAlgorithm.DFLASH` (`spec/spec_info.py:39`), handler (`arg_groups/speculative_hook.py:286-440`), worker (`spec/dflash_worker_v2.py`), model (`models/dflash.py:638`; `:1224` `DFlash2DraftModel`, "Reuses the DFLASH speculative worker"), Triton kernels (`kernels/ops/speculative/dflash.py`).

CLI (card L696-699; block alias at `fields/spec.py:70-73`):

```
--speculative-algorithm DFLASH --speculative-draft-model-path ./FP8/DFlash2-FP8
--speculative-draft-model-quantization fp8 --speculative-num-draft-tokens 8
```

Constraints: `pp_size` must be 1 (`:307-311`); dp-attention rejected on non-NPU (`:303-306`); device must be `cuda*` (`:293-295`) — ROCm reports `cuda`, so this passes; `num_steps`/`eagle_topk` forced to 1 (`:316-355`), confirming no autoregressive loop. Draft backends are limited to flashinfer/fa3/fa4/triton/ascend/trtllm_mha/intel_xpu (`choices.py:103-111`) with an explicit triton fallback on HIP (`:855-860`) — **triton is gfx1100's only viable draft backend**. The non-causal block is handled: `triton_backend.py:1747-1762` sets `causal = False` for `AttentionType.ENCODER_ONLY`, which `is_causal: false` resolves to (`models/dflash.py:157-161`).

MTP is not a separate algorithm: it runs as `--speculative-algorithm EAGLE` against the bundled head (card L678-681). `FROZEN_KV_MTP` (`spec_info.py:44`) is unrelated.

What must still be built, by risk: (1) FP8 128×128-block draft load on ROCm; (2) guarded Triton attention for 32/8/128 non-causal + window 2048 at block 8; (3) `selector_walk_triton` / `_score_edges` under wave32 (untested anywhere); (4) **`flashinfer.top_k` is absent on ROCm**, so the selector falls back to `torch.topk`, which "roughly halves end-to-end throughput on a large vocabulary" (`models/dflash.py:1119-1123`) — the largest gfx1100 tax in the path; (5) GDN verify + graph capture at block 8, flagged untested at `docs/research/14-sglang-gfx1100.md:72,123`.

## Cost on gfx1100 (estimates shown)

```
tok/s = (A + 1) / (t_draft + t_verify)      A = accepted drafts, +1 = bonus token
```

Draft step reads 2.407 GB (drafter) + 1.271 GB (lm_head, 8×5120×248320 FP8) = **3.68 GB**. At 592 GB/s/card, TP=2: 1.84 GB/card → 3.1 ms at 100 % BW, **4.8 ms at 65 %**; TP=1: 6.2 / 9.6 ms.

Verify: FP8 text weights 29.47 GB, TP=2 → 14.7 GB/card → 24.9 ms at 100 %, 38.3 ms at 65 %. The card's 46 tok/s (21.7 ms/token on 1.79 TB/s HBM) implies large non-bandwidth overhead, so **t_verify ≈ 42 ms for a block-8 verify at 100K on gfx1100 (estimate)**; unspeculated ≈ 26 tok/s.

| case | t_draft | A+1 | tok/s (est.) |
|---|---:|---:|---:|
| unspeculated | — | 1 | 26 |
| MTP, 3 steps | 8.3 ms | 2.6 | 52 |
| MTP, 3 steps | 8.3 ms | 2.9 | 58 |
| DFlash2 block 8, TP=2 | 4.8 ms | 3.0 / 3.9 | 64 / 83 |

Break-even is `A+1 > (t_draft + t_verify)/t_verify_single = 1.22` — speculation pays even if two thirds of every block is rejected. DFlash2/MTP ≈ 1.65×, against 2× on the card's hardware and 1.4× in its C8 table. Draft GEMMs are memory-bound; 592 GB/s, not FLOPs, sets the price.

Memory on 40 GB: 29.47 GB target + 2.41 GB draft + draft KV (20 KiB/token, 40 MiB/seq at the 2048 window). Speculation costs ~7 % of the box.

## Other DFlash2 drafters

`incoai/Qwen3.8-27B-DFlash2` is canonical: 3,848.8 MB bf16, 359k downloads, `config.json` **identical** to the shipped FP8 one (same 5 layers, `target_layer_ids`, block 8, rank 256, top 16) — the nerkyor draft is a block-FP8 quantization of Z Lab's, not bespoke. `z-lab/Qwen3.8-27B-DFlash2` mirrors it. Smaller: `lued/…-W8` 2,172.7 MB, `Launch80/…-Int4-baked-drafter` 1,521.0 MB, `syvai/…-W4A16` 1,280.6 MB. Everything else in the `dflash2` search (Ornith, Muse-Glimmer, Qwen3.6-35B-A3B, GLM-5.3) targets another model.

For 40 GB the shipped 2.24 GB FP8 draft is right: smallest non-4-bit option, greedy accept 5.05 at 171 tok/s per the card. Fall back to the 3.85 GB bf16 original only if ROCm FP8 block-scale loading breaks; W4/W8 save ~1 GB at unknown acceptance — the wrong trade with 7 GB of headroom.

## Recommendation and first experiment

**MTP first, DFlash2 second.** MTP is 849 MB, needs no new loader, and exercises the risky plumbing DFlash2 also needs: multi-token verify against 48 GDN layers with state rollback, graph capture at a non-1 draft width, acceptance on *this* finetune. DFlash2 adds a five-layer non-causal draft forward, a 248k-vocab selector, an FP8 draft load and the torch.topk fallback. Healthy MTP acceptance makes DFlash2 an integration job; a collapse predicts the same for DFlash2, learned for a third of the effort.

First experiment (FP8 tier, one server, no builds): (1) baseline at `--context-length 32768 --max-running-requests 1`, greedy, 6 coding prompts, 1024-token cap → tok/s; (2) card's EAGLE command (steps 3, topk 1, draft tokens 4) → accept length, tok/s, whether verify + GDN rollback is coherent; (3) card's DFLASH command at block 8 — watch for the "loads fine, accepts nothing" mode the card documents.

Go/no-go: DFlash2 ships if **accept length ≥ 2.8 at block 8 and ≥ 1.35× unspeculated tok/s**; MTP as fallback at accept ≥ 2.0 and ≥ 1.2×. Below that the drafter is not transferring to this finetune/backend and no kernel work fixes it.

**Main risk.** The drafter was trained on Qwen3.8's hidden states; this target is four rounds of SFT + RLOO downstream. The card's 3.35-4.36 at 100K says it survives *this* finetune on *that* engine — but any change of quant format or attention backend shifts hidden states, and the precedent is brutal: an older per-tensor-scale draft "loaded without error but accepted almost no draft tokens" (card L397). Cross-engine acceptance, not the algorithm, is the risk.

## Claims I could not verify

- Whether Triton on gfx1100 runs non-causal, window-2048 attention at 32/8/128 over an 8-token block (path exists, no test found), whether the 2048 window is honoured in KV *pool sizing* (`dflash_utils.py:139-172` sums a per-token cell), and the wave32 cost of `selector_walk_triton` / `_score_edges`.
- Card inconsistency: headline "~180 tok/s DFlash2 vs 91 MTP vs 46 none", but its C8 table gives 132.4 per-request DFlash2 / 91.3 MTP, its single-request note 171-178 — different runs and draft revisions.
- MTP depth: card says NInfer accepts 1-5 draft tokens; Z Lab says "seven-token MTP" and benchmarks 7. FP8 block dequant for the *draft* on ROCm: probably shared with the target body, no gfx1100 evidence.

## Source map

- Card: `huggingface.co/nerkyor/Qwen3.8-27B-Coder390-…-MTP-DFlash2` README — L696-699, L678-681, L467, L397.
- Drafter: `.../raw/main/FP8/DFlash2-FP8/{config.json,manifest.json,SHA256SUMS}` + safetensors header range read. Definition: `huggingface.co/z-lab/Qwen3.8-27B-DFlash2`, `arxiv.org/abs/2602.06036`, `z-lab.ai/projects/dflash/`, `rocm.blogs.amd.com/artificial-intelligence/dflash-on-MI355x/README.html`.
- engine: `spec_info.py:39,44`; `speculative_hook.py:286-440,855-860`; `choices.py:103-111`; `fields/spec.py:70-73`; `models/dflash.py:1-3,157-161,638,1103-1160,1119-1123,1224-1242`; `kernels/ops/speculative/dflash.py:180-181,348-367`; `dflash_utils.py:128-172`; `dflash_info.py:39`; `triton_backend.py:1747-1762`; `spec_aux_hidden_state.py:23`.
- Port context: `docs/research/14-sglang-gfx1100.md:11-13,27,72,123`. Every tok/s and ms figure is an estimate from the stated byte counts and 592 GB/s; no GPU job was run.
