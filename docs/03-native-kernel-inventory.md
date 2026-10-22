# 03 — Native-kernel inventory

Derived from the file and from what was measured here. Every row traces to
[`plan/TENSOR-INVENTORY.md`](plan/TENSOR-INVENTORY.md) or a probe in `bench/gfx1100-probes/`;
a row closes only when its path passes its test and its predecessor is **deleted**.

## What "native" means here

1. **No kernel in the measured path was written for another architecture.** Triton is a
   reference and a scaffold, `torch` only for single launches; neither is the shipped path.
2. **The old path is deleted, not flagged.** A flag is how an engine rots: two
   half-tested paths, and nobody knows which one the number came from.
3. **Enabling changes are not ports.** The arch allowlist, `SGL_ROCM_WARP_SIZE`, C++20,
   the Quark import guard and the RCCL all-reduce are what makes the tree run here at all.
4. **One weight layout, two front-ends.** GGUF is read directly into one in-memory
   layout (`R2`); a quantised safetensors container maps into the same one. No offline
   conversion and no second format on disk — the file is served as published (`R1`,
   `DECISION-packaging-iq3s.md`).

## The quantised matmul problem

Shape needs no dispatch: every role has one shape across all 64 layers. Type does — no
quantised role is single-type (`attn_qkv` spans 7, `ffn_gate` spans 9). The dispatch key
is **(role, type)**: **72 pairs** in the base file (`INV` §6, §7).

Five types are **94.5%** of the bytes (`INV` §2): **IQ3_S** 31.0% (144 tensors,
3.393 GiB), **IQ4_XS** 25.7% (96, 2.814), **IQ3_XXS** 17.6% (78, 1.933), **Q4_K** 13.4%
(39, 1.474), **IQ2_S** 6.8% (17, 0.745). Of the rest, only **IQ1_M** matters: one tensor,
`blk.13.ffn_gate`, 19.5 MB, 1.75 bits/weight, so it gets a slow path rather than a family.
Q6_K appears nowhere in the trunk; it is the MTP layer's only matrix type.

## The cost model, which is measured

Ablations on the real published blocks, 1.07 GiB footprint, every row gated on a
data-dependent checksum (`E1` §1–2). Wall = **776 GB/s** (`WALL`).

| variant | GB/s | % of 776 | isolates |
|---|---|---|---|
| `read` — block addressing, no dequant | 717.7 | 92.5% | the ceiling of this layout |
| **`full` — as published** | **516.4** | **66.6%** | — |
| `nogather` — codebook load deleted | 493.5 | 63.6% | gather: **−3.0 pts — removing it is slower** |
| `full_nosgn` — sign decode deleted | 608.2 | 78.4% | sign decode: **+11.8 pts** |
| `full_nosgn` against `read` | — | — | index + dot: **+14.1 pts** |
| `full` against `read` | — | — | dequant + dot, total: **+25.9 pts** |

**Sign decode is the first target; the codebook is not.** The gather is free — the 2 KB
table sits in L1 and deleting it costs 22.9 GB/s. What costs 11.8 points is applying 32
sign bits per 32 weights, and the probe's sign path is **already branch-free SWAR**: 11.8
is an upper bound on any rewrite, not a forecast. `E1` §6 puts the realistic prize at +5
to +9 points (sign plane re-spread at load so a mask is one AND, one negate per group),
about +4 to +6% tok/s — a hypothesis to prototype, and `A1` records it open.

Two thresholds are dead: **619 GB/s** (a short-kernel artefact, `00-facts.md`) and any
fraction of it. What matters is **752.5 GB/s**, which the clean 4-bit dequant-GEMV
already reached at 97.0% of the wall (`WALL`), and **516.4**, the IQ3_S baseline.

## The families

`INV` = `plan/TENSOR-INVENTORY.md` · `REC` = `plan/RECORD.md` · `E1`,`E2`,`WALL` =
`bench/gfx1100-probes/` · `PROF` = `research/20-decode-profile-27b.md` · `KNOBS`, `R17` =
`research/2026-10-22/05-serving-knobs.md`, `research/17-dflash2.md`. **P0**: no record
moves without it. **P1**: a settled decision requires it (`H2`, `S2`). **P2**: smaller.

| family | today | measured evidence | native target | test | pri |
|---|---|---|---|---|---|
| **1. Quantised GEMV + dequant**, 5 types, 72 pairs, 10.57 GiB/token (`INV` §4) | no GGUF weight loader; only the probes read the file | IQ3_S 516.4 = 66.6% of wall; clean 4-bit 752.5 = 97.0%; 25.9-pt penalty | one wave32 GEMV per `(role,type)`: dequant in registers, `v_dot4_i32_i8`, one pass | bit-exact vs an fp64 CPU dequant (4.8e-06 worst case today); ≥752 on the 4-bit shape; IQ3_S ≥516.4 and rising | **P0** |
| **1a. Sign path**, first slice of 1 | branch-free SWAR in `E1` | deleting it: 608.2 GB/s, +11.8 pts; 21.4 pts in the 4-bit shape | sign plane re-spread in memory at load; one AND per 4 weights | gated, bit-identical, >516.4; ceiling 11.8 pts, realistic +5–9 | **P0** |
| **2. GDN decode**, 48 of 64 layers, 9 GDN-only roles | Triton delta-rule + `_causal_conv1d_update` | GDN path 1.1 ms/step across 48 layers, recurrence alone 14.5 µs/layer (`PROF`) = 7% of a 15.2 ms step at the wall (derived); state 159 MB/seq, 317 MB/token read+write = 2.7% of traffic at B=1, 21.5% at B=8 (`R5`) | one HIP kernel per layer: 3 projection GEMVs + conv1d, dt, A, alpha/beta in one launch, FP32 state resident | 32-token parity vs Triton; state trajectory at 128/1K/4K/8K; no per-token rebuild | P1 |
| **3. Layer-fused dispatch**, 2–4 layers per launch | per-tensor: 401 launches/step | 2 layers 33 launches 774.2 = 99.7%; 4 layers 773.8; 8 layers 691.4 = 89.1%, *worse* than per-tensor (1.35 GB launch, +199 µs, `E2` §5); marginal launch 2.5–2.9 µs | 2–4 consecutive layers per launch, one sequence per step (`S2`) | 33 or 17 launches, ≥99% of wall on the read stream; never 8 | P1 |
| **4. MTP verify**, layer 64, 15 tensors, 0.325 GiB, every matrix Q6_K | nothing; the head ships in the `-mtp` file, unread | Q6_K is absent from the trunk: one new type for one layer; 4 tensors have no counterpart (`nextn.eh_proj`, `enorm`, `hnorm`, `shared_head_norm`); head shared, not duplicated (`INV` §9) | Q6_K GEMV for one layer; one batched verify launch over the draft width; GDN state rollback | accept ≥2.0 and ≥1.2× unspeculated tok/s (`R17`), else the head is off | P1 |
| **5. Graph capture** | `torch.cuda.CUDAGraph` → hipGraph per bucket | 41 s for bs=[1,2,3] (`00-facts.md`); gain **0.6%** — launch gaps only (`KNOBS`) | capture once per bucket at init, keyed by launch shape and page count | ≤41 s for 3 buckets, never per-request; replay bit-identical | P1 |
| **6. Full-attention decode**, 16 layers, 24/4 heads at 256 | Triton; `aiter` unusable (dies at capture, asserts on `cpp_itfs`) | 0.4 ms/step for 16 layers at the FP8 baseline (`PROF`); KV 64 KiB/token bf16 = 2.15 GB at 32K, 18% of the weight stream | wave32 flash-decode over paged q8 KV, no per-length specialisation | ≤1e-3 vs Triton; records at `decode/1/4096` and `/32768` | P2 |
| **7. LM head**, `output.weight` Q4_K 248320×5120, 0.666 GiB, untied | the same GEMV family | largest single tensor, 6.08% of bytes, read in full every token; `token_embd` is a row lookup (`INV` §4) | one GEMV fused with the sampler's top-k; the same matrix serves MTP, so no second head | ≥752 on its own shape; bit-exact; one read per token | P2 |
| **8. Sampler and top-k** | torch ops; `torch.topk` where `flashinfer.top_k` is absent on ROCm | 55.8 ms/step — 6.5% — outside every kernel: gaps, sampling, detokenise (`PROF`); top-k over 248,320 tokens reportedly halves throughput (`R17`) | one fused per-step kernel (softmax → filter → cumsum → sample) + native top-k for verify | exact distribution match on fixed seeds; the non-kernel residue must fall | P2 |
| **9. KV cache q8 + paged** | bf16, 45,727 tokens = 1.40 GB; 2.08 GB free/card → `max_running_requests=3` | q8: 5.42 → 4.34 GB/token at 32K × 4 (−20%); 4 → 7 sessions at 32K (`KNOBS`); 8% at batch 1 × 32K (`04`) | q8 pages, dequant fused into attention, no separate pass | logits within 1e-3 of bf16 on 8×4K prompts; capacity at 32K | P2 |

The GDN-only roles, 2.350 GiB = 21.5% (`INV` §4): `attn_qkv` 5120×10240, `attn_gate`
5120×6144, `ssm_out` 6144×5120, `ssm_conv1d` 4×10240, `ssm_a` 48, `ssm_dt` 48, `ssm_norm`
128 (F32), `ssm_alpha`/`ssm_beta` 5120×48 (BF16) — three dtypes in one block. Full
attention: `attn_q` 5120×12288 is query concatenated with its output gate, so the gate is
no separate kernel; `attn_k`/`attn_v` 5120×1024, `attn_output` 6144×5120. `attn_qkv`
10240 = 2×(16×128) + 6144, one GEMV with concatenated output slices.

## Out of scope and not scheduled

- **A second model.** `H1`, ADR 0002: one file, `Qwen3.8-27B-GSQ-RCO-IQ3_S`, with its MTP
  head; no second quant, no second config.
- **MoE.** The 27B is dense and the inventory has no expert role; DeepInfer's psum layout
  and every grouped-expert kernel are irrelevant (`plan/REQUIREMENTS.md` §5).
- **Flash-Next, ngram, tiered expert store.** A different model; dropped, `H1`.
- **mmproj / vision.** `H1` closes it: the 0.466 B BF16 visual tower is a separate file,
  never read in text decode (`INV` §3b).
- **FP8.** ADR 0004: no checkpoint, no reference, no kernel. The 857 ms/token run is
  evidence for the exclusion, never a baseline.
- **Not scheduled — HiCache host tier.** `kvcacheio` refuses gfx1100 in the AOT build.
- **Not scheduled — quantised GEMM, prefill, DFlash2.** `S1` leaves the GEMM path unbuilt
  until the sub-8K, >30-concurrent corner is measured, and `prefill/4096` has no record
  (`REC`); DFlash2 comes after MTP (`R17`).

## Not yet measured

- **MTP acceptance rate on this box** (`A3`); the publisher's 3.4–4.4 is another card.
- **Type-by-type rates for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S** (`A2`). Only IQ3_S has been
  measured in situ; the whole-file 56.1 tok/s assumes the other 69% of bytes run at 752.
- **The physical cause of the ~800 MiB per-launch cliff** (`A5`). TLB/page-walk is a
  hypothesis; the cliff is measured at ~230 µs per launch (`WALL`) and caps fusion at 4.
- **Whether a quantised GEMM can reach the wall.** 752.5 was a GEMV, and a quantised GEMM
  re-reads the scale plane per column block. It matters only if the sub-8K high-batch
  corner is ever visited (`S1`).

Where this file and a measurement disagree, the measurement wins (`H10`).
