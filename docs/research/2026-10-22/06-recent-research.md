# 06 — What actually changed in kernels and serving, 2025–2026, filtered for `rasputin`

**Status:** survey, written 2026-10-22. Nothing here was measured on `rasputin`. Every ceiling quoted is arithmetic from `docs/00-facts.md`; every external claim carries a venue, a date and a URL.

**How to read the tags**

| Tag | Meaning |
|---|---|
| **[V]** | Opened in this pass: the URL returned the page/abstract, and the claim is in the text I read. |
| **[D]** | Opened by a delegated extraction pass that returned the abstract text; URL and date checked, full text not re-read here. |
| **[U]** | Could not verify. Do not cite outside this document. |

Two claims in this document are load-bearing and are *not* citations, they are arithmetic: the KV-versus-weight crossover (§1) and the concurrency ceiling it implies. Both are derived from `docs/01-models.md` and `docs/00-facts.md`. If either input is wrong, both conclusions move.

**Relationship to the other documents in this directory.** `01`, `02`, `04`, `05` and `07` are topic deep-dives written in parallel with this survey; this file is the external-literature layer that sits above them and defers to them on verdicts. Where they overlap, they agree, and they were derived independently: §1's arithmetic reproduces `04-kv-cache.md` §1.2 (a 59.3 % KV share at 8×32K, and q8_0 strictly dominating fp8 on this chip) and `05-serving-knobs.md` §1 ("long-context decode on this box is a KV problem, not a weight problem"). Where `02-megakernels.md` concludes *do not build a megakernel*, this document does not argue: a.5 explains why the idea keeps winning elsewhere, and P-B is scoped so that it survives `02`'s verdict rather than depending on overturning it.

---

## 1. The number everything is judged against — and the premise that breaks at the product target

`R44` says decode is bandwidth-bound at weight bytes per token. That is correct at batch 1, and **incomplete at the stated concurrency target**, because the KV cache is read once per sequence per step and the weights are read once per *batch*.

Per decode step, with `B` sequences at context `L`:

```
bytes(B, L) = 11.77 GB (IQ3_S weights)  +  B × L × 64 KiB   (bf16 KV, 16 attention layers)
            = 11.77 GB                 +  B × L × 32 KiB   (q8 KV)
```

| B | L | KV dtype | bytes/step | ms/step @619 GB/s | tok/s per session | aggregate tok/s | KV share of bytes |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 4K | bf16 | 12.04 GB | 19.4 | 51.4 | 51 | 2 % |
| 1 | 32K | bf16 | 13.92 GB | 22.5 | 44.5 | 44 | 15 % |
| 8 | 32K | bf16 | 28.95 GB | 46.8 | 21.4 | 171 | **59 %** |
| 8 | 32K | q8 | 20.36 GB | 32.9 | 30.4 | 243 | 42 % |

Read the last two rows against 02-plan.md (removed) ("~28 GiB of free VRAM holds about 13 sessions at 32K"), which assumes a free-VRAM figure that `docs/00-facts.md` contradicts (2.08 GB free per card). **At bf16 KV, eight 32K sessions need 16 GiB of KV plus 10.96 GiB of weights — 26.96 GiB on a 19.98 GiB card. They do not fit.** At q8 KV the same eight sessions need 8 GiB of KV, so 10.96 + 8 + 1.24 GiB of recurrent state = **20.20 GiB — still 0.2 GiB over the card**, before any workspace, graph pool or activation buffer. The realistic one-card ceiling at 32K q8 is **~6 sessions**, not 13.

Cross-check, because this table is the one thing here that could be wrong in a way that matters: `04-kv-cache.md` §1.2 derives the identical 59.3 % KV share from the per-card TP=2 split (8 GiB of KV against 5.48 GiB of weights), and `05-serving-knobs.md` §1 reaches the same conclusion from a different direction — the compute/bandwidth crossover. Three independent derivations, one answer.

Two consequences the rest of this document depends on — but first, the caveat that stops this table being misquoted: **every row assumes 100 % of the measured read wall.** At the ~94 %-of-wall efficiency a shipping RDNA engine reaches (a.1), the batch-1 numbers become ~48 tok/s at 4K and ~42 at 32K, and the batch-8 KV shares stay the same.

1. **q8 KV (`R22`) is not a capacity nicety, it is the precondition for the concurrency target.** At bf16 the target is physically impossible on one card regardless of engine quality.
2. **As resident tokens rise, a faster GEMV stops being the main lever.** Between a few resident tokens and a few hundred thousand, the binding term moves from weights to KV. Any 2026 technique that only accelerates weight streaming moves a 40 %-of-bytes term.

---

## (a) Transfers to us

### a.1 The decode GEMV shape: one wave per row, 32 threads, no LDS, scale after the dot

`hipfire` (Apache-2.0, Rust + HIP, RDNA-first) documents the exact shape it converged on for 4-bit GEMV across RDNA1–4 and validated on gfx1100: **one full wave32 = one workgroup, one output row per workgroup, `__launch_bounds__(32, 16)` to force ≥16 resident waves per CU, dp4a inner loop, accumulator in VGPRs, zero shared memory** (the activation is quantised once to q8_1 and reused from registers). Its `PRIOR-ART.md` says copying the kernel without the thread-count/launch-bounds *pair* either spills VGPRs or under-fills the CU — i.e. on a 5376-lane card the occupancy story *is* the kernel.
**[V]** hipfire `PRIOR-ART.md` §3, first commit 2026-03-22 — https://github.com/warpfront/hipfire/blob/master/PRIOR-ART.md

Why it fits: at `R44`'s 6.6 % issue occupancy we are not compute-limited, but the measured evidence (a.2) says sub-4.5 bpw formats become *dependent-load-bound* on RDNA, which is exactly what a 32-thread wave with a single-row accumulator avoids: one wave issues one long stream of independent `buffer_load`s with nothing to synchronise.

**Calibration point for `R40`'s 60 % gate:** a mature RDNA-native engine reports 27B at 4.25 bpw on a 7900 XTX sustaining **47 tok/s at 651 GiB/s effective**, with 9B and 27B both saturating ~650 GiB/s. **[V]** hipfire `BENCHMARKS.md`. Converted honestly, 651 GiB/s = **699 GB/s = 72.8 % of that card's 960 GB/s spec** — the source's own note says "68 %", which mixes GiB into a GB ratio. Two readings follow, and both matter:
- **The gate is a floor.** `R4`'s 60 % of 619 GB/s is 46 % of our card's 800 GB/s spec — twenty-six points below what a shipping engine of the same architecture reaches. A kernel that only clears 60 % has not been tuned, it has been started.
- **The ceiling is not 52.6 tok/s either.** Matching 72.8 % of spec on our 800 GB/s card is 582 GB/s, which at 3.50 bpw is **48.4 tok/s at 4K and 41.8 at 32K** — *below* the 52.6 tok/s weight-bytes ceiling, because that ceiling assumes 100 % of peak. The number to plan against is ~48, not ~53.

**Caveat, and it matters:** hipfire's gfx906 prefill path explicitly follows llama.cpp's MMQ design, and `R77` excludes that source. hipfire is readable as an *architecture and measurement* reference (its own `PRIOR-ART.md` exists precisely because people port its shapes); it is not a permitted source of kernel code, layout or heuristic for the parts that inherit llama.cpp lineage.

### a.2 Pre-shuffle the weights at load; never gather a table in the inner loop

Two independent shipping designs agree: **the repack is the optimisation**, and it belongs at load time, not in the kernel.
- Marlin repacks into `mma.m16n8k32` fragment order at load, with permuted scales and an implicit zero point, and argues the repack is a *precondition* for using tensor cores at all, not itself a speedup. **[V]** *MARLIN*, arXiv:2408.11743, 21 Aug 2024 — https://arxiv.org/abs/2408.11743
- hipfire stores 4-bit as 136 B per 256 weights (dual-fp16 scale+zero header per 128) so the inner loop is a fixed-stride `LDG.128 → dp4a → ffma`, and stores its 16-entry codebook *in registers* synthesised with `v_perm_b32` rather than fetched from a 256/512-entry global table. **[V]** hipfire `QUANTIZATION.md`, v0.4.2 2026-10-14 — https://raw.githubusercontent.com/warpfront/hipfire/master/docs/QUANTIZATION.md

For us: `R20` already says dequantise in registers. a.2 adds the second half — **the byte stream should be laid out so that each lane's inner loop is one contiguous vector load**; if the IQ3_S block layout makes that impossible without a per-lane permutation, that cost has to be paid in `v_perm`/`ds_bpermute` ALU, not in dependent gathers.

### a.3 Rotate outliers offline instead of protecting them at runtime

hipfire's shipping 4-bit format (MagnumQuant MQ4) applies an **offline FWHT-256 rotation per group** and fuses the matching activation rotation into the RMSNorm kernel, so the matmul equals the unrotated product and the rotation costs almost nothing at runtime. This is the QuaRot/SpinQuant idea reduced to a group-local transform with no extra bytes.
**[V]** hipfire `QUANTIZATION.md` (MQ family: "offline FWHT-256 … FWHT baked into weights") and `PRIOR-ART.md` §4, 2026-04-08 — https://github.com/warpfront/hipfire/blob/master/PRIOR-ART.md

**Verdict: not usable now, usable at the next quant.** The weights are fixed (`R16`, GSQ-RCO IQ3_S) and `R26` says that recipe needs no new kernel. This is the technique to hold for a re-quant, and it is the strongest argument that 4-bit-class quality on RDNA3 does not require higher bpw — it requires rotation.

### a.4 Prefill is a WMMA problem, and WMMA on gfx1100 is real

hipfire reports prefill "WMMA-bound on the MQ4 fused projections" under a gfx1100 fixture, and uses batched WMMA GEMM to amortise dequant across prompt tokens. **[V]** hipfire `BENCHMARKS.md` (inventory 2026-07-19; historical rows) — https://raw.githubusercontent.com/warpfront/hipfire/master/docs/BENCHMARKS.md

That is the same conclusion as `R19` (compute is bf16 WMMA), arrived at independently, on the same chip. Note the consequence for the kernel inventory: GEMV below the crossover, WMMA GEMM above it, and the crossover is a *shape* fact to be measured, not a policy.

### a.5 The megakernel is the one scheduling idea 2026 validated end to end

- **Cohere, 8 Sep 2026**: a production serving engine (continuous batching, paged attention, ragged lengths, OpenAI endpoint) built as a single decode megakernel. North Mini Code 30B/3.3B-active on one H100: SoL is 470 tok/s, vLLM serves 185 (39 %), megakernel 292 (**62 % of SoL, 1.58× vLLM**), holding to 256K context. Their diagnosis of the missing 61 %: launch/sync gaps, wave quantisation, and inter-operator HBM round-trips — not arithmetic. "After the kernels are fast, the schedule is the rest of the speedup." **[V]** https://cohere.com/blog/megakernels
- **Ada-MK, 12 May 2026**: same idea on a card with no TMA and 128 KB SMEM (L20). +23.6 % over TensorRT-LLM single-batch, +50.2 % over vLLM, shipped in a production ads system. Its central move is *compile-time* specialisation, not runtime branching (see §b.10). **[V]** arXiv:2605.11581 — https://arxiv.org/abs/2605.11581

**Calibrated to us, and reconciled with `02-megakernels.md`:** the plan already holds the megakernel at P6, gated on the profiler (`R46`), and `docs/research/20-decode-profile-27b.md:78-79` puts non-kernel residue at 55.8 ms/step = 6.5 %. Cohere's 1.58× came off a baseline at 39 % of SoL; ours runs at ~3 % (17.8 GB/s of 619). `02` bounds the prize here at **2–14 % of the step, probably ~5 %**, because the mechanism that would deliver even that (a sub-microsecond device-scope barrier across 84 CUs) is unmeasured on gfx11 and the naive version costs more than the launch it replaces. This document does not overturn that verdict: the external 2026 evidence says the *idea* is sound and the *size* is small on this box, which is exactly `02`'s conclusion. Read a.5 as "the direction is validated elsewhere, the budget here is 5 %, not 58 %".

### a.6 Dynamic split-KV bin-packing — the one place "split-K" thinking applies to us

HPC-Ops (Tencent Hunyuan, integrated into SGLang main, **7 Aug 2026**) replaces a *static* per-request split-KV schedule with a **persistent kernel that builds a device-side task map from live KV lengths each decode step**: slice every sequence into 64-token tiles, sum tiles across heads and requests, divide by the number of persistent CTAs, fill bins, drain them. Report: **2.95× over its own static split-KV schedule on H20**, 2.25× average over the best of FlashInfer/FA in each case. **[V]** https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/

This is the single most directly applicable kernel-scheduling result in the survey, because our operating point is *exactly* mixed-length decode (1K–32K sessions in one batch, on 84 CUs), and the failure mode it fixes is ours: a static split sized for the longest session leaves short sessions paying for empty chunks, and a fixed chunk count lets one long session own the kernel tail. We do not need their FP8 KV, their fused FP8 prologue, or their router GEMM — only the two-phase "assign kernel + persistent drain + combine" structure, which is ~200 lines and has no architecture dependency.

### a.7 The hybrid-model plumbing lessons are already published, and they are 2–3× items

From hipfire's changelog, on RDNA3, for Qwen-family hybrid models (i.e. our architecture class):
- **Deinterleave**: replacing a per-head memcpy loop in the full-attention layers — **576 HIP memcpys → 9 dispatches per token** — took 9B Q4 from **15 → 43 tok/s**. **[V]** `CHANGELOG.md` v0.1.3-alpha, 2026-04-05.
- **Recurrent state in the prefix cache**: their LCP-keyed prefix cache had to carry the DeltaNet recurrent state *and* conv state, because restoring KV alone "silently corrupts generation". Two implementation traps recorded: re-tokenising turn N+1 does not reproduce turn N's ids at the shared prefix (**BPE re-encode drift**), and a chat template that re-renders the whole conversation perturbs bytes inside the supposedly stable region (**verbatim splice**). **[V]** `PRIOR-ART.md` §10, 2026-04-04 → 2026-06-09.
- **Stochastic rounding** in Q8/Q4 state re-quantisation fixed coherence decay after ~500 tokens. **[V]** `CHANGELOG.md` v0.1.3-alpha.

`R56` already states the first of these as a requirement, from three engines' agreement. a.7 adds the two *tokenisation* traps, which are cheaper to avoid than to debug and are not in any research paper.

### a.8 q8 KV: the field converged on it too, and sub-8-bit KV was walked back

hipfire shipped 3-bitish asymmetric KV (`asym3`) as its default in April 2026, then **deprecated it in 0.4.0 and removes it in 0.5.0**; current default resolves to `q8` everywhere except exact gfx1201, where it is native `fp8`. **[V]** `BENCHMARKS.md` header notes and `QUANTIZATION.md`, v0.4.2 2026-10-14.

An engine with every incentive to ship 3-bit KV went back to 8-bit. That is the strongest available corroboration for `R22` (q8, not fp8, not 4-bit), and it says the 2.4× compression from 3-bit KV bought less than the quality and complexity cost.

### a.9 Speculation: real, but genre-conditional, and the numbers people quote are the best case

The most useful table in the survey, because it is the same model class, the same silicon family (7900 XTX against our XT), and it reports the *bad* rows (hipfire, 27B MQ4, historical fixture, asym3 KV, `max_tokens=120`):

| Model | genre | AR tok/s | DFlash tok/s | speedup | τ |
|---|---|---:|---:|---:|---:|
| Qwen 3.5 27B | code (HumanEval/53) | 44.1 | **196.0** | **4.45×** | 9.82 |
| Qwen 3.5 27B | prose (Rome essay) | 44.0 | 49.6 | 1.13× | 1.67 |
| Qwen 3.5 27B | instruct (sky-colour) | 44.6 | 44.7 | **1.00×** | 1.39 |
| Qwen 3.5 9B | prose (federalist) | 125.3 | 99.4 | **0.79×** | 1.20 |

**[V]** https://raw.githubusercontent.com/warpfront/hipfire/master/docs/BENCHMARKS.md

Against `C6` (three different claims about speculation's payoff, 1.3–1.8×, 2.6–4.4× τ, 3.9×): the honest reading is **~4.4× on code, ~1.1× on prose, 1.0× on short instruct, and negative when the draft and target distributions diverge**. `R58`/`R59` should carry that spread, and the plan's exit criterion ("a measured multiplier, or a recorded reason it does not pay") is exactly right.

Three 2026 results bound the design space:
- **TreeWY** (21 Aug 2026) removes per-draft-position GDN state snapshots with a tree-structured WY transform, but its own claim is that wider trees become *affordable*, "though not yet a throughput win". **[V]** arXiv:2608.20961
- **DScale** (29 Sep 2026) reports block-diffusion drafters degrading at **concurrency 8–32** — our band — and fixes it with scheduling, not a better drafter. **[D]** arXiv:2609.37532
- **DDTree** (14 Apr 2026) is the algorithm hipfire implements on gfx1100 today. **[V]** arXiv:2604.12989

Read that trio carefully, because it is easy to over-read. DScale's degradation at concurrency 8–32 is a **draft-side** effect — verification padding and rejected candidates — not evidence that the bandwidth amortisation disappears as the batch grows. It does not: a `k`-token verify reads the weights once and the KV pages once for `k` queries, so the gain survives batching, and `05-serving-knobs.md` §5.2 derives exactly that for this box from the 11.9 K-token KV crossover (the MagicDec regime, arXiv:2408.11049). What DScale adds is narrower and still useful: at our batch the *drafter* needs scheduling work to hold its acceptance, which is an argument for MTP (draft in-model, no separate drafter to schedule) over a block-diffusion drafter once concurrency rises.

So: our checkpoint ships MTP heads (`R27`, the `IQ3_S-mtp` file is on the host), DFlash is first-class in the forked tree (`R58`), and a shipping RDNA engine does both on gfx1100 at τ≈9.8 on code. The engineering is de-risked; the open question is purely *our* acceptance rate, which is a measurement, not a research problem.

### a.10 Adaptation that is cheap and justified — and the one place it is not

Real and implementable in days, no ML:
- **Admission on bytes, not on request count**: the inequality in §1 evaluated against the bandwidth the kernel actually achieves, not the peak. One integer comparison per admit, and it is the only "bandwidth-budget scheduler" the hardware actually supports.
- **Frozen per-shape autotune at load** (batch bucket × sequence bucket), never on the request path. Ada-MK's whole thesis (§b.10) and the SLO-Guard result (§b.9) both point the same way.
- **Draft depth from observed τ**: DISCO-style per-step accept/reject heuristic (**[D]** arXiv:2405.04304, 2024), and D-cut's runtime cost model that adapts pruning depth to the deployment environment (**[D]** arXiv:2607.14647, 2026). Cheap, deterministic, no learned component.
- **SM clock lock, not power cap**: the one energy lever with a measured effect on decode (see §b.11).

The one place adaptation is *not* justified is per-kernel or per-step selection driven by measured wall time — see §d, prototype P-B's kill criterion, and hipfire's own bug report in §b.10.

### a.11 Hardware direction: the ISA is frozen, and that is good news for a kernel written now

- **gfx1100 has no FP8 datapath at all**, confirmed by AMD's own precision support table: FP8 E4M3/E5M2 = CDNA3 ✅, CDNA4 ✅, **RDNA3 ❌**, RDNA4 ✅; FP4/FP6 = CDNA4 only. **[V]** https://rocm.docs.amd.com/en/latest/reference/precision-support.html (fetched 2026-10-22)
- **RDNA4 (gfx1200/1201)** adds dense `V_WMMA_*_FP8_FP8/BF8_BF8` and sparse `V_SWMMAC_*`, at the same K=16 depth — roughly 2× MACs per instruction, not a new unit; **zero FP4/FP6** in the ISA text. ISA ref dated 7 Apr 2025. **[V]** https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf
- **RDNA3.5 (gfx1151)** has the identical WMMA set to RDNA3. **[V]** https://gpuopen.com/download/rdna35_instruction_set_architecture.pdf (23 Jul 2024)
- **gfx1100 is frozen, not deprecated**: still in the ROCm 10.1.0 compatibility matrix (page 2026-08-26) and notes (2026-10-05), no removal notice; the RDNA3 ISA document is still the 15 Aug 2023 revision. **[V]** https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- **CDNA4 (gfx950)** is where the direction points — `V_MFMA_*_F8F6F4`, `V_MFMA_SCALE_*` block scaling, `V_CVT_SCALEF32_*` — datacenter only, no WMMA. **[V]** CDNA4 ISA ref, 5 Aug 2025.
- **Consumer P2P**: no AMD announcement 2025–26; AMD's own collective-comm doc (29 Sep 2026) scopes LSA/P2P-over-XGMI to Instinct. Measured on dual RDNA4 (R9700), PCIe P2P = 27.6 GB/s, TP=2 bare metal only. **[D]** https://www.pugetsystems.com/labs/articles/amd-radeon-ai-pro-r9700-dual-gpu-ai-inference-performance/

**The safe baseline, and it will last:** wave32 + `V_WMMA_{F16,BF16,I8}_16x16x16` + `V_DOT*` + 64 KB LDS, expressed through `__builtin_amdgcn_*` intrinsics rather than inline asm. That exact set appears verbatim in both the RDNA3 and RDNA4 ISA documents, so a gfx11→gfx12 port needs no ISA work. The migration risk is *not* RDNA4 — it is **UDNA**, where no ISA reference is published (only `gfx1310`/`amdgpu13` targets in current LLVM), so whether it keeps WMMA or adopts the CDNA MFMA/block-scaled model cannot be priced today. **[V]** https://llvm.org/docs/AMDGPUUsage.html

---

## (b) Does not transfer, and why

### b.1 The 2025–26 kernel frontier is architecture-gated, not idea-gated

Every headline kernel number in this period is a tensor-core-utilisation number on hardware with async copy, TMA and (now) FP4. We have none of those, and — decisively — **we have no tensor-core utilisation problem**: at 6.6 % issue occupancy the arithmetic is free and the bytes are not. Reading these libraries for ideas is right; expecting any of their speedups is not.

- **DeepGEMM**: unified tensor-core library, CUDA, SM90/SM100, FP8/FP4/BF16 GEMM, Mega MoE with overlapped comms, MQA scoring for the lightning indexer, DeepJIT runtime compilation; 1550 TFLOPS on H800 (Apr 2025); localised to Ascend via a separate repo (30 Sep 2026). No AMD path of any kind. **[V]** https://github.com/deepseek-ai/DeepGEMM
- **FlashMLA**: as of **30 Sep 2026 it dropped Hopper and everything before DeepSeek-V4.x**; now Blackwell + Ascend 950, token-level sparse attention (DSA) with FP8/FP4 KV, fused norm-RoPE-attn-RoPE-cast. The AMD route is explicitly **AITER/MLA** — i.e. Instinct. **[V]** https://github.com/deepseek-ai/FlashMLA
- **AITER**: AMD's operator library, default attention backend for vLLM/SGLang on ROCm, backends Triton/CK/ASM/Opus/FlyDSL. Its news through 2026-07 is MI300X/MI355X/Kimi-K3/Kimi-K2.5. **No gfx11 admission anywhere in the README.** **[V]** https://raw.githubusercontent.com/ROCm/aiter/main/README.md
- **Marlin / Machete**: Marlin's win is a fragment-order repack for `mma.m16n8k32` with async global→smem, and its own framing is "batchsizes up to 16–32 with close to maximum 4× speedup" — a *batched* tensor-core argument. Machete generalises the repack to Hopper `wgmma` and is motivated precisely by Marlin's layout being `mma`-specific. Neither fragment layout exists on gfx11, and we have no `cp.async`. **[V]** arXiv:2408.11743; **[D]** Machete, vLLM PR #7174.

**What does transfer from DeepGEMM** is a design discipline, not code: a deliberately tiny set of core kernel functions, specialised at JIT time per shape and dtype, instead of a general GEMM. Our analogous surface is five quant types × {GEMV, WMMA GEMM} — small enough to keep closed.

### b.2 Any FP8/FP4/NVFP4/MXFP path

`R21` already says FP8 is a functional reference only. The 2025–26 literature makes this more true, not less: the interesting low-bit work is *block-scaled FP4 on CDNA4/Blackwell* (MXFP4, NVFP4), and gfx1100 has neither the datapath nor the conversion instructions. The best accuracy result in the survey — **NVFP4 W4A4 on this exact 48-GDN + 16-attention 27B shape**, matching BF16 within seed noise at 17.5 GiB — is worth reading for its *conclusions* (the recurrent half is the easy half to quantise; the gate projections are the least sensitive; calibrated FP8 KV scales are free) and is unusable as a format, because the hardware does not exist here. **[V]** arXiv:2609.04098, 3 Sep 2026.

### b.3 TMA, clusters, `cp.async`, warp-specialised pipelines

Absent from gfx11 and central to every 2025 kernel. Ada-MK's entire 50 % shared-memory problem is a Hopper→Ada port problem (227 KB → 128 KB SMEM, no TMA, pipeline depth 4 → 2). We have 64 KB LDS and no async copy; the equivalent design move is not "more pipeline stages" but "fewer, longer, independent loads per wave" (a.1).

### b.4 Prefill/decode disaggregation on two cards

The economic premise of DistServe/Splitwise/Mooncake/Dynamo/llm-d is a fleet with cheap inter-node KV movement and per-phase parallelism plans. On two PCIe cards with no P2P:
- Any PD split needs **two resident copies of the model**. bf16 twins are 100 GB — impossible. IQ3_S twins are 21.9 GiB of weights and do fit in 40 GB, so the split is physically possible; the cost is that decode then runs on **one** card (DP-per-phase), i.e. at the single-card 619 GB/s wall, and the ×1.5–1.8 that TP=2 would have bought at decode (`docs/research/18-tp-roccl.md:293-295`) is forfeited. PD here is a bet that prefill interference costs more than that forfeit.
- The handoff itself: a 32K q8 KV is 1 GiB, so ≈70 ms at the measured 14.2 GiB/s host-bounce rate (`docs/00-facts.md`) — about two decode steps at batch 8, per request, on a link that is also carrying the display. `R3` still holds: treat it as host traffic, not P2P.
- **Net:** PD on this box is a throughput *trade*, not a throughput win, and it is measurable in one afternoon. The only 2026 paper whose *mechanism* fits two devices is FluidPD (in-place worker role swap, 2 Oct 2026, +94.6 pp SLO attainment) — and it still assumes two resident replicas. **[D]** arXiv:2610.06917; Mooncake **[D]** FAST'25 best paper https://www.usenix.org/conference/fast25/presentation/qin; DistServe **[D]** OSDI'24; llm-d v1.0 **[D]** 20 Oct 2026 https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2

### b.5 Cluster-scale KV management and routing

Mooncake's KVCache pool, Dynamo/NIXL, llm-d's KV-aware routing, Llumnix's live migration, autoscaling: all monetise *choice among many workers*. At two cards the router has one bit of freedom and migration has one destination. The transferable residue is the abstraction — "KV is a first-class tiered object" — which on this box means **host DRAM over PCIe at 14.2 GiB/s** (HiCache/LMCache shape), and only for the 16 attention layers' KV plus recurrent-state checkpoints.

### b.6 Cache-aware routing and eviction policies

Routing needs ≥3 workers to have a choice. Eviction (H2O, SnapKV, DuoAttention, RazorAttention) buys *capacity*, not bytes per step, and puts quality at risk precisely in the 16 global layers that do retrieval. **Quest** is the exception worth knowing — it keeps all KV and moves only query-selected pages, i.e. it attacks bytes-per-step, which is our actual term. **[D]** arXiv:2406.10774 (ICML'24).

### b.7 stream-K and split-K for GEMM

Two reasons this is not our problem, and one reason to avoid it:
1. **Shape**: at decode M=1 with 5120-row projections we already have ~61 waves of independent rows across 84 CUs; there is no wave-quantisation tail to recover. At prefill M=8192, N=5120, K=5120 even 128×128 tiles give 2560 tiles — again no tail.
2. **Determinism**: changing the K-partition changes the reduction order. A 2026 characterisation measures up to **20 % cost to preserve a fixed order** and shows an autotuner cannot even identify which configurations are bitwise equivalent. **[V]** arXiv:2609.11356, 10 Sep 2026. Our cheap arbiter is "32 greedy tokens match a torch reference exactly" (`R73`); a split-K prefill that reorders reductions turns that arbiter into a coin flip.

Note the distinction that matters: **split-K (a GEMM tiling trick) ≠ dynamic split-KV (a load-balancing scheduler for mixed-length attention)**. The second one transfers (a.6); the first does not.

### b.8 ML-guided kernel selection and synthesis

The numbers do not support it as an engineering substitute:
- **Kevin** (Jul 2025): multi-turn RL kernel generation; mean speedup **0.53× → 1.10× of PyTorch Eager** — i.e. 10 % over eager, not over a hand-tuned HIP kernel. **[D]** arXiv:2507.11948
- **GEAK** (Jul 2025, AMD): Triton on MI300X/MI250, correctness ≤63 %, speedup ≤2.59×. **[D]** arXiv:2507.23194
- **AMDKernelVault** (Sep 2026, AMD): 62k verified HIP kernels, RL-tuned; **34.0 % Pass@1** on PyTorch→HIP, and the authors say it "does not uniformly lead compilation or speed metrics". **[D]** arXiv:2609.12471
- **AsmEvo** (Aug 2026): edits compiled AMDGPU objects; 1.35× geomean on KernelBench, but only **1.09×/1.18× geomean on real vLLM/SGLang Triton kernels**. That is the realistic ceiling for agentic work on production kernels: 9–18 %. **[D]** arXiv:2608.20711
- **AlphaEvolve** (14 May 2025) did find a 32.5 % faster FlashAttention kernel and a 23 % GEMM tiling rule — with unbounded offline search on one kernel, and no RDNA target. **[D]** DeepMind blog.

Verdict: a 9–18 % geomean on *already-good* kernels is a real but second-order return, and it does not beat a hand-written GEMV for a shape we fully control.

### b.9 Autotuning does not beat random search, and rankings do not transfer

- **SLO-Guard** (Apr 2026): crash-aware TPE + thermal annealing versus uniform random sampling, **statistically tied** on best latency (p = 0.84); its only real win is budget consistency. **[D]** arXiv:2604.17627
- **LSSP** (Sep 2026, preprint, not peer-reviewed **[U]**): 9,360 configs; cross-metric rank correlation τ_b = 0.419, top-policy agreement 31.9 %, and a simulator-predicted reversal that failed to reproduce on hardware.
- **Bitwise paper above**: an autotuner cannot distinguish configurations that differ only in reduction order.

Consequence for us: autotune **once at load**, over a small shape bucket set, and freeze the result into the graph. Do not build a tuner that runs while serving.

### b.10 Runtime adaptivity inside the decode kernel

This is the sharpest negative result in the survey, and it comes from both directions:
- **Ada-MK's thesis**: "under a fixed deployment configuration, the optimal execution path of a MegaKernel is uniquely determined, and runtime dynamic decision-making can be entirely hoisted to compile time." Runtime probing and if-else scheduling are named as the reason auto-compiled megakernels are unusable at decode latency. **[V]** arXiv:2605.11581
- **A shipped counter-example**: hipfire's *wall-time route chooser* (a runtime decision between routes) is recorded as a known defect in v0.4.2 — "with seed 123, 0.4.2 gives different token streams across fresh processes on three of the eight prompts (a known issue in the wall-time route chooser). Greedy AR and greedy MTP are bit-identical." **[V]** `README.md` v0.4.2, 2026-10-14.

That is a runtime-adaptive engine losing reproducibility to its own adaptation, in production, on our hardware class. For a project whose promotion gate is "32 greedy tokens match exactly", this is disqualifying for per-step adaptive routing and merely a caution for compile-time bucket specialisation.

### b.11 "Adapt to remaining hardware headroom" — the power version is an illusion

**The Illusion of Power Capping in LLM Decode** (12 May 2026) measures four attention paradigms on H200 — GQA, MLA, **Gated DeltaNet**, Mamba2 — and finds decode draws only **137–300 W on a 700 W GPU**: memory-bound decode saturates HBM and leaves power headroom untouched, so no cap ever triggers. Firmware clock throttling then corrupts any throughput measurement attributed to the cap. SM **clock locking** Pareto-dominates capping, recovering up to 32 % of decode energy at minimal throughput loss. **[V]** arXiv:2605.11999

Two things follow. First, "bandwidth-budget scheduling" is the only headroom that exists; the budget is bytes and VRAM, not watts. Second, on a two-card box where one card drives the display, **lock the SM clock before any measurement** — otherwise the throttling confound lands directly on `R65`'s before/after discipline.

### b.12 Anything claiming to beat the bandwidth ceiling without quantisation or speculation

Checked and rejected for *this* model:
- **Diffusion / flow LMs** (LLaDA Feb 2025, Mercury Jun 2025, Gemini Diffusion May 2025; 2026 step-efficiency work on top): real parallel token generation, but only on diffusion-trained weights. A conversion/retrain, untested on GatedDeltaNet stacks. **Different model.**
- **Trained-in MTP**: the one genuine lever, measured at **1.91–2.19×** single-request in a controlled study, with the dominant MTP GEMM still at the bandwidth limit. **[D]** arXiv:2609.35188 (28 Sep 2026). We have the heads (`R27`), so this is available — at ~2×, not 4×.
- **Early exit / layer skip**: 2026 three-seed audit finds 5–21 % gains and the winning variant is self-speculation, which is speculation. **[D]** arXiv:2608.28846
- **Host-RAM prefetch**: clean negative, "not even a trace-driven oracle", because bytes across the slowest populated tier bind — and moot, since 11.77 GB fits in 20 GB. **[D]** arXiv:2609.04238
- **Activation sparsity**: 3.86× only in the CPU-offload regime, and by approximating weights — quantisation in disguise. **[D]** arXiv:2610.02598
- **MoD**: needs pretraining. **PIM/CIM**: different chip.

Nothing in 2025–26 lifts the ceiling for an autoregressive 27B on 619 GB/s. The ceiling is real; the only levers are bytes per token (quantisation), tokens per weight pass (MTP/speculation, ~2–4×), and bytes moved per step at batch >1 (§1, KV).

---

## (c) Unknown, or unverified

**C-1. Whether IQ3_S can reach 60 % of 619 GB/s at all.** This is the project's biggest technical risk and it is not settled by any paper. The only RDNA per-quant bandwidth sweep at PR level shows achieved bandwidth *falling* with bpw (Q4_0 ≈ 409 GB/s, IQ3_S ≈ 236 GB/s, IQ2_XXS ≈ 155 GB/s on gfx1201), with two smoking guns that the loss is issue/latency rather than DRAM: a branch-free SWAR rewrite gave +23 % on IQ2_S at identical bpw, and *cutting* warps 8→1 gave +11–34 % on Q4_K on gfx1100 — impossible for a bandwidth-bound kernel. **[U] as design evidence**: those numbers live in llama.cpp's pull-request tracker, a repository `R77` excludes, so **no URL is given here on purpose** — cite the mechanism, not the source, and never lift the layout or heuristic. No IQ-type sweep exists for gfx1100 at all. If the trend holds, the 52.6 tok/s ceiling in `R4` is unreachable at 3.50 bpw and the plan's first kernel is aimed at the wrong type.

**C-2. Does bf16 WMMA actually beat vector FMA for our prefill shapes on gfx1100?** `R19` fixes WMMA as the compute path, but no measurement in the record compares a WMMA-path prefill against a well-blocked vector path at 27B shapes on this card. hipfire's "prefill was WMMA-bound" is suggestive, not decisive.

**C-3. Recurrent-state re-quantisation behaviour.** hipfire found that Q8/Q4 re-quantisation of the DeltaNet state needs **stochastic rounding** or coherence decays after ~500 tokens. Our state is FP32 (`R29`), so this is latent — but it is a landmine for any future state compression, and for any state *checkpoint* path that round-trips through a narrower type.

**C-4. gfx1100 VMM hazard.** hipfire's 0.4.2 changelog records: "On gfx1100 the VMM arm page-faulted and dropped the card off the bus." Context is a 2/4-bit V-cache decode write under their recorder, but the failure class — a VMM-armed allocation taking the card down — is exactly what a large paged-KV arena would exercise. Not replicated here; treat as a hazard to design around (prefer plain allocations for KV until proven otherwise).

**C-5. Co-tenancy determinism.** hipfire fixed a barrier bug where "greedy output changed under concurrent GPU load (any second process; on an APU, any GPU client)". Our box has a display card; a second process is the normal case. Any kernel using LDS across waves must retire in-flight loads of freed slots, or the 32-token arbiter will fail intermittently and look like a numerics bug.

**C-6. UDNA's matrix ISA.** No public ISA reference as of 2026-10-22; only LLVM targets (`gfx1310`/`amdgpu13`). Whether it keeps WMMA or moves to the CDNA MFMA model is unpriceable. **[V]** LLVM AMDGPUUsage.

**C-7. DeepInfer.** DeepSeek open-sourced its production engine on **21 Oct 2026** (Apache-2.0, NVIDIA + Huawei) — one day before this document. Not read here. Its grouped-expert `psum` layout was already assessed as irrelevant to a dense model (a donor report (removed)), but the open-sourcing may change what is inspectable. **[U]**

**C-8. Items I could not verify.**
- **SHEPHERD** (SLO-aware LLM serving, widely attributed to NSDI'25): **no arXiv record and nothing in the NSDI'25 programme. Do not cite.** **[U]**
- **BEAM / MorphServe / BOute** (MLSys 2026 orals, per the delegated pass): abstract-level only, full text not opened. **[U]** as to numbers.
- **Stream-K's arXiv identifier**: not resolved in this pass (the ID I expected belongs to an unrelated paper). Cite the ASPLOS'23 paper by venue, not by ID, until verified.
- **FlashAttention-4 / cuDNN attention on Blackwell**: not investigated. Irrelevant to gfx1100 either way.

---

## (d) Three things worth prototyping, and one measurement that decides a plan decision

Each is scoped to be falsifiable in under a week, and each has a kill criterion stated *before* the work.

### P-A. The IQ3_S GEMV, shaped like a shipping RDNA kernel — and the bandwidth test that gates the whole plan

**Hypothesis:** a one-pass, column-major, one-row-per-wave32 kernel with a load-time pre-shuffle reaches ≥60 % of 619 GB/s on IQ3_S — 371 GB/s, which is **30.8 tok/s at 4K** and 26.7 tok/s at 32K.

**Calibration, so the target is not confused with the gate.** 60 % of the wall (371 GB/s) is *below* what a shipping RDNA engine achieves on this silicon family: 72.8 % of spec (a.1) is 582 GB/s on our 800 GB/s card, i.e. 94 % of the 619 GB/s read wall, giving **48.4 tok/s at 4K and 41.8 at 32K** — and 48 tok/s at 4K is the P2 acceptance number proposed in `HANDOFF.md:210`. So: 30.8 tok/s is the floor a kernel must clear to be worth shipping, ~48 tok/s is what the best comparable engine implies is possible, and the whole gap is issue efficiency, not bytes. Note the competitor's format is 4.25 bpw with a register-resident codebook — fewer instructions per byte than IQ3_S — which is precisely why C-1 is a risk and not a detail.

**Method:** geometry from a.1 (32-thread WG, `__launch_bounds__(32,16)`, no LDS, q8_1 activations, dp4a, scale after the group dot); layout decisions from a.2 (per-lane contiguous vector loads; register-synthesised codes rather than table gathers); correctness first, per `R41`/`R42`. Then sweep warps-per-row ∈ {1,2,4} and rows-per-WG, because C-1 predicts the answer depends on it and the RDNA evidence says *fewer* warps wins for complex types.

**Kill criterion:** if no geometry exceeds **24.9 tok/s at 4K on IQ3_S (300 GB/s, 48 % of the wall)** while a Q4_K control on the same harness reaches ≥500 GB/s, then IQ3_S is issue-bound on this chip and `R4`'s 52.6 tok/s ceiling is unreachable at 3.50 bpw. That result is worth more than the kernel: it re-opens the quant choice (`R16`) with a measurement instead of a preference.

**Why now:** it is the P0 row of the kernel inventory anyway, and it is the only experiment that can invalidate the plan's central number before weeks are spent on it.

### P-B. A persistent decode step with device-side KV-tile bin-packing

**Hypothesis:** at 8 mixed-length sessions (one at 32K, seven at 1–4K), a per-step task map plus persistent CTAs beats one-kernel-per-layer plus graph replay by ≥10 %, because the tail is set by the longest session and 84 CUs cannot hide it.

**Method:** two pieces, separable and separately measurable. (1) The attention layer only: assign kernel builds a tile map from live KV lengths, persistent CTAs drain bins, a combine kernel merges partials (a.6, minus everything FP8/MoE). This deliberately avoids the device-scope barrier `02-megakernels.md` §3.3 flags — the bins are drained inside one kernel launch and the combine is a second launch, so nothing crosses 84 CUs mid-flight. (2) Only if (1) pays, extend persistence across the 64 layers with global-memory counters instead of kernel boundaries (a.5) — and at that point `02`'s conditions C1–C3 govern, not this document.

**Kill criterion:** if (1) does not beat graph-replay attention by 10 % at the mixed-length mix, the mixed-length tail is not worth scheduling around on 84 CUs and P6 should be dropped, not deferred.

**Why this survives `02`'s "do not build a megakernel":** it is not a megakernel. (1) is a scheduler *inside* one kernel for one layer — the thing HPC-Ops measured at 2.95× on a workload whose imbalance comes from length variance, which is a different mechanism from the launch-gap prize `02` correctly sizes at ~5 %.

**Hard constraint from §b.10:** the task map is rebuilt from *device-side* sequence lengths each step, deterministically. No wall-clock-driven route choice, no host-side timing input, or the 32-token arbiter stops being a gate.

### P-C. Bytes-based admission with q8 KV and page selection

**Hypothesis:** a single integer inequality — admit session *i* only while `11.77 GB + Σ_j KV_j + 0.159 GB × B ≤ BW_eff × T_step_target`, with `BW_eff` the bandwidth P-A actually achieves (not the 619 GB/s peak) — admits the right number of sessions and makes the 8×32K target honest, where request-count admission overcommits and then thrashes.

**Method:** (i) q8 KV per `R22`, which §1 shows is the precondition for the target at all; (ii) per-step page selection at the 16 attention layers (Quest shape: keep all KV, gather only query-selected pages) to cut KV bytes per step rather than KV capacity; (iii) admission/eviction on the inequality, with the recurrent-state cost (159 MB × B) charged to the same budget.

**Kill criterion:** if q8 KV plus page selection cannot hold 6 sessions at 32K with P99 step time within 1.2× of the model, then the concurrency target is a two-card problem (TP) and `R7`/`R8`'s one-card-first rule needs an explicit exception recorded as an ADR.

**Why this and not a scheduler:** it is the only "adaptive" mechanism the hardware supports (bandwidth is the budget, §b.11), it is arithmetic rather than ML, and it is deterministic — so it survives the promotion gate.

### M0. Measure MTP acceptance on our own prompts before building anything (half a day)

`R58`/`R59` rest on a publisher's 3.4–4.4 range from different hardware, and `C6` records three mutually inconsistent payoff claims. The gfx1100 evidence (a.9) says the answer is genre-conditional: 4.45× on code, 1.13× on prose, 1.00× on short instruct, 0.79× in the worst measured case. Since the MTP head is already on the host and already supported in the forked tree, τ on our own traffic is a log-line, not a project. It settles D6 with a number instead of a hope, and it costs less than the argument about it.

---

## Appendix — source table

| # | Item | Venue / type | Date | URL | Tag |
|---|---|---|---|---|---|
| 1 | hipfire (RDNA-native engine), v0.4.2 | OSS, Apache-2.0 | 2026-10-14 | https://github.com/warpfront/hipfire | V |
| 2 | hipfire `PRIOR-ART.md` (HFQ4 shape, MagnumQuant, asym KV, DDTree, LCP cache) | OSS doc | 2026-03-20 → 2026-06-09 | https://github.com/warpfront/hipfire/blob/master/PRIOR-ART.md | V |
| 3 | hipfire `BENCHMARKS.md` (7900 XTX 27B 47 tok/s; DFlash by genre) | OSS doc, historical rows | inventory 2026-07-19 | https://raw.githubusercontent.com/warpfront/hipfire/master/docs/BENCHMARKS.md | V |
| 4 | hipfire `QUANTIZATION.md` (MQ4 = FWHT-256 + G256, 4.25 bpw) | OSS doc | v0.4.2 | https://raw.githubusercontent.com/warpfront/hipfire/master/docs/QUANTIZATION.md | V |
| 5 | hipfire `CHANGELOG.md` (deinterleave 15→43 tok/s; stochastic rounding; VMM fault) | OSS doc | 2026-04-05 … 2026-10-14 | https://raw.githubusercontent.com/warpfront/hipfire/master/CHANGELOG.md | V |
| 6 | DeepGEMM (SM90/100, FP8/FP4, Mega MoE) | OSS, MIT | news to 2026-09-30 | https://github.com/deepseek-ai/DeepGEMM | V |
| 7 | FlashMLA (Hopper dropped; Blackwell + Ascend; AMD via AITER) | OSS, MIT | 2026-09-30 | https://github.com/deepseek-ai/FlashMLA | V |
| 8 | AITER (AMD operator library, Instinct) | OSS doc | news to 2026-07 | https://raw.githubusercontent.com/ROCm/aiter/main/README.md | V |
| 9 | *MARLIN* | arXiv:2408.11743 | 2024-08-21 | https://arxiv.org/abs/2408.11743 | V |
| 10 | Machete (Hopper wgmma repack) | vLLM PR #7174 | 2024 | https://github.com/vllm-project/vllm/pull/7174 | D |
| 11 | *Ada-MK: Adaptive MegaKernel … DAG-based Search* | arXiv:2605.11581 | 2026-05-12 | https://arxiv.org/abs/2605.11581 | V |
| 12 | Cohere megakernel serving engine (1.58×, 62 % of SoL) | Industry blog | 2026-09-08 | https://cohere.com/blog/megakernels | V |
| 13 | HPC-Ops × SGLang (dynamic split-KV, 2.95× over static) | Industry blog + upstream PRs | 2026-08-07 | https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ | V |
| 14 | *Why Gated DeltaNet Survives 4-Bit Quantization* (NVFP4 W4A4, 48+16 27B) | arXiv:2609.04098 | 2026-09-03 | https://arxiv.org/abs/2609.04098 | V |
| 15 | *Prefix Caching via Suffix Replay for Hybrid LLMs* | arXiv:2609.33477 | 2026-09-27 | https://arxiv.org/abs/2609.33477 | V |
| 16 | *TreeWY: Speculative Verification for Gated DeltaNet Hybrids* | arXiv:2608.20961 | 2026-08-21 | https://arxiv.org/abs/2608.20961 | V |
| 17 | *DDTree: Block Diffusion Draft Trees* | arXiv:2604.12989 | 2026-04-14 | https://arxiv.org/abs/2604.12989 | V |
| 18 | *The Illusion of Power Capping in LLM Decode* | arXiv:2605.11999 | 2026-05-12 | https://arxiv.org/abs/2605.11999 | V |
| 19 | *Taming Bitwise Behavior in GPU Kernels* (split-K reduction order) | arXiv:2609.11356 | 2026-09-10 | https://arxiv.org/abs/2609.11356 | V |
| 20 | *FASQ* (PQ weights, LUT-free GEMV, split-K) | arXiv:2605.04084 | 2026-04-22 | https://arxiv.org/abs/2605.04084 | V |
| 21 | *W4A16 on Ascend* (bottleneck is weight transfer, not dequant) | arXiv:2601.16536 | 2026-01-23 | https://arxiv.org/abs/2601.16536 | V |
| 22 | *DynaCore* (MEU reshape + temporal split-K, accelerator design) | arXiv:2610.07443 | 2026-10-05 | https://arxiv.org/abs/2610.07443 | V |
| 23 | RDNA4 ISA reference (FP8 WMMA, SWMMAC, no FP4) | AMD PDF | 2025-04-07 | https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf | V |
| 24 | RDNA3.5 ISA reference (identical WMMA set) | AMD PDF | 2024-07-23 | https://gpuopen.com/download/rdna35_instruction_set_architecture.pdf | V |
| 25 | ROCm precision support (RDNA3: no FP8) | AMD docs | fetched 2026-10-22 | https://rocm.docs.amd.com/en/latest/reference/precision-support.html | V |
| 26 | ROCm 10.1 compatibility matrix (gfx1100 still supported) | AMD docs | 2026-08-26 | https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html | V |
| 27 | LLVM AMDGPUUsage (gfx1310 targets; no UDNA ISA) | LLVM docs | fetched 2026-10-22 | https://llvm.org/docs/AMDGPUUsage.html | V |
| 28 | Dual R9700 P2P/TP measurement | Puget Systems | 2026-08 / 2026-09 | https://www.pugetsystems.com/labs/articles/amd-radeon-ai-pro-r9700-dual-gpu-ai-inference-performance/ | D |
| 29 | Mooncake | FAST'25 (best paper) | 2025-02 | https://www.usenix.org/conference/fast25/presentation/qin | D |
| 30 | DistServe | OSDI'24 | 2024-07 | https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin | D |
| 31 | Splitwise | ISCA'24 | 2024 | https://arxiv.org/abs/2311.18677 | D |
| 32 | llm-d v1.0 | Industry | 2026-10-20 | https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2 | D |
| 33 | FluidPD | arXiv:2610.06917 | 2026-10-02 | https://arxiv.org/abs/2610.06917 | D |
| 34 | HiCache | LMSYS blog | 2025-09-10 | https://www.lmsys.org/blog/2025-09-10-sglang-hicache/ | D |
| 35 | LMCache | arXiv:2510.09665 | 2025-10 | https://arxiv.org/abs/2510.09665 | D |
| 36 | Quest (query-aware page selection) | ICML'24 | 2024 | https://arxiv.org/abs/2406.10774 | D |
| 37 | RadixAttention | arXiv:2312.07104 | 2023-12 | https://arxiv.org/abs/2312.07104 | D |
| 38 | EAGLE-3 | arXiv:2503.01840 | 2025-03 | https://arxiv.org/abs/2503.01840 | D |
| 39 | P-EAGLE | arXiv:2602.01469 / vLLM blog | 2026-02 / 2026-03 | https://arxiv.org/abs/2602.01469 | D |
| 40 | DFlash | ICML'26, arXiv:2602.06036 | 2026 | https://arxiv.org/abs/2602.06036 | D |
| 41 | DScale (degradation at concurrency 8–32) | arXiv:2609.37532 | 2026-09-29 | https://arxiv.org/abs/2609.37532 | D |
| 42 | SuffixDecoding | NeurIPS'25 spotlight, arXiv:2411.04975 | 2025 | https://arxiv.org/abs/2411.04975 | D |
| 43 | Sarathi-Serve | OSDI'24 | 2024-07 | https://www.usenix.org/conference/osdi24/presentation/agrawal | D |
| 44 | Llumnix | OSDI'24 | 2024-07 | https://www.usenix.org/conference/osdi24/presentation/sun-biao | D |
| 45 | Autellix | arXiv:2502.13965 | 2025-02 | https://arxiv.org/abs/2502.13965 | D |
| 46 | JITServe (ex-Tempo) | arXiv:2504.20068 | 2025-04 | https://arxiv.org/abs/2504.20068 | D |
| 47 | DISCO (dynamic lookahead) | arXiv:2405.04304 | 2024 | https://arxiv.org/abs/2405.04304 | D |
| 48 | D-cut (adaptive draft pruning) | arXiv:2607.14647 | 2026-07 | https://arxiv.org/abs/2607.14647 | D |
| 49 | PolyGraph (adaptive draft breaks graph capture) | arXiv:2610.23861 | 2026-08 | https://arxiv.org/abs/2610.23861 | D |
| 50 | SLO-Guard (autotuning ≈ random) | arXiv:2604.17627 | 2026-04 | https://arxiv.org/abs/2604.17627 | D |
| 51 | Kevin (RL kernel generation, 1.10× eager) | arXiv:2507.11948 | 2025-07 | https://arxiv.org/abs/2507.11948 | D |
| 52 | GEAK (AMD, Triton, ≤63 % correct) | arXiv:2507.23194 | 2025-07 | https://arxiv.org/abs/2507.23194 | D |
| 53 | AMDKernelVault (34 % Pass@1) | arXiv:2609.12471 | 2026-09 | https://arxiv.org/abs/2609.12471 | D |
| 54 | AsmEvo (9–18 % on production kernels) | arXiv:2608.20711 | 2026-08 | https://arxiv.org/abs/2608.20711 | D |
| 55 | BEAM / MorphServe / BOute | MLSys 2026 orals | 2026 | https://mlsys.org/virtual/2026/session/3678 | U (abstract only) |
| 56 | SHEPHERD | — | — | no record found | U (do not cite) |
| 57 | DeepInfer open-source release | Industry | 2026-10-21 | https://www.technology.org/2026/10/21/deepseek-deepinfer-open-source-inference-engine-huawei-ascend-950/ | U (unread) |
