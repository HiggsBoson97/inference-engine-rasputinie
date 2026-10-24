# 04 — KV cache: memory, quantisation, and quality

> **Correction, 2026-10-23.** This lane's tables use `8 × 32 K` as an *illustration*.
> The owner has not chosen a session count or a context length, so that shape is not a
> target and must not be optimised for. The load-bearing variable is **resident tokens
> `B × C`**; the boundaries are derived in `docs/plan/COST-LEDGER.md`. The arithmetic
> below is unchanged — only the framing of one line was too strong.

**Question:** what are all the levers on KV memory and quality for Qwen3.8-27B on 40 GB, and which preserve quality for *this* architecture?

**Short answer:** the KV cache on this model is a **capacity** lever, not a bandwidth lever, and the interesting lever is not the KV cache at all — it is the 159 MB recurrent state, whose second copy (the prefix-cache checkpoint) is quantisable at nearly zero risk. q8 KV is worth building, but it is worth building because it takes 32K sessions from **6.5 to 11.6 per card** and makes the native 256K context fit at all — not because it makes batch-1 decode faster, where it buys 8%.

**Tag legend**

| Tag | Meaning |
|---|---|
| **[measured]** | Measured on `rasputin`, or recorded as measured in this repo (`docs/00-facts.md`, `bench/`). |
| **[documented]** | Read from primary source: engine code on disk, a format specification, a config file. |
| **[reported-by-publisher]** | The model publisher's or quant publisher's own numbers. Not reproduced here. |
| **[claimed]** | A paper's author-reported result. Not independently reproduced, and not reproduced on this architecture. |

**Constraint on the evidence base.** `docs/decisions/0001-exclude-llama-cpp.md:14-20` states that llama.cpp "is not a baseline. No number in this repository comes from it" and that its source "is not read while writing ours". The only published GGUF-style KV-quantisation quality measurements on Qwen3.5-family models come from that engine. **They are therefore inadmissible here**, and this document does not use them. That is an evidence gap created by a decision, and §6 treats it as a thing to measure rather than a thing to cite. The block *layouts* used below come from the GGUF tensor-type specification, which `docs/decisions/0001:27-29` explicitly permits.

---

## 1. The arithmetic that decides the priority

### 1.1 Bytes per token

The model keeps KV in 16 of 64 layers (`LLLF` × 16), 4 KV heads, head_dim 256 (`docs/01-models.md`).

`16 layers × 4 heads × 256 dim × 2 tensors (K,V) = 32,768 elements per token`

| Format | bpw | KiB/token | 32K | 128K | 256K | per card @ TP=2, 32K |
|---|---|---|---|---|---|---|
| bf16 (today) | 16.0 | 64.0 | 2.00 GiB | 8.00 GiB | 16.00 GiB | 1.00 GiB |
| fp8 e4m3 | 8.0 | 32.0 | 1.00 | 4.00 | 8.00 | 0.50 |
| **q8_0** | 8.5 | **34.0** | 1.06 | 4.25 | 8.50 | 0.53 |
| q8 K / q4_0 V | 6.5 | 26.0 | 0.81 | 3.25 | 6.50 | 0.41 |
| q5_1 | 6.0 | 24.0 | 0.75 | 3.00 | 6.00 | 0.38 |
| q4_1 | 5.0 | 20.0 | 0.62 | 2.50 | 5.00 | 0.31 |
| q4_0 | 4.5 | 18.0 | 0.56 | 2.25 | 4.50 | 0.28 |
| 4-bit g128 + f16 scale (QuaRot-style) | 4.25 | 17.0 | 0.53 | 2.12 | 4.25 | 0.27 |
| KVQuant nuq3 + 1% outliers | 3.33 | 13.3 | 0.42 | 1.66 | 3.33 | 0.21 |
| KIVI-2 g32 (+8 MiB fp16 window) | 3.0 | 12.0 | 0.38 | 1.50 | 3.00 | 0.19 |

Arithmetic is spec-derived. Sanity check against measurement: the FP8 baseline recorded 45,727 tokens in **1.40 GB** of bf16 KV (`docs/00-facts.md`) **[measured]**. Note what that number can and cannot be. 45,727 tokens × 64 KiB/token is 2.86 GiB = 3.07 GB in total across both cards, which is **1.53 GB per card**; 1.40 GB as a *total* would imply 30.6 bytes per token, which is arithmetically impossible for 16 layers × 4 heads × 256 dim × K and V × 2 bytes. **So `1.40 GB` is the per-card pool**, and 1.53 vs 1.40 is the log's rounding plus non-uniform head allocation across the two cards. The 64 KiB/token figure stands, and the record's own label ("total") is the thing that is loose, not the arithmetic. Worth a correction banner per the standing rule that a measurement beats a document.

Note the q8_0 penalty: 8.5 bpw, not 8.0, because of the f16 block scale. **On this model q8_0 costs 34 KiB/token against fp8's 32 — and fp8 is not available as a compute format on gfx1100 at all** (ADR 0003 `:7-10`, no FP8 matrix instruction). Storing fp8 would mean a conversion on every read with no compute benefit and fewer mantissa bits than int8 provides. q8_0 strictly dominates fp8 here. ADR 0003 `:41-42` already says this; the arithmetic confirms it.

### 1.2 Bandwidth: the crossover is at 180K resident tokens

Per card under TP=2 the KV read is **32 KiB per token of context**. The weight read at IQ3_S is 5.48 GiB/card (`docs/00-facts.md`, 10.96 GiB total).

`5.48 GiB ÷ 32 KiB = 179,568 tokens`

**That is the whole bandwidth story.** Below 180K *resident* tokens the KV read is a minority of decode traffic; above it, KV dominates. Because the figure depends on total resident tokens, "8 sessions × 32K" and "1 session × 256K" are the same KV traffic (262,144 tokens → 8 GiB/card at bf16, 4.25 GiB at q8).

| Workload | KV read/card, bf16 | Weight read | KV share | q8 speedup (bandwidth model) |
|---|---|---|---|---|
| 1 × 32K | 1.00 GiB | 5.48 | 15.4% | **1.08×** |
| 1 × 128K | 4.00 | 5.48 | 42.2% | 1.25× |
| 8 × 8K | 2.00 | 5.48 | 26.7% | 1.14× |
| 8 × 32K | 8.00 | 5.48 | 59.3% | **1.39×** |

Consequences, and they invert the intuitive ordering:

1. **At batch 1 and 32K, q8 KV is worth 8%.** Do not build it as a speed optimisation. The 857 ms/token baseline is 36× below the memory wall (`docs/00-facts.md`), so an 8% traffic saving is lost in the noise of a defect that is not yet explained.
2. **Above ~173K *resident tokens*, q8 is worth ~39 %.** That boundary is a property of
   the file and the card — not a chosen shape — and any `B × C` above it is equivalent for
   this purpose (`docs/plan/COST-LEDGER.md`). And it is worth that only after the weight
   path is fixed; before that, both numbers are hypothetical.
3. **Session count is where q8 actually pays**, and that is §1.3.

### 1.3 Capacity: the number that decides whether q8 is worth the kernels

Assume ~7 GiB/card of headroom after weights (the brief's figure; IQ3_S at TP=2 is 5.48 GiB of the 19.98 GiB card). Recurrent state splits with the 48 value heads, so ~74 MB/card per session.

| Format | KV/card @ 32K | + state | Sessions in 7 GiB |
|---|---|---|---|
| bf16 | 1.00 GiB | 0.074 | **6.5** |
| **q8_0** | 0.53 | 0.074 | **11.6** |
| q4_0 | 0.28 | 0.074 | 19.7 |

And the single-session long-context mode:

| Context | bf16 | q8_0 |
|---|---|---|
| 8K | 0.32 GiB ✔ | 0.21 GiB ✔ |
| 128K | 4.07 GiB ✔ | 2.20 GiB ✔ |
| 256K (native `262,144`) | **8.07 GiB ✘ does not fit** | 4.32 GiB ✔ |

**This is the strongest single argument for q8 in the whole document.** bf16 KV cannot serve the model's own native context length on one card. q8 can. The ADR reached the right answer; this is the arithmetic behind it.

The second-order observation: **the TP/DP choice (C1, unresolved) moves these numbers by 2×.** Under DP=2 each card carries a full weight copy (10.96 GiB) *and* the whole KV of its own sessions (64 KiB/token, not 32). At 4 sessions × 32K that is 10.96 + 8.00 + 0.15 = 19.1 GiB of 19.98 — it does not fit at bf16 and fits comfortably at q8. The plan's KV budget is not a function of the model alone.

---

## 2. The schemes, and what each costs

Everything below is a per-token cost for *this* shape. "Kernel cost" means work inside the attention inner loop, on gfx1100, in HIP, with bf16 WMMA as the only matrix path.

| Scheme | What is quantised | Bytes/token | Quality impact (best available evidence) | Kernel cost | Risk for this project |
|---|---|---|---|---|---|
| **bf16** (today) | nothing | 64 KiB | reference | none — but Triton path is 857 ms/token | capacity: 6.5 sessions @32K, no 256K |
| **q8_0** | K and V, 32-elem blocks, f16 scale along head_dim | 34 KiB | **+0.0002 PPL** on a 0.6B Qwen3 [claimed], and essentially exact on 4B/8B-class models in the same thread. Uniformly reported as transparent. | one `f16` scale per 32 elems; dequant to bf16 then WMMA; 8 blocks per head, block-aligned (`256 % 32 = 0`). Flash-decode must fuse the dequant. | **low.** The intended ADR target. Needs its own measurement because no admissible number exists on a hybrid Qwen3.5 (§5). |
| q8 K / **q4_0 V** | V at 4.5 bpw, K at 8.5 | 26 KiB | K is the sensitive tensor at every bit-width: K4V4 and K8V4 both land near the K4V8 point, i.e. **V can be cut to 4 bits once K is 8** [claimed, QuaRot tbl. and the K/V ablation in §2.1] | two block layouts in one kernel; V dequant on the P·V path | **low-medium.** −24% bytes for a measured-small cost. The one sub-8-bit step worth an experiment. |
| q5_1 / q5_0 | K and V, 32-elem, f16 scale + f16 min | 24 / 22 KiB | 5-bit is *not* a safe middle: the q5 results in the only modern-model sweep are mixed, with one model showing a catastrophic 4.5× PPL blowup at q5_1 that the same run did not show at q8 [claimed] | same block shape as q8_0 plus a bias term | **medium.** q5 is the awkward point — more kernel work than q8, no clear quality story. Skip. |
| q4_1 / q4_0 | K and V, 4-bit | 20 / 18 KiB | +0.04 PPL on Llama-2-7B K4V4 (5.51 vs 5.47) [claimed, QuaRot]; the damage is concentrated in K | 4-bit unpack + scale; no integer dot available (only via inline asm, `docs/00-facts.md`) | **high.** Buys 2.1 GiB/card over q8 at 8×32K and costs 4 K-bits. |
| **KIVI-2** | K per-channel (g32 tokens), V per-token, + fp16 window R=128 | 12 KiB + 8 MiB fixed | LongBench Llama-3-8B avg 44.37 vs 45.21 fp16 [claimed, authors]; **but** the only third-party long-context check puts passkey retrieval at 0.68–0.76 vs 1.00 fp16 [claimed, KVQuant's baseline table] | per-channel scales sit on the QKᵀ reduction axis — the awkward case; quantise/dequantise groups before the matmul; the fp16 residual window means two code paths | **high.** A recall regression that short-context PPL does not see. |
| **KVQuant** | K per-channel **pre-RoPE**, V per-token, NUQ LUT, 1% fp16 outliers, sink tokens | 13.3 KiB @ 3.33 bpw | best quality-per-bit in the literature: Llama-3-8B 5.64 vs 5.54 at 3-bit [claimed]; +0.10 PPL | heaviest of all: per-model Fisher/k-means calibration, per-layer LUT matvec, balanced sparse matvec, RoPE fused into the K matvec, CUDA-only, no HIP port | **very high.** Multi-week port to recover 1.5 GiB. Not defensible now. |
| **QAQ** | per-token, **variable** bit-width chosen from an attention-error bound | ~7.5× compression | "under 1% accuracy drop" — **but the paper's tables omit the fp16 baseline**, so the claim cannot be checked [claimed] | needs a CPU-resident fp16 mirror and per-step re-quantisation; bit allocation depends on live attention scores | **disqualifying.** Incompatible with a paged, append-only, prefix-shareable cache. |
| **QuaRot / SpinQuant** (rotation) | Hadamard on head_dim before write, then 4-bit g128 | 17 KiB | QuaRot K4V4 +0.04 PPL; **2-bit K is a cliff: 8.06 vs 5.47** [claimed]. SpinQuant W4A4KV4 Llama-3-8B 65.5 vs 69.6 — but KV is never ablated alone | **measured 1.45–1.57× slower than fp16 at batch 1** (0.713 → 1.033/1.117 ms, 3090); wins only from batch ~8-16 [claimed, QuaRot tbl. 15]. SpinQuant reports ~8% e2e overhead | **high at batch 1.** The one measured latency number in the literature says this *loses* in our exact regime. |
| **QuIP#** | weights only | n/a | n/a | n/a | **not applicable.** Does not quantise KV. |
| **fp8 e4m3 KV** | K and V | 32 KiB | sglang's own KV dtype surface is bf16/fp8_e4m3/fp8_e5m2 [documented, `mem_cache/kv_cache_dtype.py:14-19`] | storage-only on gfx1100; convert on every read | **no.** 2 KiB/token cheaper than q8_0, 4 mantissa bits, no compute path. ADR already right. |

### 2.1 The one robust finding across every scheme: K is the sensitive tensor

Independently reproduced across all four families:

- **QuaRot**, Llama-2-7B, rotation applied, only K/V bits varied [claimed]: fp16 5.47 · K4V4 5.51 · K4V2 5.75 · K3V4 5.65 · **K2V4 8.06** · K2V2 9.23. V at 2 bits costs +0.28; K at 2 bits costs +2.59.
- **KIVI** Table 1, Llama-2-13B [claimed]: per-token K at 2-bit collapses CoQA to 52.93 (from 66.37); per-channel K recovers it to 63.53. The fix is *granularity*, and it is a K-side fix.
- **KVQuant** ablation [claimed]: per-channel K + per-token V is worth 3.82 PPL at 3-bit, and every subsequent trick (NUQ 0.29, outliers 0.19) is smaller.

The mechanism is not subtle: K is what the query multiplies to produce the attention distribution, so a K error moves *which* tokens are attended; a V error only moves *what is read out* of an already-chosen set. Softmax amplifies the former.

**Therefore: any sub-q8 experiment spends its bits on K, never on V.** q8 K / q4_0 V is the shape of a defensible second gear, and q4_0 K is not.

### 2.2 A note specific to `partial_rotary_factor: 0.25`

Only 64 of 256 key channels are rotated; 192 carry position-invariant per-channel statistics (`docs/01-models.md`) **[documented]**. Every per-channel K scheme (KIVI, KVQuant) is *easier* on this model than on a fully-rotated one, because the channel statistics it relies on are not smeared by RoPE. This is a genuine architectural advantage — and it is also why KVQuant's "quantise K before RoPE" trick would be partly redundant here, removing its main justification.

The cheap version of the same idea, worth one experiment and no more: store the 192 unrotated channels at q4 and the 64 rotated ones at q8. K falls from 17.4 KiB to ~10 KiB per token, saving 7 KiB/token. It needs two block layouts in one kernel and is a ~1 GiB/card win at 8×32K. Interesting, not urgent, and **unmeasurable until the weight path is fixed.**

---

## 3. The recurrent state: the lever nobody has published on, and the one that is nearly free

This is the part of the question where the architecture, not the literature, is the answer.

### 3.1 What the state is

Per layer the GDN state is a running outer-product accumulator of shape `[n_v_heads, k_dim, v_dim] = [48, 128, 128]`, from 16 key heads broadcast to 48 value heads (`docs/01-models.md`) **[documented]**.

`48 layers × 48 × 128 × 128 × 4 B = 150,994,944 B = 151 MB` — plus ~8 MB conv1d window state, so **159 MB per sequence** (`docs/01-models.md`).

| dtype | Per sequence | 8 sessions | 12 sessions |
|---|---|---|---|
| FP32 | 151 MB | 1.21 GB | 1.81 GB |
| bf16 | 75.5 MB | 0.60 GB | 0.91 GB |
| int8 | 37.7 MB | 0.30 GB | 0.45 GB |

Note what this is *not*: at 32K it is 74 MB/card against 1.00 GiB of KV/card. **The recurrent state is 7% of the per-session memory at 32K.** It is not a capacity problem and no amount of cleverness there buys a session. It matters for exactly two reasons: it is read and written *every decode step* (so it is a bandwidth and latency item, not a capacity item), and it cannot be shared between prefixes, so it is the thing that makes prefix caching hard.

### 3.2 Is the state quantisable? The literature says: only in the right place

**Nobody has characterised GDN state quantisation.** No paper reports the state's per-channel dynamic range, its outlier structure, or an error bound for the delta rule specifically. That is the finding, and it means any move here is an experiment, not a citation.

What *is* published, and it is decisive:

**Every production kernel accumulates in FP32 and stores in whatever the cache dtype is** **[documented, verified in source]**:

- `fla/ops/common/chunk_delta_h.py`, `chunk_gated_delta_rule_fwd_h`: `b_h1 = tl.zeros([64, BV], dtype=tl.float32)`, downcast only on store; `final_state = ... dtype=torch.float32`.
- `fla/ops/gated_delta_rule/fused_recurrent.py`: `b_h += tl.load(p_h0, ...).to(tl.float32)` — accepts a bf16 `h0` and widens. The backward pass forces `dh0` to FP32 even when `h0` was bf16.
- vLLM `mamba_utils.py` / `config/cache.py`: `MambaDType = Literal['auto','float32','float16','bfloat16']`, default `'auto'`; **Qwen3.5-4B ships `mamba_ssm_dtype: float32` in its own `text_config`** **[documented, HF config]**. vLLM's fused GDN decode accepts FP32 or BF16 and **not FP16** (`FUSED_GDN_STATE_DTYPES`), and its KDA path raises on FP16.
- SGLang `configs/mamba_utils.py`: `ssm_dtype = torch.float32` as step-1 default, overridable by config or `SGLANG_MAMBA_SSM_DTYPE`; conv defaults to bf16.

**Is the state stable enough for bf16?** The recurrence is non-expansive, which is the opposite of the intuition that an outer-product accumulator blows up. DeltaNet (arXiv:2406.06484 §3.3) shows the transition `I − β_t k_t k_tᵀ` has eigenvalues 1 (multiplicity d−1) and `1 − β_t‖k_t‖²`, so with L2-normalised keys and `β = sigmoid(·) ∈ (0,1)` the spectral radius is at most 1; GDN adds a per-head decay `α < 1` **[claimed, paper]**. Overflow is not the failure mode. **Loss of the low-order bits is**: bf16 has an 8-bit mantissa, every stored entry is re-read as the base of the next update, so the rounding error re-enters at every token and accumulates rather than averaging out.

**The one paper that measures it** — ReplaySSM, arXiv:2610.13778 (Oct 2026), implemented in vLLM for Mamba-2 **and Gated DeltaNet** — separates the *recurrence-carrying checkpoint* from a *low-precision read copy*, so rounding never enters a later update **[claimed]**:

| Configuration | MMLU-Pro / GPQA-D / Math-L5 loss |
|---|---|
| shared FP16 + stochastic rounding | −0.09 / −0.24 pts — within noise |
| shared NVFP4 + SR | **−5.88 / −13.05 / −10.92**, to −28.2 in the longest-generation quartile |
| NVFP4 **read**, FP32 recurrence | −0.15 / −2.36 / −0.10 |
| NVFP4 **read**, FP16 + SR recurrence | −0.15 / −2.36 / −0.10 |

At 128K tokens, a low-precision read copy with an FP32 recurrence cuts output KL from 0.136 to 0.0038 — 36× — against the same quantiser sharing the state **[claimed]**. It also roughly halves state I/O and gives 1.20–1.47× decode at batch 256 with cheap speculative rollback.

Three conclusions, in order of confidence:

1. **The active, kernel-facing state stays FP32.** Every engine does this, every kernel accumulates in FP32, and the mechanism (error re-entering the recurrence) is the one ReplaySSM measures. Cost of keeping it: 151 MB/sequence, which is 7% of a 32K session. **This is not the place to save memory.**
2. **The cached checkpoint — the copy stored in a radix node — is a different object and is safely compressible.** It is written once on eviction and read once on a hit, so the error is a *single* rounding of S and never re-enters a loop. SGLang ships exactly this: `mamba_checkpoint_pool.py` stores the temporal state in **int8, per-(head, k-channel) symmetric**, and its own docstring gives the reasoning — int8 beats fp8-e4m3 at the same byte because "the temporal state is roughly uniformly distributed, so fp8 wastes bits on the exponent", and the scale axis reduces over `d_v` so it "matches the per-k-channel decay `diag(alpha)`, so the large state entries keep ~bf16 precision" **[documented, `mem_cache/mamba_checkpoint_pool.py:15-45`, read on disk in `_research/sglang`]**. ~2× the cached-prefix capacity at fixed memory.
3. **Do not touch FP16 and do not touch FP8 on gfx1100.** FP16 has bf16's mantissa with a 1/512th the exponent range; the only thing that makes FP16 safe in vLLM is stochastic rounding, which is gated on `current_platform.is_cuda()` and compute capability 10.0 for the Triton path (`cvt.rs` PTX) — **unavailable on ROCm**. FP8 would be storage-only with a conversion per read.

**So: FP32 is necessary for the state that runs, and unnecessary for the state that waits.** That distinction is worth more than every KV scheme below q8 combined, because it is ~500 lines of code against a multi-week kernel port, and it doubles cached-prefix capacity.

### 3.3 Why this is the real multi-session lever

The plan's P4 scope already includes "recurrent-state checkpointing so a prefix hit can be resumed" (02-plan.md (removed)). Without it, the radix cache on this model is a KV-only win on a model where — at 8K — KV is only 3× the state, and at 2K the state is *larger than the KV span it accompanies*:

| Checkpoint interval | State cost | KV cost | State as % of the KV it accompanies |
|---|---|---|---|
| every 2K tokens | 159 MB | 128 MiB | **118%** |
| every 8K | 159 MB | 512 MiB | 30% |
| every 16K | 159 MB | 1 GiB | 15% |

Int8 halves the first row. **Choose the checkpoint interval before choosing any quantiser** — the interval is the parameter that decides whether the prefix cache is memory-positive, and int8 is what makes the 2K interval affordable.

---

## 4. Attention-level tricks: what is lossless, what is lethal

The set splits on one property, and the split is sharper than the quality numbers suggest: **does a cache miss cost recompute, or does it cost fidelity?**

| Technique | What it does | Multi-turn conversation | Middle-of-context recall | Verdict |
|---|---|---|---|---|
| **GQA** (already 6:1) | 4 KV heads for 24 q heads | — | — | **Already banked.** 64 KiB/token instead of 384. MQA would be 16 KiB but is a pretraining decision; post-hoc head-averaging is not a serving knob. Nothing to do. |
| **Radix / prefix caching** | shares exact prefixes | turn N+1 prefills only the delta | **unchanged** — full attention still sees every token | **Adopt.** Lossless by construction. The only technique on this list with no quality cost. |
| **Sliding window** | local window per layer | nothing lost *that the layer ever had* | carried by the 16 global layers | **Already in the architecture.** Gemma-3's ablation (arXiv:2503.19786) shows PPL nearly insensitive to local:global ratio even at 7:1 [claimed]; Qwen3-Next uses 3 GDN : 1 gated attention [documented, config]. But applying a window to a layer *trained* full is the StreamingLLM failure case. **Our 16 full layers must stay full. Not a lever we hold.** |
| **StreamingLLM** (arXiv:2309.17453) | 4 sink tokens + rolling window | every turn older than the window is gone except 4 tokens | zero by construction | **Reject for an assistant.** Position-based, so it *is* prefix-reusable — the correct choice for an infinite log stream, the wrong one for a chat. |
| **H2O** (arXiv:2306.14048) | recent + heavy hitters by cumulative attention | a fact stated in turn 2 that did not attract attention *then* is dropped and unrecoverable — the model confabulates rather than errors | the exact failure mode | **Reject.** Also destroys the invariant a radix cache needs (§4.1). |
| **SnapKV** (arXiv:2404.14469) | observation window at prompt end picks per-head features | tuned to the *last* turn; fails aggregation and multi-hop | better than H2O, still unsafe; NIAH is its best case | **Reject as default.** Per-request opt-in at most. |
| **PyramidKV** (arXiv:2406.02069) | layer-varying budget | same query dependence, plus per-model tuning | 128 entries → 100% NIAH on Llama-3-70B [claimed] | **Reject**, and note the geometry: the pyramid has **16** layers to allocate across here, not 80, so the achievable saving is structurally ~5× smaller than the paper's setting. |

### 4.1 The incompatibility that settles it

**Any query-dependent eviction destroys the invariant a radix cache depends on: cached KV must be a pure function of the token prefix.** H2O, SnapKV and PyramidKV all choose their retained set using the queries observed, which means (a) two requests sharing a prefix cannot share the cache, (b) the retained set drifts during decode, and (c) the cache cannot live in a tree keyed by token prefix. You cannot have both. The radix cache is the one that pays on a multi-turn assistant, and the forked tree already implements it properly — leaf-only eviction, a parent evictable only when its last child is gone *and* `lock_ref == 0` (`mem_cache/radix_cache.py:552-580`), so **the middle of a live conversation can never be evicted** **[documented, read on disk]**. Policies `lru/lfu/slru/priority/tlru` (`arg_groups/choices.py:187`), with TLRU written for exactly this workload [claimed, arXiv:2510.15152].

That invariant is worth more than the 40-60% of KV that eviction would save, because eviction's saving is a *latency* optimisation on a workload where the failure is silent and the failure mode is confabulation.

---

## 5. What "quality" means, and how to measure it cheaply

### 5.1 The identity that makes thresholds defensible

For teacher-forced scoring on identical tokens, mean per-token KL in nats equals the cross-entropy gap:

`E[KL(p_bf16 ‖ p_q)] = CE_q − CE_bf16`  ⟹  **PPL ratio = exp(KL)**

So a KL budget *is* a PPL budget, and thresholds can be read off rather than invented:

| mean KL (nats) | PPL inflation |
|---|---|
| 0.005 | +0.50% |
| **0.010** | **+1.01%** |
| 0.020 | +2.02% |
| 0.050 | +5.13% |
| 0.100 | +10.52% |

Anchors: KVQuant (arXiv:2401.18079 §4) accepts ΔPPL < 0.02 at 4-bit, < 0.1 at 3-bit, < 0.5 at 2-bit on LLaMA-7B against a ~5.5 baseline → **KL ≈ 0.004 / 0.018 / 0.088 nats** [claimed]. So the community's 3-bit acceptance point is ≈0.02 nats, and 2-bit's ≈0.09 is where they themselves stop. No KV paper sets a KL threshold a priori; the derivation via the identity is the honest way to justify a gate.

**Proposed gate for this engine:**

| Test | Threshold | Rationale |
|---|---|---|
| KL, wikitext, mean | ≤ 0.010 nats | 1% PPL, ~2.5× inside KVQuant's 3-bit point |
| KL, chat-formatted, mean | ≤ 0.020 nats | chat turns are where the 4-bit schemes actually differ |
| KL, **any position bin** | ≤ 0.05 nats | see below — this is the one that catches KV quant |
| Free-running top-1 agreement, 256 greedy tokens × 100 prompts | report; flag if first-divergence median < 32 | catches argmax flips that a clean mean hides |
| NIAH 32K, depth 0.5 | no worse than bf16 depth 0.5 | compare against bf16, never against 100% |
| MMLU 2,000 (loglikelihood) | within ±1.1 pp of bf16 | the cheapest real signal; ±1.1 pp is its own noise floor |

### 5.2 Three measurement traps specific to this model

1. **Bin by position.** KV-quant error grows with the number of quantised tokens attended, so a 2K-window PPL hides the 32K behaviour entirely. This is the main methodological difference from weight quantisation. Report 0-2K / 2-8K / 8-32K / 32-128K separately, and gate on the worst bin, not the mean.
2. **The 4× dilution is structural but the recall risk is not.** Only 16 of 64 layers carry KV, so a given per-element KV error is diluted 4× relative to a dense model — PPL will look better here than the same scheme on a Llama. But those 16 layers are the *only* global-recall path; the other 48 have a fixed 159 MB state and no token-addressable memory. So the layers where the error lands are exactly the layers that carry long-range information. **A PPL pass on this model is weaker evidence than it would be on a dense one, and long-context recall is stronger evidence.** Do not let a clean PPL clear a scheme.
3. **AIME25 is unusable as a gate.** The publisher's own reference is AIME25 **100.00** at 3.50 bpw (`docs/adr/0003:52-53`) **[reported-by-publisher]**. At a saturated 100.00 the benchmark has zero headroom and cannot detect degradation, and 30 problems means one problem is 3.3 pp. Use it as a release sanity check, never as the discriminator. Same caution for the publisher's LCB v6 85.71 and GPQA-D 89.39: they are the *external reference for the weight quant*, not a KV-quant gate.

### 5.3 The per-change gate, sized for this box

Assumes ~800 tok/s prefill and 20 tok/s decode single-stream.

| # | Test | Cost | Failure action |
|---|---|---|---|
| 1 | Logit parity, 8 × 4K prompts, max-abs and RMS per layer band | **5 min** | hard fail at 1e-3 (already the inventory test, `docs/03-native-kernel-inventory.md:28`). Catches scale-layout, group-size, pre/post-RoPE bugs. Says nothing about quality. |
| 2 | KL + PPL, 128 × 2,048 wikitext, binned by position, bootstrapped **over documents** | **10 min** | the primary instrument. 64 × 2,048 is the floor; SE at 128 docs is 0.0002-0.001 nats, 10-50× below the gate. Report mean, p99, max, and fraction of tokens with KL > 0.5. |
| 3 | NIAH 32K × 100 probes at depths 0.1/0.5/0.9, batched 8 | **10 min** | depth 0.5 is the discriminating row; 0.1 and 0.9 sit near ceiling for anything competent |
| 4 | MMLU 2,000-question stratified subset, loglikelihood | **17 min** | prefill-bound, no decode, parallelises across sessions |
| 5 | GSM8K 300, 8-shot CoT | **1.6 h** | arithmetic-chain integrity; ±2.9 pp |
| — | **Total** | **~2.5 h/candidate** | anything failing 1 or 2 does not deserve the rest |

Release-only, against a bf16 baseline, quoting noise floors: GPQA-D 198 (3.3 h, ±3.6 pp), AIME25 avg@8 (~10 h, 1 problem = 3.3 pp), LiveCodeBench v6 (~13.4 h for 1,055; use 150-250 for ±3-4 pp). **LCB is the only suite that stresses long-form generation under a modified cache**, which is exactly what a quantised cache threatens, so it earns its place at release even though it is too expensive per change.

RULER (arXiv:2404.06654) at 4 lengths is the benchmark that catches eviction *and* quantisation — multi-hop tracing and aggregation are where full-cache models already drop, so a compressed cache compounds an existing weakness. ~8 h single-stream, ~1 h batched; run it once, when a scheme is proposed for default, not per change. And compare depth-0.5 misses against bf16's own depth-0.5 misses: Lost in the Middle (arXiv:2307.03172) already gives every model a U-shaped depth curve [claimed].

---

## 6. Recommendation, ordered by value for complexity

The ordering is by **value ÷ complexity**, not by bytes saved. Bytes saved is the wrong metric on a model where KV is 34 KiB/token and 48 of 64 layers have none.

| # | Lever | Buys | Complexity | Quality | Verdict |
|---|---|---|---|---|---|
| **1** | **q8_0 KV, paged, dequant fused into attention** | 6.5 → 11.6 sessions @32K; makes 256K fit; 1.39× traffic at 8×32K | one block layout, 8 per head, one f16 scale per 32 elems; already inventory row 2 (P1) | transparent in every report; needs our own measurement (§5) | **Do it.** ADR-mandated and the arithmetic backs it. But sell it as capacity, not speed. |
| **2** | **Radix prefix cache + int8 GDN checkpoint store** | lossless multi-turn; ~2× cached-prefix capacity | checkpoint-on-evict, dequant-on-hit; SGLang's `mamba_checkpoint_pool.py` is a working precedent to port | **zero for the KV** (lossless); single-rounding for the state, error does not re-enter the recurrence | **Do it second, and before any sub-q8 KV work.** It is the larger multi-session win and the lower risk. |
| **3** | **Keep the active recurrent state FP32** | avoids a silent, length-growing accuracy loss | none — it is the default | it is what every engine and every kernel does | **Do nothing, deliberately.** Record it as a decision so a future session does not "optimise" it. 151 MB/sequence is 7% of a 32K session. |
| **4** | **Checkpoint interval ≥ 8K; int8 below that** | prefix cache becomes memory-positive | a tunable, not code | lossless above 8K | **Decide the interval explicitly.** At 2K, checkpoints cost 118% of the KV they accompany unless int8. |
| **5** | **q8 K / q4_0 V (26 KiB/token)** | −24% KV bytes; ~1 more session/card | second block layout in the same kernel | small in the papers (K4V4 ≈ +0.04 PPL); **no admissible number for this model** | **One experiment, after 1-3.** Measure first (§5 gate); adopt only if bin-max KL ≤ 0.05. |
| **6** | Mixed precision exploiting `partial_rotary_factor 0.25` (192 unrotated channels at q4) | ~1 GiB/card at 8×32K | two layouts per head; fiddly | unknown, plausibly small — the unrotated channels are the well-conditioned ones | **Interesting, not now.** Unmeasurable until the weight path is fixed. |
| **7** | q5_0/q5_1, q4_1, q4_0 KV | 0.6-2.1 GiB/card at 8×32K | more than q8, for less clear quality | q5 is the awkward middle; q4 hits K | **Reject.** No defensible story at this model size. |
| **8** | KIVI-2, KVQuant, QAQ, QuaRot/SpinQuant rotation, fp8 KV | 1.5-4 GiB/card | multi-week HIP ports; QuaRot is **measured 1.45-1.57× slower at batch 1** | KIVI-2 loses passkey recall; QuaRot's 2-bit K is a cliff; QAQ has no kernel and no baseline | **Reject, with reasons recorded.** Each would need its own calibration and would still be slower per byte at batch 1 on a card with no FP8 and no integer dot. |
| **9** | H2O / SnapKV / PyramidKV / StreamingLLM | 40-60% of KV | moderate | silent, unrecoverable, and destroys radix sharing | **Reject.** Incompatible with the prefix cache, which is lossless. |

**Is KV quantisation worth the complexity at all?** Yes — but only the q8 step, and for a reason that is not the one usually given:

- As a **speed** optimisation at batch 1, 32K: **no.** 8%. Lost against an unexplained 36×-below-wall baseline.
- As a **capacity** optimisation: **yes, decisively.** 6.5 → 11.6 sessions per card at 32K, and it is what makes the model's own 256K context fit on one card at all (8.07 GiB ✘ → 4.32 GiB ✔ against 7 GiB of headroom).
- As a **traffic** optimisation at the target workload: **yes.** 39% at 8×32K, once the weight path is fixed.
- Below q8: **no.** Another 2.1 GiB/card and ~1.17× at 8×32K (q4 total 7.73 vs q8 9.73), for 4-bit K — the exact tensor every paper says breaks first — and a kernel port on a card with no integer dot and no FP8.

So the shape of the answer is: **bf16 KV is wrong not because it is slow but because it is too big; q8 is right; q4 is a solution to a problem we do not have.** And the 159 MB state, which looks like the expensive thing, is the cheap thing — as long as it is compressed where it waits and left alone where it runs.

---

## 7. What would change this answer

Stated so a future session can overturn the recommendation with evidence rather than opinion.

1. **A measurement of q4_0 K on this model showing bin-max KL ≤ 0.05.** Every reject in row 7/8 rests on Llama-2/3 evidence. This model has only 16 KV layers and 4 of 256 channels rotated; the sensitivity could genuinely be lower. The gate in §5.3 costs 2.5 h.
2. **A measured q8 attention kernel that is *slower* than bf16 at batch 1.** QuaRot's own number (0.713 → 1.033 ms at batch 1) says a quantised-KV attention kernel can lose on a card without a tensor-core path to absorb the dequant. If our q8 flash-decode comes in under 1.0× bf16, q8 survives on capacity alone — but the kernel must then be aggressively fused, and the "dequant to bf16 in registers" discipline in ADR 0003 `:36-37` becomes the critical path.
3. **A press for >12 sessions.** If the target moves to 16-20 sessions at 32K, q4 stops being over-engineering and becomes the only way there. That would need a decision, not a default.
4. **Evidence that GDN state bf16 is lossless over 32K+ generated tokens.** Would save 0.6 GB per 8 sessions — small, but it would also let the checkpoint store be bf16 and simplify the pool. ReplaySSM's KL-grows-with-length curve suggests it is not lossless, but nobody has run it on this model.
5. **TLRU or a cheaper eviction policy proving it beats LRU on real conversation traces here.** That is a latency win, and it does not touch quality — worth doing, but it is a P4 scheduling decision, not a KV-quant one.

---

## Provenance

| Claim | Status | Source |
|---|---|---|
| KV is 64 KiB/token, 2/8/16 GiB at 32K/128K/256K | **[documented]** config-derived | `docs/01-models.md,36-37`; recomputed here |
| 1.40 GB bf16 KV at 45,727 tokens; 2.08 GB free/card; `max_running_requests=3` | **[measured]** | `docs/00-facts.md` |
| Weight read 5.48 GiB/card at IQ3_S TP=2 | **[documented]** | `docs/00-facts.md` (11.76 GB / 10.96 GiB total) |
| Decode is 36× below the memory wall | **[measured]** | `docs/00-facts.md` |
| gfx1100 has no FP8 matrix instruction; no integer dot except via inline asm | **[measured]** | `docs/00-facts.md`; ADR 0003 `:7-10` |
| KV dtype surface in sglang is bf16/fp8_e4m3/fp8_e5m2 — q8 is engine work, not a flag | **[documented]** | `mem_cache/kv_cache_dtype.py:14-19` |
| Radix eviction is leaf-only; a parent is evictable only when its last child is gone and `lock_ref == 0` | **[documented]** | `mem_cache/radix_cache.py:552-580`; `evict_policy.py`; `choices.py:187` |
| SGLang stores cached GDN state as int8 per-(head,k-channel), conv at native dtype, with the fp8-vs-int8 and single-rounding argument | **[documented]** | `mem_cache/mamba_checkpoint_pool.py:15-45`, read on disk |
| fla accumulates FP32, returns FP32 `final_state`, accepts bf16 `h0` and widens | **[documented]** | `fla/ops/common/chunk_delta_h.py`; `fla/ops/gated_delta_rule/fused_recurrent.py` |
| vLLM GDN state dtype ∈ {auto,float32,float16,bfloat16}; fused decode rejects FP16; KDA rejects FP16; SR is CUDA/CC10-gated | **[documented]** | `vllm/config/cache.py`, `config/mamba.py`, `config/vllm.py`, `mamba/gdn/qwen_gdn_linear_attn.py` |
| Qwen3.5-4B ships `mamba_ssm_dtype: float32` | **[documented]** | HF `config.json` |
| SGLang temporal state defaults FP32, conv bf16, overridable | **[documented]** | `srt/configs/mamba_utils.py`, `srt/mem_cache/memory_pool.py` |
| Delta-rule transition has spectral radius ≤ 1 with L2-normalised keys | **[claimed]** | arXiv:2406.06484 §3.3; arXiv:2412.06464 §3.1 |
| FP16/FP8 read-copy with FP32 recurrence is near-lossless; shared NVFP4 loses 5.9-13 pts; KL 0.136 → 0.0038 at 128K; 1.20-1.47× decode | **[claimed]** | arXiv:2610.13778 (ReplaySSM) |
| K is the sensitive tensor at equal bits; V is nearly free | **[claimed]** | QuaRot arXiv:2404.00456; KIVI arXiv:2402.02750 tbl.1; KVQuant arXiv:2401.18079 |
| 4-bit KV attention measured 1.45-1.57× slower than fp16 at batch 1 | **[claimed]** | QuaRot arXiv:2404.00456 tbl.15 |
| KVQuant 3.33 bpw; Llama-3-8B 5.64 vs 5.54 at 3-bit; ΔPPL tolerances 0.02/0.1/0.5 | **[claimed]** | arXiv:2401.18079 |
| KIVI-2 LongBench 44.37 vs 45.21; passkey 0.68-0.76 vs 1.00 | **[claimed]** | arXiv:2402.02750; KVQuant's baseline table |
| SpinQuant ~8% overhead; W4A4KV4 Llama-3-8B 65.5 vs 69.6 | **[claimed]** | arXiv:2405.16406 |
| QAQ: no kernel, no fp16 baseline in its own tables, needs CPU mirror | **[claimed]** | arXiv:2403.04643 |
| QuIP# is weight-only and does not quantise KV | **[documented]** | arXiv:2402.04396 abstract |
| Gemma-3 local:global ablation insensitive to 7:1; Qwen3-Next 3 GDN : 1 attention | **[claimed]** / **[documented]** | arXiv:2503.19786; Qwen3-Next config |
| Eviction: H2O/SnapKV/PyramidKV are query-dependent; StreamingLLM is position-based | **[claimed]** | arXiv:2306.14048, 2404.14469, 2406.02069, 2309.17453 |
| AIME25 100.00 / LCB v6 85.71 / GPQA-D 89.39 at 3.50 bpw | **[reported-by-publisher]** | ADR 0003 `:52-53` |
| Session counts, traffic shares, speedup multipliers | **[derived]** arithmetic, §1 | this document; `kvmath.py` in scratch |
| GGUF KV block layouts and bpw (q8_0 8.5, q5_1 6.0, q5_0 5.5, q4_1 5.0, q4_0 4.5) | **[documented]** | GGUF tensor-type specification (permitted by `docs/decisions/0001:27-29`) |
| No GGUF K-quant (q4_K/q6_K) exists as a KV type | **[documented]** | KV type allow-list is the 32-element block family; K-quants are weight-only 256-element super-blocks |

**Numbers deliberately absent.** The most complete measured sweep of GGUF KV-quant quality on Qwen3-family models (including a Qwen3.5-9B run over 36 K/V type combinations at 46K) exists in the llama.cpp PR record. `docs/decisions/0001:14-20` forbids this repository from carrying any number from that engine, so it is not reproduced here and must not be re-imported by a later session. The gap is real: **there is no admissible published measurement of KV quantisation on a 48/64 GatedDeltaNet hybrid**, which is why §5.3 is a build item and not a citation.
