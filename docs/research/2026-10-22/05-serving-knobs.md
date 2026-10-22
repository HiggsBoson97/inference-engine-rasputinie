# 05 — Serving-side knobs on a bandwidth-bound box

**Scope:** every serving-layer decision that changes bytes moved per token, on 2× gfx1100
(19.98 GiB, 619 GB/s read wall each, **no P2P**: 14.2 GiB/s cross-GPU = host bounce), serving
Qwen3.8-27B at 3.50 bpw (11.81 GB of weights, 64 layers = 48 GDN + 16 full-attention, 4 KV heads,
head_dim 256, 16 KV layers → 64 KiB/token bf16, plus 159 MB of GDN recurrent state per sequence).

**Verdict in one paragraph.** This box is bandwidth-bound at *every* batch size and every context
above ~12 K tokens, so the usual serving playbook (batch harder to fill the compute units) buys
almost nothing per token and the knobs that matter are the ones that remove bytes: q8 KV (halves the
dominant long-context term), speculation (divides *all* body bytes by the accept length), and TP2
(halves every byte stream by paying a 10–20 % PCIe latency tax instead). Prefix caching and chunked
prefill are worth having but they buy *latency and session count*, not tokens/s — and on this hybrid
model prefix caching is only viable if the 159 MB GDN state is checkpointed, ideally to host RAM.
None of it is observable until the single Triton fp8 GEMV that owns 91.8 % of the current step is
fixed: measured today is 1.15 tok/s against a 52 tok/s wall.

Evidence tags: **[M]** measured on this box (local doc cited), **[D]** derived arithmetic from
measured inputs, **[L]** literature / other hardware, **[E]** estimate or judgement, **[U]** unverified.

---

## 0. The arithmetic this document is built on

| quantity | value | source |
|---|---|---|
| weight bytes | 11.81 GB (10.96 GiB @ 3.50 bpw) | local `docs/adr/0003`, `REQUIREMENTS-DRAFT.md` R16 **[M]** |
| weight read at batch 1, one card | **19.08 ms → 52.4 tok/s** | 11.81 GB ÷ 619 GB/s **[D]** |
| compute per token | 54 GFLOP (2P) → **0.63 ms** at 86 TFLOP/s | **[D]** |
| compute headroom at batch 1 | **30.4×** | **[D]** |
| KV per token | 64 KiB bf16 / 32 KiB q8 | 16×2×4×256 elem **[M]** config arithmetic |
| GDN recurrent state | 159 MB/sequence, read+write every step | `docs/01-models.md:38-42` **[M]** |
| measured decode today | **862 ms/token = 1.15 tok/s**, 91.8 % of it in one Triton fp8 GEMV at 17.8 GB/s | `docs/research/20-decode-profile-27b.md:6-9` **[M]** |
| measured RCCL all-reduce, TP2 | 3.8 ms/step (130 launches, 0.44 % of a broken step) | same **[M]** |
| measured graph on/off delta | 0.6 % | same **[M]** |
| measured cross-GPU copy | 14.23 GiB/s = host bounce, no P2P | `docs/00-facts.md:13-14` **[M]** |

Unit caveat carried from the requirements draft (**C8**): the same facts file reports a *copy* wall
of 592.9 GiB/s (= 636.6 GB/s) and a *read* wall of 619 GB/s, and the two are used interchangeably.
This document uses **619 GB/s**, the lower and the one the kernel gate is written against. If the
true streaming wall is 636.6, every time below shortens by 2.8 %; no conclusion moves.

**The elephant, before any knob.** 862 ms/token is **45× the bandwidth bound** and ~1 400× the FLOP
bound. The step is neither streaming nor computing, it is waiting: one un-tuned Triton kernel
(`_w8a8_block_fp8_matmul`, 256 launches, `BLOCK_M=64` at M=1) achieving 17.8 GB/s of weight traffic
where hipBLASLt bf16 does 472 GB/s on the same shape in the same process **[M]**. A data-only config
change is worth 5.5× (857 → 157 ms) and a gfx1100-native GEMV is worth ~30× (→ 26–31 ms) **[M/D]**.
Every knob below is denominated against the 19–23 ms step you get *after* that. Ranking them before
it is fixing the paint on a car with no engine: the three headline knobs are worth ~1.4–4× each,
the kernel is worth 30×, and adding them multiplicatively to a 45× defect is arithmetic theatre.

**The second ceiling nobody prices: 55.8 ms of CPU per step.** 6.5 % of the 862 ms step is outside
every kernel — launch gaps, sampling, detokenising (`docs/research/20-decode-profile-27b.md:78-79`
**[M]**). That cost is *per step*, not per token, so it is the real floor on per-step time until
dispatch shrinks: 256 launches at ~3 µs of gap is ~0.8 ms, but the measured residue is 56 ms, which
means the CPU path is currently ~56 ms/step of work. Projected onto a 157 ms kernel step it is
survivable (36 %); projected onto a 26 ms native-GEMV step it is a **2× wall** unless graph replay,
fewer launches and batching cut it. Batching amortises it over N tokens (7 sequences → 8 ms/token)
while graphs only remove the launch-gap component. Any claim of 30× should be read as "30× on the
kernel, with a 56 ms/step CPU floor to attack in parallel". The likely bulk of that floor is the
sampler over a **248 320-token** vocabulary — the same `flashinfer.top_k`-absent-on-ROCm problem that
`docs/research/17-dflash2.md:43` flags for the DFlash2 selector, which "roughly halves end-to-end
throughput on a large vocabulary" **[M]**. Fix it once; both paths need it.

---

## 1. The roofline that decides everything

Per step, one card moves `W + N·(ctx·kv) + N·2·state` bytes and does `N·54 GFLOP`. The step is
bandwidth-bound when

```
(W + N·ctx·kv + 2N·state)/619 GB/s   >   N·54 GFLOP/86 TFLOP/s
```

Two numbers fall out, and they are the whole argument:

**1.1 The weight-only crossover is N ≈ 30.** With no KV at all, compute catches bandwidth at batch
30 — because `W/BW = 19.1 ms` while `N` sequences cost only `0.63 N` ms of compute. Below N=30 the
step is a weight re-read; above it, weights are amortised and the FMA pipes are the limit. On an
H100 (54 GB bf16 weights, 3.35 TB/s, 990 TFLOP/s) the same crossover is at **N ≈ 290** — an order of
magnitude higher, which is why the NVIDIA serving literature treats batching as the primary
throughput knob and this box cannot **[D]**.

**1.2 At ≥ 12 K context this box is *never* compute-bound, at any batch.** The KV read is the
reason. A 32 K q8 sequence costs 1.07 GB per step against 389 MB of flop-equivalent traffic
(`54 GFLOP ÷ 86 TFLOP/s × 619 GB/s`). Set `32768·ctx = 389 MB` and the crossover context is
**11.9 K tokens**: above it, adding batch never converts the step to compute-bound; below it, it can
(8 K needs N > 98). Consequences **[D]**:

- **Batching is a latency/fairness knob here, not a throughput knob.** At 32 K q8, N=4 → 143 agg
  tok/s, N=16 → 291, N=64 → 393; it saturates at `619 GB/s ÷ (ctx·kv + 2·state)` = 445 tok/s. Going
  N=4 → 16 buys 2.0× aggregate and costs exactly 2.0× per-session latency (35.6 → 18.2 tok/s): past
  the capacity limit, batching is a pure zero-sum trade, not a free win.
- **The per-token byte budget at 32 K q8, N=4 is 4.34 GB/token**, of which weights 2.95, KV 1.07,
  recurrent state 0.32. At N=7 the KV term (1.07) has already overtaken the weight term (1.69).
  **Long-context decode on this box is a KV problem, not a weight problem** — the opposite of the
  batch-1 intuition the project's governing sentence (R44) encodes.
- **A useful corollary the literature calls out and the folklore misses**: because KV traffic is
  bandwidth-bound with *constant* arithmetic intensity while weight traffic is compute-bound above
  N≈30, at long context a larger batch makes decode *more* bandwidth-bound, not less
  (MagicDec, [arXiv:2408.11049](https://arxiv.org/abs/2408.11049)) **[L]**. That is what makes
  speculation pay at large batch (see §5).

---

## 2. The table

Baseline for the byte column: 32 K context, q8 KV, N=4, one card = **4.34 GB/token**
(weights 2.95 + KV 1.07 + state 0.32). Rows that change a component quote both endpoints.
"Latency" means per-step time unless stated; the achievable floor for that baseline step is 28.1 ms,
against 862 ms measured today.

Priority: **P0** decides the architecture or must precede the rest; **P1** worth building once P0 lands;
**P2** hygiene; **P3** deliberately not doing.

| # | Knob | Effect on bytes/token | Effect on latency | Cost | Applicable here (no P2P, 1 node) | Priority |
|---|---|---|---|---|---|---|
| 1 | **Continuous batching** (raise N) | weights `/N`: 2.95 → 1.69 GB at N=7; 0.74 at N=16 | step 28 → 35 ms at N=7; per-seq 35.6 → 28.7 tok/s | none, but capacity-capped | Yes — but it is a *capacity* knob (7 sessions at 32 K q8) and a throughput knob only below 12 K ctx. Saturation at 445 agg tok/s | **P1** |
| 2 | **Chunked prefill** (fine chunks, ~64–256 tok) | 0 on decode bytes; the chunk's weight read is shared with the running batch | bounds the TBT spike to 1.4× (C=64) instead of 46× (C=2048) | 0 % at C≥32, +90 % at C=16 | **Yes, and nearly free** — the chunk is compute-bound above **30 tokens** on this box (296 on an H100), so finer chunking than the literature assumes costs nothing | **P1** |
| 3 | **Mix prefill into the decode batch** (stall-free / piggyback) | ~0 marginal bytes (weights shared) | a 512-token chunk turns a 28 ms step into a 322 ms step: **11.5× TBT hit for the decodes** | saves the chunk its own 19.1 ms weight pass (5.5 % of that step) | **No** — you cannot compute-bound your way out on a 619 GB/s wall; serialise prefill or disaggregate instead | **P3 (reject)** |
| 4 | **Prefix caching / radix** | 0 bytes/token at decode | TTFT: 32 K prefix → 0 s instead of **20.6 s**; multi-turn becomes O(n) not O(n²) | KV pinned per node (shared, so no extra), plus metadata | Yes, *if* the GDN state is checkpointed (§4). KV dedup alone is a KV-only win on a model where KV is the smaller per-session cost | **P1** |
| 5 | **GDN state checkpoint in radix nodes** | 0 bytes/token | makes #4 resumable at all | **159 MB per snapshot** = 15 % of a 32 K q8 KV, **2.4× the KV of a 2 K prefix** | **Yes — mandatory for #4.** Sparse checkpoints (turn boundaries / every 4 K) + host-RAM residency: reload 159 MB from DDR5 = 3.4 ms vs 20.6 s recompute (6000×) | **P0 (blocks #4)** |
| 6 | **q8 KV** | **5.42 → 4.34 GB/token at 32 K, N=4 (−20 %)**; −1.07 GB/token at 32 K, −0.27 at 8 K | 21.3 vs 23.1 ms at N=1 32 K; aggregate ceiling 251 → 445 tok/s | dequant fused into attention; logits within 1e-3 **[M]** | **Yes — the single best byte lever at long context.** Also capacity: 4 → 7 sessions at 32 K | **P0** |
| 7 | **Paged KV, block 16–64, pooled allocator** | 0 bytes/token | none directly | ~1 MiB/block; internal waste ≤ ½ block per session (< 0.1 %) | Yes, but for **sharing and non-reservation**, not fragmentation: `ctx × 64 KiB` per session is 1–8 GiB, so reserving `max_len` is what kills you | **P1** |
| 8 | **Speculation — MTP (3 drafts)** | **÷ (A+1)**: 4.34 → 1.45 GB/token at A+1=3.0 | 2.70–2.99× tok/s; **break-even at A > 0.20–0.33** | 849 MB head, autoregressive: 3 × 2.35 ms draft | **Yes — best value per byte.** Needs GDN state rollback per verify (159 MB copy = 0.5 ms, ~2 %) | **P0** |
| 9 | **Speculation — DFlash2 (block 8)** | **÷ (A+1)**: 4.34 → 1.08 GB/token at A+1=4.0 | 3.68–4.01× tok/s; **break-even at A > 0.17–0.28** | 2.41 GB draft + 5.9 ms flat draft step = **1.9 sessions of VRAM at 32 K** | Yes, second. Bigger multiplier, 3× the memory, 5 unbuilt pieces (`torch.topk` fallback alone halves throughput) | **P1** |
| 10 | **HIP/CUDA graph capture** | 0 | **0.6 % measured** on this box (removes launch gaps only) | 41 s capture for 3 batch sizes **[M]**; ~2 min for 8 | Yes, mandatory for hygiene, but it is not a lever. The 1.5× graph win in the literature is a different engine | **P2** |
| 11 | **One graph per batch size vs one graph** | 0 | padding to the next captured size is **free in bytes** (weights read once) and costs compute you are not using | capture time scales with sizes | **Capture many sizes; do not fear padding.** On a bandwidth-bound step, M-padding waste is invisible | **P2** |
| 12 | **TP2 across the two cards** | **halves every term per card**: 4.34 → 2.17 GB/token/card | 23 sessions fit where 7/card did; box 402 → 530–587 agg tok/s at 39–43 ms/step, 25.5 tok/s/seq | 128 collectives/step over host-bounced PCIe: 3.8 ms floor **[M]**, ~8 ms at N=23 = 10–20 % | **Yes, if aggregate throughput and session count matter more than 10 % latency.** No P2P is a *latency* tax, not a bandwidth tax — and bandwidth is what TP2 halves | **P0 (decision, zero code)** |
| 13 | **2 replicas instead of TP2** | **doubles** weight bytes across the box (11.81 GB/step/card) | 28.7 tok/s/seq — 11 % better per session than TP2 | 14 GB of the box's 40 GB idle per replica, weights read twice | Yes — right answer for *latency-critical, few-session* use; loses 1.5× on aggregate and 14 vs 23 sessions | **P1** |
| 14 | **P/D disaggregation** (prefill card, decode card) | 0 | removes the TBT spike entirely; costs one 1.07 GB KV transfer per 32 K prompt = **75 ms** at 14.2 GB/s (= 0.4 % of a 20.6 s prefill) | both cards must hold the weights (11.81 GB each — fits) | **Technically clean here** (the transfer is trivially cheap at long context) but it gives up 2× decode bandwidth vs replicas. Only if TBT SLO cannot be met by chunking | **P2** |
| 15 | **Recurrent state in bf16** | −0.16 GB/token; capacity 159 → 83 MB/seq | ~3 % of a 32 K step | numerical risk on a 48-layer recurrence **[U]** | Yes; also halves the snapshot cost in #5 | **P1** |
| 16 | **Stream overlap of compute and copy** | 0 | already implicit: copies are 0.01–0.2 ms/step **[M]** | complexity, and HIP stream priorities are weak on gfx1100 | **No** — nothing to overlap. Decode is a single serial chain of 256 GEMVs over HBM, not compute-vs-copy | **P3** |
| 17 | **Pinned memory / async H2D prefetch** | 0 | only matters for prompt upload (bytes are µs) and for offload paths | host RAM pinned | Not for weights that fit. Matters for #5 snapshots (159 MB @ 46.4 GB/s = 3.4 ms) — pin the snapshot buffer | **P2** |
| 18 | **Async prefetch of next layer's weights** | 0 bytes moved (same bytes, earlier) | ~0 expected: at 619 GB/s the memory pipeline is saturated and a single dependent chain over 256 GEMVs already keeps it busy | occupancy/LDS headroom in the GEMV | **No for decode.** Layer-level prefetch pays only when the kernel is *latency*-limited (small grid, few CUs busy), which is exactly the pathology of the current GEMV — fix the kernel, not the schedule | **P3** |
| 19 | **KV offload / weight streaming (model does not fit)** | re-reads the streamed bytes every token | PCIe 14.2 GB/s → **832 ms/token (1.2 tok/s)**; DDR5 → 254 ms (3.9); NVMe → 2.7 s (0.37) | — | **Never.** The model fits with 9 GiB to spare. Streaming is a 45× regression disguised as a capacity fix | **never** |

---

## 3. Continuous batching: what it actually buys

Continuous batching (Orca, [OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu))
replaces request-level batching with iteration-level admission. Its whole economic case is that a
decode iteration has spare compute, so a new request joining mid-batch is nearly free. **On this box
that case is inverted**: a decode iteration at 32 K has 2.7 % compute utilisation and 100 % bandwidth
utilisation, so a new request is *not* free — it adds 1.23 GB/step of KV+state, which is 10 % of the
19.1 ms weight read **[D]**.

Therefore:

- **At 8 K and below, batching is a real throughput knob**: 49.9 → 300 agg tok/s going 1 → 8
  sequences, because the weight term still dominates (11.81 vs 4.7 GB) **[D]**.
- **At 32 K, batching is a fair-sharing knob with diminishing returns**: 47 → 143 → 291 agg tok/s at
  N = 1/4/16, saturating at 445. The marginal sequence is worth less than half of the previous one.
- **The binding constraint is not bandwidth, it is capacity**: 4 sessions at 32 K bf16 KV, 7 at q8
  (measured: `max_running_requests = 3` with bf16 KV, `docs/03-native-kernel-inventory.md:28` **[M]**).
  Scheduling policy should therefore maximise *sessions resident*, not tokens in flight: admit for
  memory, and cap the batch to whatever the kv+q8 budget allows. Chasing batch 32 is chasing a
  saturation point you cannot afford to reach.
- **Scheduling policy implication**: with a 445 tok/s ceiling at 32 K and 7 sessions, per-session
  throughput is set by N, not by anything the scheduler does. The scheduler's job is (a) never stall
  a decode for a prefill (§4), (b) prefer evicting a cold prefix over delaying a hot session, and
  (c) run prefills on a background stream of work so they never inflate TBT.

---

## 4. Chunked prefill and mixed prefill+decode: help or hurt

**Sarathi-Serve** ([arXiv:2403.02310](https://arxiv.org/abs/2403.02310), OSDI'24) introduced
chunked prefills + stall-free scheduling: split a prefill into near-equal chunks and admit new
requests without pausing running decodes, reporting 2.6× (Mistral-7B/A100), 3.7× (Yi-34B/2×A100)
and 5.6× (Falcon-180B, PP) serving capacity at fixed tail latency **[L]**. **DistServe**
([arXiv:2401.09670](https://arxiv.org/abs/2401.09670), OSDI'24) takes the other fork — put prefill
and decode on *different* GPUs so they cannot interfere, and co-optimise parallelism per phase;
7.4× more requests or 12.6× tighter SLO **[L]**.

**Does interleaving help or hurt a bandwidth-bound decode step?** It hurts, and the reason is
specific to this box. In a fused GEMM the prefill tokens and the decode tokens are the *same* M
dimension, so a mixed batch costs `max(shared bytes/BW, (N_dec + C)·0.63 ms)`. A 512-token chunk
next to 4 decodes is 516 tokens × 0.63 ms = **322 ms** for a step whose bandwidth floor is 28 ms:
the decodes take an 11.5× TBT spike to save the prefill exactly one 19.1 ms weight pass, i.e. 5.5 %
of that step **[D]**. The only regime where piggybacking is correct is one where the decode step is
already compute-bound and the prefill rides in the shadow — the H100 short-context regime. This box
is the opposite corner.

**So: chunk, don't mix.** And here chunking is unusually cheap. A chunk is compute-bound once
`C × 0.63 ms > 19.1 ms`, i.e. **C > 30 tokens**; the equivalent threshold on an H100 is ~296 tokens
**[D]**. Chunks of 64 tokens give a 1.4× TBT spike and **zero** prefill overhead (20.6 s for a 32 K
prompt, same as one 32 K chunk); chunks of 16 pay 90 %. The low arithmetic intensity that makes
batch-1 decode miserable is exactly what makes fine-grained, latency-friendly chunking free. Take it.

**DistServe-style disaggregation** is technically attractive here for a non-obvious reason: the KV
transfer that normally makes disaggregation expensive is trivial at long context — 1.07 GB of q8 KV
for a 32 K prompt over 14.2 GB/s host-bounced PCIe is **75 ms, 0.4 % of a 20.6 s prefill** **[D]**.
But it needs a second copy of the weights (fits, 11.81 GB) and it dedicates a card to a job that is
idle most of the time; against 2 replicas you give up ~2× decode bandwidth. Chunking achieves 95 %
of the latency benefit for 0 % of the hardware. Prefer chunking; keep disaggregation as the answer
if a hard TBT SLO appears.

---

## 5. Speculation: MTP vs DFlash2, and why batching changes the sign of the advice

**The mechanism.** A speculative verify of `k` tokens through a bandwidth-bound verifier reads the
same weights and the same KV pages as a 1-token step (attention is one pass over the KV tiles for
all `k` queries), and pays `k×` compute. Since this box has **30× compute headroom at batch 1**, a
`k`-token verify is nearly free until `k·N·54 GFLOP/F` exceeds the byte time. Gain:

```
speedup = (A + 1) / (1 + t_draft/t_verify),   break-even at  A > t_draft/t_verify
```

with the numbers from this box (32 K, q8, one card) **[D]**:

| N (seq) | t_verify (1 tok) | MTP-3 gain | MTP break-even | DFlash2-8 gain | DFlash2 break-even |
|---|---|---|---|---|---|
| 1 | 21.3 ms | 2.70× | A > 0.33 | **3.68×** | A > 0.28 |
| 4 | 28.1 ms | 2.88× | A > 0.25 | **3.88×** | A > 0.21 |
| 7 | 34.8 ms | 2.99× | A > 0.20 | **4.01×** | A > 0.17 |
| 8 K, N=8 | 26.7 ms | 2.85× | A > 0.26 | 3.84× | A > 0.22 |

Three things follow.

**5.1 The acceptance bar is far lower than the folklore.** Break-even is `A+1 > 1 + t_draft/t_verify`,
i.e. **A > 0.2–0.3**, not A > 2. Because the verify is bandwidth-bound and the draft is a small
fraction of the weights, speculation pays even if two thirds of every block is rejected. The plan's
go/no-go gates (accept ≥ 2.8 for DFlash2, ≥ 2.0 for MTP) are the gates for *getting the full
multiplier*, not for paying for itself — keep them as the ship/no-ship bar, but the risk of "no
payoff at all" is much smaller than the 3.4–4.4-on-other-hardware uncertainty suggests.

**5.2 Batching does not kill speculation here — it changes who wins.** The folk theorem ("speculation
is for low batch, batching is for throughput") holds when the batched step is compute-bound: then a
`k`-token verify costs `k×` the step. MagicDec
([arXiv:2408.11049](https://arxiv.org/abs/2408.11049), [blog](https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference))
shows the opposite regime: past a context threshold, `T_verify/T_decode → 1` because KV loading
dominates, so speedups *grow* with batch — 1.91× for Llama-2-7B-32K at batch 32, 1.84× for
Llama-3.1-8B at 100 K, on 8×A100 **[L]**. On this box that threshold is **11.9 K tokens** (§1.2), so
essentially the whole operating range is MagicDec's regime. At N=7 the gain is *better* than at N=1
(4.01× vs 3.68×) because the draft cost amortises over a bigger verify while the verify stays KV-bound.

**5.3 Where it stops.** A block-8 verify stops being bandwidth-bound when `8·N·54 GFLOP/F >
bytes/BW`: **N ≈ 6 at peak FLOPs, N ≈ 2–3 at realistic M=8 MMA efficiency (25–40 % of peak)** **[E]**.
That is at or below the 7-session capacity limit at 32 K q8, so **block 8 is right-sized for this
box and block 16+ is not** without measuring. DFlash2's own advantage over MTP is draft cost
flatness, not a bigger block.

**5.4 Costs nobody budgets.**
- **VRAM**: DFlash2's 2.41 GB + lm_head read is **1.9 sessions** of 32 K q8 KV. At 7 sessions that is
  a 27 % capacity cut for a 1.3× better multiplier than MTP. MTP (849 MB) costs 0.7 of a session.
- **GDN rollback**: verifying `k` drafts requires restoring 159 MB of recurrent state on rejection,
  so each verify step needs a state snapshot (159 MB copy = 0.51 ms) or an exactly-invertible update.
  At N=4 that is ~2 % of the step — cheap, but it is the same primitive as prefix-cache
  checkpointing (§6), so build it once and use it twice.
- **Batched-spec correctness**: variable accept lengths desynchronise position IDs, masks and KV
  across a batch; several widely used implementations silently emit corrupted output while reporting
  good speed ([arXiv:2510.22876](https://arxiv.org/abs/2510.22876)) **[L]**. Budget an exact-match
  test against unbatched greedy, not a ROUGE/throughput test.

---

## 6. Prefix caching: hit-rate economics and what a radix node costs

PagedAttention ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180), SOSP'23) is the origin of both
block-paged KV and copy-on-write sharing; 2–4× throughput at equal latency, more pronounced at long
sequences **[L]**. SGLang's RadixAttention ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104))
generalises sharing into a radix tree over token prefixes: up to 6.4× on multi-call programs and
multi-turn chat **[L]**.

**Hit-rate economics on this box.** A multi-turn session re-sends its history every turn, so without
caching prefill cost is quadratic in turns; a 32 K prefix recompute is **20.6 s** at peak prefill
**[D]**. A cache hit on the whole prefix is therefore worth ~20 s of TTFT per turn — the largest
user-visible win available anywhere in this document, and it does **not** change bytes/token at
decode by one byte. The economics:

- **The saving scales with prefix length, the cost does not.** A radix node costs its KV pages
  (shared, so ~0 marginal) *plus*, on this model, a GDN state snapshot at **159 MB**. At a 32 K
  prefix that is 15 % overhead — fine. At a 2 K prefix it is **2.4× the KV it protects** — not fine.
  So: checkpoint sparsely (turn boundaries, or every 4 K tokens), and only for prefixes long enough
  to be worth it. A per-block (16-token) snapshot scheme would cost 64× the KV it caches.
- **Put the snapshots in host RAM.** 159 MB at 46.4 GB/s is **3.4 ms** to reload, versus 20.6 s to
  recompute (6000× **[D]**). 60 GiB of DDR5 holds 405 snapshots; free VRAM holds 60. The KV pages stay
  on the GPU (they are needed every step); only the state checkpoints go to host. This makes prefix
  caching on a hybrid model viable at a granularity that VRAM alone forbids.
- **Eviction is a bandwidth decision, not a capacity one.** A resident session's 1.07 GB of KV costs
  1.7 ms/step at 32 K; a re-prefill costs 20.6 s, i.e. **12 000× the step-time it saved**. Policy:
  evict by *prefix age × length*, never LRU on request arrival.

**Caveat that decides whether any of this works.** KV-only dedup on a hybrid model cannot be resumed
without the GDN state (`docs/research/00-synthesis.md:36-40`, R56 **[M]**). Without §5's snapshot
primitive, prefix caching here is a KV-only win on a model where KV is the smaller per-session cost —
i.e. a rounding error. The snapshot is the feature; the radix tree is the plumbing.

---

## 7. Paging, pooling, and graph capture

**Paging buys non-reservation and sharing, not defragmentation.** PagedAttention's headline
(60–80 % waste in prior systems → < 4 %, vLLM [blog](https://vllm.ai/blog/2023-06-20-vllm);
[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) comes from systems that reserved
`max_seq_len × per-token-KV` per request. On this box that reservation is 8 GiB per session at 128 K
— the difference between 1 and 7 sessions, so paging is mandatory. But *fragmentation* is a
non-issue at 64 KiB/token: with 16-token blocks (1 MiB) the expected internal waste is ½ block per
session, < 0.1 % of a 32 K cache **[D]**. Pick block 16–64 for allocator sanity and radix sharing;
do not spend engineering on compaction.

**One pool, or per-purpose pools?** Two tensions specific to this model: KV pages (16 layers only)
and GDN state (159 MB, per sequence, non-shareable, fixed size). They should be **separate pools**:
the state pool is a fixed array of 159 MB slots (its size is `max_sessions`, not
`max_tokens`), while KV is a paged pool. A single `num_gpu_blocks` accounting that lumps them
together will either over-admit (and OOM on state) or under-admit (and waste 40 % of the KV pool).

**Graph capture with variable batch size.** Measured on this box: graph on vs off is **0.6 %**
**[M]** — capture removes launch gaps only (the tool's own profile shows 55.8 ms/step of CPU-side
residue on a pathological 862 ms step; on a healthy 35 ms step, 256 launches × ~3 µs ≈ 0.8 ms, ~2 %).
Capture cost is real: **41 s for three batch sizes** **[M]**, so 8 sizes ≈ 2 min of startup, and
`vLLM V1` moved to piecewise CUDA graphs precisely because full-graph capture is inflexible
([blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)) **[L]**.

The decision that follows is unusual: **because the step is bandwidth-bound, batch-size padding is
free in bytes.** Capturing `{1,2,4,8,16,32}` and running N=3 as N=4 costs 33 % more *compute* on a
machine with 30× compute headroom, and 0 % more bytes. So capture many sizes, accept padding, and
do **not** build a compile-per-batch-size scheme. The one thing to get right is capturing the
*speculative verify* shape too — a block-8 verify is a different graph from a 1-token decode, and
`dflash` at block 8 + GDN verify is flagged untested on this backend
(`docs/research/17-dflash2.md:43` **[M]**).

---

## 8. The rest, honestly

- **HIP graph replay**: same as CUDA graphs. 0.6 % here, mandatory hygiene, not a lever.
- **Stream overlap of compute and copy**: nothing to overlap. Copies and cats sum to ≤ 0.2 ms/step
  **[M]**; the step is a 256-deep serial chain of GEMVs over HBM. Streams matter only for the
  prefill/decode split (§4) and for snapshot DMA (§6).
- **Pinned memory**: irrelevant for weights that fit; relevant for the 159 MB state snapshots and
  for prompt upload. Pin the snapshot buffer or the 3.4 ms reload becomes a pageable-copy crawl.
- **Async prefetch of the next layer's weights**: expected ~0. The memory system is already
  saturated by a dependent chain of 256 GEMVs at 619 GB/s; prefetch moves bytes earlier, not faster.
  It helps when a kernel is *latency*-limited by too few resident workgroups — which is exactly the
  current pathology (grid = N/128 at M=1, 17.8 GB/s). Fix the kernel's occupancy; do not add a
  prefetch pipeline on top of it. **[E]**
- **Weight streaming / offload (model does not fit)**: not applicable — 11.81 GB in 19.98 GiB. For
  the record, if it were: PCIe 14.2 GB/s → 832 ms/token, DDR5 → 254 ms, NVMe → 2.7 s **[D]**. Any
  proposal that streams weights on this box is proposing a 45× regression. FlexGen
  ([arXiv:2211.01167](https://arxiv.org/abs/2211.01167))-style offload is a capacity technique for
  boxes that cannot hold the model; this one can.
- **Two streams, two models (draft on card 2)**: tempting given no P2P, but the drafter is 2.4 GB
  read per block and the verify needs the target's hidden states — that is a 410 KB/step transfer
  plus a hard synchronisation, buying at most 5.9 ms of a 28 ms step. Keep the drafter on-card. **[E]**

---

## 9. The three highest-value knobs for us

**Precondition, not ranked: fix the GEMV.** 45× of headroom sits in one un-tuned kernel; a config
file is 5.5× and a native GEMV is ~30×. Nothing below is measurable above that noise floor, and the
first experiment for each of them should be run *after* the config change, on a step of ~157 ms.

**1 — q8 KV in a paged pool (with the separate state pool).** 5.42 → 4.34 GB/token at 32 K N=4
(−20 %), the aggregate ceiling 251 → 445 tok/s, and capacity 4 → 7 sessions. It is the only knob
here that is simultaneously a bandwidth win, a capacity win, and already decided in the
representation doctrine (R22, "it is a bandwidth and capacity decision"). It needs no new algorithm,
only a dequant fused into the attention kernel, and the measured gate already exists (logits within
1e-3 of bf16). **Why first:** at long context the KV read is the largest single term in the byte
budget, and it is the term that grows with every turn of a conversation.

**2 — Speculation, MTP first then DFlash2, at block 8.** 2.7–3.0× (MTP) and 3.7–4.0× (DFlash2) at
*every* batch size we can afford, break-even at A ≈ 0.2, and it is the only knob on the list that
converts this box's 30× compute headroom into bandwidth savings. It compounds with #1 and #3
instead of competing: the multiplier applies to the whole step, so it divides the KV term too.
**Why second, not first:** it is 1.25× where #1 is 3–4×, but it is the larger multiplier and it carries the acceptance risk
that has never been measured on this finetune (`HANDOFF.md:219-220`), a 0.7–1.9 session VRAM cost, the
GDN rollback machinery, and (for DFlash2) five unbuilt gfx1100 pieces including a `torch.topk`
fallback that halves throughput. Sequence it as the plan already says — MTP as the cheap probe of
verify+rollback, DFlash2 as the payoff — and gate on measured accept length, not on the card's
numbers from other hardware.

**3 — TP2 (one 23-session instance), not two replicas.** Per-card bytes halve on every term, so the
box goes from 402 to 530–587 aggregate tok/s and from 14 to 23 sessions at 32 K, for a 10–20 % PCIe
latency tax that is *measured* (3.8 ms/step floor, `--disable-custom-all-reduce` already known to be
mandatory). The intuition "no P2P ⇒ no tensor parallelism" is wrong in the way that matters: no P2P
makes collectives *latency*-expensive, but TP2's benefit is *bandwidth* halving, and this box is
bandwidth-bound. Replicas remain the right choice only if per-session latency beats aggregate
throughput in the SLO — and if so, say so explicitly rather than defaulting to it.
**Why third:** it is a one-flag decision with a 1.4× aggregate payoff and the largest capacity
effect of anything here, but it is worthless until the per-step time is dominated by real bandwidth
rather than by one kernel, and its benefit evaporates if you only ever serve 2–3 sessions.

**Ranked below those, in order:** GDN state snapshots to host RAM (makes #4 viable, 3.4 ms vs 20.6 s);
chunked prefill at 64–256 tokens (bounds TBT at zero cost, and this box tolerates far finer chunks
than the literature assumes); radix prefix caching (a 20 s TTFT win per turn, worthless without the
snapshots); graph capture (2 %, hygiene); paged block sizing (do not over-engineer);
mixed prefill+decode batching (**reject** on this box); weight streaming (**never**).

---

## 10. What would falsify this document

| claim | the measurement that would overturn it |
|---|---|
| decode is bandwidth-bound at every affordable batch above 12 K ctx | a measured step time at 32 K, N=7 that is ≥ 2× the `bytes/619 GB/s` prediction, with the kernel at ≥ 60 % of the wall |
| a 30-token chunk is the compute/bandwidth crossover | measured prefill throughput vs chunk size on a real prefill kernel; if the GEMV tops out at 40 % of the wall, the threshold moves to ~12 tokens |
| speculation pays at A ≈ 0.2 | measured accept length on *this* finetune with *this* backend; the risk is A ≈ 0, not A ≈ 1 |
| TP2's cost is 10–20 % | a measured all-reduce trace at N=23 (predict 8 ms/step); if it is 20 ms, TP2 loses to replicas |
| q8 KV costs ≤ 1e-3 logit error | the existing gate; if long-context retrieval degrades beyond the publisher's numbers, the KV term must be paid in bf16 and capacity halves |
| graph padding is free | a measured step time at N=3 (padded to 4) vs N=4; predict identical bytes, +33 % compute, net ≈ 0 |

---

## Sources

**Measured on this box (local, authoritative for every local number):**
`docs/00-facts.md` (§bandwidth 197-198, cross-GPU 13-14, KV 151-153, instruction 199-204),
`docs/research/20-decode-profile-27b.md` (862 ms/token; 91.8 % one kernel; 17.8 GB/s; RCCL 3.8 ms;
graph 0.6 %; config 5.5×), `docs/01-models.md:36-42` (KV 64 KiB/token; 159 MB state),
`docs/03-native-kernel-inventory.md:28,26` (max_running_requests=3; 619 GB/s gate),
`docs/research/17-dflash2.md` (drafter bytes, break-even algebra, unbuilt pieces),
`REQUIREMENTS-DRAFT.md` R16/R22/R29/R30/R44/R54-R59, C8 (unit contradiction).

**Literature**
1. Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI'24 — https://arxiv.org/abs/2403.02310
2. Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*, OSDI'24 — https://arxiv.org/abs/2401.09670
3. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP'23 — https://arxiv.org/abs/2309.06180
4. Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention), NeurIPS'24 — https://arxiv.org/abs/2312.07104
5. Sadhukhan et al., *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding* — https://arxiv.org/abs/2408.11049
6. Chen et al., *Speculative decoding for high-throughput long-context inference* (MagicDec companion, with T_verify/T_decode measurements) — https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference
7. Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI'22 — https://www.usenix.org/conference/osdi22/presentation/yu
8. DeepSeek-AI, *DeepSeek-V3 Technical Report* (multi-token prediction as a production draft head) — https://arxiv.org/abs/2412.19437
9. Yang, Kautz, Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule* (the GDN recurrence behind the 159 MB state) — https://arxiv.org/abs/2412.06464
10. Zhang et al., *Correctness Forensics for Batch Speculative Decoding: Diagnosing the Ragged Tensor Problem*, EMNLP'26 Findings — https://arxiv.org/abs/2510.22876
11. vLLM, *V1 alpha release* §6 (piecewise CUDA graphs) — https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
12. vLLM, *Paged Attention* design note — https://docs.vllm.ai/en/latest/design/paged_attention.html
13. vLLM, *Easy, Fast, and Cheap LLM Serving with PagedAttention* (fragmentation: < 4 % vs 60–80 %) — https://vllm.ai/blog/2023-06-20-vllm
14. Sheng et al., *Sequoia: Scalable and Robust Speculative Decoding* (tree/adaptive block sizing) — https://arxiv.org/abs/2402.12374
15. Xiao et al., *StreamingLLM* (the fixed-window draft trick MagicDec uses) — https://arxiv.org/abs/2309.17453
16. Sheng et al., *FlexGen: High-Throughput Offloading* (cited as *not applicable* here) — https://arxiv.org/abs/2211.01167

**Reproduce the arithmetic:** `research/05-serving-knobs-calc.py` (roofline, crossover, capacity,
topology, speculation, chunking, prefix-cache and streaming tables; every input tagged with its
source).
