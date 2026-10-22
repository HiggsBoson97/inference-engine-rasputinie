# 02 — Megakernels and persistent kernels: what they are, and whether this box wants one

**Status:** research note for the PLAN loop. Not a decision record. Nothing here is measured on gfx1100 except where the repo's own bench files are cited; the tag on every claim says which.
**Question asked:** do megakernels / persistent kernels make sense for *this* machine (2× RX 7900 XT, gfx1100, bandwidth-bound decode), and under what conditions?

> **Correction, 2026-10-22 — and it closes the question this note leaves open.**
> The arithmetic below assumes a **619 GB/s** wall, giving a 19.01 ms step. The
> wall is **776 GB/s**, so the step is **15.15 ms**. That makes the launch-gap
> share *larger* than estimated, which is the direction that favours a megakernel.
> But E2 has since **measured** the launch gap directly: 401 launches at 3.3 µs
> each = **1.31 ms of a 15.86 ms step, 8.3 %** — and 2–4 layer fusion recovers it
> at **99.7 % of the wall** (`bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md`).
> So the prize this note bounds at ~5 % is real, ≈8 %, and **layer-scoped fusion
> takes it without a megakernel's barrier risk**. The verdict below — do not build
> a whole-forward or run-scoped megakernel — stands, and is now measured rather
> than inferred. See `docs/03-native-kernel-inventory.md` row 3.
**Re-examined from scratch** — the previous iteration listed the megakernel as "deferred by decision" with no decision behind it (the withdrawn requirements draft R85 / C4). This note does not inherit that verdict, and it does not inherit its opposite either.

**Tags used**

| Tag | Meaning |
|---|---|
| `[measured-on-RDNA3]` | Measured on gfx1100 / this box, with a pointer to the bench record in the repo. |
| `[measured-elsewhere]` | A real measurement, on another GPU family. Named. |
| `[documented]` | Stated by a primary source (paper, vendor doc, engine README) as design or behaviour, not as a number. |
| `[claimed]` | An assertion by an interested party, or a number whose methodology is thin. Treat as a hypothesis. |
| `[derived]` | Arithmetic done here from measured inputs. |

**One-paragraph answer.** On this box a whole-forward-pass megakernel is not worth building now, and the reason is arithmetic, not taste. The step is 19.0 ms of pure HBM streaming (`11.77 GB ÷ 619 GB/s`, `[derived]` from `docs/00-facts.md`), of which at most 3–14 % is inter-kernel gap under the most pessimistic launch-cost assumptions that the literature supplies (`[derived]`, §5.1). The literature's headline wins — 50 %→78 % of bandwidth (Hazy, H100), 4.5 µs launches removed (Kog, MI300X) — were won on models where the step was ~700 µs and the *fixed* per-pass cost dominated. At 19 ms the fixed cost is not the denominator; the weight bytes are. Worse, the closest published analogue to our exact situation — a 27B hybrid DeltaNet/attention model, quantised, on one RDNA card, per-kernel HIP, no megakernel — already reaches **83 % of the card's bandwidth** (§4.4). There is no 20-point bubble to reclaim. What that engine *does* lose bandwidth to is per-tensor kernel quality on odd quant types, which is the same pathology our own record shows (17.8 GB/s against a 619 GB/s wall, `docs/research/20-decode-profile-27b.md:6-9`) and the same thing our P0 GEMV work exists to fix. A megakernel makes fast kernels slightly faster; it cannot make a slow kernel fast. **Verdict: no whole-forward megakernel; no run-scoped persistent kernel; possible layer-scoped persistent kernel, gated on two measurements that cost a day each (§7 E1, E3); and even then the expected prize is ~1 ms of 19 ms.**

---

## 1. Taxonomy — five things that get called "megakernel"

The literature conflates these constantly, and the conflation is load-bearing because the *costs* attach to different ones. MPK's own introduction does it in one sentence: *"a single mega-kernel, also known as a persistent kernel"* ([arXiv:2512.22219v2 §1](https://arxiv.org/html/2512.22219v2)). They are not the same object.

| # | Object | Definition | Boundary count per step | Fusion of *math*? | Preemptible by the engine? |
|---|---|---|---|---|---|
| 0 | **Kernel-per-op** (baseline) | one launch per operator, implicit barrier between launches on a stream | ~450–580 | no | yes, at every boundary |
| 1 | **Fused kernel** | 2–5 adjacent ops in one launch (RMSNorm+QKV+RoPE, SiLU+up-gate). What every engine already does. | ~450–580, some of them fatter | locally | yes |
| 2 | **Persistent kernel** | a kernel launched with grid ≤ resident capacity, which loops internally until its work queue drains. Op boundaries become *software* barriers. Scope is a free parameter: per-layer, per-step, or per-sequence. | user-chosen (64 / 1 / 1) | no, by itself | **at its own scope only** |
| 3 | **Megakernel** | persistent kernel spanning the whole forward pass, with the dependency graph *inside* the kernel (counters, task lists, or a compile-time static schedule). Sub-case of (2) with model-wide scope and an internal scheduler. | 1 (or 1/graph) | yes, globally | **no — only at step or run boundaries** |
| 4 | **CUDA/HIP graph replay** | the launch *sequence* is captured once and replayed by the driver with low per-node cost. Nothing is fused; every op is still a separate kernel with a real barrier. | unchanged | **no** | yes, at every boundary (with a graph re-instantiation or node update for shape changes) |
| 5 | **PDL** (programmatic dependent launch, NVIDIA) | the next kernel's prologue starts before the previous kernel's epilogue completes; a coarse device-side handshake. | unchanged | no | yes |

The distinctions that matter, stated as rules:

- **Graph replay ≠ fusion.** It removes *host dispatch* cost and part of the launch latency; it does not remove the kernel boundary, the grid-wide drain of the previous kernel's tail blocks, or the cold restart of the memory pipeline. Hazy measures the residual precisely: on H100 a dummy kernel on a stream costs **~2.1 µs**, and **inside a CUDA graph ~1.3 µs** — the graph "only decreases" it, and "time spent with the GPU doing no useful work" remains ([Hazy, *Look Ma, No Bubbles!*](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). This is the single most-cited number in the field and it is the reason megakernels exist at all: graphs left 1.3 µs on the table per boundary.
- **Persistent ≠ megakernel.** A persistent kernel that runs one layer is (2); Kog's *monokernel* runs the entire sequence, prefill through EOS, sampling included ([Kog, MI300X](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)) — that is (3) pushed to run scope. Cohere's engine is (3) at *step* scope: one persistent kernel per decode step, walking a host-built task list ([cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel)). The difference between those two is the entire preemption story (§3.4), and it is the axis on which our verdict turns.
- **Two ways to build (3)**, and they behave differently under a profiler:
  - *Static / compile-time partition*: the host precomputes exactly which CU does what, in what order, for a fixed batch shape. Kog ("programmer-managed, compile-time partition of work, thus removing the overhead of task management and scheduling"), Hazy ("we schedule each SM's instruction sequence ahead of time on the Python side, and we can reuse each schedule for hundreds of forward passes"), Luce (hand-written for one layer pattern). Cheapest at runtime, least tolerant of shape change.
  - *Dynamic task graph*: tasks in memory queues, dependencies as counters, decentralized scheduling in-kernel. MPK (ttGraphs, worker/scheduler SM split), Cohere (host-built list, explicit global counters). Tolerant-ish, pays per-task scheduling cost.
- **The interpreter trick.** Hazy does not compile 100 fused ops into one kernel body. They define 7 *instruction types*, each an instantiation of one CUDA template, and an on-GPU interpreter that walks a precomputed per-SM instruction list ([§How to Megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). This is the only published answer to the code-size problem, and it is the design any serious attempt here would have to copy: our quant mix alone is 11 tensor types (`docs/00-facts.md`).

---

## 2. Claimed advantages, and what each is worth here

| Claim | Best evidence | Status | Worth on this box |
|---|---|---|---|
| **Launch-overhead removal** | H100: 2.1 µs stream / 1.3 µs graph per boundary `[measured-elsewhere]`; MI300X: **4.5 µs per launch**, plus ~0.5 µs HBM re-start latency and >1 µs for the intermediate-tensor round trip it also removes `[measured-elsewhere]` ([Kog](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)) | solid, two vendors | 0.6–2.6 ms of a 19.0 ms step at 448–576 boundaries (`[derived]`, §5.1) = **3–14 %**. This is the *whole* prize. |
| **Cross-layer fusion** (drop intermediate materialisation; prefetch the next layer's weights during the current layer's epilogue) | Hazy: weights for instruction *n+1* loaded while instruction *n* still stores, gated on a paged shared-memory allocator; B200 breakdown attributes only **30 µs of 600 µs** to "awaiting weights from global memory" — pipelining worked `[measured-elsewhere]` | solid | Real, but a *pipelined kernel-per-layer* gets most of it: prefetch across a boundary already happens if the next GEMV's first loads are issued before the barrier drains. This advantage is not exclusive to (3). |
| **Keeping weights resident in L2** | — | **false for us** | Our per-token working set is 11.77 GB against 80 MB of Infinity Cache: **147×** (`[derived]`, §4.2). Kog *deliberately defeats* caching for weights with non-temporal (`NT`-scoped) loads "to deprioritize storing weights" `[documented]`. Nothing to keep resident. |
| **Deterministic scheduling / no tail stragglers** | Hazy: a 512-block down-projection on 148 SMs leaves 80 SMs empty at the tail; megakernel backfills `[documented]`; Cohere: "ready MoE tiles occupy SMs idled by the tail of attention" `[documented]` | solid on NVIDIA | Partially real here and **cheap to get without a megakernel**: 84 CUs against, e.g., 5120 output rows is 61 waves — the tail is 1/61 = 1.6 % of one GEMV. Tail loss is a *small-grid* problem. Our grids are large. |
| **Fine-grained compute/communication overlap** | MPK: MatMul task → AllReduce task per-tile events, up to **10× vs PyTorch and 1.1–1.4× vs vLLM/SGLang on 8×H100** `[measured-elsewhere]` | real, but needs NVSHMEM and P2P | **Inapplicable.** No P2P DMA on this pair; cross-GPU traffic is host-bounced at 14.2 GiB/s (`docs/00-facts.md`). |
| **Energy** | Luce: 1.87 tok/J at 220 W vs llama.cpp 0.76 `[measured-elsewhere]` | real | Out of scope: the target is one desktop, not tok/J (MAP.md (removed)). |
| **Headline speedups** | Hazy 2.5× vLLM / 1.5× SGLang at bs=1, 1B model, H100 `[measured-elsewhere]`; Kog 3,000 tok/s on 8×MI300X, 2B `[claimed — vendor]`; MPK 1.0–1.7× over vLLM/SGLang, "largest gains on smaller models and newer GPUs" `[measured-elsewhere]`; Cohere 1.58× vLLM at bs=1, 62 % of speed-of-light, H100 `[measured-elsewhere]`; Luce 1.55× llama.cpp, **0.8B** model, RTX 3090 `[measured-elsewhere]` | all real, all conditioned | Every one of them is a *small-model, batch-1* result. Not one is a 27B model at 19 ms/step. Luce says it out loud: *"Megakernel fusion benefits shrink as model size grows and compute begins to dominate over launch overhead… We chose 0.8B because it's the first hybrid DeltaNet model available, not because it's representative"* `[documented]`. |

The pattern in that last row is the finding: **the published megakernel wins are all in the regime where the fixed per-pass cost is a large fraction of the pass.** Hazy's own arithmetic makes the precondition explicit — for Llama-1B on H100 the memory limit is ~1350 passes/s (740 µs), and with 7 launches × 16 layers at an optimistic 5 µs of stall each you land at ~770 passes/s, i.e. *the stalls are ~43 % of the ideal step* ([Hazy](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). Scale that model to 27B and the ideal step becomes 19 ms while the stall budget stays ~1–2 ms. The technique does not get worse; the *ratio* collapses by an order of magnitude.

---

## 3. Real disadvantages

### 3.1 Register pressure, and the way it fails
A megakernel must keep every stage's live state in one register budget, because there is no kernel boundary to spill through. Luce hit this on the one architecture they targeted: *"We attempted `S_TILE=16` for more instruction-level parallelism. Silent crash, no CUDA error, registers spilled to local memory, performance collapsed, and eventually the kernel just stopped. `S_TILE=8` was the sweet spot"* ([luce-megakernel README](https://raw.githubusercontent.com/Luce-Org/lucebox/main/optimizations/megakernel/README.md)) `[measured-elsewhere]`. This is a forced-march design point: on gfx11 the register file is the constraint that decides how many waves per CU you can keep resident, and residency is *mandatory* for a persistent kernel (§3.2), so a spill is not a slowdown, it is a correctness hazard. Kog's answer is the opposite trade — 8 waves per CU at 64 lanes "to allow a large per-wave register allocation", with weights staged through LDS precisely to "bypass the register file, which decreases register pressure", and noting LDS is only 64 KiB so "not all weights can be prefetched this way" `[documented]`.

### 3.2 Full residency is not optional, and it caps your occupancy
A persistent kernel with an internal barrier requires every participating workgroup to be co-resident; otherwise the barrier deadlocks. Luce: the "launch count is clamped to the resident-block ceiling on smaller GPUs to avoid grid-sync deadlock" `[documented]`. MPK reserves 4 SMs for schedulers and uses the rest as workers, with a fixed 32 KB shared-memory page size giving 5–7 pages per SM `[documented]`. The consequence is that the *worst* stage in the fused kernel sets the shared-memory and register budget for *every* stage — the classic "one kernel, one occupancy" problem. Our GEMV is fine at low occupancy (it is bandwidth-bound, ~6.6 % issue slots, `docs/00-facts.md`), but our attention and GDN stages want different shapes.

### 3.3 The grid barrier is now your problem, and on AMD it is expensive
Removing kernel launches means implementing the barrier yourself. The naive version is *worse than the launch it replaces* — measured on MI300X, 256 CUs publishing 4 bytes each:

| Barrier implementation | Latency |
|---|---|
| Kog sentinel-in-the-data-path (consumers poll the value itself for a NaN sentinel; `sc1` device-scope loads/stores) | **0.80–0.93 µs** `[measured-elsewhere]` |
| Arrival counter + epoch, `__threadfence()`, atomic contention | **7.59–7.88 µs** `[measured-elsewhere]` |

([Kog, Table 1](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/))

Read that table twice. The *good* barrier is 5× cheaper than a kernel launch; the *obvious* barrier is 1.7× **more** expensive. And Kog's good one depends on device-scope cache semantics (`global_load/global_store … sc1`) plus the observation that `__threadfence()` lowers to `buffer_wbl2` + `buffer_inv`, which "force broad cache movement and extra HBM traffic" — i.e. a naive barrier inside a bandwidth-bound kernel injects *extra HBM traffic*, which is the one thing we cannot afford. Whether the equivalent recipe is available and that cheap on gfx11 is **unmeasured by anyone** (§4.5).

Kog also reports the aggregate: *"grid synchronization is responsible for around 35 % of the total token generation time"* on their MI300X monokernel `[claimed — vendor, but consistent with their own 4.5 µs launch figure and their sync count]`. A megakernel does not make synchronisation free; it makes synchronisation *yours*.

### 3.4 Preemption: the honest version
The common claim "a persistent kernel cannot be preempted" is **wrong as stated and right in practice**, and the distinction matters for us.

- *Driver-level* preemption of a running kernel exists (compute preemption with instruction-level granularity has been a feature since Volta on NVIDIA, and AMD's scheduler can preempt a queue). But it is a **context-switch mechanism, not a scheduling primitive**: the engine cannot ask for it, cannot bound its latency, and cannot use it to swap a request out at a token boundary. `[documented]`
- What the engine *can* do is choose the scope. That is the whole design space:
  - **Run-scoped** (Kog): the kernel runs from prefill to EOS. Nothing else can use the GPU until the sequence ends. Incompatible with a second request, with continuous batching, and — on our box — with the display compositor. Also a single submission lasting the whole generation: NVIDIA/AMD lockup detection is a real operational concern at multi-second kernel durations, and must be checked before anyone writes one `[claimed — unverified on gfx1100, see §7 E5]`.
  - **Step-scoped** (Cohere): one kernel per decode step; the host parks between steps and is free to mutate batch state. Cohere ships continuous batching, paged KV, prefix caching **and preemption** on top of a persistent decode kernel — which is the existence proof that (3) and dynamic batching are not fundamentally incompatible ([README](https://github.com/cohere-ai/cohere-megakernel)) `[documented]`. The price is that preemption granularity becomes "one step", so a preempted request waits out the current step. On a 19 ms step that is acceptable; on Kog's 0.33 ms step it would be trivial; on a 200 ms step it would not be.

### 3.5 Dynamic batch shapes: how the state of the art actually handles it
Neither of the two shipping megakernels is dynamically shaped. They specialise and select:

- MPK: *"To handle the dynamic batch sizes intrinsic to LLM serving, MPK generates multiple ttGraphs specialized for representative batch sizes, using powers of two up to the maximum batch size. At runtime, the scheduler selects the appropriate ttGraph"* — evaluated at **max batch 1–16** `[documented]`.
- Cohere: **batch sizes 1–8 only**, and *"Prefill pauses decode; mixed prefill/decode is not yet supported"* `[documented]`. Their end-to-end gain (1.25–1.41×) is smaller than their decode-only gain (1.58×) for exactly this reason.

So the truthful answer to "can a persistent kernel coexist with dynamic batching at all?" is: **yes, at step scope, with one compiled schedule per shape bucket, and with prefill evicting decode.** That is a real engineering tax — N schedules × M quant types — and it is the tax that our two shape regimes (prefill chunk, decode batch 1..N) would pay twice over.

### 3.6 Everything else the practitioners report
- **Debugging.** `grid.sync()` inside the per-token GDN recurrence loop "= instant deadlock… No error message, just a hang"; register spills that "silently crash" with no CUDA error (Luce) `[measured-elsewhere]`. Compare our own incident culture: a load test on the display card produced a `ring gfx_0.0.0` timeout and a gnome-shell casualty (`HANDOFF.md:154-157`). A hung persistent kernel on that card is the same class of event, but harder to attribute.
- **No quantisation.** Luce: "No quantization support (GGUF/GPTQ/AWQ)… We benchmark at half-precision to isolate kernel-level efficiency from quantization tradeoffs" `[documented]`. Every published megakernel except Luce's NVFP4 Blackwell path runs bf16/fp16. **Not one runs a GGUF k/i-quant GEMV.** That is the single largest transfer gap for us (§8).
- **One model, one kernel.** "The kernel is hand-written for Qwen 3.5-0.8B's specific layer pattern… It does not generalize to other models without rewriting" (Luce); "the current megakernel task schedule is specific to North Mini Code" (Cohere) `[documented]`.

---

## 4. AMD-specific evidence

### 4.1 What exists, by architecture

| Work | Hardware | ISA family | Megakernel? | Relevance to us |
|---|---|---|---|---|
| Kog "monokernel" ([blog](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)) | 8× MI300X | CDNA3, gfx942 | yes, run-scoped | **closest AMD evidence on mechanism** (barriers, NT loads, LDS staging, DPP reductions). Wave64, MFMA, 304 CUs, 8 XCDs with **per-XCD L2**, 5.3 TB/s. Almost nothing about occupancy transfers. `[measured-elsewhere]` |
| HipKittens ([arXiv:2511.08083](https://arxiv.org/html/2511.08083v1), [blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)) | MI325X, MI355X | CDNA3/CDNA4 | no (tile DSL) | The AMD *kernel-quality* literature, not the megakernel literature. Validated only on CDNA; no RDNA claims. `[measured-elsewhere]` |
| Luce megakernel ([README](https://raw.githubusercontent.com/Luce-Org/lucebox/main/optimizations/megakernel/README.md)) | RTX 3090/2080 Ti, GB10, M5 Max | sm_75/86/121 | yes, run-scoped, **hybrid DeltaNet+attention** | **Architecturally the closest thing that exists to our model** (18 DeltaNet + 6 attention, cooperative grid sync between layers, state in F32 registers). But CUDA-only; no HIP build of the megakernel is published. `[measured-elsewhere]` |
| Luce *engine* on R9700 ([blog](https://www.lucebox.com/blog/qwen38-r9700)) | 1× Radeon AI PRO R9700, 32 GB | **RDNA4, gfx1201** | **no** — per-kernel HIP | **The closest thing that exists to our box**: same model family (Qwen3.8-27B), GGUF IQ4_XS, ROCm 7.2, one card, fused DeltaNet decode path, RDNA4-tuned tiles, paged attention, continuous batching. `[measured-elsewhere]` |
| **RDNA3 / gfx1100 megakernel** | — | gfx11 | — | **No published work. None.** Searches for gfx1100/RDNA3 persistent-kernel or megakernel LLM inference return nothing. `[absence of evidence]` |

### 4.2 Is the 80 MB Infinity Cache big enough to matter? No — and it is not one pool

Anatomy first. Navi 31 is a chiplet part: one Graphics Compute Die plus Memory Cache Dies. **Each MCD carries 16 MB of L3 (the Infinity Cache) and two 32-bit GDDR6 interfaces**, so a 64-bit slice of the memory system per MCD. The 7900 XTX has six MCDs (96 MB, 384-bit); **the 7900 XT has five (80 MB, 320-bit)** ([RDNA 3, MCDs](https://en.wikipedia.org/wiki/RDNA_3); [AMD product page](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html) lists "AMD Infinity Cache Technology 80 MB", 20 GB, 800 GB/s, "effective bandwidth up to 2900 GB/s"). The GCD's own L2 is small — "up to 6 MB" ([RDNA 3 infobox](https://en.wikipedia.org/wiki/RDNA_3)).

Consequences, in order of importance for us:

1. **The pool is not uniform.** It is five 16 MB slices, each bound to its own 64-bit of memory channels. A streaming read is address-interleaved across all five, so the effective capacity for a uniform stream is the full 80 MB — but a *targeted* residency plan has to be slice-aware, and on gfx11 there is no published address→MCD map (Kog had to reverse-engineer the equivalent for MI300X's IOD and called it out as a separate research item) `[documented]`.
2. **The arithmetic is not close.** Per token we stream 11.77 GB of weights. 80 MB / 11.77 GB = **0.68 %** of the working set `[derived]`. There is no reuse to exploit: every weight byte is touched exactly once per token and not again until the next token, by which time 11.77 GB of other bytes have passed through. AMD's own "2900 GB/s effective" marketing number is an L2-hit-rate claim for *texture* workloads with locality; a transformer's weight stream has none.
3. **The practitioners actively avoid the cache.** Kog: all weight loads use "non-temporal (streaming) requests by using NT scope bit on the load instructions. This hints the cache subsystems to deprioritize storing weights, and alleviates cache pressure for other useful data." `[documented]` That is the correct policy here too, and it is the opposite of the "keep weights resident in L2" claim in §2.
4. **What the cache *is* good for** — and this is a real, testable item: the GDN recurrent state. 159 MB per sequence (151 MB FP32 + 8 MB conv, `docs/01-models.md`), read *and written* every token across 48 layers = 317 MB of traffic per token per sequence, i.e. **2.7 % of weight traffic at batch 1 and 21.5 % at batch 8** `[derived]`. That is the one place where cache behaviour and kernel structure could move the number, and a persistent kernel neither creates nor removes that traffic. It can only overlap it (prefetch the next layer's state during the current layer's math), which a pipelined kernel-per-layer also does.

### 4.3 Does a persistent kernel help a bandwidth-bound GEMV, or is it irrelevant because we wait on HBM?

Both, in a specific order:

- **It does not raise the wall.** Nothing in this literature claims more than ~80 % of peak DRAM bandwidth, and none claims to exceed it. Hazy: 78 % of HBM bandwidth on H100 vs ~50 % for vLLM/SGLang. Kog: "4.3 TB/s empirical" on MI300X against 5.3 TB/s peak ≈ 81 %. Cohere: "62 % of theoretical speed-of-light" at bs=1 on H100. `[measured-elsewhere]` The mechanism is always the same: fill bubbles that were *not* caused by DRAM.
- **Therefore the size of the prize equals the size of your existing bubbles**, not the distance to peak. For a kernel that is already at 76–83 % of the wall, a megakernel has 0–5 points to give. For a kernel at 17.8 GB/s (2.9 % of the wall, `docs/research/20-decode-profile-27b.md:6-9`), it has nothing to give, because the problem is not a bubble: it is a kernel reading 35× slower than the memory system allows.
- **Our own instruction budget says the GEMV is nowhere near issue-limited**: ~6.6 % of issue slots at 619 GB/s (`docs/00-facts.md`) `[measured-on-RDNA3]`. So the GEMV *can* saturate HBM if its memory access pattern is right; the question is entirely whether the pattern is right, which is a kernel question.

### 4.4 The decisive AMD datapoint: an RDNA engine with no megakernel already hits 83 %

Lucebox, one Radeon AI PRO R9700 (RDNA4, 32 GB, 640 GB/s), Qwen3.8-27B, ROCm 7.2, per-kernel HIP, no megakernel `[measured-elsewhere]`:

| Config | Decode | Implied weight bandwidth | % of the card's 640 GB/s spec `[derived]` |
|---|---|---|---|
| UD-IQ4_XS dynamic quant (13.3 GiB/token) | 32.3 tok/s | 461 GB/s | **71.6 %** |
| pure IQ4_XS requant (13.3 GiB/token) | 37.4 tok/s | 534 GB/s | **83.5 %** |

Their own diagnosis of the 32.3 row is the important part: *"This quant streams only 13.3 GiB per token but runs at about 71 % of the card's bandwidth, **because dynamic quants push some tensors below 4-bit onto slower kernels**."* `[documented]`

So on RDNA, with our model shape, in 2026: **12 points of bandwidth were lost to per-tensor kernel quality on a handful of tensors, and zero points were lost to launch bubbles.** A megakernel cannot recover the 12; better kernels for the odd types recover all of it. This is the same lesson our own record already contains — the 857 ms step was 91.8 % one Triton kernel at 17.8 GB/s, not a launch problem (`docs/research/20-decode-profile-27b.md:6-9`) `[measured-on-RDNA3]`.

Sanity check against our own ceiling: IQ3_S is 11.77 GB/token, so 619 GB/s ⇒ 19.0 ms ⇒ 52.6 tok/s `[derived]`. At the 71.6 % and 83.5 % efficiencies above that is **37.3 and 43.9 tok/s**. The gap between "good kernels" and "perfect kernels" on this model is ~6 tok/s; the gap between "perfect kernels" and "megakernel" is zero by construction.

### 4.5 What does *not* transfer from CDNA to gfx11

Be blunt about the list, because the tempting mistake is to read Kog's MI300X post as an AMD green light:

| CDNA3 (Kog's box) | gfx1100 (our box) | Consequence |
|---|---|---|
| wave64, 304 CUs in 8 XCDs, per-XCD L2 | wave32, 84 CUs, one GCD + 5 MCDs | barrier cost scales with participant count; 84 participants is 3.6× cheaper than 256 *if* the atomics behave — unverified |
| MFMA matrix cores | WMMA 16×16×16 only, no MFMA | irrelevant for GEMV (Kog uses VALU dots for the same reason we do), but rules out their GEMM stages |
| LDS "relatively small (~64 KiB)" per Kog, 5.3 TB/s HBM | gfx11 LDS capacity/bandwidth is not in any source surveyed here — measure it if LDS staging is ever proposed; 619 GB/s measured read | the LDS-staging trick is unproven here, and our GEMV already dequantises in registers (`ADR 0003`), so it needs less of it |
| `buffer_wbl2` / `buffer_inv` fence lowering, `sc1` device scope | gfx11 memory model exists (`sc0/sc1/sc2` on `global_load/store`) but no published barrier recipe at 84-CU scale | **the 0.85 µs barrier is not known to be achievable here.** This is the load-bearing unknown. |
| 8 GPUs with P2P/XGMI, NVSHMEM-equivalent in-kernel comm | 2 GPUs, **no P2P**, cross-GPU = host bounce at 14.2 GiB/s | in-kernel multi-GPU fusion is off the table (§5.2) |

---

## 5. Our reality

### 5.1 The single-card arithmetic

Step = 11.77 GB ÷ 619 GB/s = **19.01 ms** ⇒ 52.6 tok/s ceiling `[derived]` from `docs/00-facts.md` `[measured-on-RDNA3]`.

Boundary count per step: 64 layers × 7–9 kernels ≈ **448–576** `[claimed — count is an estimate; §7 E2 measures it]`.

| Assumption | Gap cost | Share of 19.01 ms |
|---|---|---|
| HIP graph replay, NVIDIA-like 1.3 µs × 512 | 0.67 ms | **3.5 %** |
| Stream launch, H100-like 2.1 µs × 512 | 1.08 ms | **5.7 %** |
| Stream launch, MI300X 4.5 µs × 512 | 2.30 ms | **12.1 %** |
| + a *naive* in-kernel barrier replacing them (7.7 µs × 512) | 3.94 ms | **20.7 %** |
| + a Kog-class barrier (0.85 µs × 512) | 0.44 ms | 2.3 % |

The entire design space of this technique on our box sits between **2 % and 21 % of one step**, and the 2 % end requires a sub-microsecond 84-CU barrier that nobody has demonstrated on gfx11. Best realistic case: recover ~2 ms of 19 ms, i.e. **+11 % tok/s**; worst case: lose 2 ms to barriers, i.e. **−10 %**.

Against that, two levers we already have in the plan:
- **Speculation** (MTP / DFlash2, 02-plan.md (removed)): multiplies *tokens per forward pass*. At k = 6.4 (Luce's measured acceptance on 27B code, `[measured-elsewhere]`) the per-token time is 19.01/6.4 = 2.97 ms ⇒ **336 tok/s**. Even a conservative k = 3 gives 178 tok/s. Measured lever, ~10× bigger than the megakernel's ceiling, and it attacks the *same* fixed cost by amortising it instead of shrinking it.
- **GEMV kernel efficiency** (P0, `docs/03-native-kernel-inventory.md:26`): doubling a 17.8 GB/s kernel is 35×. The megakernel multiplies whatever you already have by ≤1.15.

**Ranking, on evidence:** GEMV kernel quality ≫ speculation ≫ megakernel. Two of those three are already P0/P?; the third is the one this note is about.

### 5.2 Two cards, no P2P

- **DP is the project rule when the quant fits one card** (MAP.md (removed), GLOSSARY.md (removed); the primary target is 11.76 GB against 19.98 GiB, `docs/00-facts.md`). Under DP there is *no cross-card communication at all*: each card runs its own copy of whatever decode loop we build. A megakernel would then be "two independent megakernels", which changes nothing about the analysis above and gives 2× the sessions, not 2× the tok/s on one stream.
- **TP=2 is where a megakernel would have to span devices**, and our pair cannot do that cheaply: no P2P, cross-GPU = host bounce at 14.2 GiB/s (`docs/00-facts.md`, `[measured-on-RDNA3]`), each all-reduce latency-bound at 20.65 µs, 2 all-reduces × 64 layers = 2.64 ms/token (`docs/research/18-tp-roccl.md:290`). MPK's multi-GPU megakernel gets its win from NVSHMEM tile-level overlap; with no P2P there is no tile-level overlap to win, only the same PCIe round trips moved inside a kernel where they are harder to profile. **Verdict for TP=2: strictly worse than DP for the megakernel case, independent of everything else.**

### 5.3 The hybrid architecture, specifically

48 of 64 layers are GatedDeltaNet, 16 are full attention (`docs/01-models.md`). What that does to each side of the ledger:

*For* the megakernel, modestly:
- No KV paging in 48 of 64 layers, so the "ragged batch" surface is smaller than for a pure-attention model — 48 layers have identical, fixed-shape work regardless of context length. That makes a *static* schedule (Hazy/Kog style) more tractable than it would be for a 64-layer attention model.
- The GDN state update is a fixed-shape sequence with a serial dependence per layer; that is exactly the kind of thing a persistent kernel pipelines well (Luce runs "DeltaNet recurrence via warp-cooperative state updates in F32 registers" and reports it as the win, `[measured-elsewhere]`).

*Against*, more strongly:
- The 159 MB/sequence state is per-sequence, so the static schedule is per-*sequence*, not per-model. With N sessions you need N state regions and your "compile-time partition" gains a runtime index. Cohere solves the analogous problem with a host-built task list per step — i.e. it gives up the static schedule.
- State traffic scales with batch while weight traffic does not: 2.7 % → 21.5 % of traffic from batch 1 → 8 `[derived]`. At the batch sizes the product actually wants (MAP.md (removed), "several long-context sessions at once"), the step is *not* purely weight-bound, so the "we are just waiting on HBM for weights" premise weakens exactly where the megakernel would be asked to help.
- Prefill is chunked and mixed with decode (02-plan.md (removed)). Every shipped megakernel either excludes prefill (Cohere: "prefill pauses decode") or excludes mixed batches. That is a product-level regression for us, not a tuning detail.

### 5.4 What the repo already recorded, read correctly

Two internal records get quoted at each other, and both readings are wrong:

- `docs/research/20-decode-profile-27b.md:38-41` — graph on 1.13–1.17 tok/s vs 1.154 off, "0.6 % apart ⇒ graph replay here removes launch overhead only" `[measured-on-RDNA3]`. **This experiment cannot answer our question.** It was run on an 857 ms step in which 91.8 % of the time was one kernel that was GPU-saturated; when the GPU is busy, launch gaps hide behind queued work. Worse, 0.6 % of 857 ms is ~5 ms in absolute terms — which would be 26 % of a healthy 19 ms step. The experiment bounds the launch cost at "≲5–11 ms absolute", which is *compatible with the megakernel being worth 20 %* and equally compatible with 2 %. It is a null result about the pathological step, not about the healthy one.
- 02-plan.md (removed) — "P6 — Megakernel. Only after P5 shows how much time is launch gap. One persistent kernel per layer, if graph replay has already taken the easy part." `[documented]` That is the *right* ordering and the *right* scope (per-layer, not whole-model), and it is the position this note ends up endorsing — with the addition that the trigger condition should be a number, not a vibe (§7).

---

## 6. Verdict

**Do not build a megakernel for this box. Do not build a run-scoped persistent kernel. Keep the door open for a layer-scoped persistent kernel, and gate it on measurement.**

Reasons, in the order they bind:

1. **The prize is bounded by 2–14 % of the step and is probably ~5 %** (§5.1). On a 19 ms memory-bound step with 64 layers of large GEMVs, there is no small-model bubble problem to fix. Every published megakernel win is at 0.3–1 ms per pass, not 19 ms.
2. **The mechanism that would deliver the 2 % end of that range is unproven on gfx11** — a sub-microsecond 84-CU device-scope barrier. The naive version measured on AMD costs *more* than the kernel launch it replaces (7.7 vs 4.5 µs) and injects HBM traffic via `buffer_inv`-style fences (§3.3).
3. **The bandwidth we are actually missing is not gap-shaped, it is kernel-shaped.** Best available RDNA evidence: 12 points of bandwidth lost to slow kernels on odd quant types, 0 points to launch bubbles (§4.4). Our own record says the same thing louder (17.8 GB/s vs a 619 GB/s wall).
4. **A megakernel cannot represent this model.** No published megakernel supports GGUF k/i-quants; the two shipping ones are bf16/fp16/NVFP4, batch ≤ 8 or ≤ 16, one model each, prefill excluded or pausing decode. Building one means first building a quantised fused GEMV family *and* a scheduler *and* N shape-specialised schedules — while the same quantised GEMV family bought on its own is worth 10× more (it is the difference between 17.8 and 400+ GB/s).
5. **It conflicts with the product**: several concurrent long-context sessions, chunked prefill, prefix caching that must resume GDN state (02-plan.md (removed), R56). Step-scoped fusion is compatible with all of that (Cohere proves it); run-scoped is not, and run-scoped is where the headline numbers come from.
6. **Speculation dominates it as a way to amortise fixed per-pass cost** — 6.4× measured on our model family, versus ≤1.15× `[measured-elsewhere]`.

### Conditions under which the verdict changes

Each of these is a measurement, not an opinion. Any *one* of them flips the answer to "build the layer-scoped version"; two or more flip it to "build the whole-forward version".

| # | Condition | Threshold | Why that threshold |
|---|---|---|---|
| C1 | Measured GPU-side inter-kernel gap on a *healthy* 19 ms step (HIP graph replay) | **> 5 µs/gap** ⇒ > 2.5 ms, > 13 % of step | at the NVIDIA graph figure (1.3 µs) the prize is 3.5 % and cannot justify the work |
| C2 | Measured non-kernel residue in a healthy step (profiler: sum of gaps + gaps-not-in-kernels) | **> 3 ms of 19 ms** | same arithmetic as C1, measured end-to-end rather than per-gap |
| C3 | gfx11 device-scope barrier, 84 CUs, sentinel-handoff style | **< 1.5 µs** | must be cheaper than the 2.1–4.5 µs launch it replaces; Kog got 0.85 on CDNA3 |
| C4 | Product shape changes to batch-1-only, single-session, no continuous batching | any | removes the whole §3.4–3.5 objection; makes run-scoped fusion legitimate |
| C5 | Achieved IQ3_S GEMV bandwidth | **< 45 % of 619 GB/s** even after the P0 kernel work | if we cannot get the GEMV above 45 %, the engine has a kernel problem that fusion will not touch, and effort belongs there |
| C6 | The 27B stops fitting one card (a bigger target model enters scope) | any | TP=2 + no P2P makes fusion worse, not better — this *hardens* the "no" |

Note C6 is a hardening condition, not a softening one: it is listed because it is the most likely scope change, and it points the wrong way.

### If C1/C2/C3 all fire, what to build (and what not to)

Not "a megakernel". The minimum object that captures the win:

1. **Layer-scoped persistent kernel**, one launch per layer (64/step, not 1/step), which is already what 02-plan.md (removed) specifies. It keeps the batch shape mutable between layers, keeps prefill able to interleave, and needs only one internal barrier per layer instead of a general dependency graph.
2. **Static per-SM instruction lists built on the host**, Hazy-style, with a small instruction set (fused-norm-GEMV, GDN-recurrence, attention, residual+norm) rather than a general task-graph runtime. 7 instruction types covers 27B; MPK needed 44K lines of C++ + 42K of CUDA to build the general version.
3. **Sentinel handoff, not counters** (Kog), because counters cost 7.7 µs and inject fences.
4. **Non-temporal weight loads**, because the weights must not pollute the 80 MB that the GDN state and activations could use.
5. **One schedule per shape bucket** (batch 1, 2, 4, 8 × context buckets), selected by the host — the same compromise MPK makes.

Do **not** attempt run-scoped fusion (whole generation in one kernel) on this box: it makes the display-card timeout surface worse, destroys preemption granularity, and buys a further ~1 % over step-scoped.

---

## 7. What to measure to decide

Six experiments, ordered by cost-to-information. E1 and E3 are the decision; E2 is nearly free and should ride along with any profiling already scheduled; E4 is the tie-breaker; E5 is a safety check that only matters if the answer is "build"; E6 is the pilot that would de-risk the build.

| # | Experiment | Method | Pass (build) / fail (drop) | Cost |
|---|---|---|---|---|
| **E1** | **GPU-side gap under graph replay** — the number the whole question turns on | Capture a 512-node HIP graph of a trivial kernel (`__global__ void nop(){}`, 1 block, plus a variant with 84 blocks touching 1 cache line each). Time N replays; divide by 512. Compare against the same chain launched from the stream. Report µs/gap for: stream launch, graph replay, and graph replay with 84-block nodes. | **> 5 µs/gap → build**; < 2 µs → the technique cannot pay for itself, stop | ~half a day, no model needed |
| **E2** | **Kernel inventory and residue on a healthy step** | `rocprof` / torch profiler trace of one decode step at IQ3_S once the native GEMV exists (or on the best current step). Count launches; sum inter-kernel gaps; sum "not in any kernel" time. | residue **> 3 ms of 19 ms → build**; < 1 ms → drop | hours, rides on existing profiling |
| **E3** | **84-CU device-scope barrier latency on gfx11** | Implement Kog's sentinel handoff (`NaN` sentinel + `sc1` loads, or `atomic` + `s_waitcnt`): 84 workgroups publish 4 B, each reads a 1 KB buffer. Report max(end)−max(start) over 200 iterations. Also measure the naive counter+epoch+`__threadfence()` version for the comparison. | **< 1.5 µs → build**; > 3 µs → drop (it is more expensive than the launch) | 1–2 days, pure kernel, no model |
| **E4** | **Occupancy ceiling of a fused layer kernel** | `hipOccupancyMaxActiveBlocksPerMultiprocessor` for a kernel that must hold the largest GEMV tile + the GDN state slice + attention scratch. Check 84 × blocks/wave ≥ grid. Report the register/LDS pair that makes full residency possible, and the resulting waves/CU. | if full residency needs < 4 waves/CU → the fused kernel will be latency-starved; drop or reduce scope to 2-layer fusion | 1 day |
| **E5** | **Lockup-timeout surface for a long kernel** | Deliberately run a 5 s and a 20 s persistent kernel on the **headless** card (`HIP_VISIBLE_DEVICES=1`, never the display card, per R9) and watch `dmesg` for `ring gfx` timeouts. Check `amdgpu.lockup_timeout` semantics on this kernel/driver pair. | a 20 s kernel survives → run-scope is *possible*; it does not → step-scope only | 2 hours, and it must be run on the headless card |
| **E6** | **Pilot: 64-launch step vs 512-launch step** | Take the *existing* step, and without fusing any math, collapse adjacent kernel launches into one kernel per layer by inlining the elementwise ops (no GEMV changes). Same math, 8× fewer boundaries. Measure tok/s. | **< 3 % gain → the whole fusion line is dead here**, and E1–E4 become unnecessary | 3–5 days; the cheapest *conclusive* experiment |

E6 is the one to schedule first despite being the most expensive, because it answers the question in the currency that matters (tok/s) with no new numerics, no correctness risk to the GEMV path, and no megakernel. If collapsing 8 boundaries per layer into 1 buys less than 3 %, then no amount of further fusion — up to and including one kernel per model — can buy more than ~3 %, and the entire research line is closed with a number instead of an argument.

---

## 8. Where the evidence is thin

Stated plainly, because a plan built on this note should know which parts are load-bearing on borrowed ground.

1. **There is no RDNA3/gfx11 work at all.** Not a paper, not a blog, not a repo. Every AMD number in this note is CDNA3 (MI300X) or RDNA4 (R9700). The two are different in exactly the dimensions that matter for persistent kernels: wave width, CU count, LDS size, cache topology, and the existence of per-XCD L2.
2. **No megakernel anywhere runs quantised GGUF weights.** Luce's is bf16/fp16 with "no quantization support"; Luce's Blackwell path is NVFP4 (a different, hardware-native format); MPK, Cohere, Hazy, Kog are all bf16/fp16. Our entire representation doctrine (`ADR 0003`: dequantise in registers, never materialise bf16) has no precedent inside a fused kernel, and the register budget that dequantisation-in-registers implies is precisely the budget a fused kernel is shortest of.
3. **The two AMD barrier numbers come from one vendor with a product to sell.** Kog's 0.80–0.93 µs vs 7.59–7.88 µs is measured, reproducible in principle, and consistent with their reported 35 % grid-sync share — but it is a single source, on gfx942, with their own cache-scope recipe. Treat the 0.85 µs as an existence proof, not an expectation.
4. **MPK's gains are multi-GPU and NVSHMEM-shaped.** Their 8×H100 result (1.1–1.4× vs vLLM/SGLang) rests on in-kernel collectives. With no P2P we get none of it. Their single-GPU numbers (1.0–1.7×) are the only transferable ones, and they are on 0.6B–30B models at batch ≤ 16 on A100/H100/B200.
5. **"35 % of token time is grid sync"** is the single most alarming number in the literature and it is a vendor claim on a run-scoped kernel with 8 GPUs. Do not import it.
6. **The Luce R9700 83 % figure is one engine, one model, one quant, one card, reported by its author.** It is the best evidence we have for "RDNA does not have a bubble problem at 27B", and it is n=1. It is also the number that most deserves an independent reproduction, because if it is wrong by 15 points the verdict here softens.
7. **Our own boundary count (448–576/step) is an estimate.** It is the denominator of every percentage in §5.1. E2 replaces it with a real number and could move the case by a factor of two in either direction.

---

## 9. References (primary first)

- Spector, Juravsky, Sul, Dugan, Lim, Fu, Arora, Ré — *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B*, Hazy Research, 2025-05-27. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles — code: https://github.com/HazyResearch/Megakernels
- Cheng, Zhang, Zhou, Ji, Jiang, Zhao, Xiao, Ye, Huang, Lai, Jin, Hou, Wu, Dong, Yip, Wang, Yang, Miao, Chen, Jia — *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*, arXiv:2512.22219v2 (CMU/Tsinghua/NVIDIA/Michigan/PKU). https://arxiv.org/html/2512.22219v2 — code: https://github.com/mirage-project/mirage
- Cohere — *cohere-megakernel* (persistent decode megakernel, single H100, continuous batching + preemption). https://github.com/cohere-ai/cohere-megakernel
- Kog AI — *Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X GPUs*, 2026-05-28. https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
- Luce — *Luce Megakernel* (hybrid DeltaNet+attention, run-scoped, cooperative grid sync, CUDA). https://raw.githubusercontent.com/Luce-Org/lucebox/main/optimizations/megakernel/README.md · https://www.lucebox.com/blog/megakernel · results https://github.com/Luce-Org/lucebox/blob/main/optimizations/megakernel/RESULTS.md
- Luce — *Qwen3.8-27B on the AMD R9700: up to 227 tok/s* (RDNA4, per-kernel HIP, 71.6 % / 83.5 % of bandwidth). https://www.lucebox.com/blog/qwen38-r9700
- Hu, Wadsworth, Ré, Arora — *HipKittens: Fast and Furious AMD Kernels*, arXiv:2511.08083. https://arxiv.org/html/2511.08083v1 · https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- Zhu et al. — *NanoFlow: Towards Optimal Large Language Model Serving Throughput*, OSDI'25 (intra-device parallelism; the throughput-oriented counter-argument to fusion). https://www.usenix.org/system/files/osdi25-zhu-kan.pdf
- AMD — Radeon RX 7900 XT product specifications (80 MB Infinity Cache, 800 GB/s, 2900 GB/s effective, 84 CU, 5376 SP). https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html
- *RDNA 3* — MCD structure: 16 MB L3 per MCD, two 32-bit GDDR6 interfaces per MCD, 5 MCDs / 320-bit on 7900 XT; GCD L2 "up to 6 MB". https://en.wikipedia.org/wiki/RDNA_3
- Internal, `[measured-on-RDNA3]`: `docs/00-facts.md` (bandwidth, no P2P), `:197-204` (619 GB/s read wall, issue-slot margin); `docs/research/20-decode-profile-27b.md:6-9,38-41` (17.8 GB/s kernel; graph on/off); `docs/research/18-tp-roccl.md:290` (20.65 µs per all-reduce); `docs/01-models.md` (state and KV); 02-plan.md (removed) (P6 ordering); `docs/03-native-kernel-inventory.md:26,37` (P0 gate, megakernel row); `docs/adr/0003-representation-doctrine.md` (bf16 WMMA, dequant in registers).
