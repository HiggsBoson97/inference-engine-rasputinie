# Requirements

**Status:** current, 2026-10-22. Replaces the mined draft of the same date, whose
15 contradictions are now resolved into decisions below.

This document states only what binds. Measured numbers live in
[`docs/00-facts.md`](../00-facts.md); the kernel set lives in
[`docs/03-native-kernel-inventory.md`](../03-native-kernel-inventory.md); the
performance record lives in [`RECORD.md`](RECORD.md).

---

## 1. Hard constraints — owner decisions, not open to a plan

| # | Requirement | Where it is recorded |
|---|---|---|
| **H1** | **One served model, 100 %: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, IQ3_S, with its MTP head.** No second quant file of the target, no mmproj/vision path. **A drafter is not a second served model:** the MTP head ships inside the file, and the DFlash2 block-diffusion drafter (`z-lab/Qwen3.8-27B-DFlash2`, 2B) is admitted as a *draft* model under A3, subject to measurement. | ADR 0002, owner-ratified; drafter amendment 2026-10-23 |
| **H2** | **Speculation is engine core from the first milestone**, not a retrofit: the decode path is built to accept a draft and verify it. MTP ships inside the file and is the baseline drafter; **DFlash2 is a first-class candidate**, not a later option, because a block-diffusion drafter proposes a whole block in one pass where MTP drafts autoregressively. A3 decides which ships first. | O11, amended 2026-10-23 |
| **H3** | **llama.cpp is excluded** — not a baseline, not a reference, no derived code, not installed. | `decisions/0001` |
| **H4** | **FP8 is excluded as a format** — no checkpoint, no reference run, no kernel. Weights are quantised; compute is bf16. | ADR 0004 |
| **H5** | **Card-specific only.** Portability is a non-goal; every decision derives from this machine. | O3 |
| **H6** | **Bandwidth efficiency is the objective.** tok/s is a measurement of how well bandwidth is spent, never a target. There is no session count, concurrency target or per-session floor. | O2 |
| **H7** | **No target number. Break the standing record every time.** Records are kept as achieved GB/s over the measured wall, per shape, in `RECORD.md`; a slice that regresses must explain why. | O12 |
| **H8** | **The PCIe link is a design input, not a ceiling.** Use both cards; do not saturate the link. | O4 |
| **H9** | **The engine contains only what this machine needs.** Bloat is a defect. | O5 |
| **H10** | **Every claim carries a measured number.** When a document and a measurement disagree, the measurement wins and the document is corrected. | `AGENTS.md` |
| **H11** | **Correctness measurement is itself designed**, not assumed. A 32-token match is an arbiter, not the method. | O7 |

## 2. Representation — settled

| # | Requirement | Basis |
|---|---|---|
| **R1** | Weights stay quantised as published. **No offline conversion, no prepared artifact, no second format on disk.** | `DECISION-packaging-iq3s.md` |
| **R2** | Kernels see **one internal layout**; front-ends (GGUF, later a quantised safetensors container) normalise into it. | O1b |
| **R3** | Compute is **bf16** (WMMA or `v_dot2_f32_f16`). No FP8 arithmetic, no int8 GEMM path unless a measurement justifies it. | ADR 0003/0004 |
| **R4** | **q8 KV** (and k8v4 — K 8-bit, V 4-bit — where V tolerates it), paged, dequant fused into attention. It is a **capacity** precondition, not a speed optimisation: at low resident tokens the KV read is a minority of decode traffic and only overtakes the weight read above `B × C ≈ 173K` resident tokens at bf16 KV (~326K at q8). | research lane 04; `COST-LEDGER.md` |
| **R5** | **The GDN recurrent state stays FP32.** 159 MB per sequence is 7 % of a 32K session; a shared low-precision state loses 5.9–13 points. The int8 *checkpoint* store for cached prefixes is the lever, and it is the same primitive speculation needs for rollback. | research lane 04 |

## 3. Structure — settled

| # | Requirement | Basis |
|---|---|---|
| **S1** | **GEMV is the engine.** At 8K context and above this box is never compute-bound at any reachable concurrency, so the GEMM path is not built until a measurement shows the sub-8K, >30-concurrent corner being visited. | `DESIGN-batch-regimes.md` |
| **S2** | **Fuse 2–4 consecutive layers per launch.** Never 8: a 1.35 GB launch crosses the ~800 MiB footprint cliff and lands below per-tensor dispatch. | E2, measured |
| **S3** | A single scheduler-side signal governs regime: **achieved GB/s against the wall**. Near the wall means bandwidth-bound; falling away means something else binds. | `DESIGN-batch-regimes.md` |
| **S4** | Python may keep the API server and the scheduler; every device-touching hot path is native, and scheduler overhead has a budget measured against a null-scheduler step. | ADR 0002, ADR 0003 |

## 4. Open — each waits on a named measurement

| # | Open question | What settles it |
|---|---|---|
| **A1** | **Is IQ3_S instruction-bound?** **Answered 2026-10-23: no.** Issue occupancy is 48.6 % of the slot rate, not ≥ 85 %, and relieving memory pressure returns +23.7 % (521.3 → 645.1 GB/s) — impossible under issue saturation. The kernel sits at a **two-bound corner**: +124 GB/s of memory headroom (16.0 % of the wall) and +131 GB/s of instruction headroom (16.9 %). The `n ≈ 1.57` model was void for a 32× unit error. | Closed by `run-artifacts/hypotheses/H-1.md` and `H-2.md`. What replaced it: `H-3` — the 521 → 645 gap is latency/MLP, not bandwidth, and is tested by widening outstanding loads per wave. See `DESIGN-instruction-budget.md`. |
| **A2** | ~~In-situ rates for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S~~ **Answered 2026-10-23.** Load floors 637–660 GB/s, each within 0.4–3.7 % of its own aligned control — no layout penalises the read. Dot: IQ4_XS **502.1**, Q4_K **452.2** (kernel-limited). IQ3_XXS and IQ2_S **ungated** and keep only floors. The 752 assumption was 30–40 % too high where measured. | Closed by `H-4.md`; whole-file re-derived 56.0 → **47.1 tok/s** (`RECORD.md`) |
| **A3** | ~~**Which drafter, and what does it buy?**~~ **Answered 2026-10-23.** DFLASH is first-class in the vendored tree (Triton, ROCm fallback, nothing CUDA-only by import). At 521 GB/s the MTP head costs **7.445 GB = 65.6 % of a sweep** for 7 tokens; DFlash2 4.5 bpw costs **1.798 GB = 15.8 %, flat in block size** → 139 / 199 tok/s. MIT code over Apache-2.0 weights. | Closed by `H-5.md`; build MTP first, DFlash2 second, verify path designed for the block from day one |
| **A4** | Topology: TP=2, two replicas, or pipeline | **Reopened 2026-10-23 — the earlier answer priced only aggregate throughput.** Weights are sharded permanently; what crosses the link is the partial sums, 128 all-reduces of ~10 KiB per token = 1.3 MB, 52 µs of bytes. Cost is `128 × round-trip latency`. At the recorded 20.65 µs: **TP=2 = 1.48× single-stream** (1.93× at 2 µs) and, at a 7 GiB KV pool, ~6.5 concurrent 32K sessions; **DP=2 = 1.00× single-stream**, 2× only at concurrency ≥2. Remaining gate: **mapped-memory round-trip latency**, which decides it. |
| **A5** | The physical cause of the ~800 MiB per-launch footprint cliff | TLB/page-walk hypothesis, untested. It constrains fusion depth, so it matters. |
| **A6** | Whether runtime adaptation (O6) earns its complexity | Every adaptive mechanism ships with an ablation build, or its gain is unattributable. Not yet prototyped. |
| **A7** | **What defines the i-quant grids, and is our inventory the publisher's recipe?** 84 % of the file's bytes are i-quant types whose dequant depends on grid tables; E1 used a **placeholder** codebook because the ordering "cannot be recovered from the file plus our docs". | The model card states the files are **standard GGUF** ("run unmodified in llama.cpp, Ollama, and LM Studio"), so the grids are the standard ggml tables for those type ids. Source them from the publisher's **Apache-2.0** code (`github.com/IST-DASLab/GSQ`), never from llama.cpp (`H3`). Validate our 72 `(role, type)` pairs against `tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S.rco-allocation.txt`. |
| **A8** | **What does the attention path actually cost?** 16 full-attention layers over paged KV — no kernel, no rate, and it is the leading term above the resident-token boundary. | A paged decode-attention kernel over a context ladder (4K/16K/64K/256K), batch 1 and higher, q8 and k8v4, checksum-gated against a CPU reference. Report GB/s, % of wall and ms/step. `COST-LEDGER.md` queue item 2. |

## 5. Settled and closed — do not reopen without a new measurement

Listed so no future session re-argues them. Each has a decision document.

- **FP8 as a fast path.** Excluded: 857 ms/token, 2.7 % of the wall. ADR 0004.
- **Lossless expansion of the IQ3_S codebooks.** At best a 0.6 % tie, and that
  row exceeds the wall. `DECISION-packaging-iq3s.md`.
- **A megakernel or run-scoped persistent kernel.** ~5 % prize, and the barrier
  costs more than the launch it replaces. What replaced it — 2–4 layer fusion —
  is measured and settled.
- **Flash-Next, MoE, ngram, expert paging.** Out of scope by owner decision.
- **A concurrency target or "N sessions at M context".** Explicitly rejected as
  a goal-shape; bandwidth efficiency is the objective.
- **A card-agnostic build path.** Withdrawn by its author.
- **The 619 GB/s wall.** A kernel artefact; corrected to 776.
