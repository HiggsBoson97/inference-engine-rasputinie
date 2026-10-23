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
| **R4** | **q8 KV**, paged, dequant fused into attention. Worth ~8 % at batch 1 × 32K and ~39 % of traffic at 8 × 32K — it is a precondition for long-context capacity, not an optimisation. | research lane 04 |
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
| **A2** | In-situ rates for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S — only IQ3_S has been measured inside a real file | One sweep per type, same gated harness. Each has its own `n` and therefore its own ceiling. |
| **A3** | **Which drafter, and what does it buy?** Acceptance and per-step cost for the shipped MTP head and for the DFlash2 block-diffusion drafter (`z-lab/Qwen3.8-27B-DFlash2`, 2B BF16, ~4 GB — or ~1.1 GB quantised). | Teacher-forced replay against the shipped MTP head gives MTP's acceptance with no engine; the same replay plus the drafter's own pass cost gives DFlash2's. Per-step cost = target sweep + draft cost, and the drafter's weight read is part of it. Decides which ships first and whether the drafter must be quantised to keep the KV budget. |
| **A4** | Topology: TP=2, two replicas, or pipeline | **Answered for our quant.** Replication costs zero link bytes and takes 2× aggregate; TP=2 pays 128 all-reduces × 20.65 µs = 2.64 ms = 27 % of the step for 1.47×; pipeline costs one 10 KiB handoff (0.67 µs) and equals replication at ≥2 concurrent requests. **DP=2 for IQ3_S; PP=2 if a larger quant is ever used.** Remaining gate: measure the mapped-memory round-trip latency, which is what decides TP's viability at any quant. |
| **A5** | The physical cause of the ~800 MiB per-launch footprint cliff | TLB/page-walk hypothesis, untested. It constrains fusion depth, so it matters. |
| **A6** | Whether runtime adaptation (O6) earns its complexity | Every adaptive mechanism ships with an ablation build, or its gain is unattributable. Not yet prototyped. |

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
