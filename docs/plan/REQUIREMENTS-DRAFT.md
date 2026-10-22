# REQUIREMENTS DRAFT — rasputinie

**Status:** draft for owner correction, not a plan. Every item is mined from the existing record and cites `file:line`. Nothing here is invented; where the record is silent the omission is stated.
**Source tree:** `development/inference-engine-rasputinie` @ `707168a`, working tree clean. Read-only; this file was written to `/tmp` only.
**Documents mined:** `AGENTS.md`, `MAP.md`, `GLOSSARY.md`, `README.md`, `HANDOFF.md`, `docs/00-facts.md`, `docs/01-models.md`, `docs/02-plan.md`, `docs/03-native-kernel-inventory.md`, `docs/adr/0001..0003` + `README`, `docs/decisions/0001`, `docs/agents/{control-plane,domain,issue-tracker}.md`, `docs/agents/{invariants,production}/README.md`, `docs/research/00,10–20`, `engine/UPSTREAM.md`.

**Classes used**

| Class | Meaning |
|---|---|
| **HARD** | An owner decision, an accepted ADR, or a rule the repo states as non-negotiable. Not open to a plan to change. |
| **DERIVED** | Follows from a measurement. The measurement is named. Re-derivable, and breakable only by a new measurement. |
| **ASPIRATIONAL** | Stated as a goal, never measured and never decided by the owner. A plan may adopt or drop it. |
| **SUPERSEDED** | An earlier decision a later one overrides. The overrider is named. Kept because it still exists in the tree and still reads as binding. |

A caution on evidence quality, stated once: several "HARD" items live in `GLOSSARY.md`, `MAP.md` or `docs/02-plan.md` rather than in an ADR. They bind, but their authority is a table row, not a signed decision — see **C11**.

---

## 0. Owner directives — round 2, 2026-10-22

Recorded the turn they were given. These override the items below wherever they conflict, and each names what it supersedes.

| ID | Directive | Supersedes |
|---|---|---|
| **O1** | **Weight formats: GGUF *and* safetensors, both in scope.** The engine is not a GGUF-only engine. | ADR 0003 cl.1 names GGUF as *the* weight format. **Open sub-question:** which safetensors precision is first-class — see O1a. |
| **O1a** | **Resolved 2026-10-22: "safetensors" means a *quantised* safetensors container, not FP8 and not bf16.** The chain: 27B bf16 is 51.77 GiB and does not fit 40 GB; the FP8 checkpoint fits at 28.8 GiB but gfx1100 has no FP8 matrix instruction, so its arithmetic is emulated — measured at **857 ms/token, 1.17 tok/s, 16.9 GB/s = 2.7% of the wall**. Emulation is not a slow path, it is a different machine. Therefore: **quantised weights + bf16 compute is the only path**, in either container. See ADR 0004. | The FP8 branch of this question, and ADR 0003 clause 3. |
| **O1b** | **One internal weight layout, two front-end parsers.** GGUF and quantised safetensors both normalise at load into a single engine-native block layout chosen for the hardware, not for the file format. Kernels see only that layout. Supporting a second container is then a parser plus one repack pass (~19 s for 11.76 GB at the wall), not a second kernel family. | — |
| **O8** | **WITHDRAWN 2026-10-22 by the owner.** The orchestrator restored Flash-Next from `AGENTS.md`'s original intent; the owner has since stated the scope explicitly as **one and only one model**. See O10. Flash-Next, its NVMe expert-paging engine and its grouped MoE GEMM are out. | This row. |
| **O10** | **One model, 100 % support: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, IQ3_S, with its MTP head.** No second model, no second quant file, no mmproj/vision path. The engine exists to run this one artifact at the highest efficiency it can reach. | ADR 0002's scope is now *owner-ratified* rather than merely asserted; `AGENTS.md` and `docs/01-models.md` are corrected to match. |
| **O11** | **The MTP head is in scope from the first milestone, not deferred.** The `-mtp` build is ~0.35 GB larger with otherwise identical weights (`IQ3_S` 11.8 GB → 12.1 GB; locally measured 11.77 / 12.12 GB). Speculative decoding is therefore part of the engine core — graph shapes, batch handling and the verify pass are designed for it, not retrofitted. DFlash2 remains a later option. | The "MTP later" framing in earlier planning. |
| **O12** | **No target number. The goal is to break the previous record every time.** Operationalised as a **measured record ledger**: every shape carries its best-achieved GB/s as a fraction of the measured wall, and a slice that does not beat the standing record must explain why. Fraction-of-wall is the primary metric because it is bounded, comparable across shapes, and cannot be gamed by changing the workload; tok/s is derived from it and always reported with its context length and batch. | Every fixed tok/s target in the record, including the ≥45 and ≥31 proposals. |
| **O9** | **FP8 is excluded as a format** — not demoted, excluded: no checkpoint loaded, no reference run, no kernel. Recorded in ADR 0004, which amends ADR 0003 clause 3. | ADR 0003 cl.3 ("functional reference only"). |
| **O2** | **The objective is bandwidth efficiency.** tok/s is a measurement of how well bandwidth is being spent — never a target. No session count, no concurrency target, no per-session floor. | The "8 sessions at 32K" recommendation, the "105 tok/s ceiling" as a planning figure, and the Q3 framing. |
| **O3** | **Card-specific only.** Portability is a non-goal. The engine is built for one machine and every decision is derived from that machine. | The orchestrator's proposal to "build card-agnostic at batch 1" — withdrawn by its author. |
| **O4** | **The PCIe link is a design opportunity, not a limit.** Take full advantage of the two-card interconnect without saturating it; treat constraints as the thing to engineer around. Strata is cited as a model of this stance. | Any framing of "no peer DMA" as a ceiling rather than a design input. |
| **O5** | **The engine contains only what this machine needs.** Bloat is a defect, not a baseline. The owner expects the final engine to be well under 1 GB. | The inherited assumption that an inference engine is a large artifact. |
| **O6** | **Runtime adaptivity is an open design axis**: an engine that adapts to workload and remaining hardware headroom dynamically rather than executing a static schedule. To be evaluated on evidence, not adopted or rejected by taste. | — |
| **O7** | **Correctness is required, and its measurement must itself be designed.** A 32-token greedy match is an arbiter, not the whole method. | The unexamined assumption that "correct" is self-evident. |

**Corrections to the record recorded at the same time**

- **Bandwidth is three numbers, not one.** 800 GB/s is the specification; **619 GB/s is the measured read wall** (`00-facts.md:197`); **472 GB/s is what one bf16 GEMM shape achieved** (`research/20:131`). Only the second is a wall. Real headroom above 472 is +31%, not +69%.
- **This box has no compute-bound regime.** At 86.4 TFLOP/s, compute can absorb 0.398 GB/token; one sequence at 8K context already reads 0.537 GB/token of KV. Compute-vs-bandwidth crosses at batch 91 at 4K context and **never** at 8K or above. Bandwidth is the only currency at every concurrency this machine will run.
- **O2 is arithmetic, not philosophy**: aggregate tokens/s = b·B/(w + b·kv) is strictly increasing in b, so no batch size maximises it. Batch is capped only by VRAM, latency and TTFT.

**Repository reset, executed 2026-10-22:** root commit `b74d457` — pristine sglang `d0a7565`, 11,063 files under `engine/`, 302 files in `srt/models` (the previous HEAD had 0). Public. Host re-cloned clean. The documents that carried the wrong directives (`02-plan.md`, `MAP.md`, `GLOSSARY.md`, `HANDOFF.md`) were not carried; this file and the evidence tree were.

---

## 1. Machine and topology

**R1 — The engine is built for exactly one machine, `rasputin`, and is allowed to assume its exact hardware.** **HARD.**
`AGENTS.md:3` — "`rasputinie` is a local inference engine built for exactly one machine, `rasputin`." Also `GLOSSARY.md:8` ("the machine … Not the engine") and `GLOSSARY.md:10` ("allowed to assume its exact hardware").
Confidence: fact. Two independent documents, one of them the operating-rules file.

**R2 — The hardware is 2× AMD RX 7900 XT, gfx1100, 19.98 GiB each, PCIe 4.0 x16, 60 GiB DDR5-5600, 4.4 GiB/s NVMe, ROCm 10.1 / hipcc 7.16, Ubuntu.** **HARD.**
`AGENTS.md:11-13` — "2x AMD RX 7900 XT (gfx1100, 19.98 GiB each, 592 GiB/s HBM per card, PCIe 4.0 x16 with **no peer-to-peer DMA**), 60 GiB DDR5-5600 at 46.4 GiB/s copy, 4.4 GiB/s NVMe, ROCm 10.1 / hipcc 7.16, Ubuntu."
Confidence: fact; every value is reproduced by `bench/00_hardware_probe.py` (`docs/00-facts.md:3-4`).

**R3 — No design may assume peer-to-peer DMA. Cross-GPU traffic is host traffic over PCIe.** **HARD** (rule) over a **DERIVED** measurement.
`docs/00-facts.md:21-23` — "`torch.cuda.can_device_access_peer` returns true, but the measured copy speed equals the host-bounce speed. Treat cross-GPU traffic as host traffic over PCIe. Do not design around peer-to-peer DMA." Measurement: `docs/00-facts.md:13-14`, cross-GPU copy 14.23 GiB/s vs host-to-GPU 14.19 GiB/s. Restated as a coder-loop invariant at `docs/agents/invariants/README.md:9-11`.
Confidence: measured, 0.04 GiB/s apart from host copy. The rule is load-bearing: `research/12-deepinfer.md:59` kills DeepEP entirely on it.

**R4 — The decode ceiling is weight bytes per token divided by 592 GiB/s per card, minus ~30% for reality.** **HARD** (grading standard) over **DERIVED** (the wall).
`GLOSSARY.md:34` — "weight bytes per token divided by 592 GiB/s per card, before overheads and minus ~30% for reality; the target every speed number is graded against". `docs/00-facts.md:47-50` — "Decode speed equals weight bytes divided by 592 GiB/s."
Confidence: copy bandwidth measured 592.9 / 577.8 GiB/s, 1182.7 both busy (`docs/00-facts.md:10-12`). **Caveat:** the same file also reports a *read* wall of 619 GB/s (`docs/00-facts.md:197-198`), and the kernel gate is written against 619 (`docs/03-native-kernel-inventory.md:26` "≥60% of 619 GB/s"). The two walls are used interchangeably — see **C8**.

**R5 — The RAM tier is 46.4 GiB/s and the NVMe tier 4.4 GiB/s single-stream / 5.6 GiB/s over four streams; residency has three tiers and the decision variable is which tier a weight block is in.** **DERIVED.**
`GLOSSARY.md:14` — "one of the three placement levels, each a *bandwidth* rather than a capacity: VRAM (592 GiB/s per card, 19.98 GiB), RAM (46.4 GiB/s, 60 GiB), NVMe (4.4 GiB/s single-stream, 5.6 GiB/s over four streams)". Measurements at `docs/00-facts.md:16-19`.
Confidence: `bench/00_hardware_probe.py`, `dd iflag=direct` for the NVMe figures.

**R6 — gfx1100 is the only architecture the engine must support; `--offload-arch=gfx1100` is on every HIP build; wave32/DPP behaviour is assumed, not detected.** **HARD.**
`GLOSSARY.md:12` — "the only architecture the engine must support; `--offload-arch=gfx1100` is on every HIP build, and lanes/wave64/DPP behaviour is assumed, not detected."
Confidence: fact; enforced in the build patches listed at `HANDOFF.md:141` and `docs/research/16-sglang-gfx1100-build.md:229-244` (`SGL_ROCM_WARP_SIZE`).

**R7 — The engine must run on one card first. Two-card support is a mode, not a requirement.** **HARD.** Repeated verbatim in three files.
`AGENTS.md:53-54` — "The engine must run on one card first. Two-card support is a mode, not a requirement." Identical at `README.md:77-78` and as a note at `MAP.md:17`.
Confidence: fact — a rule stated three times, in the operating rules, the README and the map. Contradicted by the plan's TP=2 recommendation (**C1**, **C3**).

**R8 — The 27B at the intended quant fits one card, so DP is the rule and TP is the exception.** **HARD** (the rule) over **DERIVED** (the arithmetic).
`MAP.md:29` — "TP or DP for the 27B | DP when the quant fits one card, TP when it does not". `GLOSSARY.md:32` — "TP/DP — tensor parallel splits every layer across both cards … while data parallel runs one whole copy per card …; picked per model and per quant, never hard-coded — DP whenever the quant fits one card."
Confidence: arithmetic is measured. `docs/00-facts.md:58-62` — "Data parallel beats tensor parallel when the model fits one card. The 27B at Q4_K_XL needs 16.35 GiB, which fits one 19.98 GiB card." The primary target file is smaller still: 11.76 GB (`docs/00-facts.md:114`). This rule, applied to the primary target, selects DP.

**R9 — HIP device 0 is the display card. Single-GPU work is pinned to `HIP_VISIBLE_DEVICES=1` and verified by PCI bus id.** **DERIVED** (mapping) + **HARD** (the pin, from an incident).
`HANDOFF.md:134` — "**HIP 0 = amd-smi 0 = sysfs card1 = PCI 03:00.0 = `boot_vga=1` = THE DISPLAY CARD**; HIP 1 = amd-smi 1 = sysfs card0 = PCI 07:00.0 = headless". `HANDOFF.md:154-157` — "HIP device 0 is the display card. A load test on it caused a `ring gfx_0.0.0 timeout` in a gnome-shell context … Pin single-GPU work to `HIP_VISIBLE_DEVICES=1` and verify with `torch.cuda.get_device_properties(0).pci_bus_id` (expect `7`)."
Confidence: the mapping was measured twice and the first audit got it inverted (`docs/research/16-sglang-gfx1100-build.md:3-9`, correction banner). The incident is dated 2026-10-22 12:26.

**R10 — A TP=2 run necessarily puts compute on the display card.** **DERIVED.**
`docs/research/18-tp-roccl.md:310` — "any future 'use the headless card' instruction must say `HIP_VISIBLE_DEVICES=1`, and a TP=2 run necessarily puts compute on the display card."
Confidence: mapping above; `HANDOFF.md:135` — "TP=2 inevitably uses both."

**R11 — The iGPU (gfx1036, 512 MB) is not used.** **HARD.**
`README.md:21` — "iGPU | gfx1036, 512 MB, not used by the engine".
Confidence: fact.

**R12 — The card has 5376 lanes (84 CU × 64), not 10752; any estimate built on 10752 is 2× too optimistic.** **DERIVED.**
`docs/00-facts.md:195-196` — "**The card has 5376 lanes (84 CU x 64), not 84 x 128 = 10752.** Any estimate built on 10752 is 2x too optimistic."
Confidence: instruction-throughput probe, `bench/gfx1100-probes/`, card 0, sclk 2802 MHz sustained, median of 5.

**R13 — Grid/occupancy assumptions must be re-probed when the driver, the ROCm version or the hardware changes.** **HARD.**
`AGENTS.md:50-52` — "Never plan from specifications. Regenerate the hardware facts with `bench/00_hardware_probe.py` when the driver, the ROCm version or the hardware changes."
Confidence: rule; see R43 for the process consequence.

---

## 2. Model and representation

**R14 — Scope is Qwen3.8-27B only. Flash-Next is out.** **HARD.**
`docs/adr/0002-fork-sglang.md:54` — "Scope is **Qwen3.8-27B only**. Flash-Next work is out." Restated as a table decision at `docs/02-plan.md:103` ("D2 | Target models | **Qwen3.8-27B only**; Flash-Next out"), at `MAP.md:25` ("Qwen3.8-27B (`qwen3_5`) only. Flash-Next is out of scope") and as an in-file scope note at `docs/01-models.md:6-9`.
Confidence: fact — accepted ADR, and the only scope decision in the tree. **Directly contradicted by AGENTS.md and README:** see **C2**.

**R15 — Two weight sources, both GGUF: ISTA-DASLab GSQ-RCO IQ3_S (primary) and the nerkyor LynnStyle MTP tiers (secondary).** **HARD.**
`docs/02-plan.md:104` — "D3 | Weight sources | GSQ-RCO IQ3_S and the nerkyor LynnStyle tiers, both GGUF". Table at `docs/02-plan.md:11-14` gives 11.8 GB / 3.50 bpw and 16.1 / 18.3 GB.
Confidence: fact; the files exist on the host and are byte-checked (`HANDOFF.md:136`).

**R16 — Weights are GGUF k/i-quants; the primary target is `Qwen3.8-27B-GSQ-RCO-IQ3_S` at 3.50 bpw.** **HARD.**
`docs/adr/0003-representation-doctrine.md:31-32` — "**Weights: GGUF k/i-quants**, primary target `Qwen3.8-27B-GSQ-RCO-IQ3_S` (3.50 bpw, 10.96 GiB, task-lossless against BF16 on the publisher's suite)."
Confidence: fact — accepted ADR.

**R17 — The kernel family covers IQ3_S, IQ4_XS, IQ3_XXS, Q4_K and IQ2_S (94.5% of that file's bytes), plus a slow path for the single IQ1_M tensor.** **HARD** (scope) over **DERIVED** (the shares).
`docs/adr/0003-representation-doctrine.md:33-34` — "The kernel family covers IQ3_S, IQ4_XS, IQ3_XXS, Q4_K and IQ2_S, which are 94.5% of that file's bytes, plus a slow path for the single IQ1_M tensor."
Confidence: measured — the eleven-type histogram is `docs/00-facts.md:117-129`, and `docs/00-facts.md:136-141` identifies the single type-29 tensor (`blk.13.ffn_gate.weight`, 1.75 bpw, 19.5 MB) as IQ1_M-until-confirmed.

**R18 — Dispatch is per tensor, from the type field. `general.file_type` is a label, not a promise.** **DERIVED**, stated as mandatory by the plan.
`docs/00-facts.md:143-144` — "Per-tensor dispatch is therefore not a nicety: `general.file_type` describes the target average, and the RCO search spent the budget unevenly on purpose." Plan repeats it as a requirement: `docs/02-plan.md:32-33` — "Per-tensor dispatch is mandatory; `general.file_type` is a label, not a promise."
Confidence: two measured counter-examples — a file named `UD-Q4_K_XL` holding nine types at 5.05 bpw (`docs/00-facts.md:82`), and a file labelled IQ3_S holding eleven (`docs/00-facts.md:115`).

**R19 — Compute is bf16 WMMA. Integer dot (`v_dot4_i32_i8`) is allowed only where a measurement beats the bf16 path.** **HARD.**
`docs/adr/0003-representation-doctrine.md:35-36` — "**Compute: bf16 WMMA**, with `v_dot4_i32_i8` allowed only where a measurement beats the bf16 path."
Confidence: fact — accepted ADR. Supporting measurement: the integer dot is half rate (2.08 slots/instr) and the quantised GEMV needs ~6.6% of issue slots, so it is bandwidth-bound with ~15× margin (`docs/00-facts.md:199-204`).

**R20 — Dequantise in registers; never materialise a bf16 copy of the whole tensor.** **HARD.**
`docs/adr/0003-representation-doctrine.md:36-37` — "Dequantise in registers, never through a materialised bf16 copy of the whole tensor."
Confidence: fact — accepted ADR.

**R21 — FP8 is a functional reference only. It is never a performance target and no native kernel is written for it.** **HARD.**
`docs/adr/0003-representation-doctrine.md:38-40` — "**FP8 is a functional reference only.** It is the one 27B checkpoint sglang can load today, so it serves as a smoke test of TP, serving and the API. It is never a performance target and no native kernel is written for it."
Confidence: fact — accepted ADR, and the reason is physical: gfx1100 has no FP8 matrix instruction of any kind (`docs/adr/0003-representation-doctrine.md:7-10`).

**R22 — KV cache is q8, not fp8.** **HARD.**
`docs/adr/0003-representation-doctrine.md:41-42` — "**KV cache: q8**, not fp8. It is a bandwidth and capacity decision, and q8 needs no unavailable instruction."
Confidence: fact — accepted ADR. Note the engine today allocates bf16 (`docs/03-native-kernel-inventory.md:28`), so this is target-state, not current-state.

**R23 — The representation doctrine is hardware-forced, not a preference: every design question is downstream of "which instructions does this chip have".** **HARD.**
`docs/adr/0003-representation-doctrine.md:3` — "Status: **accepted**, 2026-10-22. Hardware-forced, not a preference." `HANDOFF.md:236-240` — "any proposal that needs an instruction outside that set is dead on arrival regardless of how good it looks on paper."
Confidence: fact; instruction set measured at `docs/00-facts.md:178-204`.

**R24 — The GGUF reader is ours, written against the GGUF v2/v3 specification.** **HARD.**
`docs/decisions/0001-exclude-llama-cpp.md:24-26` — "**The GGUF reader is ours.** Written against the GGUF v2/v3 format specification. FreeToken, Strata and TensorFold contain independent GGUF readers in Python, C++ and Zig; those are the permitted source material."
Confidence: fact — accepted decision, and the consequence is priced: "a native GGUF path is now a build item of several weeks, not a thin interop layer. That cost is accepted and must not be hidden in estimates" (`docs/decisions/0001-exclude-llama-cpp.md:36-37`).

**R25 — Every dequant and GEMV kernel is ours, written against the tensor-type block layouts in the specification.** **HARD.**
`docs/decisions/0001-exclude-llama-cpp.md:27-29` — "**The dequant and GEMV kernels are ours.** Written against the tensor-type block layouts in the specification. EXL3 layouts come from exllamav3; GSQ-RCO layouts come from ISTA-DASLab's material."
Confidence: fact.

**R26 — GSQ-RCO needs no new kernel: it is a mixed-precision recipe, not a container format.** **DERIVED.**
`docs/research/15-quant-kernels.md:118-126` — "**It is not a format.** … the released files carry unmodified GGUF tensor types and no transform metadata … Kernel implication: **nothing new to implement.**"
Confidence: read from the publisher's card and the type table; the file's header was parsed directly (`docs/00-facts.md:114-115`). Not load-tested — `docs/research/15-quant-kernels.md:199` flags the card's load-compatibility claim as read, not tested.

**R27 — MTP weights ship as separate GGUF files, one head per model.** **DERIVED.**
`GLOSSARY.md:36` — "the multi-token-prediction head, one per target model, already in the weights and shipped as a separate GGUF". `docs/01-models.md:115-116` — "MTP weights ship as separate GGUF files."
Confidence: the files exist on the host: `IQ3_S-mtp` 12 GB (`HANDOFF.md:136`).

**R28 — The quality gate is the publisher's GSQ-RCO numbers, not our own guess at what 3.5 bits should cost.** **HARD.**
`docs/adr/0003-representation-doctrine.md:52-53` — "The quality gate uses the GSQ-RCO publisher's numbers (AIME25 100.00, LCB v6 85.71, GPQA-D 89.39 at 3.50 bpw) as the external reference, not our own guess at what 3.5 bits should cost."
Confidence: fact — accepted ADR. Operational reading at `HANDOFF.md:260-263`: "If our path scores far below that, the dequant is wrong, not the model."

**R29 — The 27B's 48 linear layers carry 159 MB of recurrent state per sequence (151 MB FP32 + 8 MB conv), fixed, and it cannot be shared between different prefixes.** **DERIVED.**
`docs/01-models.md:38-42` — "Recurrent state per sequence | 151 MB in FP32 (48 x 48 x 128 x 128 x 4 B) plus 8 MB conv state"; "The recurrent state of the 48 linear layers is the bigger per-user cost: 159 MB per sequence, fixed, and it cannot be shared between different prefixes."
Confidence: config-derived (48 layers × 48 value heads × 128 × 128 × 4 B). Restated at `GLOSSARY.md:26`.

**R30 — KV is small by construction: 16 of 64 layers keep one, 64 KiB/token, 8 GiB at 128K.** **DERIVED.**
`docs/01-models.md:36-37` and `docs/02-plan.md:120` — "The 27B keeps KV in 16 of 64 layers: 64 KiB/token, so 8 GiB at 128K."
Confidence: config arithmetic; measured KV in the FP8 baseline was 45,727 tokens / 1.40 GB bf16 (`docs/00-facts.md:151-153`).

---

## 3. Compute and kernels

**R31 — `engine/` is our fork of sglang, vendored at upstream commit `d0a7565`. We own that tree and optimise in place.** **HARD.**
`docs/adr/0002-fork-sglang.md:7-9` — "`engine/` in this repository is **our fork of sglang**, vendored at commit `d0a7565` (`engine/UPSTREAM.md`). We own that tree. Optimisation work happens inside it, in place." Confirmed at `engine/UPSTREAM.md:2`.
Confidence: fact — accepted ADR, tree present. **Contradicted by the repo's own layout language:** see **C7**.

**R32 — Upstream sglang is a reading reference, not a rebase target; the fork will diverge.** **HARD.**
`docs/adr/0002-fork-sglang.md:14-15` — "Upstream sglang is a **reading reference, not a rebase target.** The fork will diverge; that is the point."
Confidence: fact.

**R33 — Other engines are sources of strategy, not parts to assemble. A technique worth having is ported and implemented for gfx1100 in our tree.** **HARD.**
`docs/adr/0002-fork-sglang.md:10-13` — "The other engines are **sources of strategy**, not parts to assemble. … We do not bolt their code together and call it an engine."
Confidence: fact — accepted ADR, superseding `docs/adr/0001-engine-composition.md:32` ("Build a thin engine and borrow by licence. No fork of any candidate") and the composition table at `docs/research/00-synthesis.md:42-54`.

**R34 — The layer disposition is fixed: inherit the HTTP/API server, tokenizer, request lifecycle, model code and graph capture; rewrite architecture gating and the kernel build; replace quantised matmul and attention.** **HARD.**
`docs/adr/0002-fork-sglang.md:31-40` — the disposition table, e.g. "Architecture gating … **Rewrite.** No RDNA3 helper exists; the HIP default is a CK backend that gfx1100 cannot run" and "Attention | **Replace** the CK/aiter path with a wave32 gfx1100 path; keep Triton as the correctness fallback".
Confidence: fact — accepted ADR.

**R35 — Three enabling changes unblock everything and are the first commits: admit gfx1100 to the AOT build, add `is_rdna3_supported()`, stop defaulting HIP attention to `aiter`.** **HARD.**
`docs/02-plan.md:49-55` — the three-commit list, naming `setup_rocm.py` ("It hard-exits today for anything except gfx942/950/1250 (verified at lines 77-81)"), `utils/common.py` ("so the ~71 `is_gfx95_supported()` call sites can be audited instead of being silently false") and `model_override_base.py` ("stop defaulting HIP attention to `aiter` on RDNA3 (verified at lines 326-327)"). Same three at `docs/adr/0002-fork-sglang.md:44-48`.
Confidence: fact — both files read at the named lines.

**R36 — A kernel family's row is closed only when its native path passes its test and its predecessor is deleted, not flagged.** **HARD.**
`docs/03-native-kernel-inventory.md:5` — "A row is closed only when its native path passes its test and its predecessor is **deleted**." `:13-14` — "**The old path is deleted, not flagged.** A flag is how an engine rots: two half-tested paths, and nobody knows which one the number came from."
Confidence: fact — the inventory calls itself "The contract for '100% gfx1100 native'".

**R37 — No kernel in the measured path may be written for another architecture. Triton is a correctness reference and temporary scaffold, never the shipped path.** **HARD.**
`docs/03-native-kernel-inventory.md:9-12` — "**No kernel in the measured path was written for another architecture.** Triton is allowed as a correctness reference and as a temporary scaffold, never as the shipped path. `torch` ops are allowed only where the whole operation is one launch and there is no measurable win from fusing it." Same rule as `docs/02-plan.md:105` (D4).
Confidence: fact — stated as the definition of "native". **Caveat on authority:** its only ADR-shaped home is `docs/adr/0001-engine-composition.md:42-44`, which is superseded; the rule survives as a plan decision and an inventory contract. See **C11**.

**R38 — Enabling changes are not ports. The arch allowlist, `SGL_ROCM_WARP_SIZE`, C++20, the Quark import guard and the RCCL all-reduce are first-class code that stays.** **HARD.**
`docs/03-native-kernel-inventory.md:15-18` — "**Enabling changes are not ports.** … They are not the CDNA path kept alive; they are what makes the tree build and run here at all."
Confidence: fact.

**R39 — The quantised GEMV for GGUF k/i-quants is the first kernel family, and its target is one-pass, column-major, dequant-to-bf16-in-registers, with WMMA above the GEMV crossover.** **HARD** (priority) / **ASPIRATIONAL** (the achieved bandwidth).
`docs/03-native-kernel-inventory.md:26` — row 1, priority **P0**: "HIP wave32, column-major, one-pass, dequant to bf16 in registers, WMMA above the GEMV crossover | bit-exact vs a torch dequant reference, then ≥60% of 619 GB/s".
Confidence: the *ordering* is a decision; the *60% target* has never been achieved by any kernel in this tree. `HANDOFF.md:256-259` is explicit that it is an arbiter, not a result.

**R40 — A quantised GEMV that does not reach roughly 60% of 619 GB/s has failed.** **HARD.**
`HANDOFF.md:256-258` — "**Gigabytes per second against 619 GB/s.** A quantised GEMV that does not reach roughly 60% of that wall has failed, whatever it feels like."
Confidence: fact as a standard; the existence proof for the wall is a bf16 hipBLASLt GEMM at 472 GB/s on the same shape (`docs/00-facts.md:25-27`, `docs/research/20-decode-profile-27b.md:131`).

**R41 — Every kernel gets a correctness test against a torch reference before it gets a speed test.** **HARD.**
`AGENTS.md:48-49` — "Every kernel gets a correctness test against a torch reference before it gets a speed test." Also `README.md:75-76`, `docs/agents/invariants/README.md:7-9`.
Confidence: fact — three documents, and it is a rig gate.

**R42 — The correctness oracle is a torch / Hugging Face reference forward, never another engine's output. A slow path is acceptable for a 0.2% tensor; a wrong one is not.** **HARD.**
`docs/decisions/0001-exclude-llama-cpp.md:30-32` — "**The correctness oracle is a torch / Hugging Face reference forward**, not another engine's output. Every kernel is checked against a CPU or torch implementation of its own arithmetic before any speed number is recorded."
Confidence: fact.

**R43 — sglang is the only engine baseline (FP8, TP=1 and TP=2), plus Strata if it builds on gfx1100.** **HARD.**
`docs/decisions/0001-exclude-llama-cpp.md:33-35` — "**Baselines are sglang** (FP8, TP=1 and TP=2) and, if it builds on gfx1100, Strata." `MAP.md:31` — "sglang is the only engine baseline".
Confidence: fact; TP=1 and TP=2 baselines were both taken (`docs/00-facts.md:146-170`, `docs/research/18-tp-roccl.md:280-281`).

**R44 — Decode at batch 1 is bandwidth-bound; compute is not the limit. Every decode optimisation is a weight-byte reduction or a weight-byte speed-up.** **DERIVED.**
`docs/00-facts.md:47-50` — "**Decode speed equals weight bytes divided by 592 GiB/s.** Compute does not limit decode on either target model." Instruction margin: 6.6% issue occupancy at 619 GB/s (`docs/00-facts.md:199-204`). Restated as the governing sentence at `HANDOFF.md:19-21`.
Confidence: two probes — the bandwidth probe and the instruction-throughput probe.

**R45 — Launch overhead is real but second-order: it is not the 857 ms anomaly, and graph capture removes only launch overhead.** **DERIVED.**
`docs/research/20-decode-profile-27b.md:38-41` — graph on 1.13-1.17 tok/s vs graph off 1.154 tok/s, "0.6 % apart ⇒ graph replay here removes launch overhead only". Non-kernel residue 55.8 ms/step = 6.5% (`:78-79`). The 1.51× graph win is from a different engine: `docs/research/10-strata.md:44`, 1.585 vs 2.393 ms.
Confidence: profiled, one sequence, 8 decode steps.

**R46 — The megakernel comes only after the profiler says how much time is launch gap.** **HARD** (ordering).
`docs/02-plan.md:95-96` — "**P6 — Megakernel.** Only after P5 shows how much time is launch gap. One persistent kernel per layer, if graph replay has already taken the easy part." `docs/03-native-kernel-inventory.md:37` — row 11, "only after 10".
Confidence: fact as an ordering constraint; see **C4** for the status conflict.

**R47 — MoE and grouped-expert kernels are not part of this project; the 27B is dense.** **HARD.**
`docs/03-native-kernel-inventory.md:41-42` — "**MoE kernels.** The 27B is dense. DeepInfer's psum layout and every grouped expert kernel are irrelevant to this target." Plan agrees: `docs/02-plan.md:129`.
Confidence: fact — model config has no MoE (`docs/01-models.md:22`, "MLP | 17408 intermediate, SiLU, no MoE"). **Note:** `GLOSSARY.md:20,38` still define `router look-ahead` and `psum` as project terms, which is dead vocabulary under this decision.

**R48 — Python stays for scheduling and model code; HIP C++ for kernels; Triton as correctness fallback, not in the hot path.** **HARD** (D4) / **ASPIRATIONAL** (whether "native" also covers the host layer — owner question open).
`docs/02-plan.md:105` — "D4 | Language split | Python for scheduling and model code, HIP C++ for kernels; Triton as correctness fallback, not in the hot path". The boundary itself is unresolved: `HANDOFF.md:209` — "Q4 | Native boundary: does Python stay as scheduler/API/tokenizer | owner | **yes**; 'native' applies to kernels and every device-touching path".
Confidence: D4 is a plan decision; Q4 has no owner answer, so the boundary is **not** settled (see R64).

**R49 — Attention must become a wave32 gfx1100 path at head_dim 256 with 6:1 GQA; Triton stays as the correctness fallback.** **HARD** (direction) / **ASPIRATIONAL** (the ≥1.15× test).
`docs/03-native-kernel-inventory.md:30` — row 4: "HIP flash-decode for wave32, paged KV, no per-length specialisation | ≤1e-3 vs Triton, ≥1.15x". Direction fixed by `docs/adr/0002-fork-sglang.md:38`.
Confidence: `--attention-backend triton` is mandatory today (`docs/research/16-sglang-gfx1100-build.md:31`); the native path does not exist.

**R50 — The NVIDIA path stays behind dispatch but carries no parity obligation.** **HARD.**
`docs/03-native-kernel-inventory.md:44-45` — "**CUDA parity.** The NVIDIA branches stay behind dispatch; they are not our problem to keep working." Tension with `AGENTS.md:55-56` — "Delete NVIDIA code when it blocks a build or confuses a reader, never for its own sake."
Confidence: fact as policy; the two statements sit at different strengths (see **C9**).

**R51 — `v_dot4_i32_i8` is reachable only through inline asm on gfx1100; `__builtin_amdgcn_sdot4` does not compile.** **DERIVED.**
`docs/00-facts.md:189-192` — "**`__builtin_amdgcn_sdot4` does not compile for gfx1100.** It requires the `dot1-insts` feature, which the subtarget does not advertise … The instruction is reachable **only through inline asm**."
Confidence: probe on card 0, `bench/gfx1100-probes/`. Also `v_pk_fma_f32` does not exist on gfx11 (`:193-194`).

**R52 — The 857 ms/token FP8 decode is one Triton kernel family at 17.7 GB/s, not a systems problem; the fix is the representation, not a profile hunt.** **DERIVED.**
`docs/research/20-decode-profile-27b.md:6-9` — "**Verdict: 91.8 % of the 857 ms step is the Triton `_w8a8_block_fp8_matmul` GEMV** — 256 launches per step, 786.3 ms, at 17.8 GB/s of weight traffic where the card measures 592.9 GiB/s."
Confidence: profiled. **Conflict:** the same document's recommendation contradicts ADR 0003 — see **C5**.

---

## 4. Serving behaviour

**R53 — The engine speaks the OpenAI chat API.** **HARD.**
`MAP.md:11` — "It speaks the OpenAI chat API and serves several long-context sessions at once." Inherited with the fork: `docs/adr/0002-fork-sglang.md:33` — "HTTP/API server, tokenizer, request lifecycle | **Inherit.**"
Confidence: fact; a bench harness already speaks it (`bench/10_openai_bench.py`, `HANDOFF.md:86-87`).

**R54 — It serves several long-context sessions at once, and multi-session memory is the constraint the owner cares about.** **HARD** (that it matters) / **ASPIRATIONAL** (the count).
`docs/02-plan.md:38-42` — "**KV capacity for many sessions.** This is the constraint the owner cares about, and the arithmetic is unforgiving: at 32K context one session costs ~2 GiB of paged KV *plus* 159 MB of recurrent state, so ~28 GiB of free VRAM holds about 13 sessions. At 8K it holds ~42."
Confidence: the arithmetic is measured-input; the target count is **not decided** — `HANDOFF.md:208` lists it as owner question Q3 with a recommendation of "8 sessions at 32K". See **C10**.

**R55 — Paged KV with a q8 option, radix prefix sharing, recurrent-state checkpointing so a prefix hit can be resumed, continuous batching and chunked prefill.** **HARD** (the P4 scope).
`docs/02-plan.md:84-87` — "**P4 — Multi-session.** Paged KV with a q8 option, radix prefix sharing, recurrent-state checkpointing so a prefix hit can be resumed, continuous batching and chunked prefill. Exit: N concurrent sessions at a stated context with measured aggregate throughput."
Confidence: plan phase; the exit criterion is deliberately "N … at a stated context", i.e. the number is unset.

**R56 — A prefix hit on this hybrid model must resume the GDN state, or prefix caching is a KV-only win on a model where KV is the smaller cost.** **DERIVED.**
`docs/research/00-synthesis.md:36-40` — "**The 151 MB per-sequence recurrent state must be a currency in the prefix cache.** … Without it a prefix hit on a hybrid model cannot be resumed, and the KV dedup work is a KV-only win." `GLOSSARY.md:28` defines the radix node as owning "a KV page set *and* a GDN state snapshot *and* SWA pages".
Confidence: three independent engines agree (`docs/research/00-synthesis.md:26-40`); FreeToken implements it (`docs/research/11-freetoken.md:31`). No ADR records it — `docs/adr/README.md:8-9` lists it as a decision still owed.

**R57 — q8 KV is the lever that raises session count, and bf16 KV is what capped `max_running_requests` at 3.** **DERIVED.**
`docs/03-native-kernel-inventory.md:28` — "bf16, 45,727 tokens, 1.4 GB | 2.08 GB free per card → `max_running_requests=3` | q8 KV + paged layout, dequant fused into attention | logits within 1e-3 of bf16; tokens/session doubles".
Confidence: measured ceiling state (`docs/00-facts.md:151-154`).

**R58 — Speculation order is MTP first, then DFlash2 with the 2.24 GiB drafter.** **HARD.**
`docs/02-plan.md:107` — "D6 | Speculation order | MTP first, DFlash2 second". Phase text at `:79-82`.
Confidence: fact — a plan decision, and the mechanism exists in-tree: `docs/research/17-dflash2.md:38-44` ("DFlash/DFlash2 is first-class in the vendored tree").

**R59 — Speculation is the biggest multiplier on top of the quant level, and it is judged by measured multiplier or a recorded reason it does not pay.** **ASPIRATIONAL.**
`docs/02-plan.md:26-30` — "**Speculative decoding is the biggest multiplier on top of it.** The nerkyor card reports ~180 tok/s per request with DFlash2, 91 with MTP and 46 without, on a data-centre card." Exit at `:82` — "a measured multiplier, or a recorded reason it does not pay here."
Confidence: **unmeasured on this hardware.** `HANDOFF.md:219-220` — "MTP / DFLASH acceptance on this box — never measured; the 3.4-4.4 range is the model publisher's, on different hardware." See **C6**.

**R60 — The vision tower is loaded but out of scope for optimisation.** **HARD.**
`docs/02-plan.md:108` — "D7 | Vision | Loaded, but out of scope for optimisation". Open question alongside it: `MAP.md:67` — "Whether the visual tower is worth loading for occasional screenshots."
Confidence: fact as a decision; note it answers the map's open ticket by fiat.

**R61 — Operational invocations that are not optional: `--attention-backend triton`, `--disable-custom-all-reduce`, loopback-only test port 30100.** **DERIVED.**
`docs/research/16-sglang-gfx1100-build.md:31` — "`--attention-backend triton` is mandatory on gfx1100". `HANDOFF.md:178-181` — "**Custom all-reduce is unreachable by design** … Always pass `--disable-custom-all-reduce`". `HANDOFF.md:139` — "Test port | 30100 (127.0.0.1 only)". `HANDOFF.md:127-128` — SSH port 8374.
Confidence: measured (the aiter path crashes; quickreduce admits only gfx94/95, and the pair measures 2 hops).

**R62 — The server must be warm before any measurement: 1-2 s per Triton kernel compile, 16.5 s for the fp8 matmul, 41 s for decode graph capture.** **DERIVED.**
`HANDOFF.md:182-184` — "**Runtime Triton compiles stall the first requests**: 1-2 s per kernel, and **16.5 s** for `_w8a8_block_fp8_matmul`. Warm-up is not optional." `:184` — "**Decode graph capture took 41 s** for three batch sizes. It is not a hang."
Confidence: measured on the FP8 baseline.

---

## 5. Process and evidence standards

**R63 — This project runs the fullstack loop from PLAN onward: SPEC, slices as GitHub issues, coder subagents, gated promotion.** **HARD.**
`AGENTS.md:24-26` — "**This project runs the fullstack loop from PLAN onward.** SPEC, slices as GitHub issues, coder subagents, gated promotion." `docs/agents/control-plane.md:8-12` — "GitHub Issues is the control plane … **The orchestrator operates it alone** … workers never write to it."
Confidence: fact — operating rule.

**R64 — Nothing is built until the plan gate returns APPROVED, and the gate is a fresh context that never wrote the plan.** **HARD.**
`HANDOFF.md:23-25` — "**Phase: PLAN.** The planning loop … is open and nothing is built until its gate returns APPROVED." `HANDOFF.md:252-253` — "a plan gate that is a fresh context and never the session that wrote the plan". `docs/agents/control-plane.md:157-158` — "Promotions refuse a project whose `GATE|PLAN|PASS` line is missing".
Confidence: fact.

**R65 — Measure before and after every change. A number without a baseline is an opinion.** **HARD.**
`AGENTS.md:46-47` — "Measure before and after every change. A number without a baseline is an opinion." Also `README.md:71-72`, `docs/agents/control-plane.md:349-351` — "A speed number without its baseline in the same artifact is an opinion, not evidence".
Confidence: fact — rig gate.

**R66 — Every claim in a report carries an evidence pointer (file:line, command output, artifact path); the orchestrator presence-checks before posting.** **HARD.**
`docs/agents/control-plane.md:335-337` — "Every claim in a worker report carries an evidence pointer … a report missing a required section goes back, never posted." `:338-340` — "**Findings travel in full — a count never stands in for the list.**"
Confidence: fact.

**R67 — Receipts ride in the same commit as the code they evidence, force-added past any ignore rule.** **HARD.**
`docs/agents/control-plane.md:345-348` — "**Receipts ride in the same commit as the code they evidence**, on the slice branch, force-added past any repo-level ignore rule."
Confidence: fact.

**R68 — The gate layer is local discipline only, because branch protection is unavailable on this repo.** **HARD.**
`docs/agents/control-plane.md:24-30` — "**Branch protection is not available on this repo.** … every branch-protection call returns HTTP 403 … The gate layer is therefore **local discipline only**: `main` moves only through an orchestrator promotion after a FINAL exercise PASS, because the loop keeps that rule, not because GitHub enforces it."
Confidence: fact — verified against the API; do not write a protection setting anywhere.

**R69 — Two slots: two slices, each a coder and its reviewer.** **HARD.**
`docs/agents/control-plane.md:20` — "Slots | **2** — two slices, each a coder and its reviewer, four workers in flight". `AGENTS.md:70` — "with 2 slice slots".
Confidence: fact.

**R70 — The sync topology is Mac clone → bare `origin` on rasputin → host working tree pulls. Nothing writes into the host tree from the Mac.** **HARD.**
`AGENTS.md:30-42` — "The Mac clone is the writing surface. It pushes to `origin`, the bare repo …; the working tree at `rasputin:~/development/inference-engine-rasputinie` **pulls** from it. … Nothing writes into the rasputin working tree from the Mac directly — no rsync, no scp, no editing over SSH. A dirty tree on the host means another session is mid-edit; do not pull over it."
Confidence: fact.

**R71 — No measurement from the host can be attributed to the committed tree until the host desync is resolved.** **HARD** (blocking).
`HANDOFF.md:113-116` — "On the host: anything other than a clean clone of `f759837` means the desync above is still open, and no measurement from the host can be attributed to the committed tree until it is resolved." State at `:99-104` — host HEAD `9b604b9`, `engine/` shows **0 tracked files** ("it was rsynced by hand"), six untracked paths.
Confidence: fact — the single blocking item on the board.

**R72 — Measurement wins over documentation; the document gets a correction banner.** **HARD.**
`HANDOFF.md:245-247` — "When a document and a measurement disagree, the measurement wins and the document gets a correction banner — three documents already carry one."
Confidence: fact; three banners exist (`docs/research/16-sglang-gfx1100-build.md:3-9`, `docs/research/15-quant-kernels.md:163-179`, `docs/research/00-synthesis.md:3-6`).

**R73 — 32 greedy tokens matching a torch reference exactly is the cheap arbiter for every kernel.** **HARD.**
`HANDOFF.md:265-266` — "A third, cheaper arbiter for every kernel: 32 greedy tokens matching a torch reference exactly on the same weights." Phase exit uses it: `docs/02-plan.md:76-77`.
Confidence: fact.

**R74 — Download only what a task needs; the model files are large.** **HARD.**
`MAP.md:18` — "The model files are large. Download only what a task needs." Host reality: no `hf` CLI, no `git-lfs`, no `uv`, no `pipx` (`HANDOFF.md:188-190`).
Confidence: fact.

**R75 — Style and code ownership: use the glossary's terms; flag a conflict with an ADR instead of silently overriding it.** **HARD.**
`docs/agents/domain.md:60-66` — "it is `recurrent state`, not 'SSM cache'; it is `bandwidth ceiling`, not 'roofline'; DP and TP are not exchangeable words." `:72-78` — "If your output contradicts an existing ADR or a pre-loop decision record, surface it explicitly instead of silently overriding it."
Confidence: fact.

**R76 — Delete NVIDIA code when it blocks a build or confuses a reader, never for its own sake; speed comes from kernels and data placement.** **HARD.**
`AGENTS.md:55-56` — identical text at `README.md:73-74`.
Confidence: fact.

---

## 6. Exclusions

**R77 — llama.cpp is excluded from the project completely: not a baseline, not a reference, not read, not installed, no derived code.** **HARD.**
`docs/decisions/0001-exclude-llama-cpp.md:14-20` — "llama.cpp is excluded from this project completely: 1. It is not a baseline. No number in this repository comes from it. 2. It is not a reference implementation. No kernel, loader, layout or heuristic is copied from it, and its source is not read while writing ours. 3. It is not installed on `rasputin`. The clone and the HIP build created during the first session were deleted." Also `AGENTS.md:20-23` ("The excluded engine stays excluded"), `MAP.md:31`.
Confidence: fact — "Owner decision, not a technical preference" (`docs/decisions/0001-exclude-llama-cpp.md:3`). Enforced: `:39-43` — a slice fails if "`llama.cpp`, `llama-bench`, `ggml`, or a vendored copy appears in the tree or in a build manifest". Two donor files are also off-limits for this reason: `docs/research/15-quant-kernels.md:12-13` (FreeToken's vendored `ggml/`, TensorFold's `ggml.cu`).

**R78 — CUDA and NVIDIA hardware, and CUDA parity, are out of scope.** **HARD.**
`MAP.md:72` — "CUDA and NVIDIA hardware." `docs/02-plan.md:130` — "CUDA parity, multi-node, training, Windows, macOS." `docs/03-native-kernel-inventory.md:44`.
Confidence: fact.

**R79 — Multi-node serving, training and fine-tuning are out of scope.** **HARD.**
`MAP.md:73-74` — "Multi-node serving." / "Training and fine-tuning." `docs/02-plan.md:130`.
Confidence: fact.

**R80 — Windows and macOS are out of scope.** **HARD.**
`MAP.md:75` — "Windows and macOS." Same at `docs/02-plan.md:130`.
Confidence: fact. Note the engine is *developed* on a Mac clone and *runs* on Ubuntu — the exclusion is about target platforms, not the writing surface.

**R81 — Matching data-centre throughput is out of scope; the target is one desktop.** **HARD.**
`MAP.md:76` — "Matching data-centre throughput. The target is one desktop." `docs/02-plan.md:131` — "Beating NVIDIA hardware. The target is this box."
Confidence: fact.

**R82 — Flash-Next, the 180B model, expert tiering and the ngram table are out of scope.** **HARD.**
`docs/02-plan.md:129` — "Flash-Next, the 180B model, expert tiering, the ngram table." `docs/03-native-kernel-inventory.md:43` — "**FlashNext / ngram / tiered expert store.** Different model, dropped."
Confidence: fact. Consequence: `GLOSSARY.md:18,20,22` define `expert cache`, `router look-ahead` and `ngram table` as project terms for a model that is out of scope.

**R83 — FP8 tuning is closed; the tile configs are reference-only and unverified.** **HARD.**
`HANDOFF.md:222-225` — "FP8 tuning is closed (interrupted deliberately, superseded by `ADR 0003`)." `HANDOFF.md:194-196` — "**The fp8 tile configs are unverified.** They were committed, but the verification run was killed before it measured. Do not quote a 5.5× from them."
Confidence: fact via ADR 0003; the "5.5×" is measured (`docs/research/20-decode-profile-27b.md:143-146`) but was never verified end-to-end.

**R84 — EXL3 and every other non-GGUF quant format are not this project's problem.** **DERIVED**, currently listed as deferred.
`docs/research/15-quant-kernels.md:124-126` — "kernel implication: nothing new to implement. Support the IQ/Q_K types". Deferral stated at `HANDOFF.md:222-223` ("Deferred by decision: … EXL3"). **The deferral contradicts the quant report's own backlog**, which puts EXL3 at item 9 (`docs/research/15-quant-kernels.md:191`); and no ADR records the deferral. See **C4**.

**R85 — A list of four things is recorded as "deferred by decision": Flash-Next and every MoE kernel; EXL3; HiCache host tier; megakernel.** **CONTESTED — no ADR carries it.**
`HANDOFF.md:222-225` — "Deferred by decision: Flash-Next and every MoE kernel; EXL3; HiCache host tier; megakernel."
Confidence: **low.** Only the first item is backed by an ADR (`docs/adr/0002-fork-sglang.md:54`). The HiCache host tier is an open inventory row with a test and a priority (`docs/03-native-kernel-inventory.md:36`, row 10, P4: "HIP kvcacheio | host-tier KV at ~46 GB/s"), and the megakernel is plan phase P6 (`docs/02-plan.md:95-96`). Both read as scheduled work, not deferrals. See **C4**.

---

## 7. Contradictions to resolve

Each entry names both sides. These are where the previous iteration's wrong directives were born; none can be fixed by a plan without an owner answer.

**C1 — "DP when the quant fits one card" vs "TP=2 default".** *The central one.*
- Rule: `MAP.md:29` — "DP when the quant fits one card, TP when it does not." `GLOSSARY.md:32` — "picked per model and per quant, never hard-coded — DP whenever the quant fits one card." `docs/00-facts.md:58-62` — "Data parallel beats tensor parallel when the model fits one card."
- Rule's premise holds for the primary target: 11.76 GB fits 19.98 GiB (`docs/00-facts.md:114`).
- Recommendation: `HANDOFF.md:207` — "**TP=2 default** (105 tok/s ceiling, ~25 GB KV pool, 2.64 ms/token AR), DP=2 as measured alternative".
- Also open in the plan: `docs/02-plan.md:106` — "D5 | TP or DP | Open; measure both in P1."
- And the measurement cuts against TP as a throughput choice: `docs/research/18-tp-roccl.md:324` — "**TP=2 on this PCIe pair is a batch-1 optimization, not a throughput optimization**"; measured 0.78× at 32 concurrent requests (`:286`).
Unresolved by the record. Note the rule and the recommendation cannot both be the default.

**C2 — Two target models vs one.**
- Two: `AGENTS.md:4-7` — "It runs two target models — Qwen3.8-27B (`qwen3_5` …) and Qwen3.8-Flash-Next (`qwen4_exp`, 512-expert top-10 MoE)". Also `README.md:6` ("two target models"), `README.md:32-35` (both rows), `GLOSSARY.md:10` ("these two model architectures"), `GLOSSARY.md:36` ("one per target model").
- One: `docs/adr/0002-fork-sglang.md:54` — "Scope is **Qwen3.8-27B only**. Flash-Next work is out." `docs/02-plan.md:103` — D2. `MAP.md:25`. `docs/01-models.md:6-9` scope note.
- ADR 0002 is the accepted decision and postdates the AGENTS.md text by the same day; the AGENTS.md copy was never updated. Four documents still contradict the ADR.

**C3 — "Must run on one card first" vs a TP-default plan.**
- `AGENTS.md:53` — "The engine must run on one card first. Two-card support is a mode, not a requirement." Identical at `README.md:77` and `MAP.md:17`.
- `HANDOFF.md:207` recommends TP=2 as the default topology.
- The same handoff's own P2 gate recommendation is one-card: `HANDOFF.md:210` — "IQ3_S on one headless card". So the recommendation set is internally split, not just split against AGENTS.md.

**C4 — A "deferred by decision" list that no decision carries.**
- Claim: `HANDOFF.md:222-225` — "Deferred by decision: Flash-Next and every MoE kernel; EXL3; HiCache host tier; megakernel."
- Counter-evidence, item by item: HiCache host tier is inventory row 10 with a test and priority P4 (`docs/03-native-kernel-inventory.md:36`); the megakernel is plan phase P6 with an explicit precondition (`docs/02-plan.md:95-96`) and inventory row 11 (`docs/03-native-kernel-inventory.md:37`); EXL3 sits in the quant report's ranked backlog at #9 (`docs/research/15-quant-kernels.md:191`).
- No ADR in `docs/adr/` and no record in `docs/decisions/` mentions EXL3, HiCache or the megakernel. Only Flash-Next/MoE is genuinely decided (`docs/adr/0002-fork-sglang.md:54`, `docs/03-native-kernel-inventory.md:41-43`).
Consequence: a session reading HANDOFF.md will silently drop three work items the plan still schedules.

**C5 — "First native kernel is the fp8 GEMV" vs "no native kernel is written for FP8".**
- ADR: `docs/adr/0003-representation-doctrine.md:38-40` — FP8 "is never a performance target and no native kernel is written for it." Inventory row 1b agrees: "none. Deleted when the GGUF path serves the model" (`docs/03-native-kernel-inventory.md:27`).
- Profile report's recommendation: `docs/research/20-decode-profile-27b.md:180-181` — "the first native-kernel ticket should target **the M=1 blockwise-fp8 GEMV for the five shapes** … not attention, not the GDN path, not the all-reduce." Its ranked lever table also puts "A gfx1100-native fp8 GEMV" second (`:177`).
- Superseded arithmetic in the same direction: `docs/00-facts.md:71-74` — "FP8 weights still pay off … Plan for a fused decompress-and-multiply kernel, not for a native FP8 GEMM."
The report predates ADR 0003 and was never given a correction banner, unlike `research/15` and `research/16`. This is the most likely source of a wrong "optimise FP8 first" directive.

**C6 — Three different claims about speculation's payoff.**
- `GLOSSARY.md:36` — "should return 1.3x-1.8x at batch 1 for the cost of one verify pass"; `docs/01-models.md:56` — "The MTP layer should recover 1.3x to 1.8x at batch 1."
- `docs/research/17-dflash2.md:16` — publisher acceptance "DFlash2 3.35-4.14 / 3.72-4.33 / 3.83-4.36 … vs MTP 2.60-2.97". Acceptance length in tokens, not a speedup.
- `docs/02-plan.md:27-29` — "~180 tok/s per request with DFlash2, 91 with MTP and 46 without, on a data-centre card" (≈3.9× and 2.0×).
- `HANDOFF.md:219-220` — "never measured; the 3.4-4.4 range is the model publisher's, on different hardware."
No two of these are the same quantity. The plan ranks speculation #2 on a number nobody has measured here.

**C7 — Where the code lives: `engine/` vs `rspt/` + `kernels/`.**
- `docs/adr/0002-fork-sglang.md:7-9` — "`engine/` in this repository is **our fork of sglang** … Optimisation work happens inside it, in place." `engine/UPSTREAM.md:5-7` agrees.
- `GLOSSARY.md:10` — "the engine: the package in `rspt/` plus the HIP kernels in `kernels/`". `docs/agents/domain.md:44-58` draws the same layout (`kernels/`, `rspt/`, `tests/`). `README.md:50-52` lists them as "not started".
- Physical state: `engine/` holds the vendored fork; `rspt/`, `kernels/` and `tests/` are empty directories.
The glossary and the domain doc describe the superseded thin-engine composition (ADR 0001), not the accepted one.

**C8 — Two different bandwidth walls, used interchangeably.**
- 592 GiB/s: `GLOSSARY.md:14,34`, `docs/00-facts.md:10-12,47-50`, `docs/adr/0003-representation-doctrine.md:19-23` (all ceilings computed from it).
- 619 GB/s: `docs/00-facts.md:197-198` — "Measured read bandwidth is **619 GB/s** (copy 595 GB/s), against the 800 GB/s specification"; `docs/03-native-kernel-inventory.md:26` ("≥60% of 619 GB/s"); `HANDOFF.md:256` ("against 619 GB/s").
- A third, GB/s vs GiB/s, sits underneath: `docs/research/20-decode-profile-27b.md:168` converts 592.9 GiB/s to 636.6 GB/s.
Every derived ceiling uses 592 GiB/s; every kernel gate uses 619 GB/s. The ~4% gap is small but the SPEC cannot cite both without saying which is which.

**C9 — Two strengths of NVIDIA-code policy.**
- `AGENTS.md:55-56` — "Delete NVIDIA code when it blocks a build or confuses a reader, never for its own sake."
- `docs/03-native-kernel-inventory.md:44-45` — "The NVIDIA branches stay behind dispatch; they are not our problem to keep working."
These differ on whether a dead NVIDIA branch is *kept* or *deleted*. Low stakes, but a coder loop will pick one at random.

**C10 — Session target: three numbers, none decided.**
- `MAP.md:11` — "serves several long-context sessions at once" (no count).
- `docs/02-plan.md:40-42` — "~28 GiB of free VRAM holds about 13 sessions" at 32K; "~42" at 8K.
- `HANDOFF.md:208` — Q3 recommends "**8 sessions at 32K** with q8 KV; 128K as a single-session mode", listed as waiting on the owner.
The plan's ~13 assumes a free-VRAM figure that no longer matches the measured 2.08 GB free per card (`docs/00-facts.md:154`). The number is a product requirement nobody has stated.

**C11 — Hard rules whose only home is a table row or a superseded ADR.**
`docs/adr/README.md:6-9` says an ADR records "the hard choices … that a future slice must not silently contradict: data-placement policy, the GGUF and kernel surface, TP-versus-DP per model, the dedup scope, the multi-user memory model." Three of those six have no ADR: TP-versus-DP is a `MAP.md` row plus a `GLOSSARY.md` entry (R8); dedup and the multi-user memory model appear only in `GLOSSARY.md:28-30` and `docs/02-plan.md:84-87`; and the "no Triton in the hot path" rule survives from a superseded ADR (`docs/adr/0001-engine-composition.md:42-44`) into `docs/02-plan.md:105` and `docs/03-native-kernel-inventory.md:9-12`.
Consequence for this restart: the DP rule that the previous plan overrode has **weaker documentary authority than the TP recommendation that overrode it**, which is a plausible mechanical cause of the drift.

**C12 — ADR 0001 is superseded but still quoted as binding, in two places where it matters.**
`docs/adr/0001-engine-composition.md:3-6` — "superseded by `0002-fork-sglang.md` … The exclusion it records still binds". Its `Decision` (thin engine, borrow by licence, `:32`) and its phase reordering (`:68-69` — "the first kernel is a Q5_K GEMV, not a 'Q4' one") are dead. Live descendants: the exclusion (re-homed to `docs/decisions/0001`, still accepted), and the no-Triton rule (re-stated in `docs/02-plan.md:105`). The first-kernel-type claim is not live and is also wrong for the current target — see C13.

**C13 — "First kernel is a Q5_K GEMV" vs a target file that contains no Q5_K.**
- `docs/adr/0001-engine-composition.md:68-69` — "the first kernel is a Q5_K GEMV, not a 'Q4' one". Repeated in `docs/research/00-synthesis.md:102-105` ("First kernel: `gemv_k_q8_1`. Q5_K times int8 activations … 79% of the bytes this model reads per token").
- The primary target's type histogram: `docs/00-facts.md:117-129` — IQ3_S 31.0%, IQ4_XS 25.7%, IQ3_XXS 17.6%, Q4_K 13.4%, IQ2_S 6.8%, then Q2_K/IQ2_XS/IQ2_XXS/BF16/type-29/F32. **No Q5_K.**
- ADR 0003 re-bases the family on those five types (`docs/adr/0003-representation-doctrine.md:33-34`) and the ladder on IQ3_S bytes (`:49-51`).
The Q5_K-first ranking is correct for the *unsloth* file (`docs/00-facts.md:84-101`, Q5_K 45.2%) and wrong for the *primary* file. Both files are in scope (R15), so the order needs stating explicitly rather than inheriting.

**C14 — All-reduce cost: 86 µs/token or 2.64 ms/token.**
- 86 µs: `docs/00-facts.md:52-56` — "the two all-reduces per layer move `2 x 5120 x 2 bytes` per token per layer, which is 1.31 MB per token over 64 layers, or 86 us per token"; `GLOSSARY.md:32` repeats "86 us/token of all-reduce".
- 2.64 ms: `docs/03-native-kernel-inventory.md:34` — "20.65 µs per 5120-element reduce, 2.64 ms/token"; `docs/research/18-tp-roccl.md:290` — "2 all-reduces × 20.65 µs = **41.3 µs** … 64-layer dense stack therefore pays **2.64 ms/token**".
The 86 µs figure is a pure bandwidth model with no latency floor; the measurement says each reduce is latency-bound at 20.65 µs, ~30× the modelled cost. Both are still live in the tree, and the TP economics differ by that factor.

**C15 — TP=2's speedup: 2×, 1.5× or 1.8-1.9×.**
- `HANDOFF.md:207` — "105 tok/s ceiling" (the naive 2× of `docs/adr/0003-representation-doctrine.md:49-51`'s 52.6 one-card ceiling).
- `docs/02-plan.md:114-118` — "**The all-reduce is not free at low bit-widths.** … Against a 9.5 ms bandwidth floor at IQ3_S that is 28% of the step, so TP=2 buys about 1.5x there, not 2x."
- `docs/research/18-tp-roccl.md:293-295` — derived at bf16: "TP=2 all-reduce overhead ≈ **6%** … Expected speedup … **1.8-1.9×**".
The 1.8-1.9× figure is for a bf16 model (54 GB → 27 GB/card); at IQ3_S the all-reduce share is 28%, not 6%, so 1.5× is the applicable one and 105 tok/s is not attainable. A SPEC that cites "105" as a ceiling will miss by a third.

---

## 8. What the record does **not** contain

Stated so the owner can add it rather than have a plan infer it:

- **No stated concurrency target.** The count is an open owner question (`HANDOFF.md:208`), and no document records an answer.
- **No stated context-length product requirement.** 32K is used because it was the baseline flag (`docs/00-facts.md:149`), not because it was chosen.
- **No TTFT or latency budget.** Only throughput ceilings exist.
- **No decision on the native/host boundary.** `HANDOFF.md:209` (Q4) is unanswered; ADR 0002 inherits the server wholesale, which implies an answer but does not state one.
- **No P2 acceptance number decided.** `HANDOFF.md:210` proposes "≥45 tok/s at 4K" for the owner; nothing in the tree records agreement. Note the proposal is below the ~54 tok/s one-card ceiling in R4/R16, which is sensible, but it is a proposal.
- **No ADR for TP-versus-DP, dedup scope, or the multi-user memory model** — the three items `docs/adr/README.md:8-9` says must have one.
- **No owner-signed record of what was rejected and why**, beyond `docs/decisions/0001` (llama.cpp) and the one line at `docs/research/00-synthesis.md:3-6` recording that the thin-engine composition was rejected.
