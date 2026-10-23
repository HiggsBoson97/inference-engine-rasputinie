# AGENTS.md — operating rules for this repo

`rasputinie` is a local inference engine built for exactly one machine, `rasputin`.
It runs exactly one model — Qwen3.8-27B (`qwen3_5`, 48 gated-deltanet + 16
full-attention layers), from the `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S
file with its MTP head — faster than a general-purpose runtime by assuming this
hardware and nothing else.
Correctness is proven against a reference, speed against a measured baseline: no
claim ships without a number.

Hardware, one line: 2× AMD RX 7900 XT (gfx1100, 19.98 GiB each, **776 GB/s
measured read wall**, PCIe 4.0 ×16 with **no peer-to-peer DMA**), 60 GiB
DDR5-5600 at 46.4 GiB/s copy, 4.4 GiB/s NVMe, ROCm 10.1 / hipcc 7.16.

Read before writing anything: `HANDOFF.md`,
`docs/plan/REQUIREMENTS.md`, `docs/00-facts.md`,
`docs/03-native-kernel-inventory.md`.

## Three hard rules

1. **The excluded engine stays excluded.** llama.cpp is not a baseline, not a
   reference, contributes no code or layout, and is not installed on `rasputin`
   — see `docs/decisions/0001-exclude-llama-cpp.md`. The GGUF reader and every
   dequant kernel are ours. **Name this rule in every subagent brief**: a worker
   that does not know it will reach for the obvious reference implementation.
2. **FP8 is excluded as a format**, not demoted. No checkpoint, no reference
   run, no kernel — see `docs/adr/0004-exclude-fp8.md`.
3. **This project runs the fullstack loop from PLAN onward.** SPEC, slices as
   GitHub issues, coder subagents, gated promotion. See
   `docs/agents/control-plane.md`.

## Principles

The hard rules above are constraints. These are the direction of travel: they decide what
a *good* change looks like when nothing forbids it. Each is traceable to a record.

**1. Records, never targets.** No number is a goal. A fixed tok/s or GB/s target is a false
objective — it licenses leaving performance on the table the moment the number is met. The
objective is **bandwidth efficiency**: achieved GB/s against the measured 776 GB/s wall,
per shape. The discipline is to break the standing record every time and explain the shape.
`docs/plan/RECORD.md` is the ledger; `REQUIREMENTS.md` H6/H7 is the rule.

**2. gfx1100-native, one machine.** Every hot path is gfx1100 machine code reached through
ROCm. No portability layer, no NVIDIA fallback kept "just in case", no abstraction that
exists to serve another vendor or another card. A branch that exists for hardware we do not
have is a defect, not insurance. (H5)

**3. The engine is the minimum that runs this model on this box.** Vendored sglang is raw
material, not a dependency to preserve: strip it to what the decode path and the gate path
actually touch, delete rather than configure, and make every retained file earn its place.
Small and fast beats complete. (H9)

**4. Speculation is core, not an optimisation.** Decode pays one weight sweep per
*verification step*, so accepted tokens per step divide the per-token cost. Both drafters are
in scope — the shipped MTP head and the DFlash2 block-diffusion drafter — and the drafter's
own weight read is part of the per-step budget, never free. (H2)

**5. VRAM is the budget.** The model, the drafter and every KV cache share 19.98 GiB per
card. Any addition names what it displaces, in GB and in concurrent 32K sessions. Spilling to
host memory is not a slow path, it is a design failure. (H1)

**6. Measure, then believe.** A number without its baseline in the same artifact is an
opinion. When a document and a measurement disagree, the measurement wins and the document is
corrected that turn. (H10)

**7. Gate every probe on a checksum.** A fast wrong kernel is worse than a slow right one: an
ungated probe once reported 230 % of the wall because the compiler hoisted its loads. No row
enters a report without a data-dependent checksum replayed on the host.

**8. Validate the instrument before trusting the reading.** Ask the instrument for a known
quantity first — a canary with a countable answer. A zero from a profiler is a broken counter
until proven otherwise; on this host `rocprofv3` reads no instruction counter at all.

**9. Reproduce before recording.** Nothing enters `RECORD.md` or the hypothesis ledger without
an independent reproduction, including anything a subagent reports. Cite the row; never
re-derive a number from prose.

**10. Correctness is designed, not assumed.** Where a format or kernel claims losslessness,
greedy output matches the reference exactly; where it does not, the bounded error is stated
with its tolerance *before* the run. A 32-token match is an arbiter, not the method. (H11)

## Two cards, one machine

Two RX 7900 XT at 19.98 GiB each: **39.96 GiB of VRAM and 2× the compute**, joined by
PCIe 4.0 ×16 with **no peer-to-peer DMA**. That asymmetry is a design problem, not a
limitation to accept:

- **Bandwidth is per card, not pooled.** 776 GB/s each, 1552 GB/s aggregate — but only when
  the cards do *independent* work. Two cards cooperating on one token double nothing.
- **The link is ~25–30 GB/s and host-mediated.** Never move weights across it per token. A
  10 KiB handoff per token is affordable; a 10 GiB one is not. The measured ratio for a
  host-mediated boundary is **690:1 latency-to-bytes** — price handoffs in round-trips,
  never in bytes.
- **Replicate before you shard.** For this quant, DP=2 (a full replica per card) costs zero
  link traffic and takes 2× aggregate throughput; TP=2 spends 128 all-reduces per step
  (~2.64 ms, 27 % of it) to buy 1.47×; pipeline equals replication at ≥2 concurrent
  requests. **DP=2 for IQ3_S; PP=2 only if a larger quant is ever used.**
- **Each replica carries its own model and its own drafter**, so *per-card* VRAM — not the
  aggregate — sets the KV budget and therefore the number of concurrent 32K sessions.

## Sync topology

The Mac clone is the writing surface. It pushes to `origin`, the bare repo
`rasputin:development/inference-engine-rasputinie.git`; the working tree at
`rasputin:~/development/inference-engine-rasputinie` **pulls** from it.

```
Mac clone ──push──> origin (bare, on rasputin) ──pull──> rasputin working tree
```

Nothing writes into the host working tree from the Mac directly — no rsync, no
scp, no editing over SSH. A dirty tree on the host means another session is
mid-edit; do not pull over it. `github`
(`HiggsBoson97/inference-engine-rasputinie`, **public** since 2026-10-22) is a
second remote for issues and review, not part of the sync path. Because it is now
public, **branch protection is available** — required checks and
`enforce_admins` can be turned on, so the gate layer no longer has to be pure
discipline.

## Measurement discipline

- Measure before and after every change. A number without a baseline is an
  opinion.
- **Any speed number must carry the hash of a passing correctness receipt.** A
  fast wrong kernel is worse than a slow right one.
- Every GPU probe needs a **data-dependent checksum** that is replayed on the
  host. Without one, the compiler will delete your loads and report 230 % of the
  wall.
- Aggregate bandwidth results from kernels larger than ~800 MiB per launch: that
  is a cliff, not a launch cost.
- Never plan from specifications. Regenerate the hardware facts with
  `bench/00_hardware_probe.py` when the driver, the ROCm version or the hardware
  changes.
- The engine must run on one card first. Two-card support is a mode, not a
  requirement.
- Delete NVIDIA code when it blocks a build or confuses a reader, never for its
  own sake. Speed comes from kernels and data placement.

## Agent skills

### Issue tracker

Issues live as GitHub issues on `HiggsBoson97/inference-engine-rasputinie`, driven
by `gh`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: numbered `docs/adr/` plus a decision record. A root
`GLOSSARY.md (removed)` is created with the SPEC, not before. See `docs/agents/domain.md`.

### Build loop

The fullstack loop — **Discovery** (grill → hypotheses → SPEC → slices → rig → plan gate),
then **Development** (CODE → FINAL → MERGE, then LEARN) — with 2 slice slots. See
`docs/agents/control-plane.md`.
