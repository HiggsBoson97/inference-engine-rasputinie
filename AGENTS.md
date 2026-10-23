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
4. **`aiter` is excluded on this card, and Triton is the kernel backend.** aiter's
   CK/ASM kernels are built for gfx942/gfx950; on gfx1100 they die at capture
   (`asserts on cpp_itfs`) and its rope kernels need `ck_tile` types that do not
   build here. Worse, the HIP attention default is **aiter**, so a working path can
   still route into dead kernels — force Triton explicitly. Evidence:
   `docs/research/14-sglang-gfx1100.md:19,42,44` and
   `docs/03-native-kernel-inventory.md:72`. Do not spend effort porting CK.

## Principles

The hard rules above are constraints. These are the direction of travel: they decide what
a *good* change looks like when nothing forbids it. Each is traceable to a record.

**1. Records, never targets.** No number is a goal. A fixed tok/s or GB/s target is a false
objective — it licenses leaving performance on the table the moment the number is met. The
objective is **bandwidth efficiency**: achieved GB/s against the measured 776 GB/s wall,
per shape. The discipline is to break the standing record every time and explain the shape.
**No shape is the target either.** "B sessions at C context" is a record to keep, not an
operating point to optimise for, unless the owner has named it — optimise the surface, not a
point on it (`docs/plan/COST-LEDGER.md`). `docs/plan/RECORD.md` is the ledger;
`REQUIREMENTS.md` H6/H7 is the rule.

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

**5. Parallelise the calculations, not just the requests.** Both cards work on every token
where the latency budget allows. A shard-native kernel — row/column-parallel plus a reduction
— is the default shape, so topology is a dispatch decision rather than an architectural fork.
Replication is the fallback when a reduction cannot be hidden, never the default. (A4)

**6. VRAM is the budget.** The model, the drafter and every KV cache share 19.98 GiB per
card. Any addition names what it displaces, in GB and in concurrent 32K sessions. Spilling to
host memory is not a slow path, it is a design failure. (H1)

**7. Measure, then believe.** A number without its baseline in the same artifact is an
opinion. When a document and a measurement disagree, the measurement wins and the document is
corrected that turn. (H10)

**8. Gate every probe on a checksum.** A fast wrong kernel is worse than a slow right one: an
ungated probe once reported 230 % of the wall because the compiler hoisted its loads. No row
enters a report without a data-dependent checksum replayed on the host.

**9. Validate the instrument before trusting the reading.** Ask the instrument for a known
quantity first — a canary with a countable answer. A zero from a profiler is a broken counter
until proven otherwise; on this host `rocprofv3` reads no instruction counter at all.

**10. Reproduce before recording.** Nothing enters `RECORD.md` or the hypothesis ledger without
an independent reproduction, including anything a subagent reports. Cite the row; never
re-derive a number from prose.

**11. Correctness is designed, not assumed.** Where a format or kernel claims losslessness,
greedy output matches the reference exactly; where it does not, the bounded error is stated
with its tolerance *before* the run. A 32-token match is an arbiter, not the method. (H11)

## Two cards, one machine

Two RX 7900 XT at 19.98 GiB each: **39.96 GiB of VRAM, 2× the compute, 2× the bandwidth**,
joined by PCIe 4.0 ×16 with **no peer-to-peer DMA**. The goal is to make the pair behave as
one card that is double in everything.

- **Weights are sharded, never moved.** Tensor parallelism gives each card a permanent half
  of every tensor. What crosses the link is the partial sums: **128 all-reduces of ~10 KiB
  per token — 1.3 MB, 52 µs of link time.** The cost is `128 × round-trip latency`, never the
  bytes. Nobody moves weights across PCIe at inference time; that is a death sentence at any
  link generation.
- **TP is what makes one token faster.** At the recorded 20.65 µs round trip, TP=2 turns a
  15.1 ms/token sweep into 10.2 ms (**1.48×**); at a 2 µs round trip, 7.8 ms (**1.93×** —
  essentially perfect doubling). DP=2 gives **1.00× on a single stream**: one card does the
  work and the other idles. It reaches 2.00× only at concurrency ≥ 2.
- **TP also buys KV capacity.** Sharded weights are ~5.5 GiB per card instead of 11.8, and the
  KV shards by head, so the pair's KV pool is `VRAM − trunk/2 − drafter` per card. At a 7 GiB
  pool that is ~6.5 concurrent 32K sessions with no drafter, ~6.0 with the MTP head, ~5.9 with
  DFlash2 at 4.5 bpw, and ~4.8 with DFlash2 at BF16 — the drafter's precision costs sessions,
  which is why it must be quantised (`run-artifacts/hypotheses/H-5.md`).
- **The lever is round-trip latency.** Every microsecond off it is ~0.13 ms off the step,
  because there are 128 of them. 20.65 µs is a host-mediated path; GPU-publish /
  pre-launched-acquire is the documented alternative and is the thing to measure.
- **Build shard-native from the first kernel.** A row/column-parallel GEMV plus a reduction is
  a dispatch flag, not a rewrite. Topology then becomes a per-operating-point decision — TP
  for single-stream latency and long context, DP for coupling-free aggregate — instead of an
  architecture chosen before the measurement exists.

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
