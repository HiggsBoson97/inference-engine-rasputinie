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

The fullstack loop (PLAN, CODE, PROMOTE, LEARN) with 2 slice slots. See
`docs/agents/control-plane.md`.
