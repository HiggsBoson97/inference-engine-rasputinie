# AGENTS.md — operating rules for this repo

`rasputinie` is a local inference engine built for exactly one machine, `rasputin`.
It runs two target models — Qwen3.8-27B (`qwen3_5`, 48 gated-deltanet + 16
full-attention layers) and Qwen3.8-Flash-Next (`qwen4_exp`, 512-expert top-10
MoE) — faster than a general-purpose runtime by assuming this hardware and
nothing else.
Correctness is proven against a torch reference, speed against a measured
baseline: no claim ships without a number.

Hardware, one line: 2x AMD RX 7900 XT (gfx1100, 19.98 GiB each, 592 GiB/s HBM
per card, PCIe 4.0 x16 with **no peer-to-peer DMA**), 60 GiB DDR5-5600 at
46.4 GiB/s copy, 4.4 GiB/s NVMe, ROCm 10.1 / hipcc 7.16, Ubuntu.

Read before writing anything: `docs/00-facts.md`, `docs/01-models.md`,
`docs/02-plan.md`, `MAP.md`, `HANDOFF.md`.

## Two hard rules

1. **The excluded engine stays excluded.** llama.cpp is not a baseline, not a
   reference, contributes no code or layout, and is not installed on `rasputin`
   — see `docs/decisions/0001-exclude-llama-cpp.md`. The GGUF reader and every
   dequant kernel are ours.
2. **This project runs the fullstack loop from PLAN onward.** SPEC, slices as
   GitHub issues, coder subagents, gated promotion. See
   `docs/agents/control-plane.md`.

## Sync topology

The Mac clone is the writing surface. It pushes to `origin`, the bare repo
`rasputin:development/inference-engine-rasputinie.git`; the working tree at
`rasputin:~/development/inference-engine-rasputinie` **pulls** from it.

```
Mac clone ──push──> origin (bare, on rasputin) ──pull──> rasputin working tree
```

Nothing writes into the rasputin working tree from the Mac directly — no rsync,
no scp, no editing over SSH. A dirty tree on the host means another session is
mid-edit; do not pull over it. `github` is a second remote
(`HiggsBoson97/inference-engine-rasputinie`, private) for issues and review, not
for the sync path.

## Measurement discipline

- Measure before and after every change. A number without a baseline is an
  opinion.
- Every kernel gets a correctness test against a torch reference before it gets
  a speed test.
- Never plan from specifications. Regenerate the hardware facts with
  `bench/00_hardware_probe.py` when the driver, the ROCm version or the hardware
  changes.
- The engine must run on one card first. Two-card support is a mode, not a
  requirement.
- Delete NVIDIA code when it blocks a build or confuses a reader, never for its
  own sake. Speed comes from kernels and data placement.

## Agent skills

### Issue tracker

Issues live as GitHub issues on `HiggsBoson97/inference-engine-rasputinie`, driven by `gh`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: root `GLOSSARY.md` plus numbered `docs/adr/`. See `docs/agents/domain.md`.

### Build loop

The fullstack loop (PLAN, CODE, PROMOTE, LEARN) with 2 slice slots; the gate layer is local discipline because branch protection is unavailable on this repo. See `docs/agents/control-plane.md`.
