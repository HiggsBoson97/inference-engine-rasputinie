# Probe brief — H-1: the A1 counter readout

```
STATION: probe (hypothesis H-1)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin` (~/e1 harness, read-only) ; scratch `~/a1/`
TOOLSET: ssh, hipcc, rocprofv3 (v1.4.1, /opt/rocm/bin), llvm-objdump, amd-smi, sudo -n
BUDGET: 80 tool calls; at ~90 % switch to write-out mode; unfinished items report as failed(budget), never omitted
ROUND: 1
CONTRACT: none — this is a Discovery probe, not a slice
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-1.md
```

## GOAL

Settle, with hardware counters, whether the IQ3_S dequant-GEMV (`k_full` in
`~/e1/iq3s_valid.cpp`) at **516.4 GB/s** is limited by VALU **issue slots**.

Deliver a counter readout that a second person can re-derive from the files you leave
behind: per kernel, the VALU instruction count, the VALU pipe-busy cycles, the issue
utilisation, the wave-instruction rate, and the DRAM read traffic — with the arithmetic
that turns them into an occupancy percentage written out explicitly, not asserted.

## CONTEXT

`rasputin` is a 2× RX 7900 XT box (gfx1100, RDNA3). The relevant history, all measured
and all in the repo at `/Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie`:

- `bench/gfx1100-probes/E1-IQ3S-2026-10-22.md` — the ablation table. `full` = 516.4 GB/s,
  `read` (loads + index, no dequant) = 717.7 GB/s, `full_nosgn` = 608.2 GB/s, footprint
  1.0700 GiB, grid 336 × 256, `rp = 4`, checksum-gated.
- `bench/gfx1100-probes/WALL-2026-10-22.md` — the memory wall is **776 GB/s**.
  619 GB/s was a 1.6 ms kernel artefact; never quote it as the wall.
- `docs/plan/DESIGN-instruction-budget.md` — the claim under test: at 3.44 bits/weight
  the ceiling is *instruction issue*, `n ≈ 1.57` against a budget of `n ≤ 1.04`, so
  "memory tuning cannot move this kernel". Its own falsifier: *"if issue-slot occupancy
  is not high at 516 GB/s, the model is wrong and memory work returns to the table."*

The harness already exists and is built: `~/e1/iq3s_valid`, source `~/e1/iq3s_valid.cpp`,
manifest generator `~/e1/mkmanifest.py`, model
`~/local-inference/models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`.

## UNITS — fixed before you start, so the readout cannot repeat a unit error

- **Wave instruction** = what an issue slot carries. gfx1100: one per SIMD32 per cycle,
  **168 SIMD32** (84 CU × 2), wavefront size **32** (verify both: `rocminfo`).
  Slot rate = 168 × measured sclk.
- **Lane instruction** = one workitem's execution of it. One wave instruction covers
  32 lanes, so a `v_dot4_i32_i8` wave instruction consumes 32 × 4 = **128 weights**.
- Issue slots are shared by **all** wave instruction classes (`SQ_INSTS_VALU`,
  `SQ_INSTS_SALU`, `SQ_INSTS_SMEM`, `SQ_INSTS_LDS`, `SQ_INSTS_FLAT`), not only VALU.
  Report occupancy both ways and say which one the claim needs.
- `SQ_INST_CYCLES_VALU / SQ_INSTS_VALU` is the mean cycles the VALU pipe is occupied per
  VALU instruction — measure it, do not assume half or quarter rate.

## RULES

1. **`HIP_VISIBLE_DEVICES=1`** for every run. That is PCI `07:00.0`, the headless card.
   HIP 0 / sysfs card1 / PCI `03:00.0` is the **display** card — a load there wedged
   gnome-shell once. Confirm the mapping by readlink before you launch anything.
2. **llama.cpp is excluded** — no clone, no build, no source read, no derived code, not
   installed. Name it in your report as a compliance line.
3. **Checksum-gate every measurement.** A probe without a data-dependent checksum
   once reported 1784 GB/s (230 % of the wall) because the compiler hoisted the loads.
   `iq3s_valid` already gates every row; capture its verdict lines and report them. A
   row that reports VALID but exceeds 776 GB/s is an automatic INVALID, not a result.
4. **Pin clocks before the first capture**: `power_dpm_force_performance_level=high` on
   the 07:00.0 card (find it by `readlink -f /sys/class/drm/card*/device`), restore
   `auto` when you finish, and state both in the report. A cold launch measured 96 MHz
   memory clock once and cost 3 ms of wake-up.
5. **Never `pkill -f <pattern>`** — it matches your own ssh command line and kills the
   shell. Kill by PID, and only what you started.
6. **No page-size theories, no `cp.async`** — gfx11 has neither 2 MB pages nor async
   copy, and MUBUF has no LDS bit. Do not spend effort there.
7. The model file is opened `O_RDONLY`; **nothing in the repo is modified**, and
   `~/e1/` is read-only to you: copy `iq3s_valid.cpp` into `~/a1/` if you need to
   change a launch parameter, and say so.

## WHAT TO COLLECT

1. **Preflight:** `git log --oneline -1` on the host clone, GPU VRAM table, `rocminfo`
   SIMD count and wavefront size, `amd-smi metric --clock` while idle.
2. **Baseline, unprofiled:** regenerate the manifest with `mkmanifest.py` (it must print
   `0 mismatches` and a 1.0700 GiB footprint), then reproduce `full` and `read` within
   ±3 % of 516.4 and 717.7 GB/s. If you cannot reproduce the baseline, stop and report
   that — a counter readout of a kernel that does not reproduce is worthless.
3. **Counter capture**, kernel-filtered to `k_full` (and, if cheap, `k_read` and
   `k_full_nosgn`). Counters to attempt, in this order of importance:
   `SQ_INSTS_VALU`, `SQ_INST_CYCLES_VALU`, `SQ_INSTS_SALU`, `SQ_INSTS_SMEM`,
   `SQ_INSTS_LDS`, `SQ_INSTS_FLAT`, `SQ_WAVES`, `GRBM_COUNT`, `SQ_BUSY_CYCLES`,
   `ValuPipeIssueUtil`, `SIMD_UTILIZATION`, `OccupancyPercent`, `GL2C_EA_RDREQ_32B_sum`,
   `GL2C_MC_RDREQ_sum`, `GL2C_HIT_sum`, `GL2C_MISS_sum`, `GL2C_MC_RDREQ_STALL`.
   `rocprofv3 --pmc` with several counters may need multi-pass (`--pmc "G1" --pmc "G2"`).
   Use `-f csv` if supported, else `rocpd` + `rocpd convert`; leave the raw files in
   `~/a1/` and cite their paths. If a counter is rejected by the agent, report the exact
   error and the subset you did collect — do not substitute a different metric silently.
4. **Derive and show the arithmetic:**
   - `waves/s`, `wave instructions/s` per class, and occupancy against `168 × sclk`.
   - `slots per VALU instruction` = `SQ_INST_CYCLES_VALU / SQ_INSTS_VALU`.
   - `weights/s` from the measured GB/s at 3.44 bpw **and** independently from the
     counters (`SQ_INSTS_VALU × 32 lanes × 4 weights / elapsed`), then say whether the
     two agree. This cross-check is the point of the exercise.
   - DRAM read traffic from `GL2C_MC_RDREQ_sum` (state the request size you assume and
     why) against the footprint × repeat count: does the kernel read each byte once?
5. **Reproducibility:** a second capture (fresh process, same config) must agree within
   2 % on occupancy; report both numbers.

## HOW TO VERIFY YOU DID IT RIGHT

- The kernel name in the counter file is `k_full`, and the dispatch count matches the
  harness's own printed pass count. A capture that silently profiled a warm-up copy of a
  different instantiation is the failure mode to check for.
- The counter-derived weights/s lands within ~15 % of the GB/s-derived weights/s. If it
  does not, your normalisation is wrong — fix it or report it as unresolved, but do not
  present the first number that looks plausible.
- Total DRAM reads are within ~10 % of the number of bytes the harness says it touched.
- Every table row traces to a file under `~/a1/` that you name.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-1.md` in the repo (the Mac
checkout path above), then print its content in your final message:

```
# H-1 — VALU issue occupancy of the IQ3_S dequant-GEMV
## Question / hypothesis (verbatim from the ledger)
## Configuration (card, clocks, manifest, footprint, grid, pass count, repeat count)
## Baseline reproduction (full, read, full_nosgn — GB/s and checksum verdicts)
## Counter readout (table: kernel, counter, value, normalisation, arithmetic)
## Derived (occupancy both ways; slots per VALU instruction; weights/s two ways)
## Verdict (supports / refutes the hypothesis, one paragraph, no hedging)
## What this changes (one paragraph; if it refutes, say what becomes unblocked)
## Compliance (llama.cpp excluded; no repo file modified; clocks restored)
## Receipts (absolute paths under ~/a1/, and the raw counter file names)
```

Effort: **high**. This is the project's decisive measurement, the one the whole plan is
waiting on. Spend the effort on making the capture trustworthy — kernel filtering,
multi-pass groups that actually collect, a second capture, and the counter-versus-bytes
cross-check — not on prose. A refutation delivered cleanly is worth more than a
confirmation delivered ambiguously.
