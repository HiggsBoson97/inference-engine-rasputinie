# Handoff — rasputinie

One handoff, updated in place. A superseded section is deleted outright, never
tombstoned. This file is versioned with the repo and is the canonical copy.

**As of 2026-10-23 15:43 CEST.** Verify anything dated before relying on it.

---

## 1. What this is

A bespoke LLM inference engine for **exactly one machine and one model**:

- **Machine:** 2× AMD RX 7900 XT (gfx1100, RDNA3, 19.98 GiB each, no peer DMA),
  60 GiB DDR5, one node. Read wall **776 GB/s** per card.
- **Model:** `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, **IQ3_S**, with its MTP
  head. 851 tensors, 26.896 B params, 3.498 bpw, 64 layers (48 GatedDeltaNet +
  16 full attention), 10.57 GiB read per token.

**The governing rule has now been corrected by measurement.** It read "decode is
bandwidth-bound, so bytes per token decide the ceiling"; then "at 3.44 bpw the
ceiling is instruction issue, so memory work is wasted". **Both are wrong.** The
kernel at 521 GB/s uses 48.6 % of the issue slots and gains **+23.7 %** when memory
is made free (521 → 645 GB/s), so it is neither saturated nor bandwidth-bound: it
sits at a corner where memory costs 16.0 % of the wall and instructions cost 16.9 %.
See `docs/plan/DESIGN-instruction-budget.md` and the evidence in
`docs/agents/production/run-artifacts/hypotheses/H-1.md` and `H-2.md`.

## 2. Phase

**PLAN.** A1 is answered (refuted) and the ledger has one open row, `H-3`. The plan
gate has not passed and nothing is built. The next unit of work is **`H-3`: prove or
refute that the 521 → 645 GB/s gap is memory latency / MLP** — not the SPEC, and not
another counter readout.

## 3. Read order

1. [`docs/plan/DESIGN-instruction-budget.md`](docs/plan/DESIGN-instruction-budget.md)
   — **the corrected two-bound model**: 521 achieved, 645 with memory free, 776 wall;
   why the old `n` model was void (a 32× unit error), the five ISA facts, and the
   three falsifiers. **This is the frontier.** Its evidence is
   `docs/agents/production/run-artifacts/hypotheses/H-1.md` (the differential
   experiment) and `H-2.md` (the ISA census).
2. [`docs/plan/REQUIREMENTS.md`](docs/plan/REQUIREMENTS.md) — hard constraints,
   settled rules, six open items each naming its measurement.
3. [`docs/00-facts.md`](docs/00-facts.md) — measured physics; every number
   derives from it.
4. [`docs/03-native-kernel-inventory.md`](docs/03-native-kernel-inventory.md) —
   the kernel set, keyed `(role, type)` = 72 pairs.
5. [`docs/plan/RECORD.md`](docs/plan/RECORD.md) — standing records, measured vs
   estimated, with the four shapes that still have no number.
6. [`docs/plan/DECISION-packaging-iq3s.md`](docs/plan/DECISION-packaging-iq3s.md)
   and [`DESIGN-batch-regimes.md`](docs/plan/DESIGN-batch-regimes.md) — serve the
   file as published; fuse 2–4 layers per launch.
7. ADRs 0002 (fork), 0003 (representation), 0004 (FP8 excluded); `decisions/0001`
   (llama.cpp excluded).
8. [`docs/research/README.md`](docs/research/README.md) indexes two rounds and
   says which conclusions are superseded. **`2026-10-23/` is current.**

## 4. State, and the first check

```bash
cd "$REPO" && git log --oneline -1 && git status --short
ssh rasputin 'cd ~/development/inference-engine-rasputinie && git log --oneline -1 && git status --short'
for c in /sys/class/drm/card[0-9]*; do [ -f "$c/device/mem_info_vram_used" ] && \
  printf "%s %sMiB\n" "$(readlink -f $c/device | sed 's|.*/||')" \
  "$(( $(cat $c/device/mem_info_vram_used)/1024/1024 ))"; done
```

Clean on both sides means matching tips and empty status. **Idle GPUs are
26 MiB (07:00.0), ~99 MiB (03:00.0), ~15 MiB (12:00.0, iGPU).** Far above that is
a leftover process — check `etime` before killing, and kill by PID.

| | |
|---|---|
| Repo | public, `HiggsBoson97/inference-engine-rasputinie` |
| Remotes | `origin` = bare repo **on the host**; `github` = public |
| Branch | `main`; branch protection is **available now the repo is public, not yet enabled** — enable it before the first dispatch |
| Host tree | clean clone, synced |
| Staging | `../rasputinie-plan/` — the repo copies are authoritative |
| Environment | venv `~/venv/sglang-fork`, wheel `~/wheels/sglang_kernel-0.4.10-…whl`, 69 GB models — all outside the repo |

## 5. Baseline, exactly

| Thing | Value |
|---|---|
| Repo (Mac) | `/Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie` |
| Repo (host) | `/home/mattia-bosi/development/inference-engine-rasputinie` |
| SSH | `ssh rasputin`, port **8374**, user `mattia-bosi`, `sudo -n` |
| Engine source | `engine/` — vendored sglang `d0a7565`, 11,063 files, complete |
| Python | `~/venv/sglang-fork/bin/python` — 3.12.15, torch 2.14.1+rocm7.2, HIP 7.2.53211, sglang 0.5.9, sgl_kernel 0.4.10 |
| ROCm | 10.1 under `/opt/rocm/core-10.1`; hipcc 7.16 / clang 24; dev packages `amdrocm-{runtime,core,blas}-dev10.1` **required** |
| GPU mapping | **HIP 0 = amd-smi 0 = sysfs card1 = PCI 03:00.0 = display**; HIP 1 = amd-smi 1 = sysfs card0 = PCI 07:00.0 = headless |
| Use this card | **`HIP_VISIBLE_DEVICES=1` → PCI 07:00.0** |
| Models | IQ3_S 11.77 GB, IQ3_S-mtp 12.12 GB, UD-Q4_K_XL 17.56 GB (reference only), mmproj 931 MB (out of scope) |
| Donor checkouts | `development/_research/{strata,tensorfold,freetoken,deepinfer,sglang,exllamav3}` — read-only |
| Bench port | 30100 |
| E1 harness | `~/e1/iq3s_valid <manifest> <gguf> <NV> <rp>`, **`GRIDMUL=8 BLK=256` → `grid=336×256`**; `GRIDMUL=4` silently halves the grid and reports 436 GB/s |
| Manifest | `python3 ~/e1/mkmanifest.py <gguf> 30 <out>`; a 1-tensor manifest must be cut by hand (`awk 'NR==2'`) — the generator asserts ≥1 GiB and an empty manifest segfaults the harness |
| Profiler | `rocprofv3` **1.4.1** at `/opt/rocm/bin`. Reads only `SQ_BUSY_CYCLES`, `SQ_WAVES`, `Wavefronts`; every instruction and `GL2C_*` counter is 0. See trap 16 |

## 6. Standing traps

Each has an incident behind it. Do not treat any as folklore.

1. **HIP 0 is the display card.** A load test there caused a `ring gfx_0.0.0`
   timeout under gnome-shell. Always `HIP_VISIBLE_DEVICES=1` for single-card work.
2. **llama.cpp is excluded** — no baseline, no reference, no derived code, not
   installed. A research subagent cloned and built it on 2026-10-23 anyway and it
   had to be deleted. **Name this rule in every dispatch brief.**
3. **FP8 is excluded as a format** (ADR 0004). The 857 ms/token run is evidence
   for the exclusion, never a baseline.
4. **619 GB/s was never the wall** — a 1.6 ms kernel artefact. The wall is
   **776 GB/s**. Any ceiling derived from 619 is wrong.
5. **Below ~4 bits/weight the format is instruction-bound.** IQ3_S needs
   `n ≤ 1.04` and measures 1.57. **Do not tune memory on it.** See §1.
6. **Kernels must be checksum-gated.** A probe without a data-dependent checksum
   reported 1784 GB/s — 230 % of the wall — because the compiler hoisted the
   loads. A fast wrong kernel is worse than a slow right one.
7. **The ~800 MiB "footprint cliff" is probably the clock ramp.** E2 named it a
   cliff; the arithmetic says 800 MiB ÷ 750 GB/s = 1.12 ms, i.e. a fixed ~230 µs
   stops being amortised there. Pin clocks and re-run before designing around it.
8. **`pkill -f <pattern>` matches your own ssh command line** and kills the shell
   running it. Kill by PID.
9. **Runtime Triton compiles stall the first requests** — 1–2 s per kernel, 16.5 s
   for one. Warm-up is not optional.
10. **No `hf`, `git-lfs`, `uv`, `pipx` or `rocm-smi` on the host.** Downloads are
    HTTP + `xargs -P4 curl`; clocks are `amd-smi`.
11. **1249 MHz is the top memory step; idle is 96 MHz.** A ~3 ms wake-up after
    idle disappears with `power_dpm_force_performance_level=high`. Never measure
    one cold launch and call it the wall.
12. **4 KB pages only.** `amdgpu` maps 4 KB PTEs; no HIP or ROCm API or env var
    gives 2 MB pages. Huge-page theories are arithmetically excluded.
13. **`v_dot4_i32_i8` cannot dual-issue**, and gfx11 has no `cp.async` (MUBUF has
    no LDS bit). Two optimisation families are dead by design — check the ISA
    before believing a rate table.
14. **Dispatch must be type-sorted.** I-cache is 32 KB per WGP shared by both CUs;
    interleaving quant types in one CU costs −10 %, sorting costs 0.
15. **A GGUF parse is right when arithmetic says so**: metadata + sum of tensor
    bytes must equal the file size exactly. It closes to 0 B today.
16. **`rocprofv3` reads no instruction counter on this host.** Twenty-one counters
    were tested against a canary with a known instruction count (100 000 dependent
    fadds): only `SQ_BUSY_CYCLES`, `SQ_WAVES` and `Wavefronts` are populated;
    `SQ_INSTS_VALU`, `SQ_INST_CYCLES_VALU`, `VALUInsts`, `ValuPipeIssueUtil`,
    `OccupancyPercent` and every `GL2C_*` read **0**. SQTT/ATT gives `"code": null`.
    A zero from this profiler is not a measurement. `rocprofv3-avail pmc-check` is
    not a tiebreaker — it fails on the iGPU agent for every counter. (2026-10-23)
17. **The E1 harness needs `GRIDMUL=8`.** `grid = multiProcessorCount × GRIDMUL` =
    42 × 8 = 336. A stray `GRIDMUL=4` runs at half grid and reports `full` = 436.5
    instead of 516–523; that looked like an unstable baseline for one session.
18. **A kernel shorter than ~3 ms cannot be measured on this card**, whatever the
    footprint. The L2-resident experiment at `rp=4` (0.22 ms kernels) gave 690.3 and
    366.7 GB/s on identical input. Use `rp=64` for a 38 MB footprint. (2026-10-23)

## 7. Open — each waits on a named measurement

| # | Question | Settled by |
|---|---|---|
| A1 | ~~**Is IQ3_S instruction-bound?**~~ | **CLOSED 2026-10-23 — no.** Occupancy 48.6 %, not ≥ 85 %; +23.7 % when memory is made free; the `n ≤ 1.04` target was a 32× unit error. Evidence: `H-1.md`, `H-2.md`. Replaced by A1b. |
| **A1b** | **What costs the 521 → 645 GB/s gap?** Hypothesis: latency/MLP — 43 % of the body is `s_waitcnt`+`s_delay_alu`, and 21 global loads per 32 weights against `k_read`'s 5. | Ledger row **`H-3`**: a variant with wider outstanding loads per wave, accepting `full` ≥ 600 GB/s at 1.0700 GiB with the gate VALID. |
| A2 | In-situ rates for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S | Per-type sweep on the E1 harness — each type has its own instruction count and therefore its own two-bound corner |
| A3 | MTP acceptance on this box | Teacher-forced replay against the shipped MTP head; the highest-value measurement now, because speculation divides per-token work by the acceptance length and pays off whichever resource binds |
| A4 | Topology | **Answered for IQ3_S: DP=2.** PP=2 if a larger quant is used; TP needs a sub-6 µs all-reduce. Remaining gate: mapped-memory round-trip latency |
| A5 | The ~800 MiB anomaly's true cause | The clock-pinned re-run (trap 7) |
| A6 | Does runtime adaptation earn its complexity | An ablation build per mechanism |

---

## 8. The research prompt to run next

Copy this verbatim into a fresh session. It is the frontier as of the corrected
two-bound model: 521 achieved, 645 with memory free, 776 wall.

```
Prove or refute that the 521 -> 645 GB/s gap is memory latency, then price the
two levers that follow. Every number from a command; every probe checksum-gated
and committed.

1. LATENCY EXPOSURE. The kernel is 193 instruction words per 32 weights, of which
   19 s_waitcnt and 31 s_delay_alu, and it issues 21 global-memory instructions
   per 32 weights against k_read's 5 (which reaches 743.7 GB/s at the same
   footprint). Build a variant that widens outstanding loads per wave -- unroll
   the row loop so 2-4 block groups are in flight -- at fixed footprint,
   footprint gate and GRIDMUL=8. Claim to test: the gap is latency, not
   bandwidth. Acceptance: full >= 600 GB/s at 1.0700 GiB, gate VALID.

2. FIELD LOADS. Merge the four odd-sized block fields into one aligned 16 B read
   per lane group (a load-time permutation only; bytes unchanged). Report the
   drop in global-memory instructions per 32 weights and the GB/s it buys.

3. SIGN PATH. (a) current branch-free SWAR; (b) signs folded into a 512-entry LUT
   at load; (c) scale amortised over 256 weights. Report GB/s AND instruction
   words per 32 weights per variant. Measured fact to respect: deleting the sign
   path is worth +13.8% at DRAM but only +7.3% with memory free, so judge every
   variant at 1.0700 GiB, never in cache.

4. PRICE SPECULATION. Teacher-forced replay against the shipped MTP head to get
   the acceptance distribution on our own prompts -- no engine needed. Apply the
   depth rule run^k >= (tokens/ms)*row(k) to pick draft depth. Report effective
   tokens per weight sweep.

5. REPAIR OR REPLACE THE INSTRUMENT. Find a profiler build on this ROCm 10.1 /
   gfx1100 host that reads SQ_INSTS_VALU and GL2C_MC_RDREQ_sum -- rocprof v1,
   another rocprofiler-sdk, or a pinned rocprofv3. Do NOT block items 1-3 on it:
   the differential method needs no profiler. If it cannot be repaired, record
   that and move on.

6. VALIDATE THE WALL. Reproduce 776 GB/s with code that shares nothing with the
   current probe: a write-only kernel and a different stride. >5% disagreement
   means the wall is wrong and every percentage above must be re-read.

Deliver: the GB/s table per variant at 1.0700 GiB, the instruction words per 32
weights per variant, the latency verdict, the acceptance distribution, the
instrument status, and a verdict per item. Do not tune layout, page size or
coalescing: read is already at 96% of the wall.
```

## 9. Research assets, and what each is for

| Path | Use it for |
|---|---|
| `docs/research/2026-10-23/s3-amd-sinks.md` | the ISA survey: VOPD limits, I-cache, page size, memory hints, queue behaviour — 42 sourced URLs. Read it for the **ISA facts**, not for an issue model |
| `docs/research/2026-10-23/s4-topologies.md` | every two-card strategy priced per token: DP=2 = PP=2 = 2×, TP=2 = 1.47× |
| `docs/research/2026-10-23/s2-tensorfold-freetoken.md` | TensorFold's measured MTP depth rule; why the codebooks are not the cost; the LUT-instead-of-SWAR observation |
| `docs/research/2026-10-23/s1-strata.md` | the GPU-publish/pre-launched-acquire handoff (~200 lines) and the measured **690:1 latency-to-bytes** ratio for a host-mediated boundary |
| `docs/research/2026-10-22/` | round one: languages, megakernels, bandwidth, KV, serving knobs, 2025–26 sweep, correctness. Absolute figures ~25 % low; relative findings stand |
| `bench/gfx1100-probes/` | `WALL`, `E1` (IQ3_S ablations, checksum-gated), `E2` (launch/fusion), and the sources `bw2.cpp`, `gemv.cpp`, `iq3s_valid.cpp`, `e2_launch.cpp` |
| `bench/manifests/` | `iq3s.json`, `iq3s-mtp.json` — every tensor, role, type, dimension |
| `docs/plan/TENSOR-INVENTORY.md` | the inventory: 72 `(role, type)` pairs, byte budget, GDN-only roles |
| `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | quality anchors: IQ3_S matches BF16 on AIME25 and LCB v6 |

## 10. Skills for the next session, in order

1. `fullstack-software-development` → `references/discovery.md` and
   `references/orchestrator.md` (PLAN/Discovery is the open stage; v10 has no
   separate `plan-guide.md`).
2. `get-deeper` before any design document. The unit check that killed the `n`
   model — and the differential experiment that replaced it — both came from
   exactly that move.
3. `mattpocock-research` for any lane reading primary sources.
4. `adversarial-judge-stress` for the plan gate, dispatched fresh.
5. `mattpocock-handoff` for the next handoff.

## 11. One caution about workers

Four independent subagents in a prior session produced confident output that was
wrong or in breach: one built llama.cpp, one reported 230 % of the wall, one
reintroduced a rule that had been disproven, one recorded an open question as
settled. Each was caught by an independent check — a rebuild, a checksum, a
re-read. **Budget verification time, and never accept a subagent's number into
the record without reproducing it.**

Two more on 2026-10-23, both in this section's spirit:

- A probe wrote its own disassembler and re-derived a census another probe was
  already producing, instead of stopping when its named counter read zero. The
  stop-and-report rule in the brief is the thing that was missing; it is now in
  every brief.
- A probe reported `full` = 436.5 GB/s and called the baseline unstable. It had
  inherited a stray `GRIDMUL=4`. **Check the harness's printed config line against
  the published one before believing any % of wall.**

The instrument failure (trap 16) was found by asking for a *known quantity* —
a canary with a countable instruction stream — not by reading a report.
