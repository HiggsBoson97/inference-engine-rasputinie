# Handoff — rasputinie

One handoff, updated in place. A superseded section is deleted outright, never
tombstoned. This file is versioned with the repo and is the canonical copy.

**As of 2026-10-23 01:00 CEST.** Verify anything dated before relying on it.

---

## 1. What this is

A bespoke LLM inference engine for **exactly one machine and one model**:

- **Machine:** 2× AMD RX 7900 XT (gfx1100, RDNA3, 19.98 GiB each, no peer DMA),
  60 GiB DDR5, one node. Read wall **776 GB/s** per card.
- **Model:** `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, **IQ3_S**, with its MTP
  head. 851 tensors, 26.896 B params, 3.498 bpw, 64 layers (48 GatedDeltaNet +
  16 full attention), 10.57 GiB read per token.

**The governing rule was wrong and is now corrected.** It used to read "decode is
bandwidth-bound, so bytes per token decide the ceiling". That holds at 4
bits/weight and above. **At 3.44 bpw the ceiling is instruction issue, not
memory** — see §1 of `docs/plan/DESIGN-instruction-budget.md`. Bytes still matter;
they are no longer the binding constraint on this file.

## 2. Phase

**PLAN.** The measurement round is closed; the plan gate has not passed. Nothing
is built. The next unit of work is the counter test in §7, not the SPEC.

## 3. Read order

1. [`docs/plan/DESIGN-instruction-budget.md`](docs/plan/DESIGN-instruction-budget.md)
   — the `n` model, the three consequences, the five ISA facts, the falsifying
   test. **This is the frontier.**
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

## 7. Open — each waits on a named measurement

| # | Question | Settled by |
|---|---|---|
| A1 | **Is IQ3_S instruction-bound?** (`n ≈ 1.57` vs budget 1.04) | `rocprofv3` issue-slot occupancy. **One counter readout; run it first.** |
| A2 | In-situ rates for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S | Per-type sweep on the E1 harness — each type has its own `n` |
| A3 | MTP acceptance on this box | Teacher-forced replay against the shipped MTP head; the highest-value measurement now, because speculation divides `n` |
| A4 | Topology | **Answered for IQ3_S: DP=2.** PP=2 if a larger quant is used; TP needs a sub-6 µs all-reduce. Remaining gate: mapped-memory round-trip latency |
| A5 | The ~800 MiB anomaly's true cause | The clock-pinned re-run (trap 7) |
| A6 | Does runtime adaptation earn its complexity | An ablation build per mechanism |

---

## 8. The research prompt to run next

Copy this verbatim into a fresh session. It is the frontier as of `n = 1.57`.

```
Prove or refute that this engine is instruction-issue-bound, then price the
three levers that follow. Every number from a command; every probe
checksum-gated and committed.

1. INSTRUCTION CENSUS. Disassemble the IQ3_S dequant-GEMV
   (llvm-objdump -d --mcpu=gfx1100) and count VALU instructions per 4
   weights, split by role: index formation, sign application, scale
   application, dot. Report n. Then run it under rocprofv3 and report
   issue-slot occupancy, TCC_EA reads and L2 sector counts.
   Claim to test: n ~ 1.57, issue occupancy >= 85%, therefore memory tuning
   cannot move this kernel.

2. REDUCE n. Three variants, each at 1.07 GiB with the E1 checksum gate:
   (a) current branch-free SWAR sign path; (b) signs folded into a 512-entry
   LUT at load so application is one lookup; (c) scale amortised over 256
   weights. Target n <= 1.04, which is what 776 GB/s requires at 3.44 bpw.
   Report GB/s and n per variant, and the n-reduction attributable to each.

3. PRICE SPECULATION. Teacher-forced replay against the shipped MTP head to
   get the acceptance distribution on our own prompts -- no engine needed.
   Apply the depth rule run^k >= (tokens/ms)*row(k) to pick draft depth.
   Report effective tokens per weight sweep and the n it implies. Claim: at
   >= 2.5 tokens/sweep the kernel returns to memory-bound.

4. SETTLE THE TWO-CARD QUESTION IN ROUND-TRIPS, NOT BYTES. Measure mapped
   pinned-memory flag round-trip latency on this box (GPU publishes, GPU
   spins). Then price: replication (0 round-trips), TP=2 (128 all-reduces
   of 10 KiB), pipeline (1 handoff of 10 KiB per token). Report which wins
   at the measured latency and whether a 1 ms round-trip flips it.

5. VALIDATE THE INSTRUMENT. Reproduce 776 GB/s with code that shares nothing
   with the current probe: a write-only kernel and a different stride.
   >5% disagreement means the wall is wrong and items 1-3 must be re-read.

Deliver: the n table, the counter readout, the three variants' GB/s,
the acceptance distribution, the latency number, and a verdict per item.
Do not tune memory until item 1 returns.
```

## 9. Research assets, and what each is for

| Path | Use it for |
|---|---|
| `docs/research/2026-10-23/s3-amd-sinks.md` | **the instruction-issue model**, VOPD limits, I-cache, page size, memory hints, queue behaviour — 42 sourced URLs |
| `docs/research/2026-10-23/s4-topologies.md` | every two-card strategy priced per token: DP=2 = PP=2 = 2×, TP=2 = 1.47× |
| `docs/research/2026-10-23/s2-tensorfold-freetoken.md` | TensorFold's measured MTP depth rule; why the codebooks are not the cost; the LUT-instead-of-SWAR observation |
| `docs/research/2026-10-23/s1-strata.md` | the GPU-publish/pre-launched-acquire handoff (~200 lines) and the measured **690:1 latency-to-bytes** ratio for a host-mediated boundary |
| `docs/research/2026-10-22/` | round one: languages, megakernels, bandwidth, KV, serving knobs, 2025–26 sweep, correctness. Absolute figures ~25 % low; relative findings stand |
| `bench/gfx1100-probes/` | `WALL`, `E1` (IQ3_S ablations, checksum-gated), `E2` (launch/fusion), and the sources `bw2.cpp`, `gemv.cpp`, `iq3s_valid.cpp`, `e2_launch.cpp` |
| `bench/manifests/` | `iq3s.json`, `iq3s-mtp.json` — every tensor, role, type, dimension |
| `docs/plan/TENSOR-INVENTORY.md` | the inventory: 72 `(role, type)` pairs, byte budget, GDN-only roles |
| `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | quality anchors: IQ3_S matches BF16 on AIME25 and LCB v6 |

## 10. Skills for the next session, in order

1. `fullstack-software-development` → `references/orchestrator/plan-guide.md`
   (PLAN is the open stage).
2. `get-deeper` before any design document. The instruction-budget finding came
   from exactly that move and would not have surfaced from a broader brief.
3. `mattpocock-research` for any lane reading primary sources.
4. `adversarial-judge-stress` for the plan gate, dispatched fresh.
5. `mattpocock-handoff` for the next handoff.

## 11. One caution about workers

Four independent subagents in this session produced confident output that was
wrong or in breach: one built llama.cpp, one reported 230 % of the wall, one
reintroduced a rule that had been disproven, one recorded an open question as
settled. Each was caught by an independent check — a rebuild, a checksum, a
re-read. **Budget verification time, and never accept a subagent's number into
the record without reproducing it.**
