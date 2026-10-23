# Handoff — rasputinie

One handoff, updated in place. A superseded section is deleted outright, never
tombstoned. This file is versioned with the repo and is the canonical copy.

**As of 2026-10-23 16:52 CEST.** Verify anything dated before relying on it.

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

**And no shape is the target.** The owner has never named a session count or a context
length, so no `B × C` is "the operating point". Every per-token cost is a function of the
shape, and what governs the plan is the **surface**: `docs/plan/COST-LEDGER.md` derives the
two boundaries — KV overtakes the weights at `B × C ≈ 173K` resident tokens (bf16 KV), and
the GDN state overtakes them at ~36 concurrent sequences. Optimising for a point on that
surface is the failure `AGENTS.md` principle 1 names.

## 2. Phase

**PLAN.** A1, A2 and A3 are answered. Open rows: `H-3` (latency/MLP), `H-6` (two-card round
trip, brief written, GPU now free), `H-7` (grids + recipe), `H-8` (attention path). The plan
gate has not passed and nothing is built.

Two measurements landed on 2026-10-23 and both cut numbers that were recorded:

- **A2/H-4:** the 752 GB/s assumption for the non-IQ3_S bytes is **30–40 % too high** where it
  was measured. Whole-file decode re-derives from 56.0 to **47.1 tok/s** (a 19 %
  overstatement), and is still an upper bound over 24.5 % of the bytes that could not be
  gated. Load floors are all 82–85 % of wall; no layout penalises the read.
- **A3/H-5:** the MTP head costs **65.6 % of a weight sweep** for 7 draft tokens — 1.064 GB
  per token, because it re-reads the LM head each pass. DFlash2 costs 15.8 % at 4.5 bpw and is
  **flat in block size**. DFLASH is already first-class in the vendored tree, Triton, with an
  explicit ROCm fallback. Recommendation: MTP first, DFlash2 second, but build the verify path
  for the block from day one.

The frame for what comes next is `docs/plan/COST-LEDGER.md`: rank measurements by the term
they address, over the shape surface, never at a point. Its queue puts the **verify-width
cost curve** first, then the **attention path**, then the round trip, then prefill.

## 3. Read order

1. [`docs/plan/COST-LEDGER.md`](docs/plan/COST-LEDGER.md) — **the frame**: every per-token
   term as a function of `(B, C)`, the two boundaries, term provenance, and the ranked
   measurement queue. Read this before proposing any optimisation.
2. [`docs/plan/DESIGN-instruction-budget.md`](docs/plan/DESIGN-instruction-budget.md)
   — **the corrected two-bound model**: 521 achieved, 645 with memory free, 776 wall;
   why the old `n` model was void (a 32× unit error), the five ISA facts, and the
   three falsifiers. Its evidence is
   `docs/agents/production/run-artifacts/hypotheses/H-1.md` (the differential
   experiment) and `H-2.md` (the ISA census).
3. [`docs/research/2026-10-23/s5-donor-transfer.md`](docs/research/2026-10-23/s5-donor-transfer.md)
   — what to import from the six donor engines and what is dead on this card, each row
   with the artifact read and the gate it must pass.
4. [`docs/plan/REQUIREMENTS.md`](docs/plan/REQUIREMENTS.md) — hard constraints,
   settled rules, eight open items each naming its measurement.
5. [`docs/00-facts.md`](docs/00-facts.md) — measured physics; every number
   derives from it.
6. [`docs/03-native-kernel-inventory.md`](docs/03-native-kernel-inventory.md) —
   the kernel set, keyed `(role, type)` = 72 pairs.
7. [`docs/plan/RECORD.md`](docs/plan/RECORD.md) — standing records, measured vs
   estimated, and the shapes that still have no number (records, not targets).
8. [`docs/plan/DECISION-packaging-iq3s.md`](docs/plan/DECISION-packaging-iq3s.md)
   and [`DESIGN-batch-regimes.md`](docs/plan/DESIGN-batch-regimes.md) — serve the
   file as published; fuse 2–4 layers per launch.
9. ADRs 0002 (fork), 0003 (representation), 0004 (FP8 excluded); `decisions/0001`
   (llama.cpp excluded).
10. [`docs/research/README.md`](docs/research/README.md) indexes the rounds and says which
    conclusions are superseded. **`2026-10-23/` is current; `s3`'s issue model is refuted.**

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
| Memory terms | **VRAM** = the card's GDDR6, 19.98 GiB, where the 776 GB/s wall is measured. **L2** = 96 MB on-die. **Host RAM** never appears in a timed path — no spill, no managed allocation. A "DRAM arm" in an older report means VRAM, not system memory |
| Two-card link | PCIe 4.0 ×16, **no peer-to-peer DMA**. RCCL 2-rank all-reduce measured at **20.65 µs** amortised for 5120 elems, with a **~19.5 µs floor** up to 16 KiB that no `NCCL_*` setting improves; `pynccl` is ~2× cheaper than `torch.distributed` — never benchmark TP with the latter (`docs/research/18-tp-roccl.md`) |

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
19. **An unmeasured shape is not a target.** `RECORD.md` lists shapes with no number;
    treating one of them as "the operating point" and optimising for it is the failure
    `AGENTS.md` principle 1 names. It happened on 2026-10-23: a whole analysis was built
    around `8 × 32 K`, a shape the owner had never named. The load-bearing variable is
    resident tokens `B × C`; argue over the range, never at a point. (2026-10-23)
20. **"The rest of the file runs at 752 GB/s" was an assumption, and it was 30–40 % too
    high.** Measured: IQ4_XS 502.1 and Q4_K 452.2 against the IQ3_S 521. The whole-file
    decode estimate fell from 56.0 to **47.1 tok/s** the moment two of the four types were
    measured. Never let an unmeasured type contribute at a benchmark's rate. (2026-10-23)
21. **IQ3_XXS's measured rows looked *fast* (581–592 GB/s) and were wrong.** Its kernel
    failed the fp64 gate at 1.43 relative error and the checksum mismatched; the rate was
    above IQ3_S's gated 516 — the wrong-but-fast signature. A plausible number from a
    kernel that cannot reproduce the file is the most dangerous output this project
    produces. (2026-10-23)

## 7. Open — each waits on a named measurement

| # | Question | Settled by |
|---|---|---|
| A1 | ~~**Is IQ3_S instruction-bound?**~~ | **CLOSED 2026-10-23 — no.** Occupancy 48.6 %, not ≥ 85 %; +23.7 % when memory is made free; the `n ≤ 1.04` target was a 32× unit error. Evidence: `H-1.md`, `H-2.md`. Replaced by A1b. |
| **A1b** | **What costs the 521 → 645 GB/s gap?** Hypothesis: latency/MLP — 43 % of the body is `s_waitcnt`+`s_delay_alu`, and 21 global loads per 32 weights against `k_read`'s 5. | Ledger row **`H-3`**: a variant with wider outstanding loads per wave, accepting `full` ≥ 600 GB/s at 1.0700 GiB with the gate VALID. |
| A2 | ~~In-situ rates for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S~~ | **CLOSED 2026-10-23 — measured.** Load floors 637–660 GB/s, all within 0.4–3.7 % of their own aligned-stream controls; **no layout penalises the read**. Dot rates: IQ4_XS **502.1**, Q4_K **452.2** (kernel-limited, 344 words/16 w). IQ3_XXS and IQ2_S **could not be gated** and keep only their floors. Whole-file 56.0 → **47.1 tok/s**. Evidence `H-4.md` |
| A3 | ~~**Which drafter?**~~ | **ANSWERED 2026-10-23.** DFLASH is first-class in the vendored tree (Triton, explicit ROCm fallback, nothing CUDA-only by import). Draft cost at 521 GB/s: MTP-7 **7.445 GB = 65.6 % of a sweep**; DFlash2 BF16 40.2 %; DFlash2 4.5 bpw **15.8 %, flat in block size** → 139 / 164 / 199 tok/s. Licences: MIT code, Apache-2.0 weights. **Build MTP first, DFlash2 second, design the verify path for the block from day one.** Evidence `H-5.md` |
| A4 | Topology: TP=2, DP=2, or pipeline | **Reopened 2026-10-23: the recorded "DP=2" priced only aggregate throughput.** Weights are *sharded*, not moved; what crosses is partial sums — 128 all-reduces × ~10 KiB = **1.3 MB, 52 µs of link bytes per token**, so the cost is `128 × round-trip latency`. At the recorded 20.65 µs, TP=2 is **1.48× on a single stream** (15.1 → 10.2 ms/token) and 1.93× at 2 µs, and sharded weights free VRAM (at a 7 GiB KV pool: ~6.5 concurrent 32K sessions, ~6.0 with the MTP head — `H-5.md`). DP=2 is **1.00× single-stream**, 2× only at concurrency ≥ 2. **Gate: mapped-memory round-trip latency** — that one number decides it |
| A5 | The ~800 MiB anomaly's true cause | The clock-pinned re-run (trap 7) |
| A6 | Does runtime adaptation earn its complexity | An ablation build per mechanism |
| A7 | **What defines the i-quant grids, and is our inventory the publisher's recipe?** | The card says the files are **standard GGUF** ("run unmodified in llama.cpp"), so the grids are the standard tables; source them from `IST-DASLab/GSQ` (Apache-2.0), never llama.cpp. Diff our 72 `(role, type)` pairs against the publisher's `rco-allocation.txt` |
| A8 | **What does the attention path cost?** 16 full-attention layers over paged KV — no kernel, no rate, and it is the leading term above the resident-token boundary | A paged decode-attention kernel over a context ladder, batch 1+, q8 and k8v4, checksum-gated. `COST-LEDGER.md` queue item 2 |

---

## 8. The research prompt to run next

Copy this verbatim into a fresh session. It is the frontier as of the corrected
two-bound model, and it is shape-agnostic on purpose: no session count and no context
length has been chosen.

```
Build the cost ledger for this engine, then measure the terms it ranks. Every
term cites either a measurement on this card or an assumption labelled with its
error bar. Do not optimise anything until the ledger exists, and do not pick a
shape: the owner has named no session count and no context length.

1. LEDGER. Every byte on the critical path of one decode step, as a function of
   B (concurrent sequences) and C (context length): weight read (10.57 GiB, once
   per step, so 1/B per token), KV read (C x KV/position; 64 KiB bf16, 34 KiB
   q8, over the 16 full-attention layers only), GDN recurrent state (159 MB per
   sequence, read and written per token), drafter read (MTP head ~0.348 GB per
   pass vs DFlash2 ~1.06 GB at 4.25 bpw, one pass per block), all-reduce or
   handoff at 128x or 64x the measured round trip, launch overhead at 2-4 layer
   fusion. Report each term as a share of the step across the plane, and the two
   boundaries: B*C ~ 173K resident tokens where KV overtakes the weights, and
   B ~ 36 where the GDN state does. State every input's provenance.

2. THE VERIFY CURVE. Measure step time at every verification width k = 1..8 on
   the real dequant-GEMV, one card. ms(k) is not linear, so this curve - not a
   model - decides draft depth. Report ms(k) and the marginal cost of the k-th
   token.

3. THE ATTENTION PATH. No kernel exists, and it leads above the resident-token
   boundary. Measure a paged decode-attention kernel at 4K/16K/64K/256K, batch 1
   and 8, q8 and k8v4 (K 8-bit, V 4-bit), checksum-gated against a CPU
   reference. Report GB/s, % of wall and ms/step.

4. THE ROUND TRIP. Beat RCCL's ~19.5 us floor or report that you cannot:
   mapped-publish flag, p99 over >=1000 iterations, single-card loopback as the
   floor. Price TP=2 (128 all-reduces) and PP=2 (64 layer handoffs) at the
   measured number, at B=1 and B=8.

5. PREFILL. Price a 4K prompt: 2 x 26.9e9 params x 4096 FLOPs against measured
   achievable FLOPS on one and on two cards. Report TTFT. If it exceeds 1 s it
   is a first-class problem that S1 currently defers.

6. ARBITER. Nothing enters RECORD.md until the publisher's GSQ reference
   reproduces one dequantised block bit-for-bit, and a second kernel path with a
   different instruction mix lands within 5% of the first.

Constraints: llama.cpp excluded; no weights across PCIe; gfx1100-native; VRAM is
the budget; records never targets; no shape is the target.
```

## 9. Research assets, and what each is for

| Path | Use it for |
|---|---|
| `docs/research/2026-10-23/s3-amd-sinks.md` | the ISA survey: VOPD limits, I-cache, page size, memory hints, queue behaviour — 42 sourced URLs. Read it for the **ISA facts**, not for an issue model |
| `docs/research/2026-10-23/s4-topologies.md` | every two-card strategy priced per token: DP=2 = PP=2 = 2×, TP=2 = 1.47× — **read with `A4`'s correction in mind**: it prices aggregate throughput, not single-stream |
| `docs/research/2026-10-23/s5-donor-transfer.md` | **what to import from the donors and what is dead here**: the verify-width cost curve (tensorfold), layer split as the two-card default (strata), k8v4 KV, and the per-row applicability gate |
| `docs/plan/COST-LEDGER.md` | the shape surface: every per-token term as a function of `(B, C)`, the two boundaries, provenance, and the ranked measurement queue |
| `docs/research/2026-10-23/s2-tensorfold-freetoken.md` | TensorFold's measured MTP depth rule; why the codebooks are not the cost; the LUT-instead-of-SWAR observation |
| `docs/research/2026-10-23/s1-strata.md` | the GPU-publish/pre-launched-acquire handoff (~200 lines) and the measured **690:1 latency-to-bytes** ratio for a host-mediated boundary |
| `docs/research/2026-10-22/` | round one: languages, megakernels, bandwidth, KV, serving knobs, 2025–26 sweep, correctness. Absolute figures ~25 % low; relative findings stand |
| `bench/gfx1100-probes/` | `WALL`, `E1` (IQ3_S ablations, checksum-gated), `E2` (launch/fusion), and the sources `bw2.cpp`, `gemv.cpp`, `iq3s_valid.cpp`, `e2_launch.cpp` |
| `bench/manifests/` | `iq3s.json`, `iq3s-mtp.json` — every tensor, role, type, dimension |
| `docs/plan/TENSOR-INVENTORY.md` | the inventory: 72 `(role, type)` pairs, byte budget, GDN-only roles |
| `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | quality anchors: IQ3_S matches BF16 on AIME25 and LCB v6 |
| `https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2` | the DFlash2 drafter card: 2B BF16 block-diffusion, acceptance vs MTP (5.46 vs 5.02 GSM8K), throughput vs MTP (3.43× vs 2.59× at concurrency 1). Published on an **H200 with FA3** — the acceptance may transfer, the kernel stack will not |
| `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | **the publisher's own card**, which says the files are *standard GGUF* ("run unmodified in llama.cpp, Ollama, LM Studio"). GSQ = Gumbel-Softmax Quantization (learns grid assignments + group scales); RCO = Riemannian Constrained Optimization (picks the type per tensor under a size budget). Apache-2.0 |
| `tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S.rco-allocation.txt` (same HF repo) | **the publisher's per-tensor recipe** — the ground truth our 72 `(role, type)` pairs must match. Diff it against `bench/manifests/iq3s.json` (A7) |
| `https://github.com/IST-DASLab/GSQ` (Apache-2.0) | the legitimate source for the i-quant **grid tables** — it must implement the GGUF grids to quantise into them. E1 used a placeholder; 84 % of the file's bytes depend on the real ones. Never llama.cpp (`H3`) |

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
