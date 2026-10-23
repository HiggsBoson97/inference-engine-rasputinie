# Handoff — rasputinie

One handoff, updated in place. A superseded section is deleted outright, never
tombstoned. This file is versioned with the repo and is the canonical copy.

**As of 2026-10-23 17:40 CEST.** Verify anything dated before relying on it.

---

## 1. What this is

A bespoke LLM inference engine for **exactly one machine and one model**:

- **Machine:** 2× AMD RX 7900 XT (gfx1100, RDNA3, 19.98 GiB each, no peer DMA),
  60 GiB DDR5, one node. Read wall **776 GB/s** per card.
- **Model:** `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, **IQ3_S**, with its MTP head.
  851 tensors, 26.896 B params, 3.498 bpw, 64 layers (48 GatedDeltaNet + 16 full
  attention), 10.57 GiB read per step. **11 quant types, not one** — IQ3_S is 31 % of
  the bytes; IQ4_XS 25.7 %, IQ3_XXS 17.6 %, Q4_K 13.5 %, IQ2_S 6.8 %.
- **GSQ-RCO is a method, not a container format.** The publisher's card says the files are
  *standard GGUF that run unmodified in llama.cpp*; GSQ learns grid assignments and group
  scales, RCO picks the quant type per tensor under a size budget. So there is no custom
  block layout to parse — what is file-specific is the **recipe**, which the publisher ships
  as `tensor-allocation/*.rco-allocation.txt`.

**Three corrected rules, each measured, each previously recorded wrong:**

1. **Decode is not bandwidth-bound and not instruction-bound.** The kernel at 521 GB/s uses
   48.6 % of the issue slots and gains **+23.7 %** when memory is freed (521 → 645 GB/s), so
   it sits at a corner: memory costs 16.0 % of the wall, instructions 16.9 %. See
   `docs/plan/DESIGN-instruction-budget.md`, evidence `H-1.md`/`H-2.md`.
2. **No shape is the target.** The owner has never named a session count or a context length.
   Every per-token cost is a function of `(B, C)`; what governs the plan is the **surface**
   (`docs/plan/COST-LEDGER.md`), not a point on it.
3. **The engine is the artifact.** Every rate on record before 2026-10-23 came from scratch
   harnesses over 30 concatenated tensors. They exercise no layer sequence, dispatch, GDN
   recurrence, KV path or second card. Work now happens **in `engine/`**.

## 2. Phase — Discovery, and it is not finished

No SPEC, no slices, no rig, no plan gate. That is deliberate: the owner's position is that a
SPEC written before the engine teaches us anything would encode five invented premises. The
SPEC is Discovery's output, not its input.

**Closed by measurement (2026-10-23):**

| | finding |
|---|---|
| **A1** | Not issue-bound. 48.6 % occupancy; +23.7 % when memory is freed; the `n = 1.57` model was a 32× unit error, back-solved from the measurement. `H-1`, `H-2` |
| **A2** | The 752 GB/s assumption for non-IQ3_S bytes was **30–40 % too high**. Whole-file decode **56.0 → 47.1 tok/s**. Load floors 637–660 GB/s with no layout penalty; dot rates IQ4_XS 502.1, Q4_K 452.2; IQ3_XXS and IQ2_S **could not be gated**. `H-4` |
| **A3** | MTP head costs **7.445 GB per 7-token step = 65.6 % of a weight sweep** (it re-reads 0.666 GiB of LM head per draft). DFlash2 4.5 bpw costs 15.8 % and is flat in block size. DFLASH is already first-class in the vendored tree, Triton, with an explicit ROCm fallback. `H-5` |
| **H-9** | `ms(k) = 1.764 + 0.588k` on 1.0700 GiB of IQ3_S. **The marginal verification cost is flat at 0.51–0.72 ms with no knee**, so verification does not limit depth — **the drafter does.** Whole-file budget per drafted token ≈ 3.87 ms: MTP uses 53 %, DFlash2 BF16 32 %, DFlash2 4.5 bpw 13 % |
| **H-10** | **The engine cannot serve this file on ROCm at all.** Four ordered blockers, plus the deletion map. See §2.1 |

**Open:** `H-3` (what costs the 521→645 gap — latency/MLP), `H-6` (two-card round trip),
`H-7` (grid tables + the publisher's recipe — a *correctness* blocker), `H-8` (the attention
path has no kernel and no rate), `H-11` (engine bring-up — **interrupted mid-task**, see §4).
Plus one question not yet a row: prefill/TTFT (~1.6–3.2 s for 4K, deferred by `S1`).

### 2.1 The engine facts that decide the next work

- **Four ordered blockers, all verified:** (1) `aiter` imported at module scope under
  `if _is_hip` and not installed → `model_config` will not import; (2) `ModelConfig`
  hard-refuses GGUF on ROCm (`configs/model_config.py:2128-2133`); (3) on HIP
  `layers/quantization/gguf.py` has no import branch, so the five `ggml_*` names are
  undefined; (4) the ops are absent from the ROCm wheel — 0 of 47 `sgl_kernel` ops are ggml,
  `nm common_ops.so` = 0 ggml symbols, `setup_rocm.py` lists no gguf source.
- **At batch 1 the engine wants MMVQ** (`ggml_mul_mat_vec_a8`), a fused int8 GEMV. The
  dequant→bf16 path is only reached at ≥ 9 tokens. **So the kernel the engine needs is the
  one E1 already built** — it just isn't inside the engine yet.
- **`__dp4a` is undeclared for gfx1100** in this toolchain (verified with a two-line test:
  `error: use of undeclared identifier '__dp4a'`), so `vecdotq.cuh`'s ~45 call sites cannot
  compile as-is. The C++ route is closed anyway — see hard rule 4 in `AGENTS.md`: **aiter is
  dead here, Triton is the only working backend.**
- **Deletion surface, measured over 8,338 files:** **5,363 dead on any path (1.48 M LOC)**,
  27 CUDA-gated, 1,422 lazy-only, **1,070 live (455 k LOC)**; 456 C++ files execute nothing
  but are the source of the missing ops.

## 3. Read order

1. [`docs/plan/COST-LEDGER.md`](docs/plan/COST-LEDGER.md) — **the frame**: every per-token
   term as a function of `(B, C)`, the two boundaries, term provenance, and the ranked
   measurement queue. Read before proposing any optimisation.
2. [`AGENTS.md`](AGENTS.md) — the four hard rules (llama.cpp, FP8, the loop, **aiter/Triton**)
   and the eleven principles. Short, and it is the constitution.
3. [`docs/plan/DESIGN-instruction-budget.md`](docs/plan/DESIGN-instruction-budget.md) — the
   corrected two-bound model: 521 achieved, 645 memory-free, 776 wall; why the old `n` model
   was void; the five ISA facts; the three falsifiers.
4. [`docs/research/2026-10-23/s5-donor-transfer.md`](docs/research/2026-10-23/s5-donor-transfer.md)
   — what to import from the six donor engines and what is dead on this card, per row with
   the artifact read and the gate it must pass.
5. [`docs/plan/REQUIREMENTS.md`](docs/plan/REQUIREMENTS.md) — hard constraints, settled
   rules, and the open items each naming its measurement.
6. [`docs/plan/RECORD.md`](docs/plan/RECORD.md) — every measured rate, with provenance and
   the "superseded" rows marked.
7. [`docs/00-facts.md`](docs/00-facts.md) — measured physics; every number derives from it.
8. [`docs/03-native-kernel-inventory.md`](docs/03-native-kernel-inventory.md) — the kernel
   set, 72 `(role, type)` pairs.
9. ADRs 0002 (fork), 0003 (representation), 0004 (FP8 excluded); `decisions/0001`
   (llama.cpp excluded).
10. [`docs/research/README.md`](docs/research/README.md) — the index, with the correction
    banner: `s3`'s instruction-issue model is **refuted**, its ISA facts stand.

## 4. State, and the first check

```bash
cd "$REPO" && git log --oneline -1 && git status --short
ssh rasputin 'cd ~/development/inference-engine-rasputinie && \
  git branch --show-current && git log --oneline -1 && git status --short'
```

Clean on both sides means matching `main` tips and empty status.

**As of the write of this file:** both `main` at `35939842`, both trees clean.
**The host is checked out on branch `engine/bringup` at `bb39bf0f`, not `main`** — two
commits that are **unrun and unverified**:

| sha | what |
|---|---|
| `79157797` | `engine: remove aiter from the reachable path on gfx11` — 4 files, a helper in `utils/common.py` |
| `bb39bf0f` | `engine: WIP, UNVERIFIED — honest ROCm GGUF gate` — probes for `ggml_mul_mat_vec_a8` instead of deleting the refusal |

The branch exists because the probe writing it was interrupted before it constructed a
`ModelConfig`. **Review or delete it before building on it.** `main` is untouched by it.

**The host's git identity is repo-local and was unset** — a commit there fails with
`Author identity unknown`. It is now set in that clone to `HiggsBoson97
<mattia.6131@gmail.com>`, matching the existing history. A fresh clone will need it again.

| | |
|---|---|
| Repo | public, `HiggsBoson97/inference-engine-rasputinie` |
| Remotes | `origin` = bare repo **on the host**; `github` = public. Both `main`s pushed each time |
| Branch protection | available (repo is public), **not enabled** — enable before the first dispatch |
| Staging | `../rasputinie-plan/` — the repo copies are authoritative |
| Environment | venv `~/venv/sglang-fork`, wheel `~/wheels/sglang_kernel-0.4.10-…whl`, 69 GB models — outside the repo |

Idle GPUs are **26 MiB (07:00.0), ~88 MiB (03:00.0), 15 MiB (12:00.0 iGPU)**. Far above that
is a leftover process — check `etime` before killing, and kill by PID.

## 5. Baseline, exactly

| Thing | Value |
|---|---|
| Repo (Mac) | `/Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie` |
| Repo (host) | `/home/mattia-bosi/development/inference-engine-rasputinie` |
| SSH | `ssh rasputin`, port **8374**, user `mattia-bosi`, `sudo -n` (NOPASSWD) |
| Engine | `engine/` = vendored sglang `d0a7565`, **installed editable from `engine/python`** — the vendored tree *is* the running engine, no install step |
| Python | `~/venv/sglang-fork/bin/python` — 3.12.15, torch 2.14.1+rocm7.2, HIP 7.2.53211, sglang 0.5.9, sgl_kernel 0.4.10, triton 3.8.0 + triton-rocm 3.8.0 |
| ROCm | 10.1 under `/opt/rocm/core-10.1`; hipcc 7.16 / clang 24 |
| GPU mapping | **HIP 0 = amd-smi 0 = sysfs card1 = PCI 03:00.0 = display**; HIP 1 = card0 = PCI **07:00.0** = headless |
| Use this card | **`HIP_VISIBLE_DEVICES=1` → PCI 07:00.0** |
| Models | IQ3_S 11.77 GB, IQ3_S-mtp 12.12 GB, UD-Q4_K_XL 17.56 GB (reference only), mmproj 931 MB (out of scope) |
| Donor checkouts | `development/_research/{strata,tensorfold,freetoken,deepinfer,sglang,exllamav3}` — read-only |
| E1 harness | `~/e1/iq3s_valid <manifest> <gguf> <NV> <rp>`, **`GRIDMUL=8 BLK=256` → `grid=336×256`**; `GRIDMUL=4` halves the grid and reports 436 instead of 521 |
| Manifest | `python3 ~/e1/mkmanifest.py <gguf> 30 <out>`; a 1-tensor manifest must be cut by hand (`awk 'NR==2'`) — the generator asserts ≥1 GiB and an empty manifest segfaults |
| Profiler | `rocprofv3` 1.4.1 at `/opt/rocm/bin`. Reads only `SQ_BUSY_CYCLES`, `SQ_WAVES`, `Wavefronts`; every instruction and `GL2C_*` counter is 0 (trap 16) |
| Memory terms | **VRAM** = the card's GDDR6, where the 776 GB/s wall is measured. **L2** = 96 MB on-die. **Host RAM never appears in a timed path** — no spill, no managed allocation |
| Two-card link | PCIe 4.0 ×16, **no peer DMA**. RCCL all-reduce **20.65 µs** amortised at 10 KiB, **~19.5 µs floor** not tunable by `NCCL_*`; `pynccl` ~2× cheaper than `torch.distributed` |
| **Key measured rates** | IQ3_S `full` **521.3** (67 %) · memory-free ceiling **645.1** (83 %) · wall **776** · `read` 743.7 · IQ4_XS 502.1 · Q4_K 452.2 · whole file **47.1 tok/s** (estimate) |
| **Verify curve** | `ms(k) = 1.764 + 0.588k` per 1.07 GiB slice; marginal flat 0.567 ms; whole-file marginal ≈ 5.40 ms/token |
| **Drafter cost** | MTP 2.04 ms per draft token (53 % of the 3.87 ms budget) · DFlash2 BF16 1.25 (32 %) · DFlash2 4.5 bpw **0.49 (13 %)** |

## 6. Standing traps

Each has an incident behind it. Do not treat any as folklore.

1. **HIP 0 is the display card.** A load there caused a `ring gfx_0.0.0` timeout under
   gnome-shell. Always `HIP_VISIBLE_DEVICES=1` for single-card work.
2. **llama.cpp is excluded** — no baseline, no reference, no derived code, not installed. A
   research subagent cloned and built it anyway and it had to be deleted. Name this rule in
   every brief.
3. **FP8 is excluded as a format** (ADR 0004). The 857 ms/token run is evidence for the
   exclusion, never a baseline.
4. **619 GB/s was never the wall** — a 1.6 ms kernel artefact. The wall is **776**.
5. ~~**Below ~4 bits/weight the format is instruction-bound.**~~ **Refuted 2026-10-23** —
   48.6 % occupancy, +23.7 % when memory is freed. See `H-1`. Do not re-derive this from
   `DESIGN-instruction-budget.md` §1; that section is void.
6. **Kernels must be checksum-gated.** An ungated probe reported 1784 GB/s — 230 % of the
   wall — because the compiler hoisted the loads.
7. **The ~800 MiB "footprint cliff" is probably the clock ramp.** Pin clocks before designing
   around it.
8. **`pkill -f <pattern>` matches your own ssh command line** and kills the shell running it.
   Kill by PID.
9. **Runtime Triton compiles stall the first requests** — 1–2 s per kernel. Warm-up is not
   optional.
10. **No `hf`, `git-lfs`, `uv`, `pipx` or `rocm-smi` on the host.** Downloads are HTTP +
    `xargs -P4 curl`; clocks are `amd-smi`.
11. **1249 MHz is the top memory step; idle is 96 MHz.** A ~3 ms wake-up after idle
    disappears with `power_dpm_force_performance_level=high`. Never measure one cold launch.
12. **4 KB pages only.** No HIP/ROCm API or env var gives 2 MB pages.
13. **`v_dot4_i32_i8` cannot dual-issue**, and gfx11 has no `cp.async` (MUBUF has no LDS
    bit). Two optimisation families are dead by design.
14. **Dispatch must be type-sorted.** I-cache is 32 KB per WGP shared by both CUs;
    interleaving quant types in one CU costs −10 %, sorting costs 0.
15. **A GGUF parse is right when arithmetic says so**: metadata + Σ tensor bytes must equal
    the file size exactly. It closes to 0 B.
16. **`rocprofv3` reads no instruction counter on this host.** Twenty-one counters tested
    against a canary with a countable instruction stream: only `SQ_BUSY_CYCLES`, `SQ_WAVES`,
    `Wavefronts` populate. A zero from this profiler is not a measurement.
17. **The E1 harness needs `GRIDMUL=8`** — `grid = 42 × GRIDMUL = 336`.
18. **A kernel shorter than ~3 ms cannot be measured**, whatever the footprint. `rp=4` on 38 MB
    gave 690.3 and 366.7 GB/s on identical input.
19. **An unmeasured shape is not a target.** A whole analysis was built around `8 × 32 K`, a
    shape the owner never named. The load-bearing variable is resident tokens `B × C`.
20. **"The rest of the file runs at 752 GB/s" was an assumption and it was 30–40 % too high.**
    Never let an unmeasured type contribute at a benchmark's rate.
21. **IQ3_XXS's measured rows looked *fast* (581–592 GB/s) and were wrong** — fp64 error 1.43,
    checksum mismatched. The wrong-but-fast signature is the most dangerous output here.
22. **aiter's ASM/CK path is dead on gfx1100; its Triton subtree is not.** The sharpest fact:
    **aiter's two allow-lists disagree.** `aiter/jit/core.py:456-477` accepts `gfx1100` by name,
    while `csrc/cpp_itfs/utils.py:131-147` — the path every hand-written C++/CK per-shape module
    goes through — accepts only CDNA, and its `DEFAULT_GPU_ARCH` is literally `amdgpu-arch`'s
    output, which on rasputin is `gfx1100`, so the assert fires immediately. The MFMA paged-decode
    kernel would also fail on its own (`needs target feature mai-insts`) because MFMA does not
    exist on RDNA3. But aiter's **Triton** kernels do compile here — `unified_attention` with
    explicit `arch.is_rdna` tuning, `flash_attn_triton_amd` with gfx1100 in `RDNA_ARCHS` — so the
    working half of aiter *is* Triton. **Depend on Triton directly, never on aiter, and force the
    attention backend explicitly because the HIP default is aiter.** `19-aiter-gfx1100.md` also
    records that the crash first attributed to aiter was sglang's own
    `reset_verify_attn_plan_cache` hiding a missing install — a broken call path, not an
    architecture verdict. (2026-10-23)
23. **A probe's recommendation is not evidence.** `H-10` proposed adding `gguf_kernel.cu` to
    `setup_rocm.py`; it cannot compile, because `__dp4a` is undeclared for gfx1100. Check a
    recommendation's load-bearing fact before acting on it.
24. **The host clone needs a git identity** — it was unset and commits failed with
    `Author identity unknown`. Set `user.name`/`user.email` per clone.

## 7. Open — each waits on a named measurement

| # | Question | Settled by |
|---|---|---|
| **A1b** | **What costs the 521 → 645 GB/s gap?** Hypothesis: latency/MLP — 43 % of the body is `s_waitcnt`+`s_delay_alu`, 21 global loads per 32 weights against `k_read`'s 5 | `H-3`: a variant with wider outstanding loads per wave, accepting `full` ≥ 600 GB/s at 1.0700 GiB, gate VALID |
| A4 | Topology: TP=2, DP=2, or pipeline | `H-6`: the two-card round trip. TP=2 is **1.48× single-stream** at 20.65 µs, 1.93× at 2 µs; DP=2 is **1.00×** single-stream. Beat RCCL's 19.5 µs floor or report that you cannot |
| A5 | The ~800 MiB anomaly's true cause | Clock-pinned re-run |
| A6 | Does runtime adaptation earn its complexity | An ablation build per mechanism |
| A7 | **What defines the i-quant grids, and is our inventory the publisher's recipe?** | Grids from `IST-DASLab/GSQ` (Apache-2.0, never llama.cpp); diff our 72 `(role, type)` pairs against the publisher's `rco-allocation.txt`. **84 % of the file's bytes depend on this** |
| A8 | **What does the attention path cost?** 16 full-attention layers over paged KV — no kernel, no rate | A paged decode-attention kernel over a context ladder, batch 1+, q8 and k8v4, checksum-gated |
| **A9** | **Can the engine serve this file on this card?** (bring-up) | `H-11`: the four blockers removed, a Triton MMVQ for IQ3_S, and a coherent 32-token generation — then TTFT and decode tok/s |

## 8. The next unit of work

**Engine bring-up.** Not another measurement, not the SPEC. The sequence, in order, with the
blocker each step removes:

1. **Review or discard `engine/bringup`** (`79157797`, `bb39bf0f`) — both unrun. Decide whether
   the aiter removal and the honest GGUF gate are the right shape before building on them.
2. **Make `model_config` importable** — the aiter module-scope import is blocker 1; the branch
   has a candidate fix, unverified.
3. **Force Triton** for attention. The HIP default routes to aiter's MFMA path, which cannot
   run on RDNA3; a path that starts can still land there. Find where the default is chosen
   (`arg_groups/model_override_base.py` ~`:326-327` per `14-sglang-gfx1100.md:19`).
4. **Decide the GGUF kernel question honestly.** The engine needs `ggml_mul_mat_vec_a8` for
   13 types; the ROCm wheel has none of them; the C++ source cannot compile (`__dp4a`
   undeclared) and the C++ route is closed anyway. So: **a Triton MMVQ**, IQ3_S first,
   correctness-gated against the file. Then the other types in `RECORD.md` order.
   **The attention path is already covered** — sglang ships its own
   `layers/attention/triton_backend.py` (118 KB), independent of aiter, so nothing needs
   writing there, only selecting. The GGUF GEMV is the only missing kernel.
5. **Then, and only then, take the baseline**: TTFT, decode tok/s, VRAM, and the shape printed
   with it. That number is the "before" for everything else.

**On "native" and Triton.** Triton's ROCm backend (`triton/backends/amd`) compiles Triton IR
through LLVM's AMDGPU backend to gfx1100 HSACO and launches via the HIP runtime — so it emits
native gfx1100 ISA, and `REQUIREMENTS.md` S4's "native hot path" is satisfied in the
code-generation sense. What S4 cannot assume is *control*: Triton 3.8.0 exposes `tl.dot` and
`tl.inline_asm_elementwise` but not VGPR allocation, instruction scheduling, delay-slot
placement or load ordering — and the 521 GB/s kernel depends on all four (193 words per 32
weights, 31 `s_delay_alu`, hand-placed `v_dot4_i32_i8`). **The open question is therefore not
"Triton or native" but "can Triton's scheduler reach 521 GB/s on this GEMV", which is
unmeasured.** A third option exists and is worth reading before concluding: ADR 0001 records
that **TensorFold ships hand-written HIP GEMV kernels for gfx1100 that already serve this 27B**
(`docs/adr/0001-engine-composition.md:26`) — that is the ceiling to beat or to borrow.

```
Prompt for the next session, verbatim:

Read HANDOFF.md, then AGENTS.md. Get the vendored sglang to serve
Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf on the headless card (PCI 07:00.0) and take the
engine baseline: TTFT and decode tok/s, with the shape printed.

Four blockers are already measured, in order: model_config will not import
because aiter is imported at module scope under _is_hip; ModelConfig refuses GGUF
on ROCm at configs/model_config.py:2128-2133; gguf.py has no HIP import branch so
the five ggml_* names are undefined; and the ROCm sgl_kernel wheel has none of the
ggml ops (0 of 47).

aiter is excluded on this card - do not port CK. Triton is the only working
backend. The C++ gguf kernel cannot compile (__dp4a is undeclared for gfx1100) and
that route is closed.

engine/bringup holds two UNRUN commits from an interrupted probe: review or
discard them before building on them. main is clean at 35939842.

Deliverable: a coherent 32-token greedy generation and its tok/s, or the exact
next blocker with file:line. Not a plan, not a measurement of a proxy. Every claim
quoted from a command; no number enters RECORD.md ungated.
```

## 9. Research assets, and what each is for

| Path | Use it for |
|---|---|
| `docs/plan/COST-LEDGER.md` | the shape surface: every per-token term as a function of `(B, C)`, the two boundaries, provenance, the ranked queue, and what the verify curve changed |
| `docs/research/2026-10-23/s5-donor-transfer.md` | what to import from the donors and what is dead here: TensorFold's verify-width cost curve, strata's layer split + `k8v4`, and the per-row applicability gate |
| `docs/research/2026-10-23/s3-amd-sinks.md` | the ISA survey — 42 sourced URLs. **ISA facts only; the issue model is refuted** |
| `docs/research/2026-10-23/s4-topologies.md` | every two-card strategy priced per token — but it prices *aggregate* throughput, not single-stream |
| `docs/research/2026-10-23/s1-strata.md` | the GPU-publish/pre-launched-acquire handoff and the **690:1 latency-to-bytes** ratio for a host-mediated boundary |
| `docs/research/2026-10-22/04-kv-cache.md` | the resident-token crossover (173K bf16), q8_0's 34 KiB/position, and the capacity arithmetic |
| `docs/research/18-tp-roccl.md` | the measured RCCL all-reduce table on this box; the ~19.5 µs floor |
| `docs/research/14-sglang-gfx1100.md` | the engine's HIP defaults: aiter everywhere, flashinfer excluded by construction, the trap |
| `bench/gfx1100-probes/` | `WALL`, `E1` (IQ3_S ablations, gated), `E2` (launch/fusion), and the sources including `iq3s_valid.cpp` — **the kernel the engine needs** |
| `bench/manifests/` | `iq3s.json`, `iq3s-mtp.json` — every tensor, role, type, dimension |
| `docs/agents/production/hypothesis-ledger.md` | H-1 … H-11 with verdicts, tests, and provenance |
| `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | the publisher's card: **standard GGUF**, the Gumbel-Softmax/RCO method, Apache-2.0 |
| `…/tree/main/tensor-allocation` | the publisher's per-tensor recipe — the ground truth for our 72 pairs (`A7`) |
| `https://github.com/IST-DASLab/GSQ` | the grid tables, Apache-2.0 — never llama.cpp |
| `https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2` | the drafter: 2B BF16 block-diffusion, MIT code / Apache-2.0 weights, published on an H200 with FA3 |

## 10. Skills for the next session, in order

1. `fullstack-software-development` → `references/discovery.md` and `references/orchestrator.md`
   (Discovery is the open stage; v10 has no `plan-guide.md`).
2. `get-deeper` before any design document. The unit check that killed the `n` model and the
   differential experiment that replaced it both came from that move.
3. `mattpocock-research` for any lane reading primary sources.
4. `mattpocock-diagnosing-bugs` for the engine bring-up — it is a diagnosis, not a build.
5. `adversarial-judge-stress` for the plan gate, dispatched fresh, only when the ledger closes.
6. `mattpocock-handoff` for the next handoff.

## 11. On workers, and on drift

Subagents have produced real results in this project — the ISA census, the per-type sweep,
the drafter survey, the deletion map — and they have also cost it. The incidents, all caught
by an independent check: one built llama.cpp; one reported 230 % of the wall; one reintroduced
a disproven rule; one recorded an open question as settled; one wrote its own disassembler and
re-derived a census another probe already had, instead of stopping when its named counter read
zero; one reported a 436 GB/s baseline that was an inherited `GRIDMUL=4`; one recommended a
build step that cannot compile.

**The owner's verdict at the end of 2026-10-23, and it should be respected:** delegation made
the session *drift* — too much work on proxies, too little on the engine. The rule that follows:

- **Prefer doing the work directly.** Delegate a lane only when it is genuinely parallel and
  the premise is already checked.
- **Never dispatch a brief whose load-bearing premise you have not verified against the repo's
  own research.** Two briefs in one session carried premises the docs already contradicted
  (aiter as a missing dependency; the C++ kernel as portable by wiring).
- **Every subagent number needs an independent reproduction before it enters the record** —
  and the reproduction is cheap: one command, one grep, one file count.
- The user reviews decisions, never code. That makes the *evidence* the product of the
  session; a report whose claims cannot be re-checked in one command is not evidence.
