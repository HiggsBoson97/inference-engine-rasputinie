# Handoff — rasputinie

One handoff, updated in place. A superseded section is deleted outright, never
tombstoned. This file is versioned with the repo and is the canonical copy.

**As of 2026-10-22, late evening.** Verify anything dated before relying on it.

---

## 1. What this is

A bespoke LLM inference engine for **exactly one machine and one model**:

- **Machine:** 2× AMD RX 7900 XT (gfx1100, RDNA3, 19.98 GiB each, no peer DMA),
  60 GiB DDR5, one node. Read wall **776 GB/s** per card.
- **Model:** `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, **IQ3_S**, with its MTP
  head. 851 tensors, 26.896 B params, 3.498 bpw, 64 layers (48 GatedDeltaNet +
  16 full attention).

The governing rule: **decode is bandwidth-bound, so bytes per token decide the
ceiling and the instruction set decides which bytes are affordable.** The
objective is bandwidth efficiency; tok/s is only the measurement of how well it
is spent.

## 2. Phase

**PLAN.** The measurement phase is complete — the decisions the plan needed are
made and recorded. Nothing is built until the plan gate returns `APPROVED`.
Next: SPEC → slices → test rig → gate.

## 3. Read order

1. [`docs/plan/REQUIREMENTS.md`](docs/plan/REQUIREMENTS.md) — what binds, what is
   open, what is settled and must not be reopened.
2. [`docs/00-facts.md`](docs/00-facts.md) — the measured physics. Every other
   number derives from it.
3. [`docs/03-native-kernel-inventory.md`](docs/03-native-kernel-inventory.md) —
   the kernel set and what "native" means.
4. [`docs/plan/RECORD.md`](docs/plan/RECORD.md) — the standing performance
   records, split measured vs estimated.
5. [`docs/plan/DECISION-packaging-iq3s.md`](docs/plan/DECISION-packaging-iq3s.md)
   and [`DESIGN-batch-regimes.md`](docs/plan/DESIGN-batch-regimes.md) — the two
   structural decisions.
6. ADRs 0002 (fork), 0003 (representation), 0004 (FP8 excluded);
   `decisions/0001` (llama.cpp excluded).
7. [`docs/research/2026-10-22/INDEX.md`](docs/research/2026-10-22/INDEX.md) —
   seven research lanes, with a correction note on their absolute figures.

## 4. State, and the first check

```bash
cd "$REPO" && git log --oneline -1 && git status --short
ssh rasputin 'cd ~/development/inference-engine-rasputinie && git log --oneline -1 && git status --short'
for c in /sys/class/drm/card[0-9]*; do [ -f "$c/device/mem_info_vram_used" ] && \
  printf "%s %sMiB\n" "$(readlink -f $c/device | sed 's|.*/||')" \
  "$(( $(cat $c/device/mem_info_vram_used)/1024/1024 ))"; done
```

Clean on both sides means matching tips and empty status. **Idle GPUs are
26 MiB (07:00.0), ~99 MiB (03:00.0), ~15 MiB (12:00.0, iGPU).** Anything far
above that is a leftover process — check `etime` before killing, and kill by PID.

| | |
|---|---|
| Repo | public, `HiggsBoson97/inference-engine-rasputinie` |
| Remotes | `origin` = bare repo **on the host**; `github` = public |
| Branch | `main`, single genesis commit plus measured-work commits |
| Host tree | clean clone, synced |
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
| Bench port | 30100 |
| Working area | plan and research in `../rasputinie-plan/`; probe sources in `bench/gfx1100-probes/` |

## 6. Standing traps

Each has an incident behind it. Do not treat any as folklore.

1. **HIP 0 is the display card.** A load test there caused a `ring gfx_0.0.0`
   timeout under gnome-shell. Always `HIP_VISIBLE_DEVICES=1` for single-card work.
2. **llama.cpp is excluded** — no baseline, no reference, no derived code, not
   installed. A research subagent cloned and built it anyway on 2026-10-22 and it
   had to be deleted. **Put this rule in every dispatch brief by name.**
3. **FP8 has no matrix instruction here.** Emulation measured 857 ms/token,
   2.7 % of the wall. Excluded as a format by ADR 0004, not merely demoted.
4. **619 GB/s was never the wall.** It came from a 1.6 ms kernel. The wall is
   **776 GB/s**. Any document quoting 619 or a ceiling derived from it is wrong.
5. **A launch larger than ~800 MiB per launch costs ~230 µs**, clock-independent,
   cause unproven. This is why fusion stops at 2–4 layers.
6. **Kernels must be checksum-gated.** A probe without a data-dependent checksum
   reported 1784 GB/s — 230 % of the wall — because the compiler hoisted the
   loads. A fast wrong kernel is worse than a slow right one.
7. **`pkill -f <pattern>` matches your own ssh command line** and kills the shell
   running it. Kill by PID.
8. **Runtime Triton compiles stall the first requests** — 1–2 s per kernel, and
   16.5 s for one. Warm-up is not optional.
9. **No `hf`, `git-lfs`, `uv`, `pipx` or `rocm-smi` on the host.** Downloads are
   HTTP + `xargs -P4 curl`; clocks are `amd-smi`.
10. **1249 MHz is the top memory step; idle is 96 MHz.** A ~3 ms wake-up after
    idle is deleted by `power_dpm_force_performance_level=high`. Never measure a
    single cold launch and call it the wall.
11. **The first check that a GGUF parse is right is arithmetic**: metadata +
    sum of tensor bytes must equal the file size exactly. It now closes to 0 B;
    if it stops doing so, the reader broke.

## 7. Open

Six items, each waiting on a named measurement — no invented answers.

1. How much of IQ3_S's **11.8 % sign-decode cost** is recoverable (the probe's
   sign path is already branch-free SWAR).
2. **In-situ rates** for IQ4_XS, IQ3_XXS, Q4_K, IQ2_S — only IQ3_S measured.
3. **MTP acceptance** on this box.
4. **Topology at the product level** — TP=2 vs two replicas; conflicting evidence.
5. The **~800 MiB footprint cliff**'s physical cause.
6. Whether **runtime adaptation** earns its complexity.

## 8. References

| Path | Why |
|---|---|
| `docs/plan/TENSOR-INVENTORY.md` | every tensor, role, type, dimension; the kernel set's ground truth |
| `bench/manifests/iq3s{,-mtp}.json` | machine-readable tensor manifests |
| `bench/gfx1100-probes/WALL-2026-10-22.md` | the wall, measured twice |
| `bench/gfx1100-probes/E1-IQ3S-2026-10-22.md` | IQ3_S in-situ rates and ablations |
| `bench/gfx1100-probes/E2-LAUNCH-2026-10-22.md` | launch cost, footprint cliff, fusion table |
| `bench/02_gguf_probe.py`, `bench/03_tensor_inventory.py` | regenerate the manifests |
| `docs/research/2026-10-22/` | seven lanes: languages, megakernels, bandwidth, KV, serving, 2025–26 sweep, correctness |
| `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | the model card: quality anchors and the MTP build |
