# rasputinie — a bespoke inference engine for one machine and one model

Two RX 7900 XT (gfx1100, RDNA3, 19.98 GiB each, PCIe 4.0 ×16, **no peer DMA**), 60 GiB DDR5, one
host. It serves exactly `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S — a **mixed-precision** file,
11 quant types, IQ3_S only 31 % of the bytes — with its MTP head. The objective is **bandwidth
efficiency**: achieved GB/s against the measured **776 GB/s** wall, per shape. Nothing ships without
a number. The engine is a fork of sglang, editable in place at `engine/` (`D-04`).

## Invariants

1. **`HIP_VISIBLE_DEVICES=1`.** HIP 0 is the display card; a load there wedged the desktop once.
2. **llama.cpp is excluded** — no baseline, no reference, no derived code, not installed (`D-02`).
3. **FP8 is excluded** as a format (`D-03`).
4. **aiter is excluded; force Triton explicitly.** The HIP default resolves to aiter, whose MFMA
   kernels cannot run on RDNA3 (`D-07`).
5. **Every probe is checksum-gated.** A check that cannot fail is not a check.
6. **Measure only kernels ≥ 3 ms, clocks pinned.** A 1.6 ms kernel once reported 619 GB/s.
7. **`GRIDMUL=8`** for the E1 harness → `grid=336×256`. A stray `GRIDMUL=4` reports 436, not 521.
8. **No shape is the target.** Report `(B, C)` with every rate; the variable is resident tokens
   `B × C` (`D-06`).
9. **Delete, do not guard.** A `try/except` around a dead dependency keeps it alive and hides the
   next failure.
10. **Absent hardware is not a software gap.** Never emulate MFMA or FP8; do write the software the
    hardware supports and the tree lacks.
11. **No weights across PCIe.** Tensor parallelism shards them permanently; what crosses is partial
    sums (`D-09`).
12. **VRAM is the budget** — model, drafter and KV inside 19.98 GiB per card. No host spill.
13. **Kill by PID**, never `pkill -f` — it matches your own ssh command line.
14. **Reproduce before recording**, including anything a subagent reports.
15. **Validate the instrument with a canary.** A zero from a profiler is a broken counter until
    proved otherwise.

## Commands

```bash
ssh rasputin                                                  # port 8374, mattia-bosi, sudo -n
~/venv/sglang-fork/bin/python                                 # the engine; sglang editable from engine/python
HIP_VISIBLE_DEVICES=1 GRIDMUL=8 BLK=256 ~/e1/iq3s_valid <manifest> <gguf> 1 4
python3 ~/e1/mkmanifest.py <gguf> 30 <out>                    # must print "0 mismatches"
echo high | sudo -n tee /sys/class/drm/card0/device/power_dpm_force_performance_level
```

The frontier unit and its proof command live in `docs/status.md`.

## Map

| where | what |
|---|---|
| `docs/index.md` | the reachable map — start here |
| `docs/facts/` | **measured**: memory, compute, quant, engine, decoder, twocards, traps |
| `docs/decisions/` | **decided**: `D-01`…`D-09`, immutable, each with its rejected alternative |
| `docs/plan/` | `REQUIREMENTS`, `RECORD`, `COST-LEDGER`, `DEVELOPMENT-PLAN`, `TENSOR-INVENTORY` |
| `docs/status.md` | the frontier unit, its three fields, its last state |
| `_archive/2026-10-24/` | the pre-migration tree, provenance only — never cite it as current |
