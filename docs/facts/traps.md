# facts — traps

The residue that does not belong in `AGENTS.md`. Fifteen mechanically-checkable rules live there as
invariants; everything below is a trap whose *reason* matters but which does not need to be in every
turn's context. Each carries the incident that proved it.

| # | trap | the incident | the rule it leaves |
|---|---|---|---|
| 1 | **A short kernel reports a fake wall** | a 1.6 ms read kernel reported 619 GB/s; the real wall is 776 at ≥3 ms | measure only kernels longer than ~3 ms, and never derive a ceiling from a shorter one |
| 2 | **The ~800 MiB "cliff" is probably the clock ramp** | E2 named a cliff; 800 MiB ÷ 750 GB/s = 1.12 ms, so a fixed ~230 µs stops being amortised | pin clocks (`power_dpm_force_performance_level=high`) and re-run before designing around it |
| 3 | **`rocprofv3` reads no instruction counter on this host** | 21 counters against a canary with 200 000 known instructions: only `SQ_BUSY_CYCLES`, `SQ_WAVES`, `Wavefronts` populate. `SQ_INSTS_VALU` read **0**. SQTT/ATT gives `"code": null`. `pmc-check` fails on the iGPU for *every* counter, so it is no tiebreaker | a zero from a profiler is a broken counter until a canary proves otherwise; use the differential memory-pressure method instead |
| 4 | **Runtime Triton compiles stall the first requests** | 1–2 s per kernel, 16.5 s for one | warm up before timing anything |
| 5 | **The host has no `hf`, `git-lfs`, `uv`, `pipx` or `rocm-smi`** | each was tried and absent | downloads are HTTP + `xargs -P4 curl`; clocks are `amd-smi` |
| 6 | **1249 MHz is the top memory step; idle is 96 MHz** | a cold launch cost ~3 ms of wake-up | never measure one cold launch; the wake-up disappears with the performance level pinned |
| 7 | **4 KB pages only** | `amdgpu` maps 4 KB PTEs; no HIP/ROCm API or env var gives 2 MB | huge-page theories are arithmetically excluded — 3.15 M pages/launch would be 73 ps/page |
| 8 | **`v_dot4_i32_i8` is half rate and cannot dual-issue; gfx11 has no `cp.async`** | measured 2.08 slots/instr; `MUBUF` has no LDS bit | check the ISA before believing a rate table — two optimisation families are dead by design |
| 9 | **Dispatch must be type-sorted** | I-cache is 32 KB per WGP shared by both CUs; interleaving quant types cost **−10 %** | sort by type; it costs 0 |
| 10 | **A GGUF parse is right when arithmetic says so** | metadata + Σ tensor bytes = file size, closes to **0 B** | a parse that does not close is wrong, whatever it printed |
| 11 | **An unmeasured type must never contribute at a benchmark's rate** | 752 GB/s assumed for 69 % of the bytes; when two types were measured the whole-file estimate fell 56.0 → **47.1 tok/s** | an unmeasured term is an assumption with an error bar, not a number |
| 12 | **Wrong-but-fast is the signature to fear** | IQ3_XXS ran 581–592 GB/s with fp64 error **1.43** and a checksum mismatch — faster than IQ3_S's gated 521 | a plausible number from a kernel that cannot reproduce the file is the most dangerous output here |
| 13 | **A probe's recommendation is not evidence** | H-10 proposed adding `gguf_kernel.cu` to `setup_rocm.py`; it cannot compile — `__dp4a` is undeclared for gfx1100 | check the load-bearing fact of any recommendation before acting on it |
| 14 | **The host clone needs a git identity** | commits failed with `Author identity unknown`; the identity was never set | set `user.name`/`user.email` per clone (now `HiggsBoson97 <mattia.6131@gmail.com>`, repo-local) |
| 15 | **"DRAM" in an older report means VRAM** | a whole analysis was read as if it had measured a host-memory spill | say VRAM / L2 / host RAM explicitly; no timed path here has ever touched host RAM |
| 16 | **A stray `GRIDMUL` fakes a baseline** | a probe reported `full` = 436.5 GB/s and called the baseline unstable; it had inherited `GRIDMUL=4` (half grid) | read the harness's printed config line before believing any % of wall |
| 17 | **A probe will keep working after its instrument is dead** | one wrote its own disassembler and re-derived a census another probe already had | briefs must include a stop-and-report rule for a zero reading |
| 18 | **A hierarchy of digests drifts** | 619 → "the wall"; `n = 1.57` back-fitted; 752 assumed. Three consecutive abstractions, three errors | a number lives one layer from its instrument, or it is a rumour |
