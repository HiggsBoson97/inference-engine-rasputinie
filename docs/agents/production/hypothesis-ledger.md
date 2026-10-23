# Hypothesis ledger — rasputinie

Append-only. A row's **test is written before the run**. Verdicts are `confirmed`,
`refuted`, or the transient `open`; a row that cannot decide is a broken test, not a
third verdict. Evidence reports live in `run-artifacts/hypotheses/H-<n>.md`.

A rewritten hypothesis keeps its ID, with the new row line beneath it; the old line stays
visible.

| ID | Question | Hypothesis | Test (commands + scratch path) | Verdict | Evidence | Decision |
|---|---|---|---|---|---|---|
| H-1 | Is the IQ3_S dequant-GEMV at 516.4 GB/s limited by VALU issue slots? | On `k_full` at 1.0700 GiB, VALU issue-slot occupancy is ≥ 85 % of the SIMD32 slot rate, and `SQ_INST_CYCLES_VALU / SQ_INSTS_VALU ≥ 1.5` (the half-rate dp4a of trap 13). | `rocprofv3 --pmc <counters> --kernel-trace`, kernel-filtered, on `~/e1/iq3s_valid <manifest> <IQ3_S.gguf> 1 4`; scratch `~/a1/`; clocks pinned to `high`; second capture required to agree within 2 %. | **test void — the instrument does not exist here** | `run-artifacts/hypotheses/H-1.md` §2 | A1 |
| H-1′ | same | same | **Rewritten after the instrument failure.** If the kernel is issue-saturated, relieving memory pressure cannot speed it up. Run the same gated kernel twice: VRAM arm (30 tensors, 1.0700 GiB, `rp=4`) and L2-resident arm (1 tensor, 38 297 600 B, `rp=64` so the timed region is ≥ 3 ms). Compare `full` GB/s; convert to issue slots with the H-2 census (193 instr per 32 weights per lane). Scratch `~/a1/`. | **refuted** — `full` gains **+23.7 %** (521.3 → 645.1 GB/s) when VRAM pressure is removed; occupancy is **48.6 %** of the slot rate, not ≥ 85 %; the body would need 338 instr per 32 weights to saturate issue at this rate and has 193 | `run-artifacts/hypotheses/H-1.md` §5–8 | A1 closed; memory-side work returns, bounded at 645 GB/s |
| H-2 | Is the DESIGN note's `n` an issue-slot count or a lane-instruction count? | The note divides a **lane**-instruction count by a **wave**-instruction slot rate (32:1 in this ISA), so `n = 1.57` does not mean "50 % over an instruction budget". | `llvm-objdump -d --mcpu=gfx1100` over the emitted `k_full<0,0>`, counting **all** wave instructions per 32 weights per lane by role (index, gather, sign, scale, dot); reconcile against H-1. Scratch `~/a2/`. | **confirmed, with a refinement** — the body is 193 words per 32 weights (recounted independently), so one slot buys 5.31 weights and the note's ceiling formula is 32× short. `n = 1.57` is the residual of that error (`809.08/516.4 = 1.5668`), i.e. a slot count in disguise, not a lane-instruction count (which is 24.1). | `run-artifacts/hypotheses/H-2.md` | A1 closed; `DESIGN-instruction-budget.md` §1 void |
| H-3 | What costs the kernel the 521 → 645 GB/s gap? | Memory **latency / MLP**, not bandwidth: 43 % of the body is `s_waitcnt` (19) + `s_delay_alu` (31), and `full` issues 21 global-memory instructions per 32 weights against `k_read`'s 5, which reaches 743.7 GB/s. | A variant that widens outstanding loads per wave (unroll the row loop so 2–4 rows' block groups are in flight, or prefetch the next block group) at fixed footprint and checksum gate; acceptance is `full` ≥ 600 GB/s at 1.0700 GiB with the gate VALID. Scratch `~/a3/`. | open | — | feeds the first kernel slice |
| H-4 | Do the four unmeasured quant types reach the 752 GB/s the whole-file estimate assumes? | The load floor is stride-independent (each type within 5 % of IQ3_S's 718–744 GB/s); the dequant-dot rate differs per type and is set by its own instruction count, with IQ2_S and IQ3_XXS below IQ3_S because they carry more gathers per byte. | Brief `brief-A2-per-type-sweep.md`: generic-stride `read` at ≥ 1 GiB per type, then a gated dequant-dot kernel per type against an fp64 reference, then the ISA census. Scratch `~/a4/`. | open | — | A2 |
| H-5 | Which drafter — the shipped MTP head or DFlash2 — and what does each cost per verification step here? | DFlash2's block-diffusion draft (one pass per block) beats MTP's autoregressive draft at concurrency 1, but its ~4 GB BF16 footprint costs the KV budget ~2.5 of the ~4.1 concurrent-32K sessions per card that the MTP head leaves, so it must be quantised to be deployable. | Brief `brief-A3-drafter-research.md`: what is in the vendored engine, what is NVIDIA-only, the licence, and a cost model at 776 GB/s. Scratch: repo only. | open | — | A3 |
| H-6 | What is the round-trip latency of a 10 KiB exchange between the two cards? | A mapped-memory GPU-publish/pre-launched-acquire round trip is 2–5 µs (host-mediated `hipMemcpy` is 15–25 µs), which puts TP=2 at 1.85–1.95× on a single stream and makes it the default over DP. | Brief `brief-A4-round-trip-latency.md`: three paths (mapped flag, host-mediated copy, single-card loopback floor), 10 KiB and 80 KiB, ≥1000 iterations with p99. Scratch `~/a5/`. | open | — | A4 |

## Units, fixed once so no row re-invents them

- A **wave instruction** is what an issue slot carries; gfx1100 issues at most one
  wave instruction per SIMD32 per cycle, and the card has 168 SIMD32 (84 CU × 2),
  wavefront size 32. Slot rate at a measured 2.802 GHz is therefore 470.7 G/s.
- A **lane instruction** is one workitem's execution of that instruction. One wave
  instruction advances 32 lanes, so it covers 32 × 4 = 128 weights for a `dp4a`, and the
  193-word body advances 1024 weights per wave iteration.
- `n` per 4 weights, counted per lane, is numerically equal to wave instructions per
  128 weights. It is *not* comparable to a wave-slot rate without the 32.
- **No claim in this project may rest on a hardware counter until the instrument works.**
  `rocprofv3` on this host reads only `SQ_BUSY_CYCLES`, `SQ_WAVES` and `Wavefronts`
  (H-1 §2); the differential experiment in H-1′ is the substitute.

## Assumptions

| ID | Assumption | Why it cannot be tested yet | Risk if wrong | Review date |
|---|---|---|---|---|
| A-1 | The L2-resident arm's 645 GB/s is a true upper bound on the kernel's compute ceiling. | It is a bound, not a direct issue-slot read; a working instruction counter would measure it instead of bracketing it. | If the true ceiling is higher, memory-side headroom is larger than +124 GB/s and the balance shifts toward memory. | when a working profiler is found |
