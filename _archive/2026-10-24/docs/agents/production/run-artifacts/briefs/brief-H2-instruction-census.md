# Probe brief — H-2: instruction census of the IQ3_S dequant-GEMV

```
STATION: probe (hypothesis H-2)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin` (~/e1 harness, read-only) ; scratch `~/a2/`
TOOLSET: ssh, hipcc (clang 24), llvm-objdump (--mcpu=gfx1100), python3
BUDGET: 70 tool calls; at ~90 % switch to write-out mode; unfinished items report as failed(budget), never omitted
ROUND: 1
CONTRACT: none — this is a Discovery probe
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-2.md
```

## GOAL

Count, from the **emitted gfx1100 machine code**, how many instructions the IQ3_S
dequant-GEMV actually executes per 32 weights per lane, split by role — index formation,
codebook gather, sign application, scale application, dot — and report the totals in a
form that can be compared to a hardware issue-slot rate without a unit error.

A second, independent question rides along: **is `n` in
`docs/plan/DESIGN-instruction-budget.md` a lane-instruction count or a slot count?** The
note multiplies a slot rate (`84 CU × 2 VALU/cycle × 2.802 GHz = 470.7 G/s`) by
`bpw/8` and divides by `n`. Show, from the code and the ISA, which normalisation is
consistent — including whether one wave instruction covers 32 lanes of work
(32 × 4 = 128 weights for a `dp4a`) and whether the slot rate counts all instruction
classes or only VALU.

## CONTEXT

`rasputin` is 2× RX 7900 XT (gfx1100, RDNA3), wavefront size 32, 84 CU = 168 SIMD32,
measured sclk under load 2802 MHz. `rocminfo` and `amd-smi` are the sources for those.

- Kernel under census: `k_full` in `~/e1/iq3s_valid.cpp` (a template over layout and a
  `NOSGN` flag), the one measured at 516.4 GB/s in
  `bench/gfx1100-probes/E1-IQ3S-2026-10-22.md`. Also census `k_read` (the load-only
  floor, 717.7 GB/s) and the `NOSGN` instantiation (608.2 GB/s) so the ablation can be
  read as instruction deltas rather than asserted.
- The design note's claim: `n ≈ 1.57` per 4 weights against a budget of `1.04`, hence
  "instruction-bound". Known ISA facts, already measured, that the census must respect:
  `v_dot4_i32_i8` on this chip issues at **half rate** (≈2.08 slots per instruction in a
  16-chain loop) and **cannot dual-issue**; `v_dot8_i32_i4` is quarter rate; there is no
  packed fp32 FMA on gfx11.
- Block layout (110 B per 256 weights, IQ3_S) is documented in
  `docs/research/2026-10-22/03c-gguf-quant-layouts.md`; the harness's own `Off<>` struct
  is the authority for the offsets actually used.

## RULES

1. **Do not run the GPU at all.** This probe is CPU-only (compile + disassemble). Another
   probe holds the GPU for the counter capture; a second GPU context would corrupt both.
   If you find yourself reaching for `HIP_VISIBLE_DEVICES`, you have left the brief.
2. **llama.cpp is excluded** — no clone, no build, no derived code. Name it in your
   compliance line.
3. **The census must come from the emitted ISA**, not from reading the C++. `hipcc
   --offload-arch=gfx1100 -O3` changes the loop body materially; count what the machine
   executes. `--save-temps` or `-S` plus `llvm-objdump -d --mcpu=gfx1100` on the object
   is the way in.
4. `~/e1/` is read-only to you: copy into `~/a2/` and compile there.
5. **Distinguish inner-loop from preamble.** The per-weight cost is the steady-state
   loop body; prologue, bounds checks, and the reduction tail are reported separately and
   never folded into `n`.
6. Handle unrolling explicitly: the loop is unrolled to 8 `dp4a` per 32 weights. State
   the unroll factor you actually observe in the ISA and count per **one** 32-weight
   group.

## WHAT TO COLLECT

1. **Build:** compile `iq3s_valid.cpp` at `-O3 --offload-arch=gfx1100`, keep the
   `.s`/`.hsaco`, and record the `hipcc --version` line and the exact command.
2. **Loop identification:** locate the steady-state 32-weight body of `k_full` in the
   disassembly (it contains 8 `v_dot4_i32_i8`). Quote the first and last instruction of
   the body, give its address range, and state how you decided where it starts and ends.
3. **Census table** — instructions per 32 weights per lane, per role:
   | role | mnemonics | count | note |
   The roles: index formation (`qs`/`qh` assembly, shifts, masks), codebook gather
   (LDS/SMEM/LDG or `v_perm`), sign application (the SWAR mask build and negate), scale
   application (`d` load, fp16 convert, multiply), dot (`v_dot4_i32_i8`), loop overhead.
   Count **every** instruction in the body and classify it; an unclassified residue is a
   number you must explain.
4. **Issue-slot weighting:** give the count three ways — raw instructions, "slots at the
   measured per-instruction cost" (using the half-rate figure for `dp4a` and stating what
   you assume for the rest), and **all-class** (including SALU/SMEM/LDS/FLAT, because
   issue slots are shared). Say which number the design note's `n` could be.
5. **Same census for `k_read` and `k_full_nosgn`**, so the sign-path cost and the
   dequant cost appear as instruction deltas. Compare those deltas to the measured rate
   deltas (516.4 → 717.7 → 608.2 GB/s) and say whether the instruction share of the body
   explains the rate share.
6. **Reconciliation:** instructions/32 weights × weight rate implied by 516.4 GB/s
   → wave instructions/s, versus the 470.7 G/s slot rate (and versus 168 × the measured
   clock). State the resulting occupancy. If it lands far from the `n = 1.57` reading,
   say so plainly and show the arithmetic both ways.

## HOW TO VERIFY YOU DID IT RIGHT

- The body you counted contains exactly 8 `v_dot4_i32_i8` and its start/end are quoted
  with addresses; a census of a preamble is the failure mode.
- The sum of your per-role counts equals the total instruction count of the body you
  delimited, and the residue is zero or named.
- Your instruction-derived wave rate, multiplied by your census, reproduces the measured
  GB/s within a factor you state. If it is off, the unit normalisation is the suspect —
  fix it, and say which normalisation was wrong.
- No GPU was touched: `ls -l /sys/class/drm/card*/device/mem_info_vram_used` before and
  after, and `dmesg` shows no new amdgpu line.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-2.md` in the repo, then print it:

```
# H-2 — instruction census of the IQ3_S dequant-GEMV
## Question / hypothesis (verbatim from the ledger)
## Build (compiler line, flags, artifacts in ~/a2/)
## Loop delimitation (addresses, first/last instruction, unroll factor)
## Census (role table; k_full, k_read, k_full_nosgn)
## Slot weighting (raw / per-instruction cost / all-class)
## Reconciliation (wave instr/s vs slot rate; occupancy; the n question answered)
## Verdict (supports / refutes H-2, one paragraph)
## Compliance (llama.cpp excluded; GPU untouched; repo unmodified)
## Receipts (absolute paths)
```

Effort: **medium-high**. One clean build, one complete and *closed* census (nothing
unclassified), one reconciliation that actually multiplies out. The value is in the
completeness of the body and the honesty of the unit arithmetic — not in extra variants.
