# Design note — what bounds the IQ3_S dequant-GEMV

**Status:** replaced 2026-10-23 by measurement. This note previously claimed the kernel was
*instruction-issue-bound* at 516.4 GB/s, with `n ≈ 1.57` against a budget of `1.04`. Both the
number and its units were wrong. The measured picture is below, and it is a **two-bound
corner, not a single wall**.

Evidence: [`../agents/production/run-artifacts/hypotheses/H-1.md`](../agents/production/run-artifacts/hypotheses/H-1.md)
(the differential experiment) and
[`H-2.md`](../agents/production/run-artifacts/hypotheses/H-2.md) (the instruction census).

## 1. The three numbers

`k_full`, the real IQ3_S dequant-GEMV, 30 real tensors, checksum-gated, clocks pinned:

| configuration | GB/s | % of the 776 wall |
|---|---:|---:|
| as published, 1.0700 GiB, DRAM | **521** | 67 % |
| **same kernel, 38.3 MB, L2-resident** | **645** | **83 %** |
| the read wall | 776 | 100 % |

The second row is the important one. It is the same binary, the same instruction stream and
the same checksum, with the weight stream served from the 96 MB L2 instead of DRAM — so it
measures what the kernel does when memory is free. `read` in that arm reaches 1.76 TB/s,
which is how we know the footprint really was cache-resident.

Two consequences, both arithmetic from those three numbers:

- **Memory pressure costs +124 GB/s** (521 → 645), 16.0 % of the wall. A kernel at issue
  saturation cannot gain throughput from memory being made free, so it is **not** saturated.
- **Instructions cost +131 GB/s** (645 → 776), 16.9 % of the wall. With memory free the
  kernel still stops 17 % short, so instructions **do** cap the best case.

The two headrooms are within 6 % of each other. Neither resource is saturated; both are
close, and the honest statement is that this kernel sits at a corner.

## 2. Why the old `n` model was void

`n` was defined as VALU instructions per 4 weights and compared to a slot rate of
`84 CU × 2 VALU/cycle × 2.802 GHz = 470.7 G/s`. That slot rate counts **wave**
instructions. One wave instruction advances 32 lanes, so it covers 32 × 4 = 128 weights for
a `dp4a`. The note's formula `4 · slots · (bpw/8) / n` therefore treats one slot as buying
4 weights when it buys 128 — **a unit error of exactly the wavefront width, 32**.

Two independent consequences, both measured:

1. **The census.** The emitted gfx1100 body of `k_full<0,0>` is **193 instruction words per
   32 weights per lane** (8 `dp4a`, 48 sign, 34 index, 31 `s_delay_alu`, 27 loop control, 21
   global-memory, the rest address/scale/checksum). At 521 GB/s that is 228.7 G
   wave-instructions/s — **48.6 %** of the slot rate, not ≥ 85 %. To saturate issue at this
   throughput the body would need 338 instructions per 32 weights; it has 193, i.e. 57 % of
   what the claim required.
2. **`n` was solved for, not measured.** `809.08 / 516.4 = 1.5668`. Any throughput can be
   fitted by some `n`; the fit was never evidence. Correctly normalised, the same
   measurement is 0.754 issue slots per 4 weights against the note's own 1.04 budget — 28 %
   *under* budget, the opposite of the claim.

`n ≤ 1.04` is therefore not a target, and "the format gets no faster as it gets smaller"
does not follow from this machine.

## 3. What actually binds, and where to look

Not memory bandwidth: `read` reaches 743.7 GB/s at the same footprint, so the stream can be
delivered at 96 % of the wall. Not issue slots: 48.6 % used. The kernel is losing time
**between** the two, and the census says where:

- **43 % of the body is wait machinery** — 19 `s_waitcnt` + 31 `s_delay_alu` in 193 words.
- **21 global-memory instructions per 32 weights**, against `k_read`'s 5 — the dequant reads
  four odd-sized fields per block group instead of one contiguous 16 B unit.
- **The sign path's cost depends on the arm**: removing it is worth **+13.8 %** at DRAM
  (521 → 593) but only **+7.3 %** with memory free (645 → 692), although it deletes 25 % of
  the instruction words. A pure issue cost would help by the same fraction in both arms; the
  difference is the signature of an instruction–latency interaction — it costs more when
  memory is in play, because it reduces the loads in flight.

That is a **memory-latency / memory-level-parallelism** signature. It is open as `H-3` in
the ledger: widen outstanding loads per wave (unroll the row loop so several block groups are
in flight, or prefetch the next group) at fixed footprint, with the gate VALID.

## 4. What this licenses, and what it forbids

| lever | headroom | verdict |
|---|---|---|
| memory latency / MLP (`H-3`) | up to +124 GB/s | **the first thing to try** |
| instruction count (sign 48, index 34, loop 27) | up to +131 GB/s | still worth it, but it cannot fix a latency gap |
| raw bandwidth / layout permutations / page size | ~0 | `read` is already at 96 % of the wall |
| codebook gather | 0 | removing it is *slower* (E1, −3 %); it is hidden |
| lossless 4-bit expansion | negative | 100.4 % of the wall to break even (`DECISION-packaging-iq3s.md`) |

What is **forbidden** is the sentence the old note carried: *"memory-side optimisation
cannot move this kernel."* It can, by up to 124 GB/s. What is **also** forbidden is the
opposite over-correction: bandwidth tuning alone cannot reach 776, because instructions cap
the memory-free ceiling at 83 % of it.

## 5. The five ISA facts behind the arithmetic (all gfx1100-specific, all still true)

1. **`v_dot4_i32_i8` cannot dual-issue.** RDNA3's VOPD opcode set is 17 FP32-ish operations
   with three integer ops available only in the Y slot. A "half rate / full rate" table can
   never tell you that.
2. **4 KB pages only.** `amdgpu` maps 4 KB PTEs; no HIP or ROCm API or env var produces 2 MB
   pages. The page-size theory for the ~800 MiB anomaly is arithmetically excluded.
3. **No `cp.async` on gfx11; MUBUF has no LDS bit.** LDS staging is not slow, it is
   unavailable.
4. **1536 VGPRs per SIMD32 at allocation granularity 24** — full occupancy means ≤96
   registers. The kernel uses 40, so occupancy is not the constraint.
5. **I-cache is 128 B lines, 32 KB per WGP, shared by both CUs.** Sorting dispatch by quant
   type costs 0 %; interleaving different types in one CU costs −10 %.

## 6. What would falsify this note

1. A working instruction counter reading > 70 % issue occupancy at 521 GB/s — that would
   mean the differential experiment was confounded and the old claim survives.
2. A variant with the *same* instruction count and byte count reaching > 690 GB/s — that
   would mean the memory-free ceiling was an artefact of the small footprint rather than the
   kernel's compute limit.
3. A latency variant (`H-3`) reaching 776 without touching instructions — that would prove
   the instruction ceiling is not binding after all.

## 7. The instrument gap this uncovered

`rocprofv3` on this host reads only `SQ_BUSY_CYCLES`, `SQ_WAVES` and `Wavefronts`; every
instruction-count counter and every `GL2C_*` counter reads zero **on a kernel with a known
instruction count**. SQTT/ATT produces no ISA mapping for gfx1100 here. No claim in this
project may rest on a hardware counter until that is repaired — the differential experiment
is the substitute, and it is stronger because it needs no profiler. Recorded in `H-1` §2.

## Related

- `RECORD.md` — the measured rows this note adds.
- `03-native-kernel-inventory.md` — the kernel set; row 1a is the sign path.
- `../research/2026-10-23/s3-amd-sinks.md` — the ISA survey this rests on.
- `../research/2026-10-23/s2-tensorfold-freetoken.md` — the MTP depth rule; speculation
  divides per-token work regardless of which resource binds, so it is untouched by this
  correction.
