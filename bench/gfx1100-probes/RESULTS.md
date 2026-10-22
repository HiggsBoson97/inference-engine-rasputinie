# gfx1100 dot-product throughput on a Radeon RX 7900 XT

> **Note, 2026-10-22:** the read bandwidth quoted below (619 GB/s) was later shown
> to be a kernel artefact — the real wall is **776 GB/s**
> (`WALL-2026-10-22.md`). The instruction conclusions stand and strengthen: at
> 776 GB/s the GEMV needs *even fewer* issue slots, so "bandwidth-bound with a
> large margin" is more true, not less. Card 0 here means `HIP_VISIBLE_DEVICES=0`,
> which on this machine is the **display** card — use card 1 for load tests.

Host `rasputin`, card 0 only (`HIP_VISIBLE_DEVICES=0`), ROCm 7.16 / clang 24,
`hipcc --offload-arch=gfx1100 -O3`. Card 1 was never opened.

**Headline:** `v_dot4_i32_i8` is **half rate** (≈2.08 issue slots per instruction), not
quarter rate. At the measured memory bandwidth the planned Q5_K GEMV needs **~6% of the
machine's issue slots**, so it is **bandwidth-bound with roughly a 15× issue margin** —
the design note's "72% issue occupancy" is off by more than an order of magnitude, and
"fewer dots per weight" would buy nothing.

---

## 1. What compiled and what did not

Everything was first tried through the documented builtins. Most of them do not work on
this toolchain, which is itself a finding for the project.

| builtin | result | compiler output |
|---|---|---|
| `__builtin_amdgcn_sdot4(int,int,int,bool)` | **does not compile** | `error: '__builtin_amdgcn_sdot4' needs target feature dot1-insts` |
| `__builtin_amdgcn_sdot2(short2,short2,int,bool)` | **does not compile** | `error: '__builtin_amdgcn_sdot2' needs target feature dot2-insts` |
| `__builtin_amdgcn_sudot8(...)` | **does not compile** | `error: too few arguments to function call, expected 6, have 4` and, with 6 args (tried `0,0,0`, `0,0,false`, `false,0,0`, `(int)0,0,0`), `error: argument to '__builtin_amdgcn_sudot8' must be a constant integer` |
| `__builtin_amdgcn_fdot2(half2,half2,float,bool)` | **compiles** | emits `v_dot2_f32_f16`; results identical to the hand-written asm (checked case by case) |
| `v_pk_fma_f32` (packed 2×f32 FMA) | **instruction does not exist on gfx1100** | `llvm-mc: error: instruction not supported on this GPU (gfx1100)` |

The gfx1100 subtarget advertises `+dot5-insts +dot7-insts +dot8-insts +dot9-insts
+dot10-insts +dot12-insts` (dumped from the emitted `target-features` attribute) but
**not** `dot1-insts`/`dot2-insts`, while the clang front end still gates `sdot4`/`sdot2`
on those two old feature names. The MC layer assembles the mnemonics for gfx1100 without
complaint (`llvm-mc -mcpu=gfx1100` accepts `v_dot4_i32_i8` and `v_dot8_i32_i4`).

Two consequences:
* `v_dot4_i32_i8` is reachable on gfx1100 **only through inline asm** (or OpenCL); the
  builtin the project planned to use cannot be compiled. All rates below are measured
  from `v_dot4_i32_i8` written as inline asm, and the ISA was verified with `-S`.
* On gfx11 there is no `v_dot2_i32_i16` at all (`instruction not supported on this GPU`),
  so item 2 of the brief was measured with **`v_dot2_f32_f16`** (2×f16 MAC → f32),
  not the int form.
* There is no packed-fp32 FMA on gfx11, so item 3's baseline is **scalar `v_fma_f32`**
  (1 MAC = 1 issue slot). That is the correct "one slot, one MAC" reference here.

## 2. Machine model (this is where the design note's arithmetic breaks)

| quantity | value | source |
|---|---|---|
| `hipDeviceProp.multiProcessorCount` | **42** | HIP runtime |
| `amd-smi` / `rocminfo` "compute units" | 84 | `NUM_COMPUTE_UNITS: 84`, 84 per GPU |
| RX 7900 XT stream processors (spec) | **5376** | 84 CU × 64 lanes = 42 × 128 lanes |

There are **5376 fp32 lanes**, i.e. 168 SIMD32 units — not 84 × 128 = 10752. The design
note's "128 VALU lanes per CU × 84 CU" double-counts: a gfx11 CU has 2 SIMD32 (64 lanes).
Everything below is normalised against 5376 lanes × the *measured* clock.

Clock under load (`amd-smi metric --clock`, sampled by a thread inside the benchmark,
836 samples during the timed runs): **median 2802 MHz, range 2435–2929 MHz**. The idle
DPM level reads 4–128 MHz and the `amd-smi static` "max level" is 2075 MHz; the card
actually boosts to ~2.8 GHz. Theoretical peak therefore 5376 × 2.802 GHz =
**15.06 T lane-ops/s** = 471 G wave-instructions/s = **382 G wave-instr/s at the 81%
efficiency this harness reaches** (below). The note's "800 G issue slots/s" is 1.7–2.1×
too high for this chip.

## 3. Method

`probe.cpp`. One workgroup per 128-lane unit (`84 blocks × 1024 threads` = 43008 threads
= 1344 waves → 8 waves/SIMD), 16 independent accumulator chains per lane so the result is
throughput, not latency. Two loop bodies measured for every instruction:

* `dots + 1 v_mov` — the varying operand is the loop counter, so one extra `v_mov_b32`
  per 16 dots (ISA verified: `v_mov_b32_e32 v33, s3` + 16 ops + `s_add/s_cmp/s_cbranch`).
* `pure` — no varying operand at all, 16 dots + scalar branch in the body. `asm volatile`
  prevents the compiler from folding or hoisting the chain, and the ISA was disassembled
  (`--offload-arch=gfx1100 -S`) to confirm the loop contains exactly the intended ops.

2^24 iterations per lane, 3 untimed warm-up launches per instruction, then **median of 5
timed runs**. Accumulators are folded into an XOR and stored, so nothing is eliminated.
Occupancy was swept (2/4/8/16 waves per SIMD); 8 waves/SIMD is where it converges (the
register file caps it there), and results are stable to <1% across runs.

Reproduce: `HIP_VISIBLE_DEVICES=0 ./probe 16777216 1024 1` (needs `render` group or root),
`HIP_VISIBLE_DEVICES=0 ./bw 1024`, then `python3 make_json.py`.

## 4. Correctness (gates every number below)

Reference: scalar C on the host, run over thousands of randomised inputs plus targeted
edge values (`-128`, `127`, `0`, `0x80808080`, `0x7f7f7f7f`, int32 accumulator
overflow/wrap). Integer ops compared bit-exactly; float ops compared against a
correctly-rounded `fmaf` / exact integer arithmetic.

| instruction | cases | result |
|---|---|---|
| `v_dot4_i32_i8` | 8202 (incl. -128/127/0 edges, acc wrap) | **PASS, bit-exact, 0 mismatches** |
| `v_dot8_i32_i4` | 8202 | **PASS, bit-exact, 0 mismatches** |
| `v_fma_f32` | 8198 | **PASS, bit-exact vs `fmaf` (max 0 ULP)** |
| `v_dot2_f32_f16` | 8198 half2 (integer-valued, products exact in fp16) | **PASS within an 8·eps bound; NOT bit-exact**: 6057/8198 bit-exact, 0 outside the bound, worst abs err 1.53e-05 vs a 4.59e-04 bound. Deviations are last-bit (results that should be exactly 0 come back as ±6e-8) — consistent with a shared/truncated reduction path, not a mis-encode. Do not expect bit-identical fp16 dot results. |

## 5. Throughput

Clock 2802 MHz, peak 15.06 T lane-ops/s. "slots/instr" = 1 / (fraction of peak), so 1.0
would be a free-riding full-rate instruction and 2.0 a half-rate one.

### Pure loop bodies (16 ops, no `v_mov`) — the clean numbers

| instruction | MAC/instr | median ms | spread (5 runs) | Ginstr/s | GMAC/s | lane-op/clk/lane | % of peak | **slots/instr** |
|---|---|---|---|---|---|---|---|---|
| `v_dot4_i32_i8` | 4 | 3194.35 | 3193.44 – 3195.71 | 7228.3 | **28913** | 0.480 | 48.0% | **2.08 → half rate** |
| `v_dot8_i32_i4` | 8 | 6018.39 | 6016.78 – 6019.55 | 3836.5 | 30692 | 0.255 | 25.5% | **3.92 → quarter rate** |
| `v_dot2_f32_f16` | 2 | 1905.15 | 1904.94 – 1906.39 | 12119.6 | 24239 | 0.805 | 80.5% | 1.24 → full rate |
| `v_fma_f32` | 1 | 1890.31 | 1889.61 – 1890.91 | 12214.8 | 12214.8 | 0.811 | 81.1% | 1.23 → full rate |

### With the counter `v_mov` in the body (matches the as-built comparison)

| instruction | median ms | spread | Ginstr/s | GMAC/s | % of peak | slots/instr | rel. to full-rate FMA |
|---|---|---|---|---|---|---|---|
| `v_dot4_i32_i8` | 3536.95 | 3531.48 – 3537.20 | 6936.2 | 6528.2 | 46.0% | 2.17 | 0.568 |
| `v_dot8_i32_i4` | 6154.91 | 6153.79 – 6155.14 | 3985.9 | 3751.4 | 26.5% | 3.78 | 0.326 |
| `v_dot2_f32_f16` | 2169.35 | 2165.85 – 2172.29 | 11308.8 | 10643.6 | 75.1% | 1.33 | 0.926 |
| `v_fma_f32` | 2284.85 | 2283.02 – 2295.13 | 10737.2 | 10105.6 | 71.3% | 1.40 | 1.000 |

Self-consistency: FMA at 81% of the theoretical peak (and dot2 at 80%) is what a
correctly written 16-chain loop should reach; nothing exceeds 100%. The residual 19% is
the scalar loop iteration and issue-boundary effects, which is why full-rate ops land at
1.23 "slots" rather than 1.00 — the ratios in the last column are the meaningful part.

**Bandwidth, for the verdict below** (`bw.cpp`, 1 GiB stream, 6 timed runs, card 0):
read-only **619.0 GB/s** [616.7–620.8], copy (r+w) 595.3 GB/s. Spec is 800 GB/s
(20 Gbps × 320-bit), so the card reaches 77% of its memory spec — it is not throttled,
which is the cross-check that the ~81% instruction number is also a real ceiling and not
a sick GPU.

## 6. Verdict: bandwidth-bound, with a large margin

Q5_K is 5.5 bits/weight (0.6875 B/weight). At the measured **619 GB/s** a decode GEMV can
consume **900 G weights/s**. With the note's own instruction mix (20 lane-instructions
per 32 weights: 2 loads, 8 unpack, 8 dot, 2 FMA) and the *measured* cost of
`v_dot4_i32_i8` (2.08 slots, not 4), the slot cost is 2×1 + 8×1 + 8×2.08 + 2×1 = **28.6
slots per 32 weights**, i.e. 900/32 × 28.6 = **804 G lane-slots/s**. This harness sustains
12.2 T lane-slots/s, so the kernel would sit at **~6.6% issue occupancy** — bandwidth-bound
by about **15×**. To become issue-bound on this chip it would need ~9.4 TB/s of memory
bandwidth. Even if every dot were quarter-rate (4 slots, so 44 slots per 32 weights) the
occupancy would be ~10%: still bandwidth-bound. And in MAC terms the dot path is healthy —
`v_dot4_i32_i8` delivers 28.9 T MAC/s versus 12.2 T MAC/s for scalar FMA, i.e. 2.4× the
FMA array, so the half-rate issue cost is more than paid for by the 4 MACs per slot.

What this means for the design note's "578 G instr/s vs 800 G issue slots/s = 72%": the
comparison mixes units and overstates the machine. 578 G is a *lane*-instruction rate
(20 per 32 weights × 900 G weights/s / 32 ≈ 560 G, their 578 G with slightly different
bpw), while "800 G issue slots/s" is a *wave*-instruction rate (4/CU/clk) — a factor of 32
apart. In the same units the machine has 471 G wave-instr/s theoretical and ~382 G
achievable, and the kernel needs **~18 G wave-instr/s = 4.7%** of it, not 72%. Against the
achievable 12.2 T lane-slots/s the note's 578 G is 4.7%. Either way the conclusion flips:
the kernel is not issue-bound, the "IF quarter-rate" premise is false (it is half rate),
and the remedy in the note ("fewer dots per weight, not more parallelism") targets a
bottleneck that is not there. The work that matters is bandwidth: coalesced Q5_K block
layout, one pass over the weights, no re-reads of scales, and enough workgroups to reach
619 GB/s — not dot-instruction reduction.

## 7. Caveats

* The 42-vs-84 CU question: HIP says 42 units, amd-smi/rocminfo say 84 CUs. The physical
  lane count is what matters and both descriptions agree on **5376 lanes**; launching 42
  blocks × 1024 threads and 84 blocks × 1024 threads give the same throughput (11.87 vs
  12.25 T lane-ops/s), which is consistent with 42 units of 128 lanes at 8 waves/SIMD.
* During the session (12:26 host time) card **1**, the display GPU, hit a
  `ring gfx_0.0.0 timeout` in a `gnome-shell` context and recovered via ring reset
  (`device wedged, but recovered through reset`). My workloads never touched card 1
  (always `HIP_VISIBLE_DEVICES=0`, and card 0 is headless). The reset re-created
  `/dev/kfd`/`/dev/dri/*` via udev without an ACL for `mattia-bosi`, who is **not in the
  `render` group**, so after that point ROCm tooling over ssh returns `no ROCm-capable
  device is detected` unless run as root (`sudo usermod -aG render mattia-bosi` fixes it;
  not done, since it changes system config). VRAM on card 0 is back to its 27 947 008 B
  baseline.
* `dot2` numbers are for `v_dot2_f32_f16` (the int form does not exist on gfx11), so they
  are not a direct stand-in for an int16 dot path.
* Absolute % of peak depends on the clock: 2802 MHz is the median of `amd-smi` samples
  taken during the timed runs; at the `amd-smi static` "max level" of 2075 MHz every
  percentage above would be 1.35× higher (and would exceed 100% for FMA, which is why
  the sampled-under-load clock is used).
