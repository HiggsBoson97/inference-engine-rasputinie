# 03 — The bandwidth wall on gfx1100: reconciling 800 / 619 / 472, and what a fused dequant-GEMV can actually reach

**Status:** research note for the PLAN loop. Not a decision record. Everything in §2 was measured on `rasputin`, card 1 (headless, PCI 07:00.0), on 2026-10-22 with the harness in `research/bench/` (`bw2.cpp`, `gemv.cpp`) and can be re-run from `~/bwprobe` on the host.
**Question asked:** how close to the memory wall can a fused dequant+GEMV decode kernel actually get on this chip, and what is the honest attainable number?
**Companion notes:** `03b-gfx11-microarchitecture.md` (ISA-level detail: cache bits, LDS banks, VGPR/occupancy arithmetic) and `03c-gguf-quant-layouts.md` (per-format block layouts and measured per-quant speed on AMD).
**It supersedes the bandwidth premise of the earlier notes.** `02-megakernels.md` computes the decode step as `11.77 GB ÷ 619 GB/s = 19.0 ms`; §1 and §8 below replace that divisor; §8 says whether the megakernel verdict moves with it (it does not — but the argument that the fixed cost is irrelevant gets stronger).

**Tags used**

| Tag | Meaning |
|---|---|
| `[measured-on-RDNA3]` | Measured on this box, this session, with the command in the note. |
| `[measured-elsewhere]` | A real measurement, on other hardware or by another party. Named, with its method. |
| `[documented]` | Stated by a primary source (AMD ISA, LLVM, ggml source) as behaviour or layout, not as a number. |
| `[claimed]` | Assertion by an interested party, or a number whose method is thin. Hypothesis. |
| `[derived]` | Arithmetic done here from measured inputs. |

---

**One-paragraph answer.** 800 GB/s is the pin rate and nothing reaches it; **619 GB/s was never the wall** — it was one badly-shaped kernel measured over too short a window. On this card a hand-written 128-bit streaming read reaches **768–777 GB/s** (96–97 % of spec, `[measured-on-RDNA3]`), and a fused 4-bit dequant+GEMV with int8 `dp4a` reaches **733–743 GB/s** (92 % of spec, 95 % of the read wall, `[measured-on-RDNA3]`). **Two things that cost bandwidth are mundane: too few warps for the per-lane loads in flight** (375 → 636 GB/s going from 1 to 8 blocks/SM on a kernel with one load in flight per lane, and most of that inside the first four) **and a second dependent load stream every 32 weights** for the quant scale (697 → 773 when it is removed). **Load width per lane, non-temporal loads, LDS staging, and the row-per-warp access order are all worth zero** on this chip once bytes-in-flight is right. The honest number to plan against is a kernel at **~735 GB/s** and an engine at **620–680 GB/s** sustained end-to-end, i.e. **11.76 GB ÷ 650 GB/s ≈ 18 ms/token ≈ 55 tok/s** on one card. The 472 GB/s figure in the repo is a torch GEMM artefact (61 % of the wall) and is not evidence about the hardware.

---

## 1. The three numbers, reconciled

| Number in the record | What it actually is | Verdict |
|---|---|---|
| **800 GB/s** | 320-bit GDDR6 at 20 Gbps. AMD's product page states "Memory Bandwidth: Up to 800 GB/s" and separately an "Effective Memory Bandwidth" of **2900 GB/s** which is Infinity-Cache accounting and applies to no stream in this engine ([amd.com](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html)). `[documented]` | **Unreachable.** No configuration in §2 exceeded 777 GB/s. Nothing above ~750 GB/s is reported by anyone for this card. |
| **619 GB/s** | A `float4` read-reduce over a 1 GiB buffer, 42×8 blocks × 256 threads, 6 timed runs, with a serial `acc.x += v.x` dependency and no unrolled loads (`bench/gfx1100-probes/bw.cpp`). | **A kernel artefact.** Same card, same session: 656 (same footprint, 8 loads in flight), 698 (2 GiB), 748 (6 GiB), **777** (6 GiB, 4 passes in one kernel). |
| **592.9 GiB/s = 636.6 GB/s** | torch copy, counted read+write. | **A different quantity.** Copy is 599–644 GB/s r+w here; the read path is ~1.3× faster than copy. Never compare it to a read number. `[measured-on-RDNA3]` |
| **472 GB/s** | torch/hipBLASLt bf16 GEMM at M=1, 1×5120×17408 (`docs/00-facts.md`). | **A kernel-quality number**, 61 % of the read wall. It says the library GEMM leaves 35 % on the table at batch 1, not that the card stops at 472. |

The 4 % "GB/s vs GiB/s" confusion flagged as **C8** in `REQUIREMENTS-DRAFT.md` resolves the same way: 592.9 GiB/s is 636.6 GB/s, and the kernel gate should be written in GB/s against a measured read wall, not against either.

### 1.1 Why the same kernel reports 640 one day and 750 the next

Measured on one binary, one buffer, only the buffer size changed (`[measured-on-RDNA3]`; run-to-run spread on the medians is ±3 %, largest at 1 GiB):

| working set | 128 MiB | 256 | 512 | 1 GiB | 1.5 GiB | 2 GiB | 3 GiB | 4 GiB | 6 GiB |
|---|---|---|---|---|---|---|---|---|---|
| read GB/s | 717.7 | 743.3 | 712.7 | **640.6** | 677.7 | 698.3 | 722.1 | 734.4 | **747.6** |

And with the *same footprint*, only the number of passes inside one kernel changed `[measured-on-RDNA3]`:

| 1 GiB buffer | 1 pass (1.6 ms) | 4 passes (6.4 ms) |
|---|---|---|
| read GB/s | 662 | **746** |

| 6 GiB buffer | 1 pass (8.6 ms) | 4 passes (33 ms) |
|---|---|---|
| read GB/s | 749 | **768** |

A fixed cost of ~0.2–0.3 ms per launch-or-ramp, plus a few per cent of Infinity-Cache reuse on repeats, explains the whole spread `[derived]`. Consequences, both of which matter more than any micro-optimisation:

- **Any bandwidth number measured on a kernel shorter than ~3 ms under-reports this card by 10–15 %.** The repo's 619 came from 1.6 ms passes.
- **The effect recurs per launch; it does not warm up.** Eight back-to-back 1 GiB launches (13 ms of continuous queued work, no gaps) gave 641 GB/s — not 735 `[measured-on-RDNA3]`. So a decode step built from ~200 small per-tensor kernels cannot be assumed to reach the steady-state number. **What is not established is the size of the per-launch penalty**: attributing the whole gap to a fixed ~0.2 ms/launch would predict >100 ms/token for a 450-launch step, which shipping engines visibly do not pay (`[derived]`, contradicted by llama.cpp's 31 ms/token at 374 GB/s). The mechanism is unidentified — DPM state is the leading candidate, since the card idles at `memclock 1249 MHz` and only ramps under load `[measured-on-RDNA3]`, and no clock sampling was done inside the timed loop. **This is the one number in this note that must be measured on the real launch pattern before it is designed around (§9 item 4).**

Launch overhead itself is small and is not the cause: a 2.6 MB kernel costs 16 µs at one launch and 8 µs amortised over 100 `[measured-on-RDNA3]`.

---

## 2. What was measured

**Method.** `hipcc --offload-arch=gfx1100 -O3`, `HIP_VISIBLE_DEVICES=1` (headless card), 336–2688 blocks × 256 threads, `hipEvent` timing, warm-up pass discarded, median of 3–7. GB/s = 10⁹ bytes/s. Weight traffic counted as bytes actually requested (quants + scale plane where present). The fused kernel's arithmetic was validated bit-close against a host reference built from the same bytes: **worst relative error 4.8 × 10⁻⁷** over 128 rows `[measured-on-RDNA3]`.

### 2.1 Streaming reads (the wall)

| Kernel | GB/s |
|---|---|
| `uint4`/lane, 8 in flight, 336 blocks, 6 GiB, 4 passes in one kernel | **768 – 777** |
| same, 2 GiB, 4 passes | 755 |
| same, 1 GiB, 4 passes | 746 |
| same, 1 GiB, 1 pass | 662 |
| `uint4`, non-temporal (`slc=1 dlc=1`) | 662 vs 656 (+0.8 %, noise) |
| 32-bit/lane (`uint`), 8 in flight, 2 GiB | 720 |
| 64-bit/lane (`uint2`), 8 in flight, 2 GiB | 722 |
| 128-bit/lane (`uint4`), 8 in flight, 2 GiB | 722 |
| copy, counted r+w | 599 (NT store: **644**, +7.4 %) |

### 2.2 The fused dequant-GEMV (M=1)

4-bit weights, 32-weight blocks, fp16 scale per block in a separate plane (4.06 bpw), one warp per row, `v_dot4_i32_i8` via inline asm (the builtin does not compile — `bench/gfx1100-probes/RESULTS.md`), int8 activations, 1 GiB footprint (400 000 × 5120):

| variant | GB/s |
|---|---|
| single launch, one pass over 1 GiB | **635** |
| 2 / 4 / 8 in-kernel passes | 696 / **721** / **733 – 743** |
| same addressing, quants only, no scale, no ALU (`read` mode) | 767 – 772 |
| fused but the scale comes from a register (no second load stream) | **773** |
| fused with the x-vector load removed | 773 (identical — x loads are free) |
| scale amortised over 256 weights (`Q4_K`-style super-block, same load count) | 697 |
| scale read 16 B/lane and shuffled to the group (1 load per 8 blocks) | 687 |
| GGUF Q4_0 exactly: 18 B blocks, scale inline, 16 B unaligned quant load | 676 – 682 |

Occupancy, single pass, with the inner loop holding one load in flight per lane (blocks per 128-lane unit, 256 threads each):

| blocks/SM | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| GB/s | 375 | 538 | 632 | 636 | 638 | 645 | **650** |

Adding in-kernel passes at the shipped occupancy (64 blocks/SM): 1 pass 650, 4 passes 721, 8 passes 733–743. Occupancy and duration are separate levers and both are needed.

---

## 3. Why 619 was not the wall — four mechanisms, each with its measured price

**1. Not enough warps to cover the latency of one load per lane.** This is a cliff, not a slope. On a kernel whose inner loop has one load in flight per lane, the only MLP is the warp count: 1 block of 256 threads per 128-lane unit gives **375 GB/s** (48 % of the wall), 2 gives 538, 4 gives 632, and everything from 8 to 64 blocks/SM sits at 636–650 `[measured-on-RDNA3]`. Below ~4 blocks/SM the kernel is latency-bound and reports a "hardware wall" that is three times too low — which is very likely what happened to the 472 GB/s torch GEMM. For the pure stream the sweep is flat (709 → 717 from 1 to 32 blocks/SM) because the template already had 4–8 loads in flight per lane. **Corollary: with ≥4 independent loads per lane, 2 blocks/SM is enough** (the sweep in `bw2` shows x16 ILP at 714 GB/s on 42 blocks); the 64-blocks/SM configuration is only needed because the naive inner loop has ILP 1.

**2. A second dependent load stream per 32 weights.** The scale is 2 bytes per 32 weights — 12.5 % of bytes — but removing the load entirely is worth **+11 %** (697 → 773), while making the same load 8× smaller (one scale per 256 weights) is worth **+0.7 %** (697 vs 692) and batching it into one 16 B load per 8 blocks is worth **nothing** (687). So the cost is the *existence of a second dependent stream on the critical path*, not its bytes and not its request count `[measured-on-RDNA3]`. Practical reading: the quant metadata must arrive **in the same load as the weights**, or the kernel must have enough independent work in flight to hide one more stream.

**3. The buffer was too small to measure in one pass.** §1.1.

**4. Almost nothing else.** Load width per lane: 720 vs 722 vs 722 for 4/8/16 bytes at equal bytes-in-flight `[measured-on-RDNA3]`. Non-temporal loads: nil. Access order (row-per-warp 2.5 KB runs vs a flat sweep at the same footprint): 656 vs 657, i.e. **the GEMV's natural access pattern costs nothing** `[measured-on-RDNA3]`. LDS staging: not tested because there is no reason to test it — every weight is read once, so staging adds a hop (gfx11 LDS is 32 banks × 4 B, `[documented]`, [RDNA3 ISA §12](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf)). llama.cpp's MMVQ also uses **zero shared memory** at batch 1 `[documented]` (`mmvq.cu`).

The one thing the earlier session got right, and it is load-bearing: **the kernel is never issue-bound.** At 735 GB/s the 4-bit path consumes ~10 % of the 382 G wave-instruction/s the card sustains `[measured-on-RDNA3]`, `[derived]` from `bench/gfx1100-probes/RESULTS.md`. Instruction-mix work (`v_dot8_i32_i4`, packed fp16, fewer dots) buys nothing until the bandwidth side is solved.

---

## 4. Is 619 the honest ceiling? No — and this is corroborated off-box

- **The same card, two kernels, one session.** On a reference RX 7900 XT, clpeak 1.0.1's read test measured **618.37 GB/s** (22.8 W average — the card was barely out of idle), while cl-mem's read test measured **714.34 GB/s** (89.3 % of spec, 88.4 W) ([OpenBenchmarking 2212135-NE-RDNA3OPEN49](https://openbenchmarking.org/result/2212135-NE-RDNA3OPEN49)). 618 vs 714 on identical hardware and driver is the same 13 % gap this note measures between a short single-pass kernel and a well-shaped one. `[measured-elsewhere]`
- **Cross-vendor sanity check on the same suite**: 7900 XTX 90.1 %, 6800 XT 91.9 %, RTX 3090 88.2 %, RTX 3080 88.5 % of spec ([2212153-NE-RDNA3OPEN26](https://openbenchmarking.org/result/2212153-NE-RDNA3OPEN26)). ~90 % of pin rate is the normal GDDR6/HBM read efficiency; 77 % is not a hardware property. `[measured-elsewhere]`
- **Nothing above ~750 GB/s is published for this card** and no BabelStream result for Navi 31 exists at all ([BabelStream results tree](https://github.com/UoB-HPC/BabelStream)) — so 777 here is at the top of the known range, not mid-range. `[documented]`
- **Infinity Cache cannot help a decode stream.** 80 MB of IC against 11.76 GB of weights is a 0.7 % hit rate. It *does* poison small-scale benchmarks: a single 50 MB layer-shaped matvec measured **870–1124 GB/s** here because the matrix fits in the IC `[measured-on-RDNA3]`. Any per-tensor kernel benchmark in this repo that uses one layer's weights is measuring the IC, not HBM. This is a live hazard for the P0 gate in `docs/03-native-kernel-inventory.md:26`.

**The wall is ~775 GB/s.** Design against 735 for a kernel and 620–680 for an engine.

---

## 5. Techniques, ranked by measured gain

| # | Technique | Measured gain | Cost | Note |
|---|---|---|---|---|
| 1 | **Get off the occupancy cliff** — at least 4 blocks of 256 threads per 128-lane unit (8+ if the inner loop has only one load in flight per lane) | **+68 %** (375 → 632) from 1 to 4 blocks/SM; +2 % more to 64 | none | `[measured-on-RDNA3]`; below the cliff the kernel reports a fake "wall" |
| 2 | **Ship long kernels, not 1.6 ms ones** — fuse per-layer matvecs, or budget the 0.2 ms/launch tax | **+13 %** (650 → 735 at the same footprint) | architecture | §1.1 |
| 3 | **Get the scale into the weight load**, or failing that give each lane 2–4 independent weight loads so one extra stream hides | **+11 %** (697 → 773 with the stream removed) | a repack pass at load time | `[measured-on-RDNA3]` |
| 4 | 128-bit per lane | 0 % bandwidth, −75 % load instructions | trivial | `[measured-on-RDNA3]`; take it for the instruction budget, not for GB/s |
| 5 | int8 `dp4a` on Q4_0's nibble packing (`low nibbles = first 16 weights`) | 0 % bandwidth, but it is what keeps the ALU at ~10 % occupancy | inline asm | `[documented]` layout + `[measured-on-RDNA3]` |
| 6 | Non-temporal **stores** (only relevant to a copy/repack pass) | +7.4 % on copy | none | `__builtin_nontemporal_store` → `glc=1 slc=1 dlc=1` `[documented]` |
| 7 | Non-temporal **loads** (`slc=1 dlc=1`) | 0 % | none | `[measured-on-RDNA3]`; do not bother |
| 8 | Grid-stride order instead of per-block slabs | +2.7 % (696 vs 716) | none | `[measured-on-RDNA3]` |
| 9 | LDS staging of weights or activations | ≤ 0 | registers + occupancy | activations are L1-resident and free: removing x's loads changed nothing `[measured-on-RDNA3]` |
| 10 | Amortising the scale over 256 weights (Q4_K-style) | **+0.7 %** at the same load count | ALU (SWAR 6-bit unpack: +8 `dp4a`/32 w) | only pays if it *removes a load*, which in Q4_K it does not |
| 11 | Batched/shuffled scale load (1 × 16 B per 8 blocks) | **−1.4 %** (687 vs 697) | complexity | `[measured-on-RDNA3]` — it does not help, and the reason is not the shuffle (see §6) |

Read together: 1 + 2 + 3 account for essentially the entire 635 → 735 gap, and 1 also explains 619. Everything else in the literature's RDNA3 tuning list is noise at this scale.

---

## 6. First kernel design sketch

Numbers below are the configuration that measured 733–743 GB/s `[measured-on-RDNA3]`.

**Mapping.** One warp per output row. Lane *l* owns 32 consecutive weights of the row (`uint4` = 16 B), so a wave-load covers 512 B of one row — four full 128 B lines. Per 5120-wide row: 5 iterations of `b = l + 32·it`. No cross-lane exchange of weights, no transposes, no LDS.

**Registers and occupancy.** Target **4–8 waves per SIMD32** (VGPRs allocated in 16s, 256 max/thread → 64 regs = 8 waves, `[documented]`, RDNA3 ISA §3.3.2.1). Launch **8–64 blocks of 256 threads per 128-lane unit** (grid 336–2688 for the card): the measured curve is 632 GB/s at 4 blocks, 636 at 8, 650 at 64 `[measured-on-RDNA3]`. Do not launch 1–2 blocks/SM — that is the 375–538 GB/s cliff. The x-slice for a lane's current block lives in 8 dwords (32 int8) loaded as **two `uint4`** — measured free. Rows per warp (RB) = 4 for load ILP; RB ≥ 8 gains nothing.

**Load width.** `uint4` per lane. Keep `__restrict__` + `const` so the compiler emits `global_load_dwordx4` and not `flat_load` (FLAT burns both `vmcnt` and `lgkmcnt`, `[documented]`, ISA §5.6). Never `s_waitcnt vmcnt(0)` inside the loop.

**Layout on disk (the part that matters).** Per row: quants packed as 16 B per 32 weights, **and the scale for that 32-weight block inside the same 16 B** — or, if 4.06 bpw is acceptable, in a separate plane read as a *separate stream* — that is the −7 % case measured in §2.2 (721 with the plane, 773 with the scale in a register). The measured trade-off:
- 4.06 bpw with a scale plane: 721–743 GB/s.
- 4.5 bpw with the scale inline (GGUF Q4_0, 18 B blocks): 676–682 GB/s — *worse*, because 18 B blocks misalign every 16 B load. **If the scale is inline, the block must stay a power of two**: e.g. 4 quads (128 weights) + 2 B scale padded to 72 B, or 8 quads + 2 B padded to 136 B.
- The recommended shape: **32 weights per 16 B quad, one fp16 scale per 256-weight super-block, super-block header 16 B (8 scales + padding) aligned**, so a lane's 16 B load and the super-block header are both naturally aligned and the header load happens once per 8 quads.

**Reading the scale without a second stream — what was tried and what failed.** Keeping the scale in a separate plane costs 721 vs 773 (a register constant) `[measured-on-RDNA3]`. Every attempt to win it back underperformed, and the three results together say *why*:

| variant | GB/s | conclusion |
|---|---|---|
| scale in a separate plane, one 2 B load per 32 w | 692–721 | baseline |
| same, but the 8 lanes of a group load the *same* header (one scale per 256 w, hardware broadcast, no shuffle) | 697 (+0.7 %) | bytes are not the problem |
| header batched to one 16 B load per 8 blocks, shuffled to the group | 687 (−1.4 %) | instructions are not the problem either |
| no scale load at all (register constant) | 773 (+11 %) | it is the second *dependent stream* |

So the cost is neither bytes nor instruction count nor the shuffle: it is that two outstanding streams force `s_waitcnt` to wait on the slower one each iteration, and the dot inherits that latency. The two fixes that remain untested are (a) **software pipelining** — issue iteration *i+1*'s weight and scale loads before consuming *i*'s, with RB ≥ 2 giving the register budget; RB sweep to date is within noise (704–723), so this needs an explicitly pipelined loop, not just a bigger RB; and (b) folding the scale into the 16 B quad itself, which at 32 weights per quad means dropping to a 4-bit scale (4.25 bpw, IQ4_XS-style). Do **not** spend time on shuffles or wider header loads; both were measured and both lose.

**Dequant + dot.** Per 16 B: 4 × `v_and` + 4 × `v_lshrrev` + 8 × `v_dot4_i32_i8` (inline asm) + 1 fp32 FMA for the scale. That is ~10 % of issue at 735 GB/s. Nibble packing identity: byte *j* of a dword holds weights *j* and *j+16*, so masked-low pairs with `x[4k..4k+3]` and masked-high with `x[16+4k..]` — no deinterleave, no permute, no table.

**Write.** One `float` per row from lane 0 after a 5-step `__shfl_xor` tree. 17 408 rows × 4 B is noise; write-allocate is irrelevant here.

**What this design does not do:** no LDS, no tensor cores, no FP8 (absent on gfx1100), no wave64, no persistent kernel needed to hit 735.

---

## 7. What this means for quantum formats at 3.5 bpw

The measured cost model is **one dependent load stream per 32 weights ≈ 6–11 %** (§3.2). Apply it to the formats, using the layouts documented in `03c-gguf-quant-layouts.md` (`[documented]`, `ggml-common.h`):

| format | bpw | streams per 32 w beyond the weight load | predicted penalty | evidence |
|---|---|---|---|---|
| Q4_0 | 4.5 | 1 (2 B scale, inline → same cache line but a second request) | −6 % | measured 676–682 with a misaligned inline layout `[measured-on-RDNA3]` |
| Q4_K | 4.5 | same one-per-32-w load as Q4_0, just 6-bit packed → **no saving on the stream cost**, plus 16 `dp4a`/32 w and a SWAR unpack for the min term | ≈ −6 %, same as Q4_0 | ALU is 10 % occupied, so the extra `dp4a` is free `[derived]` |
| IQ4_XS | 4.25 | 0 if the 16-entry table is synthesised in registers (6 × `perm` per 8 w) | ≈ 0 | only i-quant whose table is not gathered `[documented]` |
| IQ3_S | 3.44 | **8 dependent scalar grid loads** per 32 weights | −25 % or worse | IQ3_S delivers 236 GB/s eff. vs Q4_0's 409 on RDNA4 `[measured-elsewhere]`, llama.cpp #22821 |
| IQ2_S | 2.56 | 4 × 8 B grid loads + sign plane per 32 w | −30 % | 184 GB/s eff. vs 409 `[measured-elsewhere]`, same table |

The empirical shape of this is already visible in public AMD numbers: on gfx1100 (W7900) llama.cpp's effective weight bandwidth **falls** as bpw falls — Q8_0 487, Q6_K 422, Q4_0 374, Q3_K_S 257, Q2_K 202 GB/s ([PR #20831](https://github.com/ggml-org/llama.cpp/pull/20831)) `[measured-elsewhere]`. Below ~5 bpw that kernel is issue-bound, and the cause is exactly the dependent-load chain above. Two independent confirmations that it is instruction issue and not bytes: a branch-free SWAR `__vsub4` gave **+23 %** on IQ2_S at identical bpw (#28616), and Q4_K gained **+11–34 %** from *cutting* warps 8 → 1 on gfx1100 (#23528) — impossible for a bandwidth-bound kernel.

**Consequence for the plan.** 3.5 bpw is worth having for bytes/token (11.76 GB), but only in a shape whose per-32-weight metadata arrives with the weights and whose LUT — if any — lives in registers. On the evidence, that shape is **Q4_0's nibble packing with an IQ4_XS-style register-synthesised table and super-block scales**, not IQ3_S/IQ2_S's grid format. Nobody has published a LUT i-quant GEMV for RDNA; there is no AMD reference to copy, and the one public per-quant decode table on gfx1100 contains no IQ types at all `[documented]`.

---

## 8. The number to plan against

Weight bytes per token at the target quant: 11.76 GB `[documented]`, `docs/00-facts.md:114`.

| sustained weight bandwidth | ms/token | tok/s, one card | what this is |
|---|---|---|---|
| 472 | 24.9 | 40.1 | torch bf16 GEMM at M=1 — the repo's current reference point |
| 619 | 19.0 | 52.6 | the old "wall"; a short-kernel artefact |
| 650 | 18.1 | 55.3 | **engine-realistic: kernel at 735 minus attention, KV, tails, per-layer launches** |
| 735 | 16.0 | 62.5 | the fused kernel measured in §2.2, alone, on resident weights |
| 775 | 15.2 | 65.9 | the read wall; no dequant kernel will do this |

Cross-check against a shipping engine on this hardware class: llama.cpp at Q4_0 on a gfx1100-class card achieves 374 GB/s of weight traffic end-to-end (48 % of the wall), which is 32 tok/s for this model `[measured-elsewhere]`. A purpose-written engine at 650 GB/s would be **1.7× faster than llama.cpp on the same silicon**. That is the prize, and it is a kernel-quality prize, not a hardware one.

Two corrections this forces on existing notes:
- `02-megakernels.md` uses 19.0 ms (619 GB/s) as the step time. It is 16.0 ms at the kernel ceiling and ~18 ms realistically. The megakernel's ~1 ms prize is unchanged, so its verdict is unchanged — but the argument that "the fixed cost is not the denominator" gets *stronger*, not weaker.
- `docs/03-native-kernel-inventory.md:26` gates the P0 GEMV at "≥60 % of 619 GB/s" = 371 GB/s. That gate is 50 % of what the kernel can do. **A gate of 650 GB/s (89 % of the measured 735) is the one that catches a bad kernel**; at 371 a kernel passes while leaving half the card idle.

---

## 9. What to measure first

In order. Each is a small edit to `research/bench/gemv.cpp`; all were run for this note except where marked.

1. **Establish the wall on this box, in the shipping shape.** `HIP_VISIBLE_DEVICES=1 RP=4 ./bw2 6144 5` → expect 767–777. If a future ROCm/driver changes that by >5 %, every number below moves. *(Done: 768–777.)*
2. **The same fused kernel at the real row count and the real bpw of the chosen quant, over ≥1 GiB, with in-kernel repeats.** `./gemv 400000 5120 soa 4 5 64 8 1` → 733–743 at 4.06 bpw. A 50 MB single-layer benchmark will report >1000 GB/s and mean nothing (IC).
3. **The scale-arrival experiment, before any repack work is scheduled.** Compare the current 4.06 bpw plane against an inline-scale layout that keeps 16 B alignment (72 B or 136 B super-blocks). This decides the on-disk format, and it is worth 6–40 GB/s. *(Half done: the "no scale load at all" ceiling is 773; the aligned-inline variant is not built yet.)*
4. **End-to-end launch tax — the only measurement here that can still change the architecture.** Run the real per-layer kernel sequence for one token, back-to-back, and time the *step*. §1.1 establishes that a 1.6 ms kernel reaches 635–662 GB/s and the same bytes in a 6.4 ms kernel reach 746, and that eight separate back-to-back launches stay at 641. What it does not establish is the per-launch cost, and the two readings have opposite consequences: if the penalty is per *kernel*, ~450 launches cost 10–15 % and layer fusion is worth building; if it is per *millisecond of warm-up* only, it mostly disappears in a real step and the megakernel argument loses its new evidence. Sampled `amd-smi` clock and a step-level timeline settle it in an afternoon. Nothing in the repo has made this measurement.
5. **`i-quant` decode cost, on this card, at fixed model** — the public tables have no gfx1100 IQ data. One llama-bench sweep of Q4_0 / IQ4_XS / IQ3_S / IQ2_S at fixed size tells you whether the 3.44 bpw formats are usable here or whether 3.5 bpw costs more in effective bandwidth than it saves in bytes. Expect IQ3_S to lose ~25 %; verify before designing around it.
6. **Only then** the ALU questions (dp4a vs dot8 vs fp16 dot2), register pressure, and wave-shape tuning. There is 10× issue headroom; these are the last 2 %.

---

## 10. What this note does not establish

- **One card, one session, one driver** (ROCm 10.1 / clang 24, HIP 7.16). §1.1's size/duration dependence is not fully explained; a mechanistic account (DPM state vs DRAM page state vs tail drain) would need `amd-smi` clock sampling inside the timed loop, which was not done.
- **The 735 GB/s figure is a stripped benchmark**, not a GGUF kernel: no bias, no min term, no super-block 6-bit unpack, one fixed shape, activations already quantised to int8. A production Q4_K GEMV should be expected to land 5–10 % lower — which is why §8 plans on 650, not 735.
- **Only the SoA variant was value-validated** (4.8 × 10⁻⁷). The 18 B AoS variant was measured for its memory pattern only; its numbers are bandwidth evidence, not a correctness claim.
- **Attention, KV cache, sampling and the prefill path are untouched here.** At batch 1 the matvec is 90 %+ of the bytes, but a step at 650 GB/s of weight traffic still has to pay for everything else.
- **No IQ-format kernel was built.** §7 is a cost model from public per-quant measurements plus the measured dependent-load penalty; it is the strongest available inference, not a measurement of this card.
- **Grids above 64 blocks per 128-lane unit produced impossible numbers** (1175–1269 GB/s, above the measured read wall) on the fused kernel with in-kernel repeats, and are excluded everywhere in this note. Almost certainly a bookkeeping or partial-execution artefact of that configuration rather than real traffic; it is recorded here rather than silently dropped, and nothing in the note depends on it. Re-derive with `rocprof` counters before trusting any configuration above 2688 blocks.

---

### Appendix — reproduction

```bash
# on rasputin, from the repo: scp research/bench/*.cpp ~/bwprobe/ ; cd ~/bwprobe
hipcc --offload-arch=gfx1100 -O3 -o bw2   bw2.cpp
hipcc --offload-arch=gfx1100 -O3 -o gemv  gemv.cpp

HIP_VISIBLE_DEVICES=1 RP=4 ./bw2 6144 5                 # read wall, 6 GiB, 4 passes  -> 768-777
HIP_VISIBLE_DEVICES=1 ./bw2 1024 40                     # the 1 GiB dip                -> ~640-664
HIP_VISIBLE_DEVICES=1 ./gemv 400000 5120 soa 4 5 64 8 1  # fused GEMV, steady state    -> 733-743
HIP_VISIBLE_DEVICES=1 ./gemv 400000 5120 read 1 5 64 8 1 # quants-only floor            -> 767-772
HIP_VISIBLE_DEVICES=1 ./gemv 400000 5120 d1 1 5 64 4 1   # scale from a register        -> 773
VERIFY=1 HIP_VISIBLE_DEVICES=1 ./gemv 128 1024 soa 1 1 8 1  # bit-exactness            -> 4.8e-07
```
`gemv` args: `<rows> <cols> <soa|aos|seq|read|d0..d6> <rows-per-warp> <reps> <blocks/SM> <in-kernel-passes> <launches>`. `d1` = no scale load, `d4` = scale per 256 w, `d6` = batched shuffle scale.
