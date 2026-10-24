# gfx11 (RX 7900 XT) streaming-kernel notes: width, cache bits, occupancy, LDS

Tags: **[D]** documented (primary source), **[M]** measured elsewhere, **[C]** claimed/inferred by me.
Local measurements assumed: torch copy 592.9 GiB/s, hand-written float4 read 619 GB/s, 42*8 blocks x 256 threads.

Sources
- **[ISA]** AMD, *"RDNA3" Instruction Set Architecture Reference Guide*, 609 pp — https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf
- **[LLVM]** AMDGPU backend user guide (memory model lowering tables) — https://llvm.org/docs/AMDGPUUsage.html
- **[TD]** `llvm/include/llvm/IR/IntrinsicsAMDGPU.td` (cachepolicy bit layout) — https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
- **[CLANG]** `__builtin_nontemporal_load/store` — https://clang.llvm.org/docs/LanguageExtensions.html#non-temporal-load-store-builtins
- **[SPEC]** ROCm "AMD GPU specifications" (gfx1100 row) — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- **[CC]** Chips and Cheese, *Microbenchmarking AMD's RDNA 3* — https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture
- **[GPUOPEN]** RDNA Performance Guide — https://gpuopen.com/learn/rdna-performance-guide/
- **[BABEL]** BabelStream — https://github.com/UoB-HPC/BabelStream

## 1. Load width per lane
No public load-width sweep on gfx11 exists that I could find; width advice below is **[C]**.
- Reference point: BabelStream issues one scalar element per thread and still reaches a high fraction of peak on AMD because it runs huge grids **[D, project design]** — 32-bit lanes are not disqualifying.
- The relevant quantity is bytes in flight. Little's law at 619 GB/s, 84 CUs, 300-600 ns latency: 181-363 KiB in flight total = **2.2-4.3 wave-wide `dwordx4` loads per SIMD32** (arith **[C]**); at 2 GHz that is one 512 B wave-load per CU every ~139 cycles.
- Conclusion **[C]**: `dwordx4` per lane is not magic; it wins because it needs 4x fewer VMEM instructions and 4x fewer address registers for the same MLP. Anything >=64-bit per lane plus >=4 independent chains reaches the same plateau. Start at `float4`, then A/B a `float2` build.
- 619/800 = 77% of spec. That is likely ~90% of the achievable read ceiling (GDDR6 read efficiency is not 100%); sweep grid size/ILP before changing width **[C]**.

## 2. Expressing 128-bit loads; global vs buffer
- `float4` deref → `global_load_dwordx4`; `__builtin_nontemporal_load((const float4*)p)` **[D, CLANG]** lowers on GFX10-GFX11 to `buffer/global/flat_load slc=1 dlc=1` **[D, LLVM table]**; `__builtin_nontemporal_store` → `glc=1 slc=1 dlc=1` **[D, LLVM]**.
- **Global/flat path**: 64-bit VGPR address, optional SGPR base + **13-bit signed byte** imm offset **[D, ISA field table]**; no bounds check. **Buffer/MUBUF path**: 128-bit V# in SGPRs, "sent to the texture cache when the instruction is executed" **[D, ISA 9.1]**, 32-bit VGPR offset + SGPR `soffset` + 12-bit unsigned imm, bounds-checked, and the only path with V#.`llc_noalloc`.
- Buffer intrinsics: `llvm.amdgcn.raw.buffer.load.<ty>(<4 x i32> rsrc, i32 offset, i32 soffset, i32 cachepolicy)`, cachepolicy **bit0=glc, bit1=slc, bit2=dlc** on gfx10/gfx11 **[D, TD]**. `llvm.amdgcn.raw.buffer.load.lds` (= `buffer_load_dword ... lds`) exists, gated on the `vmem-to-lds-load-insts` feature **[D, TD]**.
- **There is no idiomatic HIP spelling for the buffer path** — it needs the IR intrinsic or inline asm. That alone justifies starting with `global_load_dwordx4` **[C]**. Which of the two is faster on gfx11 for pure streaming is **undocumented**; I would expect parity, with buffers winning only when 32-bit offsets free VGPRs.
- `amdgcn.cvt.pk*` is **not** a cache control — it is fp8/f16 packing (`__builtin_amdgcn_cvt_pk_f16_fp8` etc.) **[D, TD]**. `__builtin_amdgcn_exp` is the graphics export instruction. Both are red herrings for this task.

## 3. Write-allocate avoidance
- Store cache bits are only *temporal hints*: SLC=1 → `STREAM` at GL2, DLC=1 → non-temporal at the MALL; GLC on a store is a **scope** bit, not an allocate policy **[D, ISA 4.1.1 tables]**. (The per-instruction field table says DLC "controls L1", contradicting 4.1.1 — trust 4.1.1 **[D, inconsistency noted]**.)
- So gfx11 has no "no-allocate" store bit on the flat path; the only real no-allocate control is `V#.llc_noalloc` on buffer stores **[D, ISA 4.1.1]**.
- What actually matters: write **whole 128 B lines** (32 lanes x 16 B = 512 B covers 4 lines) so no fill is needed; `__builtin_nontemporal_store` then keeps the lines from displacing the read stream in the 6 MB L2 **[C]**.
- For a copy kernel expect single-digit % from this; for dequant+GEMV (read-heavy, small writes) it is nearly irrelevant **[C]**.

## 4. Occupancy
- ISA defines a CU as **half a WGP, 2 SIMD32** **[D, ISA glossary]**; the 84 "CUs" in `rocminfo`/SPEC are those units. So "65536 VGPRs per CU" in the brief is really **per WGP** (=32768 per CU, 16384 per SIMD32) **[C, arithmetic]**.
- VGPRs: allocated in blocks of 16 (wave32) / 8 (wave64), max **256 VGPRs per shader** **[D, ISA 3.3.2.1]** → 256 regs ⇒ 2 waves/SIMD32; 64 regs ⇒ 8 waves/SIMD32; 32 regs ⇒ 16 waves/SIMD32.
- LDS: **128 KiB per WGP** (2x64 KiB blocks, 32 banks each), max 64 KiB per workgroup **[D, ISA 12.1]**; matches SPEC's 128 KiB column and [CC].
- "32 waves per SIMD" is wrong: 32 waves is a **wave32 count per CU** (2048 threads); per SIMD32 it is 16 **[C, 2048 threads / 32 lanes / 2 SIMD32]**.
- SPEC lists gfx1100 "VGPR File 768 KiB" per CU, i.e. 3x the 65536-dword figure; I could not reconcile it (gfx906=256 KiB, gfx90a=512 KiB match VGPR+AccVGPR). **Do not build occupancy math on that column** — read VGPRs from `--save-temps`/rocprof instead.
- Bandwidth needs almost no occupancy: [CC] measures RDNA2's 256-bit GDDR6 saturating with **10 WGPs** and RDNA3 ramping further. Your 619 GB/s needs only ~3-4 128-bit wave-loads in flight per SIMD32, i.e. **1-2 waves/SIMD32 with 4 independent loads each**. Practical target: **4 waves/SIMD32 (128 VGPRs)**, and do not go below 2. Beyond 4-8 waves the only effect is more L2/DRAM queueing and more s_waitcnt traffic — "more waves helps only when per-thread ILP is low" is the right mental model **[C]**.

## 5. ILP
- 4-8 independent load chains per thread, each `dwordx4`, is the sweet spot **[C]**: 4 chains x 4 waves/SIMD32 = 16 wave-loads outstanding vs the ~4 needed.
- Hard ceiling you must respect: `vmcnt` is a **6-bit** counter (`SIMM16 = {VMcnt[5:0], LGKMcnt[5:0], 1'b0, EXPcnt[2:0]}`) **[D, ISA 5.6]** → at most 63 outstanding VMEM loads per wave can even be tracked; unrolling past ~16 loads/thread buys nothing and burns VGPRs.
- Structure: prologue issues N loads, steady state consumes one and reissues one; `s_waitcnt vmcnt(N-1)` — never `vmcnt(0)` in the loop.

## 6. LDS on gfx11
- 32 banks x 4 B per CU, 1R/1W per bank per cycle; one wave32 b32 access = 1 cycle, up to 64 cycles with conflicts **[D, ISA 12.1/12.5]**; "possible to service a wavefront-wide load every cycle" **[M, CC]**. B64/B128 take 2/4 cycles minimum by bank width **[C]**; explicit broadcast rules for `ds_load_b64/b128` are **not documented** — only direct load broadcasts, and that is CU-mode pixel-only **[D, ISA 12.4]**.
- `[rows][cols]` tile, row stride 32 dwords: bank index = dword index mod 32 ⇒ lane=column, iterate rows is conflict-free; lane=row with fixed column is a **32-way conflict**. Fix by padding stride to 33 dwords (skews each row by one bank) or by `ds_bpermute`-based transpose **[C, standard practice; padding advice also in GPUOPEN]**.
- For a GEMV: **don't stage**. Each weight is consumed once, so LDS adds a hop (global→VGPR→LDS→VGPR) and steals occupancy capacity from nothing. Staging only pays if the tile is reused across a batch of activations (multi-row decode), or if you use `raw.buffer.load.lds` to skip the VGPR round trip **[C]**.

## 7. Pitfalls checklist
- **wave32 default**, wave64 halves instruction count and wave count; VOPD dual-issue is wave32-only **[D, ISA 7.6]**. For streaming, wave32 + 128-bit loads (512 B/instr) is the easy choice.
- **FLAT decrements both LGKMcnt and VMcnt** **[D, ISA 5.6]** — generic accesses cost you two counters and extra address math. Check `llvm-objdump -d` for `flat_load` vs `global_load`; fix with `__restrict__`, `const`, and explicit `global` (addrspace 1) pointers.
- **`s_waitcnt vmcnt(0)` in the inner loop** serialises the wave; wait on `N-k`. Same for `lgkmcnt(0)`.
- **SGPR/scalar loads for scales** are good: one SMEM access serves the whole wave, 16 KiB 4-way scalar cache with ~15 ns load-to-use **[M, CC]**, and SMEM always has SLC=0 **[D, ISA 4.1.1]**. But SMEM returns use LGKMcnt, so over-issuing them adds waitcnt pressure; hoist per-row scales out of the column loop.
- **VGPR bank conflicts** matter for VALU, not VMEM: 4 banks indexed by `SRC[1:0]`, each with 3 read ports, one per operand slot **[D, ISA 7.6]** — only relevant to the dequant arithmetic, not the loads.
- No documented per-wave outstanding-load limit beyond the 6-bit counters; treat >32 as pointless **[C]**.
