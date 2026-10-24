# s3 — RDNA3/gfx1100 performance sinks a CUDA-trained engineer would not look for

**Status:** research note for the PLAN loop. Everything here is either quoted from a primary source,
derived arithmetically from the numbers already in `03-bandwidth.md`, or explicitly marked unverified.
Nothing here was measured on the box in this session — the box was not reachable, so every claim that
needs a measurement carries the command that would settle it.
**Question asked:** which gfx1100-specific costs are invisible to someone reasoning from CUDA, what is
their magnitude for a bandwidth-bound dequant-GEMV, and how do we measure each one.
**Companions:** `03-bandwidth.md` (the 776 / 752 / 516 GB/s measurements this note builds on),
`03b-gfx11-microarchitecture.md` (occupancy, LDS banks, cache bits — this note supersedes its §4
occupancy arithmetic and extends its §2 and §6), `03c-gguf-quant-layouts.md` (per-format instruction cost).

**Tags**

| Tag | Meaning |
|---|---|
| `[D]` | Documented in a primary source (AMD ISA, LLVM source/tables, Linux amdgpu, ROCm docs), quoted. |
| `[D-neg]` | Documented by *absence*: I grepped the primary source and the feature/opcode/field is not there. |
| `[M]` | A real measurement, ours (cited from `03-bandwidth.md`) or someone else's (named). |
| `[C]` | Claimed/inferred/derived here. Arithmetic shown so it can be checked. |
| `[U]` | Unverified. I could not confirm it and we should not design on it. |

---

## 0. TL;DR — ranked by expected magnitude for our kernel

| Rank | Sink | § | Real for a dequant-GEMV? | Expected magnitude | How we'd know |
|---|---|---|---|---|---|
| 1 | Instruction issue ceiling (2 VALU/cycle/CU; no VOPD for any integer op) | §1 | **Yes — the binding constraint below ~4.5 bpw** | **−25 % to −40 %** of the DRAM wall at 3.44 bpw. Explains 516 vs 752 GB/s exactly | `rocprofv3` VALU busy %; instruction count per weight from `-S` |
| 2 | Per-launch fixed cost / clock ramp (the "800 MiB cliff") | §4.3 | Yes, but it is *not* a footprint effect | ~230 µs/launch = 3 % of an 8 ms token, up to 20 % of a 1 ms launch | Pin clocks, re-run the footprint sweep |
| 3 | Quant-scale dependent scalar load stream | §6 | Yes | 697 → 773 GB/s when removed (**+11 %**) `[M]` | A/B with scales preloaded into SGPRs |
| 4 | I-cache thrash from per-CU type heterogeneity | §5 | Only if waves in one CU run different quant types | up to −10 % (issue-slot loss); 0 % if sorted | rocprof I$ hit rate; A/B sorted vs interleaved dispatch |
| 5 | Launch path / queue depth (`sched_hw_submission=2`, `sched_jobs=32`) | §8 | Only for the multi-kernel engine, not the microkernel | 1–5 ms/token at 200–450 launches; hipGraph recovers most | 100-kernel loop with/without `hipGraphLaunch` |
| 6 | No global→LDS path on gfx11 (`cp.async` equivalent absent) | §2.2 | Design constraint, not a runtime cost | kills any LDS-prefetch design; explains the measured 0 % | `llvm-objdump -d` for `buffer_load … lds` / `global_load_lds` |
| 7 | Memory hints (`glc`/`slc`/`dlc`) | §7 | Almost not | 0 % measured; `glc=1` on a load is *negative* | A/B 4 hint combinations on the read wall |
| 8 | MCD interleave / partition camping | §3 | Probably not for a dense sweep | <2 % `[U]` | stride sweep 128 B → 8 MB; per-GL2-channel counters |
| 9 | Second stream / SDMA overlap | §8 | No | 0 % for one GEMV; helps only tail/ramp overlap | 2-stream A/B at 50 % grid each |
| 10 | TLB reach / page size | §4 | **No** — arithmetically excluded as the 800 MiB mechanism | ≤ a few %, and it cannot be 230 µs | §4 arithmetic + the page-count experiment |

The one-line conclusion: **the sinks that matter on this card are instruction *count* and per-*launch*
fixed cost. The sinks a CUDA engineer would predict (LDS staging, cache hints, huge pages, partition
camping, dual-issue) are worth ~0 here, and two of them do not exist on gfx11 at all.**

---

## 1. VOPD dual-issue: real, wave32-only, FP32-only, and useless to us

### 1.1 What the hardware actually accepts

The RDNA3 ISA is unusually explicit, because breaking these rules makes the instruction *not work*:

> "The VOPD instruction encoding allows a single shader instruction to encode two separate VALU
> operations that are executed in parallel. … This instruction format is legal only for wave32. It must
> not be used by wave64's. **It is skipped for wave64.**"
>
> Restrictions: each of the two instructions may use up to 2 VGPRs; at most 1 SGPR each (or share one
> literal); SRC0 may be VGPR/SGPR/constant but **VSRC1 can only be a VGPR**; SRCX0 and SRCY0 must use
> **different VGPR banks**; VSRCX1 and VSRCY1 must use different banks; if both use SRC2, one must be
> even and the other odd; **"Dest VGPRs: one must be even and the other odd"**; the two instructions must
> be independent; **"Must not use DPP."**
> — RDNA3 ISA §7.6, p. 68 of 600
> ([PDF](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf),
> [mirror](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0))

The encoding is 8 bytes (64-bit), i.e. two ops for the price of one 64-bit VOP3 slot `[D]` (ISA Table
"instruction formats", `VOPD 64 bit`).

**The complete opcode set is 14 X-slots and 17 Y-slots, and it is all floating point**
`[D]` (ISA Table 91 / Table 92, p. 169):

```
X: V_DUAL_FMAC_F32  FMAAK_F32  FMAMK_F32  MUL_F32  ADD_F32  SUB_F32  SUBREV_F32
   MUL_DX9_ZERO_F32  MOV_B32  CNDMASK_B32  MAX_F32  MIN_F32
   DOT2ACC_F32_F16  DOT2ACC_F32_BF16
Y: the same 14, plus  ADD_NC_U32  LSHLREV_B32  AND_B32
```

Three consequences that matter here:

1. **`v_dot4_i32_i8` does not participate. Neither does `v_dot8_i32_i4`, `v_perm_b32`, `v_cvt_*`,
   `v_lshrrev_b32`, `v_bfe_*`, or any 3-input integer op.** The only integer instructions that can ever
   appear in a VOPD are `ADD_NC_U32`, `LSHLREV_B32` and `AND_B32`, and only in the **Y** slot. Every
   other integer op in a dequant inner loop is single-issue, full stop.
2. The only dot products in VOPD form are `DOT2ACC_F32_F16/BF16` — *2*-element, FP32-accumulate. That is
   the 123 TFLOPS FP16 number's whole basis (Chips and Cheese make the same point from the ISA:
   only two packed-math VOPD opcodes exist).
3. No DPP ⇒ **a kernel that uses DPP/permute-based nibble unpacking can never dual-issue at all**, and a
   kernel written for VOPD cannot use DPP.

### 1.2 Does the compiler emit it? Partially, and sometimes harmfully

`GCNCreateVOPD` is a **post-RA** pass that pairs *adjacent* operations, gated on
`hasVOPD(*ST) && ST->isWave32()` `[D]`
([GCNCreateVOPD.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNCreateVOPD.cpp),
lines 10 and 310). Post-RA means it cannot re-allocate registers to satisfy the even/odd-destination and
bank rules; it can only take what the allocator happened to leave adjacent.

- Chips and Cheese, measuring RDNA3 through OpenCL: *"We only see convincing dual issue behavior with
  FP32 adds, where the compiler emitted `v_dual_add_f32` instructions. … Fused multiply add, which is used
  to calculate a GPU's headline TFLOPs number, saw very few dual issue instructions emitted."* `[M]`
  ([C&C](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture))
- LLVM #186179: on gfx11 `v_dot2acc_f32_bf16` exists **only** in VOPD form, and the pass could not form it
  from the VOP3P `v_dot2_f32_bf16`; filed Mar 2026, fixed Aug 2026 `[D]`
  ([#186179](https://github.com/llvm/llvm-project/issues/186179)). Any ROCm older than that loses the
  bf16 dot2 dual-issue entirely.
- **LLVM #222620 (open): post-RA VOPD pairing can make a bandwidth kernel *slower* on gfx1100.** The pass
  moves a consumer of buffer-load #1 ahead of independent buffer-load #2 when the two loads reuse the
  resource-descriptor SGPRs, which forces `s_waitcnt vmcnt(0)` before the second load is issued. In the
  reporter's reduced case the pair does not even form, so the serialisation buys nothing.
  `-amdgpu-enable-vopd=0` restores the original order `[D]`
  ([#222620](https://github.com/llvm/llvm-project/issues/222620)). This is the exact shape of a
  multi-load dequant loop: several `buffer_load`s off one V#.

### 1.3 The arithmetic that says VOPD cannot matter for us — and what does

RDNA3 issues **2 VALU instructions per cycle per CU** — one per SIMD32 `[C, arithmetic from the
128 FLOPs/CU/cycle figure]`. VOPD does not create a third issue slot; it packs two ops into *one* slot. So
the quantity that governs an integer dequant loop is **instruction slots per byte**, and VOPD only helps
inside the **17 distinct opcodes** it supports — none of which is an integer dot product. `[D]` rocprofiler-compute
RDNA3 states the same thing from the FLOP side: "peak VALU throughput per compute unit per cycle doubles …
FP32 from 128 to 256 FLOPs/CU/cycle", i.e. the *work per slot* doubles, not the slot count
([ROCm docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/system-speed-of-light.html)).

Now convert bandwidth into an instruction budget. At our measured wall, per CU per cycle:

```
750 GB/s ÷ (84 CU × 2.2 GHz) = 4.06 B/cycle/CU
2 VALU instr/cycle/CU  ⇒  1 instruction per 2.03 B of stream  ⇒  0.49 instr/byte
```

At 4.25 bpw (0.53 B/weight) one `dp4a` consumes 4 weights = 2.13 B. So a *pure* dp4a loop needs
1 instr / 2.13 B, which is **just under budget** ⇒ issue ceiling **785 GB/s**, i.e. DRAM-bound first.
At exactly 4.0 bpw (2.0 B per `dp4a`) the issue ceiling is **739 GB/s** — so the observed 752 GB/s on a
clean 4-bit dequant sits right between the two `[M]`, which is what an "almost free" dequant looks like.

At 3.44 bpw (IQ3_S, 0.43 B/weight) one `dp4a` consumes 4 weights = 1.72 B ⇒ 1 instr / 1.72 B, i.e.
**1.18× over budget** ⇒ predicted instruction-limited ceiling **636 GB/s**. Observed 516 GB/s `[M]`, i.e.
another 1.23× of instruction overhead from the actual dequant arithmetic. `[C, arithmetic]`

So: **at 3–4.5 bpw this kernel is issue-bound, not DRAM-bound, and the missing bandwidth is instruction
count.** VOPD is the only hardware lever that raises the 2-instr/cycle ceiling — and it cannot touch a
single instruction in an integer dequant loop.

**Real sink?** The VOPD *feature*: no. The underlying issue ceiling: yes, and it is the biggest number in
this document. Ranked first.

**Measure.**
```bash
# does the compiler even pair anything, and what does pairing cost?
hipcc -O3 --offload-arch=gfx1100 -S gemv_iq3.hip -o a.s
grep -c 'v_dual_' a.s; grep -c '^ *v_' a.s          # pairing rate
llvm-objdump -d --mcpu=gfx1100 gemv.hsaco | grep -c 'v_dual_'
# A/B the pass (it is on by default; #222620 says it can hurt)
hipcc ... -mllvm -amdgpu-enable-vopd=0
# is the kernel issue-bound at all?
rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS SQ_WAVES SQ_ACTIVE_INST_VALU -- kernel
```

---

## 2. LDS and the buffer path: the CUDA `cp.async` analogue does not exist on gfx11

### 2.1 LDS is 30× DRAM — it can never be the bandwidth constraint

> "There are 128 kB of memory per work-group processor split up into 64 banks of DWORD-wide RAMs. These
> 64 banks are further sub-divided into two sets of 32-banks each where 32 of the banks are affiliated
> with a pair of SIMD32's … Each bank is a 512x32 two-port RAM (1R/1W per clock cycle). … it is possible
> to concurrently execute 32 store or load instructions, each nominally 32-bits"
> — ISA §12.1, p. 118 `[D]`

So per CU: 32 banks × 4 B × 1 access/cycle = **128 B/cycle read**, 256 B/cycle per WGP. At 2.2 GHz:

```
128 B/cycle × 2.2 GHz × 84 CU = 23.6 TB/s   vs   0.75 TB/s DRAM   ⇒ 31×
```

A wave32 `ds_read_b32` with no conflicts completes in 1 cycle; worst case 32-way conflict = 64 cycles
(ISA §12.5: "as little as one cycle … or take as many 64 cycles"). `[D]`

That is the reconciliation of the measured 0 %: staging a stream that is read **once** through LDS costs
one extra instruction per dword (`ds_write` + `ds_read`, or `buffer_load`→VGPR→`ds_write`) and buys
nothing, because the resource that is scarce is *issue slots* (§1), and LDS is 31× oversubscribed.
`[C; consistent with the A/B in 03-bandwidth.md]`

Two more gfx11 LDS facts worth knowing (both from LLVM's GFX11 feature list, `[D]`
[GCNFeatures.td](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td)):
- `FeatureLDSAllocGranularity1024` — LDS is allocated in **1 KB** units per work-group, so a WG that
  declares 1 dword still costs 1 KB against the 64 KB/WG budget.
- `FeatureHalfAddressablePhysicalLocalMemory` — a WG can address only half the physical LDS block
  (64 KB of the 128 KB), i.e. the per-CU half. In WGP mode the two CUs' halves are *not* one pool for a
  single WG.

### 2.2 The load that isn't there

The GCN/gfx9/gfx10 trick for feeding LDS without burning VGPRs is the `lds` bit:
`buffer_load_dword … lds`, `global_load_lds`, i.e. memory → LDS with no VGPR round trip. This is the
closest thing AMD has to CUDA's `cp.async`/`LDGSTS`, and it is what a CUDA engineer reaches for.

**It does not exist on gfx11.** `[D-neg]`

- LLVM defines the feature `VMemToLDSLoad` (`vmem-to-lds-load-insts`): *"The platform has memory to lds
  instructions (global_load w/lds bit set, buffer_load w/lds bit set or global_load_lds…)"*
  ([GCNFeatures.td:1621](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td)).
  It is listed in `FeatureGFX9` (line 1784) and `FeatureGFX10` (line 1819) and appears in **no GFX11
  feature set** — `FeatureISAVersion11_Common` (line 2290) and `11_0_0..11_0_3` do not contain it.
  Nor does `FeatureAsyncLoadToLDSInsts` (that is gfx12+/gfx1250).
- Consistent with the ISA: the MUBUF microcode-format table for gfx11 lists
  `OP, VADDR, VDATA, SOFFSET, SRSRC, FORMAT, OFFSET, OFFEN, IDXEN, GLC, DLC, SLC, TFE` — **no LDS bit**
  (ISA Table 37, p. 81), and the ISA's instruction list has no `*_LOAD_*_LDS` opcode.
- Therefore `llvm.amdgcn.raw.buffer.load.lds` will not select for `-mcpu=gfx1100`; if a design doc for this
  chip says "stage with `buffer_load … lds`", the compiler silently went through VGPRs (or failed).

**Real sink?** Zero at runtime, but it invalidates a whole design family: there is no way to prefetch into
LDS on gfx1100 without paying an instruction per dword, which is exactly what we cannot afford (§1). Any
"stage the weights in LDS to hide latency" plan is dead on this chip, not just slow.

**Measure.** `llvm-objdump -d --mcpu=gfx1100 x.hsaco | grep -E 'lds|ds_write'` — if you expected
`buffer_load … lds` and see `global_load_dwordx4` + `ds_write_b128`, that is the answer. Also
`clang -mcpu=gfx1100 -S -emit-llvm` on a call to `__builtin_amdgcn_raw_buffer_load_lds` should fail to
select.

### 2.3 Cross-lane movement: `ds_swizzle` is free, DPP conflicts with VOPD

- `DS_SWIZZLE_B32` — *"Swizzles input thread data based on offset mask and returns; note does not read or
  write the DS memory banks"* `[D]` (ISA instruction description, p. ~537). It is a pure crossbar: no
  bank conflict is possible, unlike a `ds_read`/`ds_write` round trip.
- `DS_BPERMUTE_B32` / `DS_PERMUTE_B32` do not touch LDS memory either (`[D]`, ISA §12.5.2: "Does not write
  any LDS memory"), but they *do* consume the LGKM counter and an issue slot.
- **DPP is incompatible with VOPD** (§7.6) `[D]`. On gfx11 you also have DPP16 (16-lane rows, ISA
  §15.3.8) — the mask semantics differ from DPP on gfx9, a classic silent-wrong-answer port.

For a GEMV that reduces along the row, the reduction tree is the one place cross-lane movement shows up:
`ds_bpermute`-free reduction (shfl_xor via `ds_swizzle` + `v_add`) is 1 instr per stage, and 5 stages for
32 lanes; it is noise against ~8 dp4a per 32 weights.

---

## 3. Infinity Cache and the 5-MCD structure

### 3.1 The topology (documented)

| Level | Size on our card | Where | Source |
|---|---|---|---|
| L0 vector (TCP) | 32 KB **per CU**, 32-way | GCD | `[M]` C&C |
| L1 (GL1) | 256 KB, 16-way, per shader array | GCD | `[M]` C&C |
| L2 (GL2) | **6 MB**, 16-way | GCD | `[M]` C&C; matches AMD's ROCm table |
| Infinity Cache (MALL) | **80 MB = 5 × 16 MB** | **5 separate MCDs** | `[D]` product page / teardown |
| DRAM | 20 GB GDDR6, **320-bit = 5 × 64-bit**, 800 GB/s | 5 MCDs | `[D]` |

Navi 31 is 1 GCD + up to **6** MCDs, each carrying 16 MB of Infinity Cache and one 64-bit GDDR6 PHY;
the 7900 XTX uses all six (96 MB / 384-bit) and **the 7900 XT ships with five enabled (80 MB / 320-bit)**
`[D]` (AMD product page; consistent with `[M]` third-party teardown tables, e.g.
[hnts03](https://hnts03.github.io/2026-07-13-amd-gpu-arch-3-rdna3-en/),
[zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
The GCD↔MCD link is an Infinity Fanout Link, ~5.3 TB/s aggregate `[C, secondary]`.

Two structural consequences:

- **Every L2 miss crosses a die boundary.** L2 is on the GCD and is only 6 MB; a stream of 10.5 GiB is 1750×
  L2 and 130× Infinity Cache, so essentially 100 % of it is fetched from GDDR6 through an MCD. The IC is
  not a cache you can exploit at this working set; it is a bandwidth amplifier for the *activation* and
  *scale* streams only.
- **L2 and IC line sizes differ from what CUDA intuition expects** — 128 B at GL0/GL1/GL2, 64 B at the
  IC in RDNA4 `[M, secondary]`; for RDNA3 I only have 128 B documented at GL2. `[U]` for IC line size.

### 3.2 Interleaving and camping: what is knowable, and what is folklore

AMD has never published the MCD-selection hash for Navi 31. `[U]` What can be said from the structure:

- With **5** MCDs the select cannot be a power-of-two mask, so it must be an XOR fold of physical-address
  bits `[C]`. Practical implication: **there is no "align the buffer to a channel" trick that works**, and
  a stride that is pathological on a 2ⁿ-channel part (e.g. 4 KB on a 16-channel HBM part) is *less* likely
  to camp here.
- The hash operates on **physical** addresses (it is downstream of translation), so `hipMalloc`'s page
  scatter — not your virtual layout — decides MCD balance. You cannot control it from user space. `[C]`
- Balance is a law-of-large-numbers property: one 128 B line goes to one MCD; 200K pages of 800 MiB is
  ~6.6M lines, so *aggregate* imbalance over any launch ≥ a few MB is statistically nil `[C]`. This is why
  I rank camping last-but-two: it needs a *deliberately strided* access, not a dense sweep.
- The 4 KB page size (§4) gives another safety margin: each page holds 32 lines, so even a hash that only
  folds bits below 4 KB still spreads each page over all MCDs `[C]`.

**Real sink?** For our dense sequential GEMV: **no**, <2 % `[U]`. For a *strided* variant (e.g. reading one
column of a row-major weight matrix, stride = row length) it becomes plausible and worth the 20-minute
sweep.

**Measure** (cheap, no counters):
```bash
# bandwidth vs stride, one binary: read with stride S from a 2 GiB buffer, S = 128B..8MB in 2^n,
# and S = 2^n + 64. A dip >5% that is not a page/line-size artefact = camping.
# then: same total bytes as 1 stream vs 5 streams each restricted to 1/5 of the buffer.
# If 5 restricted streams == 1 stream, the hash is fine at that granularity.
rocprofv3 --pmc GL2C_...  # per-channel GL2 read counters, if the gfx11 metric set exposes them
```

---

## 4. TLB and page size: the facts are boring, and they exclude the 800 MiB cliff

### 4.1 What the driver actually does (all `[D]`, from Linux `amdgpu`)

```c
/* gmc_v11_0.c:791 */
amdgpu_vm_adjust_size(adev, 256 * 1024, 9, 3, 48);
                          /* min VM 256 TB, fragment 9 (=2 MB), 3 levels+root, 48-bit VA */
```
([gmc_v11_0.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/amdgpu/gmc_v11_0.c))

- **PTE size is 4 KB and there is no larger option.** `amdgpu_vm.c` contains no huge-page path: no
  "PDE-as-PTE", no 2 MB PTE emitter, no page-size parameter. `amdgpu_vm_adjust_size()` sets VM size,
  block size and *fragment* size — nothing else
  ([amdgpu_vm.c:2372](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/amdgpu/amdgpu_vm.c)).
  `[D-neg]`
- **`amdgpu.vm_fragment_size` is not a page size.** Module doc: *"Override VM fragment size in bits (4, 5,
  etc. 4 = 64K, 9 = 2M)"*. It is the granularity at which PTE *updates* are batched and PTE pages are
  allocated. Setting it to 9 (the gfx11 default) does **not** create 2 MB translations; it makes each
  `amdgpu_vm_update_ptes` run cover a 2 MB window. This is the single most misread knob in this area.
  ([module params](https://www.kernel.org/doc/html/latest/gpu/amdgpu/module-parameters.html))
- `amdgpu.vm_size` is the *virtual address space* size in GiB. Unrelated to page size.
- `amdgpu.vm_update_mode` picks who writes PTEs (CPU / graphics / compute / both). Perf-relevant only if
  mappings churn; ours do not.
- **No HIP API exposes a page size.** `hipMalloc`, `hipExtMallocWithFlags` (flags are host-mapped /
  uncached / use-Host, not page size), `hipMemCreate`+`hipMemMap` (the VMM API) all land on the same 4 KB
  PTE mapping; the VMM API gives you *address control and aliasing*, which is useful for other reasons,
  but not bigger pages `[D]`
  ([HIP VMM docs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html)).
- **`HSA_ENABLE_LARGE_PAGE` does not appear in ROCm's environment-variable reference.** I could not find
  it, or any equivalent, in the current env-var index `[U→D-neg]`
  ([env vars](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html)). If it is in
  a design note, it is folklore.
- **`hipMalloc` backing, concretely:** a TTM VRAM buffer object, mapped once into the process's KFD VM with
  4 KB PTEs. Physical pages come from TTM's buddy allocator, so they may be contiguous in runs, but the
  *page tables* always map 4 KB at a time — coverage per PTE is 4 KB regardless. `[D]`

### 4.2 Why the ~800 MiB cliff cannot be TLB reach

Take the hypothesis seriously and do the division.

- 12 GiB / 4 KB = **3.15 M pages per launch**.
- 230 µs / 3.15 M = **73 ps per page**. A page walk is ≥ 100 ns of latency and needs ≥1 memory access; even
  at an implausible 500 M walks/s the compulsory misses alone would cost **6 ms**, not 230 µs. `[C]`
- Upper bound on the *bandwidth* cost of translation, for a full 12 GiB pass: PTEs are 8 B, so 3.15 M × 8 B
  = 25 MB of page-table data = **0.2 %** of the stream. Add a 128 B line fetch per page in the worst case
  (upper levels cold, one PT line per 32 pages in the common case) and the range is **0.01 % – 12 %** of
  the pass. At 750 GB/s that is *at most* ~2 ms per 18 ms token, and typically nothing. `[C]`

So translation is worth at most a few percent and **cannot produce a fixed 230 µs that appears at a size
threshold**. The cliff is something else.

### 4.3 The much better explanation, already in our own data

```
800 MiB ÷ 750 GB/s = 1.06 ms
```
and the already-measured fixed cost per launch-or-ramp is **0.2–0.3 ms** (`03-bandwidth.md` §1.1), while
the memory clock sleeps at 96 MHz and takes **~3 ms** to wake `[M]`.

A fixed ~230 µs is *invisible* when the kernel runs 8 ms and dominates when it runs 1 ms. `03-bandwidth.md`
already shows exactly that shape with the same buffer, varying only passes-per-launch (1 GiB: 662 GB/s one
pass → 746 GB/s four passes). **The "800 MiB footprint cliff" is most likely not a footprint effect at all —
it is the known per-launch/ramp cost crossing over against a shrinking kernel.** `[C]` The 800 MiB ↔ 1.06 ms
coincidence is the tell: the "cliff" is where fixed cost stops being amortised, and 800 MiB is what 1 ms of
streaming weighs on this card.

**Decisive, 10-minute experiment** (do this before designing around either story):

```bash
# 1. pin the clock and re-run the footprint sweep. Sweep 256MiB..4GiB, 1 pass, same binary.
rocm-smi --setperflevel high          # or: --setsclk 4 ; verify pp_dpm_mclk stays 1249
#    -> cliff survives          => not DPM; go to step 2
#    -> cliff vanishes          => it was the ramp; stop here, budget 230us/launch
# 2. is it pages or bytes? Same total bytes, three shapes:
#    (a) one 1.2 GiB buffer  (b) 3 x 400 MiB  (c) 1500 x 800 KiB
#    -> cost follows total bytes touched = translation; follows allocation count = allocator
# 3. one 1.2 GiB buffer, touch only a rotating 200 MiB window per launch
#    -> if cost follows the *window*, it is translation/reach; if the *allocation*, it is not
# 4. kernel-side knobs (each answers a different mechanism):
amdgpu.vm_fragment_size=4   # 64K fragments instead of 2M -> moves PTE-update batching only
amdgpu.vm_update_mode=0
# 5. read /sys/kernel/debug/dri/*/amdgpu_vm_info and dmesg for VM flush / retry messages
```

If step 3 says translation after all, the ceiling is the 0.01–12 % bound above, and the only real lever is
*spatial locality within a launch* (one kernel per layer, each touching a contiguous slice, keeps the PTE
working set small) — which happens to be what a per-tensor engine does anyway.

**Real sink?** As a TLB story: no. As a per-launch fixed cost: yes, and it is #2 in the ranking.

---

## 5. Instruction cache: 128 B lines, 32 KB per WGP, and the failure mode is *heterogeneity*

- **Line size on gfx11 is 128 B**, not 64. `[D]` `FeatureInstCacheLineSize128` is in `FeatureGFX11`
  (GCNFeatures.td:1858 vs 1892 for gfx10); LLVM's `getInstCacheLineSize()` returns 128 when that bit is set.
- **Capacity: 32 KB per WGP**, per AMD's own RDNA architecture table — *"Instruction Cache (I$) 32KB per
  WGP (~2CUs), 64B"* for RDNA1 `[D]`
  ([RDNA_Architecture_public.pdf](https://gpuopen.com/download/RDNA_Architecture_public.pdf), p. ~13). AMD
  has not restated it for RDNA3 and the 609-page RDNA3 ISA contains no cache capacities at all (`[D-neg]`:
  grep for "instruction cache" returns only `S_ICACHE_INV`), so treat 32 KB as family-level and the *line
  size* as the gfx11-specific fact. It shares the SQC with the 16 KB scalar (K$) cache.
  `[U]` for exact RDNA3 capacity.
- That is **256 lines shared by every wave in both CUs of a WGP**.

Now size an 11-type dispatch kernel. A dequant block body is 150–400 VALU instructions, 4–8 B each
(VOP2 = 4 B, VOP3P/VOPD = 8 B):

```
per type:  150-400 instr × 4-8 B  =  0.6 - 3.2 KB
11 types:                           6.6 - 35 KB   vs 32 KB of I$ and 256 lines
```

So the *code* nearly fills the cache at the top end, before loop bodies, epilogue and the shared
prologue. But the important question is not total size — it is **which types are resident in one WGP at the
same time**:

- If all waves in a CU execute the *same* type (blocks sorted by type, one kernel per type, or a persistent
  kernel with contiguous per-type runs), the streaming footprint is one type ≈ 1–3 KB = 8–24 lines. No
  thrash. The 11-type dispatch costs nothing.
- If blocks of different types are interleaved in the grid (the natural `switch (type)` megakernel), the
  hardware distributes consecutive WGs round-robin, so a WGP routinely holds waves from 2–4 different types;
  it then ping-pongs between 2–4 bodies plus the epilogue and evicts on almost every fill. Each miss costs an
  issue slot and a 128 B fetch from GL1.

This is a real gfx11 trap because the I$ is *shared per WGP*, not per CU (CUDA's I$ is per-SM, and SMs are
independent schedulers — the sharing makes AMD's failure mode strictly worse).

**Real sink?** Conditional: 0 with type-sorted dispatch, up to ~10 % with a naive interleaved dispatch at
3–4 bpw (where every issue slot is already contended, §1).

**Measure.**
```bash
rocprofv3 --pmc SQ_IC_.../SQ_INSTS_CACHE... -- kernel     # I$ hit rate (rocprof-compute shows "Instruction
                                                         # Cache Hit Rate" + "Wave Issue Wait" for RDNA)
# A/B: same total work, grid ordered by type vs interleaved. If interleaved is slower, this is why.
```

---

## 6. Scalar path and constant loads

- **Kernel arguments are free at runtime.** HSA/ROCm preloads kernarg pointers into user SGPRs; `s_load` of
  an argument is typically 2 instructions in the prologue, once per wave. Not a sink. `[D]`
  (ABI: user SGPRs for kernarg ptr/queue, ISA §"User SGPRs")
- **SMEM is 16 KB, 4-way, ~15.4 ns load-to-use** `[M]` (C&C), and **SMEM operations always have SLC=0**
  `[D]` (ISA §4.1.1) — you cannot make a scalar load streaming at the L2. Only `S_BUFFER_LOAD` can carry
  `llc_noalloc`, via the V# bit; `S_LOAD` always has `LLC_NOALLOC=0` `[D]` (ISA §4.1.1 note). If your scale
  stream is large (per-block scales over 10 GiB of weights is order 100 MB), that stream **allocates
  normally in L2 and IC and cannot be marked non-temporal**. Use `s_buffer_load` with `llc_noalloc` if you
  care.
- **The real sink here is the second dependent load stream** — already measured: 697 → 773 GB/s when the
  per-32-weight scale load was removed `[M]` (`03-bandwidth.md`). +11 %.
- gfx11 accounting: `s_load`/`s_buffer_load` complete on **LGKMcnt**, `ds_*` writes on **VSCNT**
  (`FeatureVscnt`), VMEM on **VMCnt**. Three counters, not one. A loop that interleaves scalar loads,
  LDS writes and global loads needs three `s_waitcnt` clauses and the compiler will happily serialise on
  the wrong one. Look for `s_waitcnt lgkmcnt(0)` in a hot loop. `[D]` (ISA §5.6; `FeatureVscnt` in the
  GFX11 feature list)

---

## 7. Memory hints: three bits, and only two of them do anything

### 7.1 There is no `sc0`/`sc1` and no `NT` bit on gfx11

LLVM's own intrinsic documentation gives the per-generation encoding in one place `[D]`
([IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td),
lines 1591-1596):

```
bit 0 = glc, bit 1 = slc, bit 2 = dlc          (gfx10/gfx11)
gfx942:  bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
gfx12+:  bits [0-2] = th, bits [3-4] = scope, bit 6 = swz
```

`sc0`/`sc1`/`NT` are **gfx942 and gfx12+** encodings. On gfx1100 there are exactly three control bits in
the instruction plus one in the descriptor. Any design note that assigns work to `sc0`/`sc1`/`NT` on this
card is a transcription from a newer ISA and will be silently ignored (worse: the immediate will land in an
unrelated bit field if you hand-write asm). `[D]`

### 7.2 What the three bits actually do (ISA §4.1.1, pp. 35-37)

| Bit | Load behaviour | Verdict for our stream |
|---|---|---|
| `GLC=1` | "forces a miss in the first level cache and reads data from the L2 cache. If there was a line in the GPU L0 that matched, it is invalidated; L2 is reread." Also flips scope CU→DEVICE. | **Negative.** It is a *coherence* bit, not a no-allocate hint. Bypassing+invalidating L0 costs nothing when there is no reuse, but it also forbids L0 write-combining across the wave. Leave 0. |
| `SLC=1` | GL2 policy → `STREAM` / `HIT_EVICT` ("Temporal Hint for graphic client caches: 1 = Stream (non-temporal)"). | The one that matches our intent. Cheap. Measured 0 % on the read wall `[M]`. |
| `DLC=1` | "Temporal Hint for Infinity Cache: 1 = Non-temporal" → MALL marked evict-first. | Correct intent, tiny effect: the stream is 130× IC, so it evicts itself anyway. ≤1 % `[C]`. |
| `V#.llc_noalloc` (buffer path only) | MALL NOA. The **only** true no-allocate on a load, and the only no-allocate on a store. | Only reachable via `buffer_load`/`buffer_store` (IR intrinsic or asm), not via `global_load`. |

`__builtin_nontemporal_load` on gfx10/11 already lowers to `slc=1 dlc=1` `[D]` (LLVM's equivalence table,
see `03b` §2), i.e. the maximum hint available on the flat path — and it measured 0 % `[M]`. That closes
the question: **hints are not where the missing bandwidth is.**

One asymmetry worth remembering: **on stores, `GLC` is not an allocate policy at all** — it is a scope bit,
and all stores are device scope (ISA §4.1.1) `[D]`. So there is no "write-through, no-allocate" store on the
flat path; only the V# bit.

---

## 8. Async compute, SDMA, and what a graph buys

- **SDMA is a separate engine** and runs concurrently with compute; a `hipMemcpyAsync` on another stream does
  not steal CU throughput. Its only use in a decode loop is weight upload / embedding gather / all-reduce
  staging, not the GEMV. Device count = `hipDeviceProp_t.asyncEngineCount` (query it; typically 2 on big
  Navi, `[U]` for Navi 31).
- **A second HIP stream cannot help a bandwidth-bound kernel**, because DRAM and the MCDs are the shared
  resource. It can help in two narrow ways: (a) overlapping the *ramp/tail* of two half-size kernels (if the
  ~230 µs cost is per launch and not per clock-ramp); (b) overlapping an SDMA copy. Test (a) is cheap and
  directly probes the §4 mechanism.
- **Queue depth is a real, documented limit.** `amdgpu.sched_hw_submission` default is **2** — at most two
  hardware submissions in flight per ring; `amdgpu.sched_jobs` default **32** — software queue depth
  `[D]` ([module params](https://www.kernel.org/doc/html/latest/gpu/amdgpu/module-parameters.html)).
  A 200–450-launch decode step against a 32-deep SW queue means the host submitter blocks, and the GPU can
  drain before the next launch is programmed. These are the first two knobs to try (`sched_jobs=1024`,
  `sched_hw_submission=8`) — free to test, restart-only, plausible 1–3 ms/token at 450 launches.
- **`GPU_MAX_HW_QUEUES`** is the ROCm device-wide cap on work queues; exceeding it aliases streams onto the
  same queue and serialises them `[D]`
  ([HIP execution contexts](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/execution_context.html),
  [ROCm env vars](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html)). If a
  multi-stream design shows no overlap, check this before blaming the hardware.
- **hipGraph** removes per-node host cost: argument marshalling, dependency resolution, doorbell writes
  `[D]` ([HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)).
  ROCm is explicit that for direct C++ HIP launches the per-launch overhead is small — which makes our
  measured **8–16 µs device-visible per launch** (`03-bandwidth.md`) suspicious: either that number is the
  ramp (§4), or it is host-side and a graph will delete it. Discriminate with 100 identical kernels
  (`T_100/100 - T_1`) vs one `hipGraphLaunch` of 100 nodes.

**Real sink?** For the microkernel: 0. For the engine: the launch path is 1–5 ms/token out of ~18 ms, i.e.
the same order as the whole "cliff". Worth an afternoon before any microarchitectural work.

---

## 9. Everything else that bites on gfx10/gfx11

**Occupancy arithmetic — `03b` §4 needs correcting.** gfx1100 (LLVM `FeatureISAVersion11_0_0`) has
`Feature1536VGPRs`, documented in-tree as:

> "1536 physical VGPRs for wave32 and 768 VGPRs for wave64 **with allocation granularity 24 for wave32 and
> 12 for wave64**" `[D]`
> ([GCNFeatures.td:1276](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td))

With `FeatureMaxWavesPerEU16` (16 waves/SIMD32 = 1024 threads/CU, also in `FeatureGFX11`):

| VGPRs/thread (wave32) | Rounded to (×24) | Waves/SIMD32 = 1536/regs | Effective |
|---|---|---|---|
| ≤ 96 | 96 | 16 | **16 (slot-limited)** |
| 97–120 | 120 | 12.8 | 12 |
| 121–144 | 144 | 10.6 | 10 |
| 145–192 | 192 | 8 | 8 |
| 193–256 | 264 | 5.8 | 5 |

So the full-occupancy threshold is **≤96 VGPRs**, and the granularity that matters is **24**, not 8 or 16.
Notably `Feature1536VGPRs` is on gfx1100/gfx1101 **only** — gfx1102 (7900 GRE / 7800-class) has 1024, so an
occupancy model tuned on one Navi 31 SKU does not transfer. `[D]`

**XNACK does not exist on this part, and that is a design constraint.** `gmc_v11_0_early_init` sets
`adev->gmc.noretry_flags = AMDGPU_VM_NORETRY_FLAGS_TF;` `[D]` (gmc_v11_0.c:635), and the module doc for
`noretry` says on ASICs without per-process XNACK this "also disables retry page faults". No demand paging,
no lazy SVM migration: a 12 GB weight buffer must be genuinely resident, and a fault is fatal. Any plan that
relies on `hipMallocManaged` oversubscription or on touching more than VRAM is dead on gfx1100. `[D]`

**Wave scheduling granularity / tail.** Resident capacity is 84 CU × 2 SIMD32 × 16 waves = **2688 waves**
(86,016 threads). A grid that is not a multiple of the resident work-group count pays a
partial second dispatch; for a 1–3 ms launch that is a percent, for a 100 µs launch it is 30 %. Prefer
grid = k × resident WGs, or a persistent kernel. `[C]`

**`s_waitcnt` in the loop.** `vmcnt(0)` serialises; wait on `N−k`. Also `vmcnt` is 6 bits — at most 63
tracked outstanding VMEM ops, so unrolling past ~16 loads/chain buys nothing. `[D]` (ISA §5.6; in `03b` §5)

**FLAT costs two counters.** A `flat_load` decrements both LGKMcnt and VMCnt; `global_load` (addrspace 1)
does not. This is free to check and free to fix. `[D]` (`03b` §7)

**`S_ICACHE_INV` exists and is per-WGP** — "Invalidate first-level shader instruction cache for the WGP
associated with this wave" `[D]` (ISA). Do not put one in a loop; it is the one way to *manufacture* the §5
problem.

**One genuinely large lever nobody has tried on this card:** `v_dot8_i32_i4` (gfx11 `FeatureDot8Insts` `[D]`)
does 8 4-bit MACs per instruction instead of dp4a's 4. At 3.4 bpw, repacking IQ3_S codes to 4 bits raises
bytes/weight by ~16 % but halves the dp4a count, moving the issue ceiling from 636 GB/s to ~1.2 TB/s — i.e.
back to DRAM-bound. This is the only way to get 2× instruction headroom without VOPD. `[C, arithmetic;
unverified that IQ3_S's 3-bit codebook can be 4-bit packed without a table lookup]`

---

## 10. What I would actually do, in order

| # | Experiment | Cost | Answers |
|---|---|---|---|
| 1 | Pin clocks (`--setperflevel high`), re-run the 256 MiB–4 GiB footprint sweep | 30 min | Is the 800 MiB cliff DPM/ramp or real? Kills or confirms §4 in one shot |
| 2 | Count instructions per weight for each quant type; compute `bytes/instr` vs 2.03 B budget | 1 h | Which types are issue-bound, and by how much (§1) |
| 3 | `-mllvm -amdgpu-enable-vopd=0` A/B | 15 min | Is VOPD helping or (per #222620) hurting |
| 4 | Preload scales into SGPRs / restructure the second load stream | 1 day | The known +11 % (§6) |
| 5 | Type-sorted vs interleaved dispatch for the 11-type kernel | 2 h | §5 magnitude, 0 % or −10 % |
| 6 | `sched_jobs=1024 sched_hw_submission=8`; hipGraph of 100 nodes | 1 h | Launch-path share of the token budget (§8) |
| 7 | Stride sweep for MCD camping | 30 min | Closes §3 as <2 % or not |
| 8 | Try `v_dot8_i32_i4` on a repacked 3.4 bpw format | 2 days | The only 2× instruction lever (§9) |

## 11. Explicitly unverified

- MCD address-hash structure and whether any stride camps on our 5-MCD card. (§3)
- Infinity Cache line size on RDNA3 (64 B is documented for RDNA4 only). (§3)
- RDNA3 I$ capacity (32 KB is the RDNA1 figure; the RDNA3 ISA publishes no capacities). (§5)
- `asyncEngineCount` on Navi 31. (§8)
- Whether `HSA_ENABLE_LARGE_PAGE` exists in any ROCm version; it is absent from the current env-var index.
  (§4)
- The actual mechanism of the ~230 µs per-launch cost. §4 gives the arithmetic that *excludes* translation
  and points at ramp; that is a hypothesis with a 30-minute test attached, not a finding.

## Sources

1. AMD, *"RDNA3" Instruction Set Architecture Reference Guide*, 600 pp — [gpuopen.com PDF](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf) · [AMD docs mirror](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0). §7.6 p.68 (VOPD), Tables 91/92 p.169 (opcode set), §4.1.1 pp.35-37 (GLC/SLC/DLC + policy tables), §1.2.2.1 p.7 + §12.1 p.118 (LDS banks), §12.5 (conflicts, 1–64 cycles), Table 37 p.81 (MUBUF fields), §5.6 (waitcnt).
2. LLVM AMDGPU backend, `GCNFeatures.td` — feature sets per generation, `VMemToLDSLoad`, `1536VGPRs`+granularity, `MaxWavesPerEU16`, `InstCacheLineSize128`: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td
3. LLVM, `GCNCreateVOPD.cpp` (post-RA, wave32-gated): https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNCreateVOPD.cpp
4. LLVM, `IntrinsicsAMDGPU.td` (cachepolicy bit layout per generation): https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
5. LLVM #186179 — bf16 `v_dot2acc` VOPD never formed on GFX11: https://github.com/llvm/llvm-project/issues/186179
6. LLVM #222620 — VOPD pairing serialises independent buffer loads on gfx1100: https://github.com/llvm/llvm-project/issues/222620
7. LLVM, AMDGPU instruction syntax (VOPD `::` encoding): https://llvm.org/docs/AMDGPUInstructionSyntax.html
8. Chips and Cheese, *Microbenchmarking AMD's RDNA 3 Graphics Architecture* (VOPD measured, L1/L2/IC sizes, SMEM latency, LDS structure): https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture
9. AMD, *RDNA Architecture* whitepaper, cache table (I$ 32 KB/WGP, K$ 16 KB/WGP): https://gpuopen.com/download/RDNA_Architecture_public.pdf
10. ROCm Compute Profiler, RDNA3 system speed-of-light (VOPD definitions, 2 instr/cycle basis, cache rows, "Wave Issue Wait" as I$ symptom): https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/system-speed-of-light.html
11. Linux `amdgpu` module parameters (`vm_size`, `vm_fragment_size`, `vm_update_mode`, `sched_jobs`, `sched_hw_submission`, `noretry`): https://www.kernel.org/doc/html/latest/gpu/amdgpu/module-parameters.html
12. Linux `gmc_v11_0.c` (`amdgpu_vm_adjust_size(adev, 256*1024, 9, 3, 48)`, `noretry_flags`): https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/amdgpu/gmc_v11_0.c
13. Linux `amdgpu_vm.c` (no huge-page path; fragment semantics): https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/amdgpu/amdgpu_vm.c
14. HIP virtual memory management (VMM API, no page-size control): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html
15. HIP graphs: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
16. ROCm environment variables (`GPU_MAX_HW_QUEUES`): https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html
17. AMD RX 7900 XT product page (800 GB/s, 320-bit, 80 MB Infinity Cache): https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html
18. hnts03, *AMD GPU Architecture #3: RDNA 3* (MCD = 16 MB + 64-bit PHY, 6-on-XTX / 5-on-XT, fanout link): https://hnts03.github.io/2026-07-13-amd-gpu-arch-3-rdna3-en/
19. zolotukhin.ai, *AMD RDNA3/RDNA4 GPU Reference for Inference* (independent cross-check of 7900 XT specs and occupancy table): https://zolotukhin.ai/zinc/docs/amd-gpu-reference/
20. Local: `03-bandwidth.md` (776/752/516 GB/s, per-launch fixed cost, launch overhead 8–16 µs), `03b-gfx11-microarchitecture.md`, `03c-gguf-quant-layouts.md` (ISA-level per-quant instruction cost).
