# 01 — What we can write gfx1100 kernels in, in 2026, and what we should choose

Target: bespoke inference engine for Qwen3.8-27B on 2× RX 7900 XT (gfx1100, RDNA3, wave32, 84 CU, 19.98 GiB each).
Research date: 2026-10-22. ROCm current release at time of writing: **10.1.0** (LLVM 24, HIP 7.16).

Claim tags:
- `[measured-on-RDNA3]` — measured on this box (or on gfx1100-class hardware by a source we trust enough to quote)
- `[measured-elsewhere]` — measured on other hardware (CDNA, RDNA4, NVIDIA) and being extrapolated
- `[documented]` — vendor docs, headers, compiler source, ISA manual
- `[claimed]` — vendor/community assertion without a number we can verify

---

## 0. Bottom line first

| Need | Write it in | Why |
|---|---|---|
| Bandwidth-bound GEMV / dequant (decode, batch 1) | **HIP C++ (amdclang/hipcc), hand-tiled, inline asm for `v_dot*`** | Only option with full control of LDS layout, `s_waitcnt` placement, and the packed-int dot instructions that do dequant+dot in one issue |
| Large bf16 GEMM (prefill, batched verify) | **HIP C++ with `__builtin_amdgcn_wmma_*_w32`**, hipBLASLt as the baseline to beat | WMMA is reachable from HIP as a compiler builtin on gfx1100; nothing higher-level exposes the RDNA3 matrix-replication fragment layout well enough to tune |
| GatedDeltaNet recurrent path (48 layers) | **HIP C++ single fused kernel**, wave32, state in VGPR/LDS | Recurrence + short K/V means you want one wave to own one head's state across all timesteps; no tile DSL expresses that well |
| Fast iteration / exploration | **Triton (ROCm)** | Real WMMA lowering for `ISAFamily::RDNA3`, live backend, but hits codegen bugs and cannot express LDS swizzle |
| Rust | **Rust host + HIP C++ kernels** (hip-sys/raw FFI or cubecl-hip). Do **not** expect Rust-authored gfx1100 kernels | Rust→SPIR-V (rust-gpu) cannot emit WMMA or AMD ISA; cubecl-hip is real and shipping but a compiler you don't control |

Everything else (CK, rocWMMA, IREE, rocMLIR, OpenCL, Vulkan compute, hand-written ISA, Loom) is either a library you consume, a fallback, or a research bet. Details below.

---

## 1. The five primitives that actually matter on this box

Grading each language option on these, because these are what separate 300 GB/s from 560 GB/s `[measured-on-RDNA3]`:

1. **Packed dot instructions.** `v_dot2_f32_f16`, `v_dot2_f32_bf16`, `v_dot4_i32_iu8`, `v_dot8_i32_iu4`, `v_dot4_f32_fp8_*`. On gfx1100 these are plain VALU ops (no MFMA, no FP8 matrix path) `[documented]` — the Optimization Guide documents `__builtin_amdgcn_sdot4/udot4/sdot8/udot8/sudot4/sudot8` and `__builtin_amdgcn_dot4_f32_fp8_{fp8,bf8}_{fp8,bf8}`, gated on `dot1-insts` / `dot7-insts` / `dot11-insts` target features (clang `BuiltinsAMDGPU.td`).
2. **Wave32 control.** gfx11 runs wave32 by default; you need to *keep* it that way and size blocks in 32-lane units.
3. **LDS layout control.** 64 KB/CU, 32 banks × 4 B; padding/swizzle and `ds_read_b128`/`ds_bpermute` are how you avoid bank conflicts in a dequant staging loop.
4. **WMMA intrinsics.** 16×16×16 only, `f16`/`bf16`/`iu8`/`iu4`, `f32`/`f16`/`bf16`/`i32` accumulate, **no FP8** — documented arch support matrix in rocWMMA shows `f8/bf8` WMMA only on gfx1200/gfx1201 and gfx1250, while `f16`, `bf16`, `i8` at BlockM=N=16 exist for `gfx11` `[documented]`.
5. **Occupancy control.** VGPR budget (≤96 for 16 waves/SIMD), `amdgpu-waves-per-eu`, `__launch_bounds__`.

Anything that cannot express (1), (2) and (3) cannot win the decode kernel. That single sentence eliminates most of the ecosystem.

---

## 2. Option-by-option

### 2.1 HIP C++ via hipcc / amdclang++ → gfx1100 — **the reference implementation**

What it gives you:
- WMMA as a *compiler builtin*, not a library call: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, `..._bf16_w32`, `..._f16_16x16x16_f16_w32`, `..._bf16_16x16x16_bf16_w32` (+ `_tied_w32` variants that pin srcC/result to the same physical register so chained `opsel` writes don't cost a copy), `..._i32_16x16x16_iu8_w32`, `..._iu4_w32`. All require the `wavefrontsize32` feature and target `gfx1100`–`gfx1103`, `gfx1150`–`gfx1153` `[documented]` — <https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html>
- The gfx1100 fragment layout is **not** RDNA4's: 16×16 tile per wave32, lanes 0–15 hold even output rows and 16–31 odd rows, `lane = (i%2)*16 + j`, `VGPR = i/2`; **srcA/srcB must be replicated** across lanes 0–15 and 16–31 (`lane % 16` selects the row/column, the high bit is ignored) `[documented, AMD]`. CUDA muscle memory will produce silently wrong results here. The same page ships a complete `matrix_multiply_rdna3_wmma.hip` with a `ComputePolicy` struct — a good starting skeleton.
- Inline asm to anything: `asm volatile("v_dot4_i32_i8 ...")`, `s_waitcnt`, `ds_bpermute`, `s_sched_barrier`. Plus the builtin route (`__builtin_amdgcn_sdot4`, `ds_bpermute_b32`, `permlane16`, wave reduce/vote) so most kernels never need literals. Documented hub: <https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/introduction.html> (integer dot, float dot, packing, global-to-LDS, DPP/permlane builtins).
- Occupancy: `__launch_bounds__`, `__attribute__((amdgpu_waves_per_eu(M,N)))`, and full LDS control (`__shared__` extern arrays, manual padding). `-mwavefrontsize64` exists but you never want it here.
- Tooling: `rocprofv3` PC sampling + kernel replay, ROCm Compute Profiler roofline, `RGP` for wave-level, `llvm-objdump --mcpu=gfx1100 -d` on the HSACO to *read the assembly you got* `[documented]` — ROCm 10.1 release notes: <https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html>

Costs:
- Compile time is C++ template time. One small `.cpp` with one kernel: seconds. CK-style templated instantiation: minutes, and CK's own build guide warns about build duration `[documented]`. Keep the hot kernels in *flat, non-templated* translation units and you keep the seconds.
- Manual everything: no autotuner, no layout inference. You will write the swizzle by hand and read the SASS to check it.
- gfx1100 is still in the ROCm 10.1 support matrix (RX 7900 XT/XTX, PRO W7900/W7800, RX 7800 XT…) `[documented]` <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html> — this is the single most important maintenance fact in this document: **the box is not orphaned in 2026.**

Verdict: **primary language. Write everything that matters here.**

### 2.2 Triton on ROCm — real, better than its reputation, still not a bespoke-kernel tool

Current state, from the backend source rather than blog posts:
- `third_party/amd/backend/compiler.py` sets `warp_size = 32 if gfx_major >= 10 else 64` — wave32 is the default on gfx11 `[documented, source]` <https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py>
- `HIPOptions` exposes `num_warps`, `waves_per_eu`, `num_stages`, `matrix_instr_nonkdim`, `kpack`, `wgp_cu_mode` ("WGP/CU execution mode for gfx11 and gfx120x"), plus an `llvm_fn_attrs` escape hatch `[documented, source]`. So occupancy and CU-mode *are* controllable.
- WMMA is genuinely lowered, not emulated: `getWmmaVersion()` maps `ISAFamily::RDNA3 → 1`, and `BlockedToWMMA`/`chooseWmmaInstruction` pick a 16×16×16 WMMA intrinsic for `tt.dot` `[documented, source]` `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`. Separately, `kWmmaRestrictedInstsFeature` gates instructions that some SKUs lack (relevant on gfx11).
- The backend is *actively* fixing RDNA3-specific bugs in 2026: "Enable buffer atomics on RDNA3" (#11535), "Disable XCD swizzle on RDNA" (#11992), "Fix atomic_poll timeout crash on gfx11/gfx12" (#10755), "Add mixed FP8/BF8 WMMA lowering tests" (#11579), plus repeated `true16` enable/disable churn on gfx11 (`#10301`, `#10510`, `#11188`) `[documented, GitHub]`.

What it cannot express, that we need:
- **No LDS swizzle / no explicit staging.** You get tiles; you do not get to say "pad row 17, read `ds_read_b128` with this swizzle". For a bandwidth-bound GEMV the whole game is the load pattern.
- **No inline ISA.** `tl.inline_asm_elementwise` is element-wise, not a way to write a load pipeline. You cannot hand-place `s_waitcnt`.
- **No persistent-state kernels.** The GatedDeltaNet recurrence wants one wave to own a head state across many steps; Triton's model is tile-in/tile-out.
- **Open correctness bugs on exactly our chip**: `triton-lang/triton#10808` "AMD backend: fused-MoE triton kernel JIT produces GPU memory access fault on gfx1100/RDNA3" (open since 2026-07-06, reported on 4×W7900) and `ROCm/triton#909` "[triton 3.6]: NaNs and infs in mm kernels on gfx1100" (open) `[documented, GitHub]`. A JIT that compiles fine and faults at runtime is the worst failure mode for a kernel you can't inspect.
- **Compile cost per specialization.** Kernels are cached on disk keyed by a hash (`python/triton/runtime/cache.py`, `TRITON_CACHE_DIR`) `[documented, source]`, so warm runs are free — but autotune multiplies cold compiles by the config count, and every shape/quant variant is a new key. For a 64-layer model with several quant formats this is a build-time engineering problem, not a runtime one. We did **not** measure cold-compile seconds on this box; treat any "Triton compiles in X s" figure as unverified until we do `[unverified]`.

Verdict: **keep it as the exploration and baseline language** (it is the fastest way to find out whether an idea is worth hand-writing), and as the fallback for ops you don't want to hand-tune. Do not build the decode path on it.

Also note: **TileLang** (tile-ai/tilelang) added RDNA3/RDNA3.5 WMMA support in 2026 (`[AMD][Radeon] Add the Support of RDNA3/RDNA3.5(gfx11) WMMA`, PR #2044; release notes list "RDNA3/RDNA3.5 WMMA … and RDNA gfx1151 target") `[documented]` <https://github.com/tile-ai/tilelang/releases>. It is a *better-shaped* tile DSL than Triton for GEMM (explicit `T.copy`, fragment control, more layout say), and AMD's own HRX repo vendors a TileLang importer (`requirements-importers-tilelang.in`) `[documented, source]`. Worth an afternoon; same limitation class as Triton for the recurrent path.

### 2.3 HIPRTC / runtime compilation — cheap, cacheable, and it can be used at build time

Facts that matter for the decision:
- Since HIP 7.0, hipRTC is its own library (`libhiprtc.so`) `[documented]`.
- **Offline compilation is supported**: "This library can be used for compilation on systems without AMD GPU drivers installed (offline compilation)." So the "runtime compiler" is also an *ahead-of-time* compiler with a code-object output you can ship — `hiprtcGetCode` → `hipModuleLoadData` `[documented]` <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html>
- It goes through Comgr, and **Comgr caches compilations** (`AMD_COMGR_CACHE=0` forces recompile) `[documented, same page]`. So there is a real second-level cache; you are not paying clang startup per launch.
- rocWMMA is explicitly hipRTC-compatible (they ship a `hipRTC_gemm.cpp` sample) `[documented]` <https://rocm.docs.amd.com/projects/rocWMMA/en/latest/conceptual/programmers-guide.html>

Costs and the honest framing: hipRTC is *not* fast in the way a modern JIT is. It is clang + LLVM + lld in a shared library: expect hundreds of ms to seconds per translation unit for a real kernel, not the ~**1–15 ms to HSACO** AMD claims for its new Loom compiler `[claimed, AMD]` (see §2.9). The right pattern for us is therefore:

1. Ship kernels as a **build-time artifact** (hipcc → `.hsaco`, embedded in the binary, loaded with `hipModuleLoadData`) — zero runtime risk, best debuggability;
2. Keep hipRTC only for the things that genuinely must specialize at runtime (arch-conditional tile shapes, quant-format fan-out), with `AMD_COMGR_CACHE` on and a cache dir under our control.

Verdict: **use it as an AOT path and a specialization escape hatch, never as the hot launch path.**

### 2.4 rocWMMA and hipBLASLt — what a hand-written kernel gains

**rocWMMA 2.2.x** is a header-only C++17 library wrapping the same `amdgcn_*` intrinsics `[documented]`. On gfx11 it supports `f16`, `bf16`, `i8` at BlockM=N=16 with K as low as 8/16, and **no FP8** `[documented]` <https://rocmdocs.amd.com/projects/rocWMMA/en/develop/api-reference/data-type-support.html>. It is wavefront-centric: "Undefined behaviour is expected if not all threads in each wavefront are active"; fragments carry a *fragment scheduler* describing how many waves in the block collaborate.

What you gain over hand-written: correct fragment layouts for free, cooperative global→LDS loads, `mma_sync` that handles partial/large tiles piecewise since 2.0, and portability to CDNA shapes.
What you lose: the load path. rocWMMA decides how data reaches LDS. For a bandwidth-bound dequant GEMV that is exactly the part you need to own — and note rocWMMA is a *GEMM* abstraction; using it for a GEMV (n=1) means padding fragments you never fill. The library's own docs recommend larger fragments "to maximize memory bandwidth", which is the opposite of what a decode kernel wants.

**hipBLASLt 1.4.1** ships in ROCm 10.1 `[documented]`. It covers more architectures than the legacy rocBLAS/Tensile path — the practical guidance in 2026 HIP ports is "route GEMM through hipBLASLt, rocBLAS does not ship Tensile kernels for every arch" `[claimed, third-party HIP port doc]` <https://github.com/0xShug0/audio.cpp/blob/main/docs/build/HIP.md>. Its value to us is as **the number to beat**, not as a building block: it will not fuse our dequant, our RMSNorm, our RoPE, or our delta-rule recurrence.

Verdict: **rocWMMA = optional scaffolding for the prefill GEMM; hipBLASLt = benchmark target.** Neither replaces hand-written loads. We must measure `hipblaslt-bench` at our exact shapes on gfx1100 before believing any RDNA3 GEMM claim `[unverified]`.

### 2.5 Composable Kernel (CK) — **not CDNA-only**, but it is a library, not a language

Direct evidence, from CK's own build system in `ROCm/rocm-libraries` (CK moved there; the standalone `ROCm/composable_kernel` repo is marked `[DEPRECATED] Moved to ROCm/rocm-libraries`) `[documented]`:
- `CMakeLists.txt` enumerates `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`–`gfx1153`, `gfx11-generic`, `gfx1170/1171`, `gfx1200/1201`, `gfx1250` as first-class targets `[documented, source]`.
- It carries **separate `FORCE_DISABLE_XDL` and `FORCE_DISABLE_WMMA` switches** — i.e. it has WMMA (RDNA) instance families distinct from XDL (CDNA MFMA) ones `[documented, source]`.
- ROCm 10.1 release notes: "Composable Kernel: … speeds up attention on Radeon GPUs and brings more quantized matrix-multiply types to its kernel dispatcher" `[documented]` <https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html>
- Community evidence that the RDNA3 path is usable but not turnkey: a from-scratch RDNA3 flash-attention fork had to modify the CK backend substantially "to realize full support for the RDNA3 architecture" `[claimed]` <https://github.com/HUSRCF/flash-attn-rocm-rdna3>

So: the "CK is CDNA-only" belief is **out of date** (it was roughly true through ROCm 5.x). But CK is a templated kernel *factory*, not a way to write our kernels: adopting it means adopting its tile policies, its build, and its 1.3.0 API churn. Its realistic role is (a) a source of truth for RDNA3 GEMM tile shapes we can read, and (b) a fallback prefill GEMM if our hand-written one loses.

Verdict: **read it, benchmark against it, don't build on it.**

### 2.6 MLIR / IREE / AMD's own compilers

- **IREE** has a first-class `rocm` target that lowers to HSACO through the LLVM AMDGPU backend, accepts `--iree-rocm-target=gfx1100` (and SKU aliases `rx7900xt`/`rx7900xtx`), and runs on a HIP HAL driver; there is also a Vulkan/SPIR-V path `[documented]` <https://iree.dev/guides/deployment-configurations/gpu-rocm/>. It is the most credible "compile the whole model" option on this list, and it is the thing RDNA3 stable-diffusion demos used with WMMA back in 2023 `[documented, GPUOpen]`. For a hand-designed 27B engine with a custom quant format and a custom recurrent layer, IREE's value is as a **reference implementation of the non-hot ops** and as a sanity check on our fusions — not as the delivery vehicle.
- **rocMLIR** is alive but quiet (195 stars, last push 2026-10-15) `[documented, GitHub API]`, and the strategic signal is unambiguous: **MIGraphX 2.18 replaced rocMLIR with rocMLIRTriton as its backend compiler in ROCm 10.1** `[documented, ROCm blog]`. AMD is consolidating on Triton's stack, not on rocMLIR. Betting our engine on rocMLIR in 2026 is betting against the vendor's own direction.
- **LLVM AMDGPU backend directly** (i.e. generating LLVM IR or MLIR ourselves and calling `llc -mcpu=gfx1100`): fully supported, and it is what IREE/Triton/cubecl all do underneath. As a *project* it means writing our own frontend — months of work to reach what HIP gives us on day one. Only justified if we later want to auto-generate kernel *families* (quant-format × tile-shape) from a Python/Rust generator; and even then, emitting HIP C++ strings and calling clang is the cheaper version of the same idea.

### 2.7 Rust — the honest verdict

**The user's premise needs splitting in two.** "CUDA can be written in Rust" is itself only half true: `cust` (last release 2022-02-16) and `RustaCUDA` (2021-07-28) are dead; the live path is `cudarc` (10.5M downloads, updated 2026-09-24) `[measured, crates.io API]` — and cudarc is *bindings + NVRTC*, i.e. Rust host code with kernels still written in CUDA C. Nobody writes production CUDA kernels in Rust today either.

For AMD, here is every real option, graded:

| Path | What it actually is | Kernels in Rust? | gfx1100 WMMA? | Status |
|---|---|---|---|---|
| **rust-gpu** (`rustc_codegen_spirv`) | Rust → SPIR-V → Vulkan (RADV/AMDVLK) | Yes | **No** | 0.10.0-alpha.1 (2026-04-13), nightly-pinned; `spirv-builder` 241k downloads, ~11k recent `[measured, crates.io]` |
| **cubecl / cubecl-hip** | Rust `#[cube]` DSL → own IR → LLVM AMDGPU → HIP runtime (or `hip-cpp` source gen) | Yes, in a Rust-embedded DSL | Partly — cubecl-hip's matmul path is "based on rdna3 by default", rocWMMA for other arches, and rocWMMA path compiles slowly `[documented]` | **Production-viable**: 0.11.3, 1.69M downloads, 764k recent, releases within days `[measured, crates.io]` |
| **hip-sys / raw bindings** | `bindgen` output over `amdhip64` + hiprtc | No — you still write HIP C++ | n/a | Maintained but tiny: 13.9k downloads total, 3.6k recent `[measured, crates.io]` |
| **crate `hip`** | old high-level wrapper | n/a | n/a | **Dead**: version 0.0.0, last update 2024-04-24, 1.6k downloads `[measured, crates.io]` |
| **Rust → SPIR-V → Vulkan compute by hand** (ash/vulkano + glslang/`spirv-builder`) | You write SPIR-V-generating Rust or GLSL/HLSL | Partly | Only via `VK_KHR_cooperative_matrix`, which **RADV does expose on gfx11** (Mesa 23.3) `[documented]`; BF16 needs `RADV_PERFTEST=bfloat16` `[claimed, third-party]` | Works, but you're writing a graphics API, not a compute runtime |
| **AMD's own Rust efforts** | Rust bindings for AMD SMI (`[documented]`: "First-class bindings for Python, Rust, and Go"), AMD hiring Rust engineers for GPU compilers/firmware `[claimed, Phoronix/LinkedIn]` | No | No | Not a kernel language |

The decisive technical point: **rust-gpu targets SPIR-V for Vulkan, and SPIR-V gives you no way to emit `v_wmma_f32_16x16x16_f16`, no way to place `s_waitcnt`, and no way to choose LDS swizzle.** Its own changelog's headline 2026 features are mesh shaders, ray-tracing builtins, `GL_KHR_shader_subgroup` intrinsics, and a `naga`/WGSL target `[documented]` <https://rust-gpu.github.io/changelog/> — a graphics-shader roadmap. There is no cooperative-matrix capability in its changelog, and no cooperative-matrix/WMMA work in its issue tracker (search over 37 relevant open issues surfaces none) `[documented, GitHub search]`. The project's own framing: "we haven't even really started optimizing performance either" `[documented, rust-gpu blog, 2025-07-25]` <https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/>.

Counter-evidence that Rust *around* AMD kernels is real and shipping: **hipfire** — a Rust RDNA-native inference engine with its own runtime, quant formats (MQ3–MQ6, MFP4), HIP kernels, and a Rust crate (`peacemaker-isa`) that **emits gfx1100/gfx1201 machine code directly**, imports/exports ACO disassembly, and produces "byte-identical twins of the hipcc" kernels in its test suite `[documented, repo]` <https://github.com/warpfront/hipfire>. That is the shape of a Rust-on-AMD project that works: Rust for the engine, HIP C++ (or emitted ISA) for the kernels.

**Rust verdict, plainly:** *Rust for AMD GPUs is real as a **host** language and as an **embedded DSL** (cubecl), and it is a dead end as a way to hand-write gfx1100 kernels.* If we want Rust, the honest architecture is Rust host + HIP C++ kernels behind a thin FFI (or cubecl-hip if we accept its compiler), and we should budget for the fact that the C++ kernels are where the performance lives. Anyone promising "write the kernels in Rust, same as CUDA" is describing a workflow that does not exist for AMD in 2026 — and barely exists for NVIDIA.

### 2.8 Everything else real

- **OpenCL.** The compiler path exists (clang OpenCL → AMDGPU) but OpenCL is **not in the ROCm 10.1 component list** (Runtimes/compilers listed are HIP, HIPIFY, LLVM, SPIRV-LLVM-Translator, hipThreads) `[documented, compat matrix]`. No WMMA builtins, no wave32 story you control, commodity vendor support. Not worth a day.
- **Vulkan compute.** Genuinely viable and *shipping* for LLMs (llama.cpp Vulkan is competitive on RDNA3 `[claimed]`), because RADV exposes `VK_KHR_cooperative_matrix` on gfx11 and `VK_EXT_subgroup_size_control` lets you request wave32 per pipeline `[documented]` <https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html>. Costs: SPIR-V toolchain version sensitivity is brutal (one third-party report: newer `glslc` produced a 5× regression, 110 → 19–25 tok/s on RADV, pinned to the distro shaderc `[claimed]`), no `s_waitcnt` control, and shader-compile latency at pipeline creation. For us it is a **fallback portability target**, not the main path.
- **Hand-written RDNA3 assembly.** Fully possible: LLVM MC/`llvm-mc -arch=amdgcn -mcpu=gfx1100`, `ld.lld` → hsaco → `hipModuleLoadData`. hipfire proves the workflow end to end, including round-tripping ACO output `[documented, repo]`. The machine-readable ISA (GPUOpen `isa_spec_manager`) plus the RDNA3 ISA PDF make it tractable `[documented]`. But it is the *last* 5–10% tool: use it only if a specific inner loop's scheduling is provably wrong, and even then prefer inline asm *inside* HIP so the surrounding kernel stays C++.
- **Numba HIP** (AMD publishes `numba-hip` on PyPI, last release 2026-09-23 `[documented, PyPI org page]`) — interesting for prototyping, same class as Triton (no LDS swizzle), and Python in the loop. Skip.
- **hipThreads** (new in ROCm 10.1, maps C++ `std::thread` concepts onto GPU) `[documented]` — irrelevant to kernels, but a signal that AMD is investing in lowering the entry cost rather than in peak-performance ergonomics.

### 2.9 The 2026 wildcard: HRX + Loom (AMD's own client-side stack)

This is new, AMD-authored, and directly aimed at our problem, so it deserves its own section even though it is not production-ready.

- AMD's RFC on llama.cpp (2026-08-17) proposes a **`ggml-hrx` backend** built on **HRX** ("Hip Runtime Extended"): a ~2 MiB `libhrx` runtime, explicit command-buffer API, optional HIP compatibility at "88% HIP API compliance", and a **"Real-time Embedded Loom JIT compiler … capable of specializing and optimizing kernels for the exact hardware on a machine to HSACO in ~1–15 milliseconds"** `[claimed, AMD]` <https://github.com/ggml-org/llama.cpp/discussions/27219>
- Claimed performance vs the better of the two existing backends: **30–50% tok/s on prefill, parity to +15% on non-MTP decode**, measured on RDNA3 (W7900) and RDNA3.5 (Strix Halo) `[claimed, AMD]`. Note the shape of that claim — it is a *prefill/dispatch-overhead* win, exactly what you'd expect from a thinner runtime, and decode is where the gains are smallest. For us, prefill is the less important half.
- **Loom** is a real language, not slideware: `.loom` source files in MLIR-ish textual IR, with `kernel.def`/`launch` regions, `scf.if`, `view.load`, `scalar.fmaf`, and — unusually — **correctness cases and benchmarks embedded in the kernel file** (`check.case`, `check.benchmark`, run by `loom-test`/`loom-benchmark`), plus deployment linking that strips checks from the shipped artifact `[documented]` <https://rocm.github.io/hrx-system/loom/getting-started/first-kernel/>
- The repo labels itself: *"This is early-access runtime infrastructure … This is not an official component within the ROCm stack"* `[documented, README]`. 60 stars, 33 forks, very active `[measured, GitHub API]`. The build command in its README configures `-DIREE_HAL_AMDGPU_TARGETS='gfx942;gfx1151;gfx1201'` — **gfx1100 is absent from the documented configuration** `[documented, source]`, which tells you where AMD's attention is.
- The one thing that matters most: **the llama.cpp `ggml-hrx` PR ships Loom-authored kernels** (`benchmarks/loom/*.tg8.json`, a `dispatch/` layer with command programs, transient allocator, scheduler) `[documented, PR #27218 file list]`. So AMD is writing its next-generation client kernels in Loom, for RDNA3.

Verdict: **watch it, spike it in Q1, do not depend on it.** If Loom reaches "documented WMMA + stable ABI + gfx1100 in the supported target list", it becomes the most interesting option on this page — a kernel language with ms-scale specialization and embedded tests is exactly what a 64-layer engine with per-shape variants wants. Until then it is a 60-star early-access repo.

---

## 3. Comparison table

Grades: ●●● = best in class, ●● = usable, ● = painful, ○ = cannot express.

| Option | dot/ISA asm | wave32 ctrl | LDS swizzle | WMMA on gfx1100 | occupancy ctrl | compile time | debug | maintenance risk | for us |
|---|---|---|---|---|---|---|---|---|---|
| **HIP C++ (amdclang)** | ●●● builtins + literal asm | ●●● | ●●● | ●●● builtins, documented layouts | ●●● | seconds (flat TU) → minutes (templated) | ●●● rocprofv3/RGP/objdump | ●● low; gfx1100 in ROCm 10.1 matrix | **primary** |
| **HIP+rocWMMA** | ●●● (via HIP) | ●●● | ● (library owns loads) | ●●● | ●● | ●● header-only instantiations | ●● | ●● rocm-libraries consolidation | scaffolding only |
| **Triton (ROCm)** | ○ (element-wise asm only) | ●● `warp_size=32` auto, `waves_per_eu` | ○ | ●●● lowered (`RDNA3→wmma v1`) | ●● `num_warps`/`waves_per_eu` | per-specialization, disk-cached | ● open gfx1100 miscompiles | ●● live backend, moving fast | exploration + baseline |
| **TileLang** | ○ | ●● | ●● tile-level layouts | ●● RDNA3 WMMA added 2026 | ●● | per-specialization | ● early | ●● small team | GEMM experiments |
| **HIPRTC** | inherits HIP | inherits HIP | inherits HIP | ●●● rocWMMA-compatible | inherits HIP | 100s ms–s, Comgr cache | ●● logs only | ●● separate lib since 7.0 | AOT + specialization |
| **CK 1.3.0** | ●●● | ●●● | ●●● | ●● WMMA instances exist for gfx11 | ●●● | **minutes** (templates) | ●● | ●● API churn, repo move | read/benchmark |
| **hipBLASLt** | n/a | n/a | n/a | ●● (arch coverage needs measurement) | n/a | n/a | ●● | ● | baseline only |
| **IREE (rocm)** | ○ | ●● | ●● | ●● via LLVMGPU | ●● | whole-model, slow | ●● | ●● | reference/fallback |
| **rocMLIR** | ○ | ●● | ●● | ● | ●● | slow | ● | ●● MIGraphX moved off it | no |
| **Vulkan compute (SPIR-V)** | ○ | ●● subgroup-size ext | ●● workgroup mem | ●● `VK_KHR_cooperative_matrix` (RADV) | ●● | pipeline creation latency | ● RenderDoc/RGP | ●● shaderc version traps | fallback |
| **OpenCL** | ○ | ● | ●● | ○ | ● | fast | ● | ● de-emphasized | no |
| **rust-gpu** | ○ | ○ | ●● workgroup | ○ (no coop-matrix) | ● | nightly-pinned toolchain | ● | ●● 0.10-alpha, shader-focused | no |
| **cubecl-hip** | ○ (DSL) | ●● `PLANE_DIM` | ●● | ●● rdna3 matmul path | ●● | JIT + cache + bundle | ●● | ●● 1.7M downloads, active | viable if Rust host |
| **hand-written ISA** | ●●● | ●●● | ●●● | ●●● | ●●● | instant (no compiler) | ●● objdump only | ●●● you own it, forever | last 5% only |
| **Loom / HRX** | ? | ? | ? | ? (unverified) | ? | **1–15 ms to HSACO** `[claimed]` | ● embedded checks/benchmarks | ●●● early-access, not official | spike, don't depend |
| **MLIR/LLVM DIY** | ●●● | ●●● | ●●● | ●●● | ●●● | you build the frontend | ● | ●●● | no |

---

## 4. Ranked recommendation by workload

### (a) Bandwidth-bound GEMV / dequant — decode at batch 1

**Rank 1: HIP C++ with hand-written load pipeline.** The kernel is: wave32 block, each wave owns a slice of an output row-block; weights streamed as 128-bit loads (`ds_read_b128` staging optional), dequantized inline with `v_dot4_i32_i8` / `v_dot2_f32_bf16` / `v_dot4_f32_fp8_*`, accumulate in VGPRs, `s_waitcnt` placed to keep 4–8 outstanding loads per wave. Everything that matters — vector width, LDS padding, prefetch distance, wave count per CU — is a literal you type. Target: **≥90% of the 619 GB/s we measured**, i.e. ~560 GB/s sustained `[measured-on-RDNA3]`.

**Rank 2: Triton.** Can get close on simple GEMV shapes (its `RDNA3→WMMA v1` path plus `waves_per_eu`), and is the fastest way to get a *number* on day one. Loses because the load pattern is inferred, and because gfx1100-specific miscompiles are open.

**Rank 3: cubecl-hip, if the host must be Rust.** JIT'd, cached, and it does have an RDNA3 matmul path, but you are one abstraction away from the load pattern and its hiprtc/rocWMMA compile costs are documented as slow.

**Do not use:** rocWMMA (GEMM-shaped, pads to 16×16 for n=1), hipBLASLt (no fusion, GEMV is not its shape), Vulkan (no `s_waitcnt`), CK (it will not beat a purpose-built GEMV).

### (b) Large bf16 GEMM — prefill, and batched verify

**Rank 1: HIP C++ + `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`.** There is exactly one matrix instruction family available and AMD has published the fragment layout in detail; the winning move is to read the `matrix_multiply_rdna3_wmma.hip` skeleton, then re-tile for our shapes (hidden 5120, MLP 17408). Remember **srcA/srcB replication across lanes 0–15/16–31** — get it wrong and results are wrong, not slow.

**Rank 2: hipBLASLt** as the baseline; **CK 1.3.0 WMMA instances** as a second opinion. If hand-tiling does not beat both by >10% after a week, keep the library and spend the time on decode.

**Rank 3: Triton / TileLang.** Fastest path to "a GEMM exists and is roughly right". Triton's `bf16` lowering had a real bug in 2026 (`v_dot2_bf16_bf16` rounding; fixed by *removing* the lowering, #11227) `[documented]` — a reminder to check numerics, not just speed.

**Reality check on FP8:** there is no FP8 matrix instruction on gfx1100 and rocWMMA's own support matrix confirms `f8/bf8` WMMA only on gfx1200/1201/1250 `[documented]`. Any FP8 prefill plan has to be software-emulated or abandoned. With 800 GB/s spec / 619 measured, prefill arithmetic intensity is the only place we have headroom beyond bandwidth — plan for bf16, and let quantized *weights* (not activations) carry the win.

### (c) GatedDeltaNet recurrent path — 48 of 64 layers

**Rank 1: HIP C++, one fused kernel per layer-type, state resident in VGPRs/LDS.** The recurrence wants a wave (or a small wave-group) to own one head's state across the whole chunk: no global round-trip per timestep, `ds_bpermute`/`permlane` for the small cross-lane reductions, scalar unit for the per-head constants. This is a *control-flow* kernel with short vectors — precisely where a tile DSL has the least to offer and where hand-written code wins most.

**Rank 2: HIP C++ + inline asm** for the specific inner products if the compiler schedules the dependency chain badly (this is the one place hand-written ISA may be justified).

**Rank 3: Loom** — *if* it matures. Its model (target facts resolved at compile time, embedded correctness cases, ms-scale specialization) fits a kernel with 48 instances and several quant variants better than anything else on this page. Spike it, don't ship it.

**Do not use:** Triton (no persistent per-wave state, no cross-step control), rocWMMA/CK (GEMM abstractions), IREE (won't express the scan without a custom op anyway).

---

## 5. What this means for our box

Measured facts, and what they force `[measured-on-RDNA3]`:

- **619 GB/s measured vs 800 GB/s spec (77%).** Every kernel target should be expressed against 619, not 800. A GEMV at 560 GB/s is *excellent*; at 400 GB/s it is leaving 30% on the floor.
- **11.76 GB of weights per token, at batch 1 → 19.0 ms/token → 52.6 tok/s ceiling on one GPU** `[computed from measured BW]`. That single number is the engine's design constraint: *bytes-per-token is the only currency*. Every fusion that avoids re-reading an activation is worth more than any FLOPs optimization.
- **Two GPUs, 5.88 GB each, if perfectly sharded: 9.50 ms → ~105 tok/s ceiling** — but the measured cross-GPU copy is 14.2 GiB/s (host bounce, no P2P) `[measured-on-RDNA3]`. Byte cost of layer-pipelining is trivial (64 hops × 10 KB = 655 KB = 0.04 ms), so **the cost is latency, not bandwidth**: 64 hops × 20/40/80 µs = 1.3/2.6/5.1 ms, i.e. **7%/13%/21% of the compute time**. That makes pipeline (layer) parallelism over PCIe viable-but-not-free, and tensor parallelism (all-reduce per layer) much worse. Decide this before writing kernels, because it changes the kernel interfaces.
- **KV cache is not the problem at moderate context**: 65.5 KB/token of context across the 16 full-attention layers → 0.43 ms at 4K, 3.5 ms at 32K, 13.9 ms at 128K `[computed]`. At 128K, KV reads rival the weight stream — which is an argument for quantized KV and for keeping attention kernels in the same hand-written family.
- **No FP8, no MFMA, no TMA, no clusters.** RDNA3's WMMA is 16×16×16 only, needs replicated fragments, and gives 512 FLOP/clk/CU for f16/bf16 vs 1024 for iu4 `[documented, GPUOpen]`. Design for bf16 + int4/int8 dequant, not FP8.
- **Wave32 is mandatory and good.** Stay at 32; size blocks in multiples of 32; budget VGPRs to ≤96 for 16 waves/SIMD when the kernel is latency-bound, and let VGPRs grow when it is bandwidth-bound.
- **gfx1100 is still in ROCm 10.1's support matrix** `[documented]` — the vendor has not abandoned the box, which de-risks HIP as the primary language. But note where the *new* investment goes: the documented HRX build configuration targets `gfx942;gfx1151;gfx1201`, not gfx1100 `[documented, source]`. Expect RDNA4/Strix-Halo to get the new toys first.

---

## 6. What to test first (ordered, with pass/fail)

Each of these is a day or less and kills a decision.

1. **Bandwidth reality check on our own kernels.** Write the simplest possible HIP GEMV (bf16 weights, no dequant, `float4` loads, wave32, 4 waves/CU) and measure GB/s with `rocprofv3`. *Pass: ≥550 GB/s.* This calibrates everything else — if a trivial GEMV cannot hit 550, the dequant kernel won't either, and the problem is the load pattern, not the math.
2. **Dequant GEMV with `v_dot4_i32_i8` vs `v_dot2_f32_bf16` vs plain `fma`.** Same kernel, three inner loops, our real quant format. *Pass: the int-dot path is measurably faster than fma at equal bytes.* This is the single highest-value experiment in the project.
3. **hipcc compile-time budget.** Time a cold `hipcc -O3 --offload-arch=gfx1100` on one flat kernel TU, and the same source through `hipRTC` cold and with a warm `AMD_COMGR_CACHE`. *Pass: <5 s per kernel, <60 s for the whole set.* If it fails, move to a `-O2` + manual-asm strategy now rather than later.
4. **`hipblaslt-bench` at our shapes on gfx1100** (M=1 GEMV and M=512 bf16 GEMM at hidden 5120 × 17408). *Pass: we have a number to beat, and we know whether gfx1100 kernels actually exist in hipBLASLt 1.4.1* — currently `[unverified]`.
5. **Triton baseline for the same three ops.** `triton.compile` the GEMV/GEMM in 30 lines, run `rocprofv3`, and record cold-compile seconds and steady-state GB/s. *Pass: Triton within 15% of hand-written on GEMM.* If it is, we only hand-write the GEMV and the recurrent path. Also deliberately run the `ROCm/triton#909` NaN shape to see if it bites us.
6. **Hand-written WMMA GEMM.** Port AMD's `matrix_multiply_rdna3_wmma.hip` policy, re-tile for 5120×17408, and verify against a CPU reference *before* timing (the replication bug produces plausible-looking garbage). *Pass: beats hipBLASLt at prefill shapes.*
7. **GatedDeltaNet single-kernel prototype**, one layer, one head, state in LDS, timed against a naive multi-kernel decomposition. *Pass: >2× over naive.* This decides whether the recurrent path is a fusion win or a scheduling problem.
8. **Loom/HRX spike (timeboxed, 1 day).** Build `hrx-system`, run its SAXPY example with `--device=amdgpu`, then try to express one of our kernels. *Pass: it compiles for gfx1100 and documents matrix ops.* Expected outcome: informative failure, but it tells us whether to revisit in six months.
9. **Two-GPU hop latency measurement** (empty-kernel ping-pong over PCIe, 64 hops). *Pass: <40 µs/hop.* This decides pipeline vs single-GPU-resident.

---

## 7. What is outdated, and where the ecosystem actually moved

- **"CK is CDNA-only."** Out of date. CK 1.3.0 in `rocm-libraries` enumerates gfx1100–gfx1153 and gfx12xx, with separate WMMA (RDNA) and XDL (CDNA) instance switches `[documented, source]`. It was true through ROCm 5.x.
- **"Triton on AMD is CDNA-only / doesn't do RDNA3 matrix ops."** Out of date. `ISAFamily::RDNA3 → wmma v1` lowering exists and RDNA3-specific bugs are being fixed weekly in 2026 `[documented, source + issue tracker]`.
- **"rocMLIR is AMD's MLIR path."** Superseded. MIGraphX 2.18 replaced rocMLIR with **rocMLIRTriton** in ROCm 10.1 `[documented, ROCm blog]`. AMD's compiler gravity is now Triton-shaped.
- **"hipcc is the compiler."** Drifting. rocWMMA's docs state `amdclang`/`amdclang++` are preferred and *"hipcc is also supported, however may be deprecated in future ROCm releases"* `[documented]`. Write our build against `amdclang++ --offload-arch=gfx1100`.
- **"Rust can write GPU kernels, so it can write AMD kernels."** Half-true in a way that matters: rust-gpu is a *shader* compiler with a graphics roadmap, and cubecl is a DSL that compiles through LLVM to AMDGPU. Neither gives us `v_wmma` and `s_waitcnt`.
- **New since 2025 and genuinely relevant:** HRX + Loom (AMD's own thin client runtime with a ms-scale JIT, aimed at RDNA3, kernels authored in `.loom`), and TileLang's RDNA3 WMMA backend. Both are 2026 arrivals and neither is stable.
- **New and relevant to *scheduling*:** wave32/wave64 guidance that used to be folklore is now measured — on RDNA4, wave64 beat wave32 for memory-bound DMMV `[measured-elsewhere]`, while gfx11 has no choice in practice for WMMA. Do not port RDNA4 wave64 conclusions to gfx1100.

---

## 8. Sources

**AMD / vendor documentation**
- RDNA3 dense WMMA builtins (fragment layouts, full kernel example): https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html
- ROCm Optimization Guide index (dot builtins, DPP/permlane, global-to-LDS): https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/index.html
- WMMA on RDNA3 (arch background, FLOPS/clk/CU table, opsel): https://gpuopen.com/learn/wmma_on_rdna3/
- rocWMMA precision/architecture support: https://rocmdocs.amd.com/projects/rocWMMA/en/develop/api-reference/data-type-support.html
- rocWMMA programming guide (hipRTC compatibility, fragment model): https://rocm.docs.amd.com/projects/rocWMMA/en/latest/conceptual/programmers-guide.html
- HIP runtime compilation (offline compile, Comgr cache, `AMD_COMGR_CACHE`): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
- ROCm 10.1 compatibility matrix (gfx1100 support, component versions): https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- ROCm 10.1 release blog (LLVM 24, CK on Radeon, MIGraphX→rocMLIRTriton, hipThreads, profiler improvements): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html
- CK build system (gfx11/gfx12 targets, WMMA vs XDL switches): https://github.com/ROCm/rocm-libraries/blob/develop/projects/composablekernel/CMakeLists.txt
- hipBLASLt 1.4.1 docs: https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html
- IREE ROCm deployment (gfx1100 targets, HIP HAL): https://iree.dev/guides/deployment-configurations/gpu-rocm/
- RDNA3 ISA reference: https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
- Machine-readable ISA: https://gpuopen.com/machine-readable-isa/
- HRX system (libhrx, "not an official ROCm component"): https://github.com/ROCm/hrx-system
- Loom programming guide (first kernel, embedded checks/benchmarks): https://rocm.github.io/hrx-system/loom/getting-started/first-kernel/
- AMD's `ggml-hrx` RFC (Loom 1–15 ms HSACO, 30–50% prefill claim, RDNA3 targets): https://github.com/ggml-org/llama.cpp/discussions/27219

**Compiler source (primary)**
- clang AMDGPU builtins (sdot4/8, dot4_f32_fp8, all `wmma_*_w32/w64` with feature gates): https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td
- Triton AMD backend options / wave32 default: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Triton AMD WMMA selection (`RDNA3 → wmma v1`): https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
- Triton kernel cache: https://github.com/triton-lang/triton/blob/main/python/triton/runtime/cache.py

**Rust**
- Rust GPU ("we haven't even really started optimizing performance"): https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/
- Rust GPU changelog (shader-focused roadmap, subgroup intrinsics): https://rust-gpu.github.io/changelog/
- cubecl (platform/compiler table, AMD = LLVM→AMDGPU + HIP runtime): https://github.com/tracel-ai/cubecl
- cubecl-hip (RDNA3 matmul path, rocWMMA for other archs): https://lib.rs/crates/cubecl-hip
- hipfire (Rust RDNA engine; `peacemaker-isa` emits gfx1100/gfx1201 ISA): https://github.com/warpfront/hipfire
- crates.io API download/version data for `hip`, `hip-sys`, `cubecl`, `cubecl-hip`, `cudarc`, `cust`, `rustacuda`, `spirv-builder` (queried 2026-10-22)

**Issue trackers (failure modes, dated)**
- triton-lang/triton #10808 — MoE kernel faults on gfx1100 (open): https://github.com/triton-lang/triton/issues/10808
- ROCm/triton #909 — NaNs/infs in matmul on gfx1100 (open): https://github.com/ROCm/triton/issues/909
- triton-lang/triton #11227 — bf16 mul wrongly lowered to `v_dot2_bf16_bf16` (fixed by removing lowering): https://github.com/triton-lang/triton/issues/11227

**Third-party, used with caution**
- RDNA3/RDNA4 inference reference (WMMA tables, `RADV_PERFTEST=bfloat16`, shaderc regression report): https://zolotukhin.ai/zinc/docs/amd-gpu-reference/ `[claimed]`
- hipBLASLt-vs-rocBLAS arch coverage, gfx1103 verification: https://github.com/0xShug0/audio.cpp/blob/main/docs/build/HIP.md `[claimed]`
- TileLang release notes (RDNA3/3.5 WMMA): https://github.com/tile-ai/tilelang/releases
- `VK_KHR_cooperative_matrix` spec: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html
- Mesa 23.3 release notes (RADV exposes cooperative matrix): https://docs.mesa3d.org/relnotes/23.3.0.html
