# aiter on gfx1100 — 9127c94a1 (v0.1.17.dev110, 2026-06-25)

**Read first-hand from** `/sgl-workspace/aiter` inside container `sglang-build`
(`rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0`), git HEAD `9127c94a1`,
`aiter/_version.py` = `0.1.17.dev110+g9127c94a1`, commit date 2026-06-25.
That checkout is **1121 commits behind `ROCm/aiter` main** (GitHub compare API,
`f495a610...9127c94a1`). Upstream claims below are against main as of 2026-10-22.
sglang side is our vendored tree at `d0a75654368a470d8484e3ea45b06e500fee57af`.
No GPU work was run; the ISA claims are from a 3-line `hipcc -c` probe.

## Verdict

1. **aiter cannot serve sglang's default attention path on gfx1100** — the paged-decode kernel is hand-written MFMA assembly, and MFMA does not exist on RDNA3; the file is additionally rejected by a second allow-list that has no gfx1100 entry at all.
2. **But aiter is not uniformly dead on gfx1100**: its Triton subtree — including `unified_attention`, which carries explicit `arch.is_rdna` tuning — compiles for gfx1100, and upstream main lists the W7900 as *Experimental* on exactly that basis.
3. **Our crash is neither of those.** `reset_verify_attn_plan_cache` is sglang's own symbol, collateral damage from a `try/except ImportError` that hides a missing aiter; the real fault is that aiter was never installed in `~/venv/sglang-fork`. It is a broken call path, not an architecture verdict.

So: **(b) for the HIP/ASM/CK kernels sglang needs, (a) for the Triton ones, (c) for what we actually observed.**

## Architecture gates in aiter

| module | gate | gfx1100 outcome | evidence |
|---|---|---|---|
| `paged_attention_ragged` (**sglang's default decode**) | `csrc/cpp_itfs/utils.py:131-147` allow-list = `native,gfx90a,gfx940,gfx941,gfx942,gfx950` — **no gfx1100** | **(b)** `AssertionError: One of GPU archs of ['gfx1100'] is invalid or not supported` before any codegen | `csrc/cpp_itfs/pa/pa_ragged.py:65` → `compile_template_op` → `compile_lib` → `validate_and_update_archs()` |
| `paged_attention_ragged` kernel body | hard-coded MFMA builtins, no gfx11 branch | **(b)** clang: `'__builtin_amdgcn_mfma_f32_16x16x16f16' needs target feature mai-insts` | `csrc/cpp_itfs/pa/pa_common.cuh:96,100,127,141-151`; only arch switch is `#if defined(__gfx950__)` at `pa_kernels.cuh:358,744` |
| `mha_batch_prefill_func` (extend) | `get_gfx() in ("gfx942","gfx950")` / `"gfx1250"` | falls to CK (`mha_varlen_fwd`); CK build for gfx11 **unverified** | `aiter/ops/mha.py:1573-1574` ("fmha v3 is hand-written gfx9 ASM; non-gfx9 must fall back to ck-tile"), gates at `:1556,1631,2465,2482,2506,2738` |
| `unified_attention` (Triton) | none — explicit RDNA branch | **(a)** compiles; tuned per-arch, correctness not measured here | `aiter/ops/triton/attention/unified_attention.py:72-73,84,91` (`TILE_SIZE 16 if is_rdna else 64`, `waves_per_eu 6`, `BLOCK_M 64`) |
| `flash_attn_triton_amd` | `RDNA_ARCHS` = {gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201} | **(a)** compiles, with `BLOCK_N = 64 if arch.name == "gfx1100"` | `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py:56-70`, `fwd_prefill.py:82,199` |
| FP8 paths (`e4m3`/`e5m2` attention, fp8 GEMM) | `FP8_ARCHS = {gfx942,gfx950,gfx1200,gfx1201}` — gfx1100 excluded | disabled by design; RDNA3 has no FP8 hardware | `.../flash_attn_triton_amd/utils.py:72`, `GpuArch.supports_fp8` at `:99-102` |
| `module_norm`, `module_rope_*` (`compile_ops`) | `aiter/jit/core.py:461-477` **does** allow gfx1100 | builds are attempted; upstream says norm/RoPE run on RDNA | `README.md:77`; sources `csrc/py_itfs_ck/norm_kernels.cu` (CK templates, no MFMA literals) |
| `mla_decode_fwd` / `mla_prefill_ps_asm_fwd` / `mla_reduce_v1` | `get_gfx() == "gfx950"` / `gfx942` | **(b)**, and irrelevant here — Qwen3 is GQA, not MLA | `aiter/ops/attention.py:1055,1115,1130-1156,1465-1476` |

The two allow-lists disagree, and that is the single most misleading fact in this codebase:
`aiter/jit/core.py:456-477` accepts `gfx1100` by name (as do `gfx1101/1102/1103/115x/120x`), while
`csrc/cpp_itfs/utils.py:131-147` — the path every hand-written C++/CK per-shape module goes through —
accepts only CDNA. `DEFAULT_GPU_ARCH` there is literally `amdgpu-arch`'s output
(`csrc/cpp_itfs/utils.py:42-49`), which on rasputin prints `gfx1100`, so the assert fires immediately.

## Does it compile for gfx11?

Two build systems, two answers.

**`compile_ops` (jit/core.py)** — flag list built at `:911` as `--offload-arch={arch}` with
`GPU_ARCHS` defaulting to `native` (`:456`), i.e. it genuinely JITs from source for the local arch.
`build_targets.py:17` maps `GFX_MAP[7] = "gfx1100"`. So *arch-wise* these modules are permitted and
would be compiled for gfx1100. Whether the HIP source survives that is per-kernel; norm/RoPE are
CK template instantiations with no MFMA literal and upstream reports them working.

**`cpp_itfs` (utils.py)** — never reaches the compiler on gfx1100: the assert at `:145-147` fires first.
That is what kills `paged_attention_ragged`, sglang's default decode.

Even if that assert were lifted, the kernel would not build. I verified the ISA split directly with
`/opt/rocm/bin/hipcc` (ROCm 10.1, clang 24) on a two-line probe:

```
=== gfx1100
error: '__builtin_amdgcn_mfma_f32_16x16x16f16' needs target feature mai-insts
=== gfx942
(the same MFMA line compiles; the WMMA line instead errors:
 'needs target feature wmma-256b-insts,wavefrontsize32')
```

and symmetrically, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` compiles cleanly for gfx1100 and
fails for gfx942. **RDNA3 has WMMA and no MFMA; CDNA3 has MFMA and no WMMA.** `pa_kernels.cuh`
is written entirely in the MFMA idiom — `ROWS_PER_WARP = WARP_SIZE / 16` (`:71`), 16-lane row
layouts, `gcn_mfma16x16x16_instr` for QK and softmax·V (`:342,728`). Note `WARP_SIZE` there is the
CDNA 64, so even a hypothetical wave32 port would need the lane layouts rewritten, not just the
intrinsic swapped. This is a kernel-authoring project, not a flag flip.

The Triton kernels do not have this problem: Triton's AMD backend lowers `tl.dot` to the target's
own matrix op, so the same source becomes WMMA ops on gfx11 — which is exactly why aiter carries
`is_rdna` branches in `unified_attention.py` and a gfx1100 `BLOCK_N` in `flash_attn_triton_amd`.
The known RDNA3 hazards that remain are LDS (64 KB workgroup budget) and wave32 occupancy —
upstream has an open fix trail for the first ("Guard RDNA unified attention against LDS overflow",
`6eda1cce`, 2026-09-15).

## The reset_verify_attn_plan_cache failure

`reset_verify_attn_plan_cache` is **sglang's**, not aiter's:
`python/sglang/kernels/ops/attention/unified_attention_3d_mtp.py:68`. Its job is to drop the asm
kernel's split-KV segment plan cache once per forward; it is a no-op when the gfx950 asm kernel
is absent (`:56-64`).

It reaches `aiter_backend.py` through the import block at `:48-76`, which is a single `try:`
containing **both** the aiter imports and the `sglang.kernels.ops.attention.unified_attention_3d_mtp`
imports. That module imports aiter at its own module scope — `make_kernel_repr` from
`aiter.ops.triton.utils._triton.kernel_repr` and `e4m3_dtype` from `aiter.ops.triton.utils.types`
(`unified_attention_3d_mtp.py:44-45`). No aiter ⇒ that whole module fails to import ⇒
`reset_verify_attn_plan_cache` is never bound in `aiter_backend`'s namespace. The `except ImportError`
at `:63-68` only *prints* "aiter is AMD specific kernel library…" and continues.

So the backend is constructed in a half-initialised state, and the first use is a bare
`reset_verify_attn_plan_cache()` at `:1976` inside `init_forward_metadata_out_graph` — i.e. on the
CUDA-graph capture path, which is why the traceback reads like a capture failure. It is a
name-resolution failure that happens to occur during capture.

Why aiter was missing: our venv never had it. `docs/research/16` §6 records the startup line
`Ignore import error when loading sglang.srt.models.deepseek_v4: No module named 'aiter'`, and §4.5
records that we deliberately quarantined the Quark configs because aiter was absent. The
`NameError` is therefore **evidence that aiter is not installed, and evidence about nothing else.**
It says nothing about gfx1100.

`SGLANG_USE_AITER_UNIFIED_ATTN=1` cannot help for the same reason: the flag only changes which
*kernels* decode uses (`use_triton_unified_attention`, set at `:528-534`), while `:1976` runs
unconditionally in graph-capture metadata setup, before any kernel is chosen.

If aiter *were* installed, the failure would move, not vanish: `import aiter` is cheap
(`aiter/__init__.py` imports only torch/os/sys/logging at top level), so sglang's import block would
succeed and the backend would construct — then the default decode path
(`aiter_backend.py:4852 paged_attention_ragged`) would hit the cpp_itfs assert on the first forward.
A loud `AssertionError` instead of a `NameError`, same outcome.

## Upstream RDNA3 work

aiter's own README, current main, `README.md:65-77`:

| GPU | Architecture | Status |
|---|---|---|
| AMD Pro W7900 | gfx1100 (RDNA3) | Experimental¹ |
| AMD AI Max and Max Pro 400/300 | gfx1151 (RDNA3.5) | Experimental¹ |
| AMD Radeon AI PRO R9700 | gfx1201 (RDNA4) | Experimental¹ |

> ¹ "On RDNA, Triton and most FlyDSL kernels run, as do most HIP kernels (norm, RoPE, quant,
> activation, plus some GEMM/attention). Most CK and ASM kernels are CDNA-only."

That is AMD's own statement of the split I measured. But **that table did not exist at our image's
commit** — `README.md` at `9127c94a1` has the "Supported Hardware" heading and only the four Instinct
rows; the RDNA rows landed in "[Docs] Add RDNA support in README (#4303)", 2026-07-22, ~4 weeks after
our checkout. The image's aiter predates the entire RDNA documentation and tuning push
(`f495a610` gfx1100 attention tuning 2026-10-08; `d7531271` gfx1100 A16W16 GEMM configs 2026-10-16).

**No RDNA CI.** `.github/workflows/aiter-test.yaml:21` sets `GPU_ARCH_LIST: "gfx942;gfx950"`;
runners are `linux-aiter-mi35x-1` / `linux-aiter-build-mi250`. Nothing exercises gfx11 per-commit.

**The gate that blocks us is still open upstream.** `csrc/cpp_itfs/utils.py` on main, `:234-243`,
now allows `gfx1151` and `gfx1250` but **still not `gfx1100`**. So the hand-written C++/CK JIT path
remains closed for RDNA3 discrete today, months after the README started calling it Experimental.

Roadmap status, from [ROCm/aiter#4604](https://github.com/ROCm/aiter/issues/4604) (open, Aug 2026):
the reporter demonstrates a working 4-step gfx1100 patch (adding gfx1100 to the cpp_itfs allow-list
among them) and notes `gemm_a8w8_CK` is "XDL-only, cannot compile on RDNA3". A collaborator replied
that they are "actively working on scoping machine / CI resource" and welcome RFCs, but committed to
no date; the linked vLLM issue records "RDNA support on AITER is not part of the vLLM roadmap".

sglang side, [sgl-project/sglang#30599](https://github.com/sgl-project/sglang/issues/30599) (open):
"AITER is CDNA-only", no RDNA runner, and the one merged allow-list PR (#30415) went to the
`amd_march` branch, not `main`. The same issue's linked #30245 reports the Triton fused-MoE kernel
numerically correct on gfx1100 (cosine 0.999992) — the blockers are builds, not numerics.

The in-tree `docker/patches/sgl-kernel-gfx1151.sh:3-4` confirms the shape of sglang's RDNA support:
"applied at image-build time … because gfx1151 is not a supported SGLang target". It patches
`sgl-kernel`'s arch allowlist and `WARP_SIZE` only. It says nothing about attention backends, and
implies no aiter involvement.

## What this means for rasputinie

**Delete aiter outright — no.** Not because aiter is valuable, but because those files are upstream's;
deleting them converts every future `git merge` into a conflict map. Make it *unreachable*, which is
a 20-line change, not a deletion.

**Fix the call path — no.** Installing aiter and lifting the cpp_itfs allow-list gets us to a second
wall (MFMA-in-asm) that needs `pa_kernels.cuh` ported to WMMA lane layouts. Upstream, with the
hardware and the maintainers, has not done this. And there is no payoff: at 3.5 bpw, ~11.8 GB of
weights over 619 GB/s is bandwidth-bound at ~50 tok/s/card, and the attention kernel is a small
fraction of that budget. aiter cannot move the number that matters.

**Keep it gated — yes, this one.** Concretely, in priority order:

1. `srt/utils/common.py` — add `is_gfx1100_supported()` / `is_rdna3()` beside the existing
   `is_gfx95_supported` / `is_gfx942_supported` / `is_gfx1250_supported` (`:1231,1244,1256`).
2. `srt/arg_groups/model_override_base.py:326-327` — the `elif get_platform().is_hip: return "aiter"`
   branch is arch-blind and package-blind. Gate it on `find_spec("aiter")` **and** not-RDNA3.
   Triton then becomes primary by construction, and `--attention-backend triton` stops being tribal
   knowledge.
3. `aiter_backend.py:48-76` — the `except ImportError: print(...)` must raise. A `NameError` four
   thousand lines later during graph capture is the worst possible failure mode, and it cost us a
   bug hunt. Splitting the sglang-symbol imports out of the aiter `try:` block is the real fix: the
   two have different failure semantics and must not share a guard.

Cost of that: under an hour, no kernel work, and it makes the default correct on any ROCm box without
aiter — which is also the state of every consumer Radeon box today.

**The one bounded experiment worth running:** aiter from *current main* (not the June image),
`GPU_ARCHS=gfx1100`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`. That routes both extend and decode through
`unified_attention` (Triton, `is_rdna`-tuned) and never touches `paged_attention_ragged`, so it is the
only aiter configuration on gfx1100 whose kernels actually exist. Predict: it compiles and captures;
it will not beat Triton by anything we can measure on a bandwidth-bound decode. Budget one afternoon,
and only after (1)-(3) land so the rig cannot silently regress to the broken default.

## Claims I could not verify

- **Whether `import aiter` fully succeeds on gfx1100.** Not run (no GPU jobs). `aiter/__init__.py`
  looks cheap, but `module_aiter_core.so` is loaded during import and its kernels are arch-resolved
  at call time; the `.so` contains no `gfx*` strings, so I could not read its target list statically.
- **Whether `module_norm` / `module_rope_*` actually build for gfx11.** No build attempted. I have
  upstream's README assertion and the absence of MFMA literals in `norm_kernels.cu`, nothing more.
- **Whether the CK `mha_varlen_fwd` fallback builds for gfx11.** `mha_fwd_kernels.cu` has no
  gfx11/mfma strings — arch handling is inside CK, which I did not inspect.
- **Numerical correctness of `unified_attention` on gfx1100 at head_dim 256 / 6:1 GQA.** Compiles ≠
  correct; the RDNA LDS-overflow guard (#4868) suggests this area is still moving.
- **The aiter#4604 claim that README carried the W7900 row at 0.1.16.post3.** At `9127c94a1` it does
  not, and it does on main; I confirmed the add-PR title and date but did not bisect the exact commit.
- **A gfx1100 decode hang reported in a sglang#30599 comment** (prefill batch, no decode batch,
  HTTP 000, with `--attention-backend triton`). Not reproduced — our Triton run served coherent
  output — but it is a live report against the backend we are about to make primary.
- **Any aiter-vs-Triton performance number on gfx1100.** None exist in this repo and I took none.

## Source map

**aiter @ 9127c94a1** (in `sglang-build:/sgl-workspace/aiter`)
- `aiter/_version.py` — `0.1.17.dev110+g9127c94a1`
- `aiter/jit/core.py:456` (`GPU_ARCHS` default `native`), `:461-477` (allow-list **with** gfx1100), `:911` (`--offload-arch`)
- `aiter/jit/utils/build_targets.py:10-30` (`GFX_MAP[7]="gfx1100"`), `:38-43` (CU counts, no gfx1100)
- `csrc/cpp_itfs/utils.py:42-49` (`DEFAULT_GPU_ARCH` from `amdgpu-arch`), `:131-147` (allow-list **without** gfx1100), `:230-231`
- `csrc/cpp_itfs/pa/pa_ragged.py:1-80` (JIT entry), `csrc/cpp_itfs/pa/pa_common.cuh:96,100,127,141-151`, `pa_kernels.cuh:71,342,358,728,744`
- `aiter/ops/mha.py:1573-1574`, `:1556,1631,2465,2482,2506,2738`
- `aiter/ops/triton/attention/unified_attention.py:72-73,84,91`
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py:56-70,72,99-102`; `fwd_prefill.py:82,199`
- `aiter/ops/attention.py:730,1326,1055-1156,1465-1476`; `aiter/ops/norm.py:29-81`; `aiter/ops/rope.py:11-137`
- `README.md:65-77` (main; RDNA rows absent at `9127c94a1`)

**sglang @ d0a7565** (local `engine/python/sglang/srt/…`)
- `layers/attention/aiter_backend.py:48-76` (shared try/except), `:1976` (the call), `:528-534` (unified flag), `:4619`, `:4852`
- `kernels/ops/attention/unified_attention_3d_mtp.py:44-45`, `:56-64`, `:68`
- `arg_groups/model_override_base.py:326-327`; `utils/common.py:1231,1244,1256`
- `engine/docker/patches/sgl-kernel-gfx1151.sh:3-4`

**External**
- https://github.com/ROCm/aiter/issues/4604 · https://github.com/sgl-project/sglang/issues/30599
- https://raw.githubusercontent.com/ROCm/aiter/main/README.md · `.../main/csrc/cpp_itfs/utils.py:234-243`
- https://raw.githubusercontent.com/ROCm/aiter/main/.github/workflows/aiter-test.yaml:21
- ISA probe: `hipcc -x hip --offload-arch=gfx1100|gfx942 -c` on rasputin, ROCm 10.1 / clang 24
