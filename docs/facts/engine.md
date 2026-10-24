# facts — the engine

What the vendored sglang is, what it can and cannot do on this card, and what must be built.
`_archive/2026-10-24/…` paths are pre-migration locations.

## What it is

| fact | value | verified-by |
|---|---|---|
| Source | vendored sglang `d0a7565`, `engine/`, 11,059 files / 164 MB | `git log` + `find` |
| **Install** | sglang 0.5.9 installed **editable from `engine/python`** — the vendored tree *is* the running engine, no install step | `pip list` on the host |
| Versions | torch 2.14.1+rocm7.2, HIP 7.2.53211, triton 3.8.0 + triton-rocm 3.8.0, sgl_kernel 0.4.10 | same |
| Composition (measured) | **8,338 files** by extension: 7,882 `.py`, 263 `.cuh`, 87 `.h`, 44 `.cpp`, 55 `.cu`, 7 `.hip` | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-10.md` |
| Deletion map | **5,363 dead on any path (1.48 M LOC)** · 27 CUDA-gated (10.6 k) · 1,422 lazy-only (637 k) · **1,070 live (455 k)** | same |
| C++ that executes nothing | 456 files / 175 k LOC — but they are the source of the missing ops | same |
| Attention backend | sglang ships its own **`layers/attention/triton_backend.py`** (118 KB) with **zero** aiter references | `grep -c aiter` = 0 |

## The four blockers — ordered, each verified

The engine **does not serve this file on ROCm at all**. Every one of these fires before a token:

| # | blocker | evidence |
|---|---|---|
| 1 | `aiter` is imported at **module scope** under `if _is_hip` and is not installed → `import sglang.srt.configs.model_config` fails | `layers/quantization/quark/schemes/quark_w4a4_mxfp4.py:163` |
| 2 | `ModelConfig` **hard-refuses GGUF on ROCm** | `configs/model_config.py:2128-2133`; allow-list at `:1977-1991` has 13 methods, no `gguf` |
| 3 | on HIP, `layers/quantization/gguf.py` has **no import branch** — the chain is `_is_cuda → _is_musa → _is_npu → else`, so the five `ggml_*` names are undefined → `NameError` on the first linear | `gguf.py:50-75` |
| 4 | the ops are **absent from the ROCm wheel**: 0 of 47 `sgl_kernel` ops are ggml; `nm common_ops.so` = 0 ggml symbols; `gguf_kernel.cu` is not in `setup_rocm.py`'s source list | `nm`, `grep gguf setup_rocm.py` = 0 |

## Which kernel decode actually wants

At batch 1 the engine selects **MMVQ** — `ggml_mul_mat_vec_a8`, a fused int8 GEMV — because
`x.shape[0] = 1 ≤ mmvq_safe = 8` and IQ3_S is in `MMVQ_QUANT_TYPES`. The
`dequantize → bf16 matmul` fallback is reached only at **≥ 9 tokens**, i.e. prefill and speculative
verify. `gguf.py:213-238` tests MMVQ first.

**Consequence:** the kernel the engine needs is the one this project already wrote by hand
(`bench/gfx1100-probes/iq3s_valid.cpp`, 521 GB/s) — it is simply not *inside* the engine.

## aiter: partly dead, and the reason it fools you

| piece | gfx1100 |
|---|---|
| `paged_attention_ragged` (**sglang's default HIP decode**) | **dead** — hand-written **MFMA** assembly (clang: *needs target feature `mai-insts`*); RDNA3 has no matrix cores; the file is also rejected by `csrc/cpp_itfs/utils.py:131-147`, whose allow-list is CDNA-only |
| `jit/core.py:456-477` | **accepts** `gfx1100` by name — so aiter looks enabled and then asserts |
| `unified_attention` (Triton) | compiles; explicit `arch.is_rdna` tuning (`TILE_SIZE 16`, `waves_per_eu 6`) |
| `flash_attn_triton_amd` (Triton) | compiles; gfx1100 in `RDNA_ARCHS`, `BLOCK_N = 64` |
| `module_norm`, `module_rope_*` | permitted; CK templates with no MFMA literals |
| FP8 paths | excluded by design — RDNA3 has no FP8 hardware |
| the crash first attributed to aiter | sglang's own `reset_verify_attn_plan_cache`, hiding a missing install — a broken call path, **not** an architecture verdict |

**The working half of aiter is Triton.** Depend on Triton directly. See `D-09`.

## Triton is native here — verified by running one

| fact | value | verified-by |
|---|---|---|
| Backend | `triton/backends/amd` → LLVM AMDGPU → **gfx1100 HSACO**, launched via HIP | `triton/backends/amd/include/hip/hip_runtime.h` |
| Smoke test | vector add, 1 M elements: `gcnArchName gfx1100`, **max abs err 0.0**, asm sections `amdgcn+hsaco+llir+ttgir+ttir`, ISA shows `buffer_load_b128` | `~/.hermes/.../scratch/triton_probe.py`, 2026-10-23 |
| What it does not give | VGPR allocation, instruction scheduling, delay slots, load ordering. It has `tl.dot` and `tl.inline_asm_elementwise` (3.8.0) | `triton.language` introspection |

**The open question is scheduling, not portability:** a Triton IQ3_S GEMV has never been measured
against the 521 GB/s hand-written kernel. ADR 0001 notes TensorFold ships hand-written HIP GEMV
kernels for gfx1100 — the ceiling to beat or borrow.

## `__dp4a` is undeclared for gfx1100

The CUDA GGUF source (`csrc/quantization/gguf/vecdotq.cuh`) uses `__dp4a` at ~45 sites.
`hipcc --offload-arch=gfx1100` reports **`use of undeclared identifier '__dp4a'`** — not a feature
gate, an absent identifier. The C++ route is closed; the replacement is a Triton MMVQ.

`assumed:` that a Triton implementation can reach the asm kernel's rate. Unmeasured; the check is the
first per-type Triton kernel against 521 GB/s.
