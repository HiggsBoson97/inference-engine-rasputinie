# 00 — Measured facts

> **Correction, 2026-10-22 — the read wall is 776 GB/s, not 619.**
> Every ceiling in this file was derived from a **619 GB/s** figure that came from
> a 1.6 ms kernel with a serial accumulator dependency. Re-measured with kernels
> longer than ~3 ms — twice, independently — the card reads at **776 GB/s**
> (`bench/gfx1100-probes/WALL-2026-10-22.md`), and a **fused 4-bit dequant-GEMV
> reaches 752 GB/s** (97 % of that wall) with arithmetic verified to
> **2.6 × 10⁻⁷**. Consequences: the IQ3_S single-card ceiling is **66 tok/s
> weights-only and ~55 tok/s realistic**, not 52.6; the "60 % of the wall" kernel
> gate is **466 GB/s**, not 371. **Every tok/s figure below is ~25 % low.**
> Relative comparisons, mechanism findings and negative results all stand.

Every number here comes from `bench/00_hardware_probe.py` on 2026-10-22. Run it
again if the hardware, the driver or the ROCm version changes.

## Link and memory

| Quantity | Measured | Note |
|---|---|---|
| HBM copy bandwidth, GPU 0 | 592.9 GiB/s | copy counts read + write |
| HBM copy bandwidth, GPU 1 | 577.8 GiB/s | |
| HBM copy bandwidth, both busy | 1182.7 GiB/s | cards do not slow each other |
| Cross-GPU copy | 14.23 GiB/s | same as host copy: no real P2P win |
| Host to GPU | 14.19 GiB/s | pinned memory |
| GPU to host | 14.31 GiB/s | pinned memory |
| CPU DRAM copy | 46.42 GiB/s | DDR5-5600 dual channel |
| NVMe read, 1 MB blocks | 3.90 GiB/s | `dd iflag=direct` |
| NVMe read, 4 MB blocks | 4.44 GiB/s | |
| NVMe read, 4 parallel streams | 5.59 GiB/s | closest to a prefetcher |

`torch.cuda.can_device_access_peer` returns true, but the measured copy speed
equals the host-bounce speed. Treat cross-GPU traffic as host traffic over
PCIe. Do not design around peer-to-peer DMA.

## Compute

| Shape | Time | Throughput |
|---|---|---|
| BF16 GEMM 1x5120x17408 | 0.405 ms | 410 GiB/s of weights, 0.44 TFLOP/s |
| BF16 GEMM 8x5120x17408 | 0.295 ms | 563 GiB/s of weights, 4.8 TFLOP/s |
| BF16 GEMM 32x5120x17408 | 0.295 ms | 562 GiB/s of weights, 19.3 TFLOP/s |
| BF16 GEMM 2048x5120x17408 | 4.23 ms | 86.4 TFLOP/s |
| BF16 GEMM 1x2560x640 (one expert) | 23.9 us | 128 GiB/s, launch-bound |
| FP8 E4M3 matmul, torch | fails | `normal_kernel_cuda` not implemented for Float8 |

The 27B MLP uses the 5120x17408 shape. The Flash-Next expert uses the
2560x640 shape.

Decode at batch 1 reaches 410 GiB/s of 592 GiB/s. A batched decode at 8 or 32
rows reaches 563 GiB/s. A hand-written GEMV has roughly 30% of headroom at
batch 1.

## Consequences

These are the five facts that shape the engine.

1. **Decode speed equals weight bytes divided by 592 GiB/s.** Compute does not
   limit decode on either target model. Every decode optimisation is a
   weight-byte reduction (quantization) or a weight-byte speed-up (better
   GEMV, better residency).

2. **Tensor parallel 2 costs 14.2 GiB/s of PCIe.** For the 27B, the two
   all-reduces per layer move `2 x 5120 x 2 bytes` per token per layer, which
   is 1.31 MB per token over 64 layers, or 86 us per token. That is harmless
   at batch 1. It is not harmless during prefill: a 4096-token prompt moves
   5.4 GB and takes 0.35 s. Prefill on this box is PCIe-bound when TP=2.

3. **Data parallel beats tensor parallel when the model fits one card.** The
   27B at Q4_K_XL needs 16.35 GiB, which fits one 19.98 GiB card. Two
   replicated copies give two independent streams, zero PCIe traffic, and full
   prefill speed. Tensor parallel wins only for single-stream latency, or when
   the quantized model does not fit one card (Q6 and above, FP8).

4. **MoE decode dies from launch overhead unless experts are batched.** One
   expert GEMM takes 23.9 us, which is almost all launch overhead. Flash-Next
   needs 10 experts x 3 matrices x 48 layers = 1440 GEMMs per token. At 23.9 us
   each that is 34 ms, or 29 tokens/s. One grouped GEMM per layer instead of
   480 per token removes this ceiling entirely. This is the highest-value
   single kernel in the project.

5. **FP8 has no torch GEMM on gfx1100.** RDNA3 has no FP8 matrix
   instructions. FP8 weights still pay off, because decode is bandwidth-bound,
   but the kernel must decompress FP8 to BF16 and use the BF16 WMMA path. Plan
   for a fused decompress-and-multiply kernel, not for a native FP8 GEMM.

## What is actually inside the 27B test file

Measured with `bench/02_gguf_probe.py` on `Qwen3.8-27B-UD-Q4_K_XL.gguf`
(17,559,178,144 bytes, 866 tensors, 50 metadata keys, `arch=qwen35`). The tensor
sizes add up to the data section exactly, so the parse is correct.

The name says Q4; the file is a nine-type mixture at about 5.05 bits per weight.

| Type | Tensors | GiB | Share | bit/weight |
|---|---|---|---|---|
| Q5_K | 191 | 7.38 | 45.2% | 5.50 |
| IQ4_XS | 70 | 2.91 | 17.8% | 4.25 |
| Q4_K | 69 | 2.85 | 17.4% | 4.50 |
| Q6_K | 56 | 2.67 | 16.3% | 6.56 |
| IQ4_NL | 6 | 0.26 | 1.6% | 4.50 |
| Q8_0 | 110 | 0.12 | 0.8% | 8.50 |
| Q3_K | 3 | 0.11 | 0.7% | 3.44 |
| IQ3_S | 1 | 0.04 | 0.2% | 3.44 |
| F32 | 360 | 0.01 | 0.1% | 32.00 |

Two consequences:

1. **"A Q4_K kernel" is not a kernel set.** This one file needs nine decode
   paths. The `Q8_0` tensors are the small ones (norms and biases are F32/Q8_0),
   so the hot path is Q5_K, IQ4_XS, Q4_K and Q6_K: four types cover 96.7% of the
   bytes. Build those four first, in that order.
2. **The last eight layers are Q6_K** (`blk.56`–`blk.63` carry Q6_K
   `ffn_down`/`ffn_gate`), and `output.weight` is Q6_K at 0.97 GiB. Unsloth's
   "XL" recipe raises precision near the output, so the per-layer cost is not
   uniform and a layer-uniform scheduler would mispredict.

The effective 5.05 bits per weight means the 16.35 GiB / 592 GiB/s = 35 tok/s
ceiling already accounts for the mixture. A true 4.0-bit file would be about
12.9 GiB and would raise the ceiling to about 45 tok/s, which is why the quant
level is a first-order design decision and not a detail.

### The GSQ-RCO IQ3_S file, which is the real target

`Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`: 11.76 GB, 851 tensors, `arch=qwen35`, name
"Qwen3.8-27B GSQ-RCO", size check exact. **Eleven** types:

| Type | Tensors | GiB | Share | bit/w |
|---|---|---|---|---|
| IQ3_S | 144 | 3.39 | 31.0% | 3.44 |
| IQ4_XS | 96 | 2.81 | 25.7% | 4.25 |
| IQ3_XXS | 78 | 1.93 | 17.6% | 3.06 |
| Q4_K | 39 | 1.47 | 13.4% | 4.50 |
| IQ2_S | 17 | 0.74 | 6.8% | 2.56 |
| Q2_K | 13 | 0.22 | 2.0% | 2.62 |
| IQ2_XS | 9 | 0.21 | 1.9% | 2.31 |
| IQ2_XXS | 5 | 0.09 | 0.8% | 2.06 |
| BF16 | 96 | 0.04 | 0.4% | 16.00 |
| type 29 | 1 | 0.02 | 0.2% | 1.75 |
| F32 | 353 | 0.01 | 0.1% | 32.00 |

Two facts the plan depends on:

1. **Five types cover 94.5% of the bytes**: IQ3_S, IQ4_XS, IQ3_XXS, Q4_K,
   IQ2_S. Those five kernels are the whole quantised matmul problem for this
   file. Q8_0 does not appear at all.
2. **The allocation is non-uniform down to a single tensor.** Exactly one
   tensor, `blk.13.ffn_gate.weight` [5120x17408], uses type id 29 at 1.75
   bits/weight. That id is not in the type table we hold; 1.75 bpw matches
   IQ1_M exactly, so treat it as IQ1_M until its block layout is confirmed. It
   is 19.5 MB, 0.2% of the file — a slow path is acceptable for it, but the
   loader must not assume a uniform type.

Per-tensor dispatch is therefore not a nicety: `general.file_type` describes the
target average, and the RCO search spent the budget unevenly on purpose.

## First real-model baseline: 27B FP8, TP=2 (2026-10-22)

Server: `--tp 2 --attention-backend triton --disable-custom-all-reduce
--mem-fraction-static 0.88 --context-length 32768`, `HIP_VISIBLE_DEVICES=0,1`.

Startup facts: weights **14.52 GB per card** (fp8 e4m3), load 11.66 s.
KV cache in **bf16**, 45,727 tokens, 1.40 GB total. After weights, KV and
graphs only **2.08 GB free per card**, so `max_running_requests=3`. Decode
graph capture for bs=[1,2,3] took **41 s**.

| Measurement | Value |
|---|---|
| Prefill, 4,334 tokens | ~0.94 s → **~4,600 tok/s** |
| Decode, 4,334-token context | **857 ms/token = 1.16 tok/s** |
| Decode, 319-token context | **857 ms/token = 1.16 tok/s** |
| Effective weight bandwidth | 14.52 GB / 0.857 s = **16.9 GB/s = 2.7% of 619 GB/s** |

The decode cost is **independent of context length**, so it is not attention,
not KV, and not the linear-attention state. It is the per-step weight path, and
it runs 36x below the measured memory wall. Graph replay is active
(`cuda graph: True`), and the log carries no fallback or "slow path" warning.

This is the anomaly the native-kernel work exists to fix. It is recorded as
measured, not explained: the next step is a kernel-level profile of one decode
step, not a hypothesis.

## Instruction throughput (measured 2026-10-22)

Source: `bench/gfx1100-probes/` (`probe.cpp`, `RESULTS.md`, `results.json`). Card 0
only, sclk 2802 MHz sustained under load (2435–2929 range), median of 5 runs,
2^24 iterations per lane.

| Instruction | MAC/instr | Ginstr/s | GMAC/s | slots/instr | Verdict |
|---|---|---|---|---|---|
| `v_dot4_i32_i8` | 4 | 7228 | 28913 | 2.08 | half rate |
| `v_dot8_i32_i4` | 8 | 3837 | 30692 | 3.92 | quarter rate |
| `v_dot2_f32_f16` | 2 | 12120 | 24239 | 1.24 | full rate |
| `v_fma_f32` | 1 | 12215 | 12215 | 1.23 | full rate |

Peak is 5376 lanes x 2.802 GHz = 15.06 T lane-ops/s.

Five facts that change the kernel plan:

1. **`__builtin_amdgcn_sdot4` does not compile for gfx1100.** It requires the
   `dot1-insts` feature, which the subtarget does not advertise even though
   `llvm-mc` assembles `v_dot4_i32_i8` for gfx1100. The instruction is reachable
   **only through inline asm**. `sdot2` and `sudot8` also fail.
2. **`v_pk_fma_f32` does not exist on gfx11.** Packed-fp32 FMA is not an option;
   the baseline is scalar `v_fma_f32`.
3. **The card has 5376 lanes (84 CU x 64), not 84 x 128 = 10752.** Any estimate
   built on 10752 is 2x too optimistic.
4. Measured read bandwidth is **619 GB/s** (copy 595 GB/s), against the 800 GB/s
   specification — **SUPERSEDED 2026-10-22: this is a kernel artefact. The read
   wall is 776 GB/s; see the correction banner at the top of this file.**
5. **The quantised GEMV is bandwidth-bound with about 15x of issue margin.** At
   619 GB/s a Q5_K GEMV consumes ~900 G weights/s; the arithmetic costs ~805 G
   lane-slots/s against 12.2 T available, which is **6.6% issue occupancy**.
   Even a dequantise-to-fp16 path on full-rate `v_dot2_f32_f16` has more than
   25x headroom. So the work is layout, coalescing and reading each byte once —
   not minimising dot instructions.

## Toolchain (measured 2026-10-22, second session)

The box could not compile a single HIP kernel when this session started. The
runtime was installed; the development files were not.

| Fact | Detail |
|---|---|
| ROCm packaging | ROCm 10.1 ships as `amdrocm-*` packages under `/opt/rocm/core-10.1` |
| Missing | `/opt/rocm/include/hip/hip_runtime.h` and `lib/cmake/hip-lang/hip-lang-config.cmake` |
| Symptom | `hipcc` exists and runs, but CMake dies with "ROCm root does not contain the HIP runtime CMake package", and every `#include <hip/hip_runtime.h>` fails |
| Fix | `sudo apt install amdrocm-runtime-dev10.1 amdrocm-blas-dev10.1` (pulls `amdrocm-hipblas-common-dev10.1`, `amdrocm-llvm-dev10.1`) |
| After | HIP compiler detected as clang 24.0.0, hipBLAS found, gfx1100 targets build |

Other toolchain facts:

- **Python venv had no pip.** `~/venv/sglang-rocm` held torch, numpy and sympy
  only. Fixed with `python -m ensurepip --upgrade` (pip 25.0.1). There is no
  `uv`, no `pipx`, no `git-lfs` and no `hf` CLI on the host.
- **Version skew is real, and it works.** torch is `2.14.1+rocm7.2` (HIP
  7.2.53211); the system headers are HIP 7.16 / clang 24. Measured: a
  `load_inline` HIP extension with `--offload-arch=gfx1100` compiles against the
  system headers and runs correctly on a GPU tensor under torch's bundled
  runtime. Two dev packages are required for that to work:
  `amdrocm-runtime-dev10.1` and `amdrocm-core-dev10.1` (the latter supplies
  `thrust/` and `hipcub/`, without which torch's own headers do not compile).
- **GGUF containers exist for both models.** unsloth k-quants (27B
  UD-Q4_K_XL is 16.35 GiB and byte-verified against the Hugging Face size) and
  ISTA-DASLab GSQ-RCO quants (27B from IQ2_S to IQ3_S; Flash-Next at IQ2_XS,
  IQ3_XXS and IQ3_S). The GGUF reader and every dequant kernel are ours to
  write; see `docs/decisions/0001-exclude-llama-cpp.md` for the excluded source
  and why.
- **sglang exists only as a container image.**
  `rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0` (71.6 GB) imports
  `sglang.srt` on gfx1100 and JIT-builds aiter core in about 17 s. The venv has
  no sglang, and there is no sglang checkout on the host.
- **sglang 0.5.15 does know these models.** `Qwen3_5ForConditionalGeneration`
  and `Qwen4ExpForConditionalGeneration` appear in
  `python/sglang/srt/arg_groups/*`, and `hybrid_linear_attn_backend.py` plus a
  `mega_moe_hook` exist. The fork option is therefore stronger than "sglang is
  NVIDIA-only" suggests.

