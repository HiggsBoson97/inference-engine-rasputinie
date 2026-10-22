# Decode profile, 27B FP8 TP=2

Qwen3.8-27B-FP8, 2× gfx1100 (42 CUs each as reported by HIP / `numSms`), `--tp 2 --attention-backend triton
--disable-custom-all-reduce --mem-fraction-static 0.88 --context-length 32768`, 14.52 GB of fp8 weights per card.

**Verdict: 91.8 % of the 857 ms step is the Triton `_w8a8_block_fp8_matmul` GEMV — 256 launches per step, 786.3 ms,
at 17.8 GB/s of weight traffic where the card measures 592.9 GiB/s.** The linear-attention path, RCCL, the
dequantisation glue and the copies are all ≤ 0.4 % each. This is one kernel family in one code path, not a
distributed-systems problem.

---

## Setup actually used

Model shape (from `config.json`, confirming why 4 GEMMs/layer is the right count): 64 layers, `layer_types` =
48 × `linear_attention` + 16 × `full_attention` (`full_attention_interval: 4`), `hidden_size` 5120,
`intermediate_size` 17408, `vocab_size` 248320, `head_dim` 256, fp8 e4m3 with `weight_block_size [128,128]`,
`lm_head` / `linear_attn.{A_log,conv1d,dt_bias,in_proj_a,in_proj_b,in_proj_ba,norm}` / `mlp.gate` / norms
**not** quantised (i.e. bf16).

Server, second run (same flags as the measured 857 ms run, plus two profiler-only flags):

```
SGLANG_TORCH_PROFILER_DIR=/tmp/prof27b HIP_VISIBLE_DEVICES=0,1 \
  ~/venv/sglang-fork/bin/python -m sglang.launch_server \
  --model-path .../Qwen3.8-27B-FP8 --tp 2 --attention-backend triton \
  --disable-custom-all-reduce --host 127.0.0.1 --port 30100 \
  --context-length 32768 --mem-fraction-static 0.88 \
  --disable-decode-cuda-graph --decode-log-interval 1        # <- profile-only
```

**Why graphs were disabled.** torch/kineto does not emit per-kernel events for dispatches that happen inside a
HIP graph replay, so a trace of the production path would show `hipGraphLaunch` and nothing else. I therefore
profiled the eager path and measured, first, that the eager path has the same cost as the graph path:

| path | measured |
|---|---|
| decode CUDA graph on (production) | 1.13 – 1.17 tok/s (855 – 885 ms/token) |
| decode CUDA graph off (this profile) | **1.154 tok/s = 862 ms/token**, 34.66 s / 40 tokens |

0.6 % apart ⇒ graph replay here removes launch overhead only, and the kernel mix is the same.

**Capture.** torch.profiler started through sglang's own control plane
(`POST /start_profile {"activities":["CPU","GPU"],"with_stack":false,"record_shapes":false}`) **9 s into** a
32-token request, so the window contains decode steps only (prefill finished long before). Stopped with
`POST /stop_profile`, which writes `{id}-TP-{rank}.trace.json.gz`. Analysed: TP-0,
19 943 kernel events, 6 409.8 ms of kernel time, **8 decode steps** (established independently by three
launch counts: 128/8 = 16 full-attention layers, 384/8 = 48 linear-attention layers, 8 lm_head GEMMs).
`with_stack:false` was necessary — with stacks the same trace is 1.8 M events / 100 MB gz and unparseable.

No source changes were made. Nothing under `engine/` was edited.

---

## Top kernels by self time

Self time = sum of `dur` over all dispatches of that symbol in an 8-step decode window, divided by 8.
Percentages are of the **857 ms** per-step budget measured by the harness, not of the captured window.

| # | kernel | launches/step | ms/step | % of 857 ms | suspected source |
|---|---|---|---|---|---|
| 1 | `_w8a8_block_fp8_matmul.kd` | 256.0 | 786.3 | **91.8 %** | `engine/python/sglang/kernels/ops/gemm/fp8_kernel.py:55` — every fp8 linear in the model |
| 2 | `ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)` | 130.0 | 3.8 | 0.4 % | RCCL all-reduce, 2 per layer (attn + mlp) |
| 3 | `Cijk_Alik_Bljk_BBS_BH_MT64x64x32_MI16x16x16x1_SN_1LDSB1_APM1_ABV0_...` | 1.0 | 2.3 | 0.3 % | hipBLASLt/Tensile **bf16** GEMM, 1 per step ⇒ inferred `lm_head` (1.27 GB / 2.32 ms = 547 GB/s) |
| 4 | `Cijk_Alik_Bljk_BBS_BH_MT32x32x32_MI16x16x16x1_SN_1LDSB1_APM1_...` | 48.0 | 1.6 | 0.2 % | bf16 GEMM for the non-quantised `linear_attn` projections (`in_proj_ba`), 1 per linear layer |
| 5 | `vectorized_elementwise_kernel<4, bfloat16tofloat32_copy_kernel...>` | 259.0 | 0.9 | 0.1 % | bf16→fp32 casts around RMSNorm/residual |
| 6 | `reduce_kernel<512,1,ReduceOp<float,MeanOps<float>>>` | 129.0 | 0.7 | 0.1 % | RMSNorm mean |
| 7 | `fused_recurrent_gated_delta_rule_packed_decode_kernel.kd` | 48.0 | 0.7 | 0.1 % | GDN recurrence, 1 per linear-attention layer |
| 8 | `_per_token_group_quant_8bit.kd` | 256.0 | 0.7 | 0.1 % | activation quantisation feeding each fp8 GEMM |
| 9 | `elementwise_kernel_manual_unroll<128,4,...nocast>` | 129.0 | 0.6 | 0.1 % | RMSNorm scale-multiply |
| 10 | `vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<float>>` | 258.0 | 0.6 | 0.1 % | residual adds (fp32) |
| 11 | `vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::BFloat16>>` | 128.0 | 0.4 | 0.0 % | residual adds (bf16) |
| 12 | `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda>` | 129.0 | 0.4 | 0.0 % | copies / `.contiguous()` equivalents |
| 13 | `sgl_hip::activation::act_and_mul_kernel<__hip_bfloat16, silu>` | 64.0 | 0.4 | 0.0 % | SwiGLU, 1 per layer |
| 14 | `vectorized_elementwise_kernel<4, rsqrt_kernel_cuda>` | 129.0 | 0.3 | 0.0 % | RMSNorm rsqrt |
| 15 | `vectorized_elementwise_kernel<4, pow_tensor...>` | 129.0 | 0.2 | 0.0 % | RMSNorm square |

Totals: **801.2 ms/step inside kernels (93.5 %)**, of which kernel #1 is 98.1 %. The remaining
**55.8 ms/step (6.5 %)** is not in any kernel — CPU-side launch gaps, sampling and detokenising. The attention
kernels proper (`_fwd_grouped_kernel_stage1` 12.4 µs × 16/step, `_fwd_kernel_stage2` 3.8 µs × 16/step,
`_causal_conv1d_update_kernel` 3.7 µs × 48/step) sum to **≈ 0.4 ms/step**, which is the mechanical reason decode
time is *identical* at 319 and 4 334 tokens: nothing in the step scales with context.

### Where kernel #1's 786 ms goes

The trace carries `grid` per dispatch, which identifies the shape without guessing. Per-step launch counts and
the isolated timings (measured separately, same config) agree to 9 %:

| layer op | per rank (N,K) | launches/step | grid | ms/step |
|---|---|---|---|---|
| MLP `gate_up_proj` | 17408, 5120 | 64 | 136 | 323.0 |
| MLP `down_proj` | 5120, 8704 | 64 | 40 | 241.9 |
| `o_proj` | 5120, 5120 | 64 | 40 | 145.0 |
| linear-attn `in_proj` | 8192, 5120 | 48 | 64 | 109.6 |
| full-attn `qkv_proj` | 7168, 5120 | 16 | 56 | 36.4 |

4 GEMMs × 64 layers = 256 launches/step, 856.0 ms isolated. This is the whole step; there is no third thing.

---

## What this rules in and out

| candidate | verdict | evidence |
|---|---|---|
| **(a) Triton `_w8a8_block_fp8_matmul` fp8 GEMM** | **RULED IN — 91.8 %** | 786.3 ms/step over 256 launches, 98.1 % of all kernel time; block=(128,1,1), grid=(N/128,1,1) ⇒ `BLOCK_SIZE_M` covers M=1 with padding |
| (b) GDN / mamba linear-attention path | **RULED OUT** | the entire GDN cost is `fused_recurrent_gated_delta_rule_packed_decode` 0.7 ms/step (0.1 %) + `_causal_conv1d_update` 0.2 ms + `_layer_norm_fwd_1pass` 0.2 ms ⇒ **< 1.2 ms/step, 0.14 %**. 48 layers × 14.5 µs. Nothing here can hide 857 ms. |
| (c) elementwise conversion / dequantisation | **RULED OUT** | every cast and add in the table sums to ≈ 3.1 ms/step (0.36 %). `_per_token_group_quant_8bit` (the activation quant that feeds the fp8 GEMM) is 0.7 ms/step. Nothing dequantises weights per step. |
| (d) RCCL all-reduce | **RULED OUT** | 3.8 ms/step (0.44 %), 130 launches. TP=2 costs 0.4 % of the step — the comms are not the problem, which also means `--disable-custom-all-reduce` was not the mistake. |
| (e) memory copies / cat / contiguous | **RULED OUT** | `CatArrayBatchedCopy_contig` 48 launches/step totalling 0.2 ms/step; `__amd_rocclr_copyBuffer` 0.01 ms/step. |
| (f) something else | **RULED OUT as the anomaly, 6.5 %** | the only non-kernel residue is 55.8 ms/step of launch gaps/sampling; significant for a healthy step, irrelevant next to 786 ms. |

---

## Isolated fp8 GEMM measurement

Shape M=1, K=5120, N=17408 (one rank's `gate_up` half) → **89.2 MB** of weights + block scales per call.
Called through the production entry point `w8a8_block_fp8_matmul_triton`, configs injected by monkeypatching
`get_w8a8_block_fp8_configs` so the real strides/launch path are used. Median of 10 event-timed calls.

| config | ms | GB/s of weight bytes |
|---|---|---|
| **default, as shipped: BM=64 BN=128 BK=128 w4 s3** | **5.049** | **17.7** |
| same, live in the server (512 dispatches in trace) | 4.99 | 17.9 |
| BM=32 BN=128 BK=128 w4 s3 | 5.100 | 17.5 |
| BM=16 BN=128 BK=128 w4 s4 | 3.717 | 24.0 |
| BM=16 BN=64 BK=128 w4 s3 | 2.961 | 30.1 |
| BM=16 BN=32 BK=128 w4 s3 | 3.326 | 26.8 |
| **BM=16 BN=64 BK=64 w4 s3 (best found)** | **0.939** | **94.9** |
| BM=16 BN=128 BK=64 w4 s3 | 1.207 | 73.9 |
| BM=128 BN=128 BK=128 w8 s3 | 7.156 | 12.5 |
| **reference: hipBLASLt bf16 GEMM, same M/N/K** | **0.378** | **471.9** (178.3 MB bf16 weights) |
| reference from `bench/results/hardware_probe.json` (`gemm_bf16_m1_k5120_n17408`) | 0.4047 | 410.2 GiB/s |

Four facts fall out:

1. **The hardware is fine at M=1.** A bf16 GEMM of exactly this shape streams 178 MB at 472 GB/s on the same card
   in the same process. 17.7 GB/s is a property of this kernel, not of the memory system.
2. **There is no tuned config for this device.** Startup log, for all five production shapes:
   `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at
   .../gemm/configs/N=17408,K=5120,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json`.
   The configs directory has 181 entries, 9 of them AMD — MI300X, MI325X, MI455X — **zero gfx1100** for any shape
   this model uses. The generic default (BM=64, tuned for prefill-sized M) is what decode gets.
3. **Config is worth 5.5×, and it is free.** BM=16/BN=64/BK=64 at 94.9 GB/s on the big shape. Verified not to be a
   numerics artefact: against a block-dequantised fp32 reference, `max_rel_err` is 2.86e-6 for the default config
   and 2.59e-6 for the fast one; the two agree to 7.6e-5. Applied to all five production shapes and summed per
   step: **856.0 ms → 156.3 ms, i.e. 5.48×, projecting 857 ms → 157 ms ⇒ 1.17 → 6.4 tok/s.**
4. **There is no native fp8 path to fall back on.** `torch._scaled_mm` on fp8: *"only supported on CUDA devices
   with compute capability >= 9.0 or 8.9, or ROCm MI300+"*. The repo's own probe already recorded
   `gemm_fp8_e4m3_m1_k5120_n17408: NotImplementedError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'`.
   Triton is the only fp8 GEMM on gfx1100, which is why this kernel is the whole story.

One dead end, recorded so nobody re-walks it: `fp8_kernel.py:675` `use_w8a8_block_fp8_matmul_unrolledx4()` — the
AMD small-grid kernel specifically written for decode-sized grids — **computes `num_workgroups <= get_device_core_count()`
and then never returns it** (line 683). It returns `None`, so `select_w8a8_block_fp8_matmul_kernel` always picks
`_w8a8_block_fp8_matmul`; I confirmed `use_w8a8_block_fp8_matmul_unrolledx4(1, 17408, {...})` → `None`. Forcing it
is *not* a free win either: at BK=128 it overflows LDS (80 KB required, 64 KB limit), and at BK=64/s2 it runs
2.769 ms (32.2 GB/s) with **max abs diff 80.8 against the default kernel's output** — the scale advance
`scale_step_k = BLOCK_SIZE_K // group_k` is 0 when BK < group_k, so the per-block scales never step. It is both
unreachable and wrong. Fixing the missing `return` would ship incorrect numbers.

---

## The single biggest lever this reveals

**The decode step is a 14.52 GB weight re-read per card executed at 17.8 GB/s by one Triton kernel family.**

Arithmetic, all inputs measured: 14.52 GB/card ÷ 786.3 ms = **18.5 GB/s** effective. At the card's measured
592.9 GiB/s (636.6 GB/s) the same bytes take **22.8 ms**; at the 471.9 GB/s that hipBLASLt actually achieves for
an M=1 GEMM on this card, **30.8 ms**. Both are ~25–35× below where we are. The 857 ms is 34× the bandwidth
bound and 1 000× the FLOP bound — the kernel is neither streaming nor computing, it is waiting.

Ranked, with what each is worth and what it costs:

| lever | measured effect | confidence |
|---|---|---|
| Tune the 5 production shapes (BM=16/BN=64/BK=64 or per-shape best), i.e. add `configs/N=…,device_name=AMD_Radeon_Graphics…json` | 786 → **156 ms/step, 5.5×**, → **6.4 tok/s** | high — measured per shape, numerics verified |
| A gfx1100-native fp8 GEMV that upcasts fp8→bf16 and streams at the 410–472 GB/s hipBLASLt already demonstrates | 156 → **~26–31 ms/step ⇒ ~30–38 tok/s** | medium — the ceiling is measured, the kernel is not written |
| Everything else in the profile (RCCL 0.4 %, GDN 0.14 %, casts 0.36 %) | ≤ 9 ms/step combined | high |

So the first native-kernel ticket should target **the M=1 blockwise-fp8 GEMV for the five shapes
(17408,5120) (5120,8704) (5120,5120) (8192,5120) (7168,5120)** — not attention, not the GDN path, not the
all-reduce. Two things make that ticket unusually cheap to validate: the isolated harness in this report already
reproduces the in-server per-shape time to 9 %, and any candidate kernel can be checked for correctness against
the fp32 block-dequantised reference that both current configs already pass at 2.9e-6.

The config-file lever deserves to be taken first and separately: it is a data-only change, it is 5.5×, and it
tells us how much of the remaining gap a better *scheduling* of this kernel closes before anyone writes HIP.

---

## Claims I could not verify

- **Why** the default config is 5.4× slower than BM=16/BN=64/BK=64. I have no hardware counters (no SQ/L2
  occupancy, no instruction-issue or LDS-pressure data); rocprofv3 was available but I profiled with
  torch/kineto only. The cause is very likely occupancy/LDS at BLOCK_M=64 with M=1 (64× the useful rows of
  arithmetic) plus no fp8 MFMA on gfx1100, but that is inference from config sensitivity, not a measurement.
- **Kernels inside the decode CUDA graph.** Not traced. I rely on the eager/graph parity measurement
  (862 vs 855–885 ms, 0.6 %) rather than on a graph-internal trace.
- **Only 8 decode steps** in the window, one sequence, one prompt length. Nothing here tests batch > 1, and the
  cudagraph capture set includes bs ∈ {1,2,4,8} — the tuned config's behaviour at M=8 is untouched by this report.
  (M=8 should be *easier*; the default config may even be reasonable there, which would explain why nobody
  noticed.)
- **`Cijk_…MT64x64x32` = `lm_head`** is an inference from launch count (1/step), duration (2.32 ms) and byte
  count (1.271 GB bf16 → 547 GB/s). The kernel symbol does not name the tensor.
- **Whether the missing `return` at `fp8_kernel.py:683` is a fork regression or inherited.** No upstream tree
  available offline to diff against.
- **The 55.8 ms/step** outside kernels is attributed to launch gaps/sampling from the absence of kernel events,
  not decomposed from the CPU-side trace (I did not aggregate `cpu_op`/`cuda_runtime` self times).
- Prefill (~4 600 tok/s) was not profiled; it is consistent with the same kernel at M≥2048 amortising the weight
  read 2 000×, but that is arithmetic, not this measurement.
