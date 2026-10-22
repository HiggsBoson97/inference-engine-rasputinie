# DeepInfer — 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9 | Apache-2.0 | Python + CUDA C++ (SM90/SM100) + Ascend C++ + TileLang

## What it is

DeepSeek's production engine, released as an offline-inference slice, targeting **NVIDIA SM100 and Ascend 950 only** (`README.md:35-36`) — not H800, so the framing is worse than the brief assumed: every kernel is written for `tcgen05`/TMEM or Ascend cube units, and nothing here has ever run on RDNA3. The 5,908 lines of `csrc/` contain **zero `__global__` kernels** — scheduler and CPU-side metadata packing only; the math lives in `vendor/` submodules. What transfers is *contracts and layouts*, ~15% of the design and almost none of the code.

## Architecture in one text diagram

```
requirements-nvidia.txt (torch 2.13+cu132, flash-attn-4)   requirements-ascend.txt (torch-npu)
             \                                              /
              build.sh:5-11   DEEPINFER_ACCELERATOR ∈ {nvidia, ascend}
                              |
        setup.py:12  IS_NPU ? NpuExtension : CUDAExtension     <- binary fork, not runtime plugin
   ┌──────────────────────────┴───────────────────────────┐
   │ csrc/engine/*.cpp        CPU, no CUDA API: scheduler, ZOS decode prep, block table
   │ model_executor/models/v4_modules/  mla_attn, index_attn, moe, mlp, engram, hc
   │ layers/tilelang_ops/ops.py:15  → ONE backend dir: cuda/ (30 ops) or asc/ (30 ops)
   │ parallel_utils/userbuffer.py:63  → EPBuffer: dispatch/combine = DeepEP, only option
   └──────────────────────────┬───────────────────────────┘
                              |
   vendor/DeepEP (EPv2.5, NCCL-Gin, NVSHMEM deleted)   vendor/DeepGEMM (wgmma | tcgen05)
   vendor/FlashMLA (imported unconditionally)           vendor/DeepSelect
```

## Techniques worth stealing

| technique | where (file:line) | what it buys on gfx1100 | effort | verdict |
|---|---|---|---|---|
| Expanded contiguous MoE layout keyed by a **cumulative psum** (`psum[i]` = aligned prefix sum; `align(psum[i]) − align(psum[i−1])` = real count) | `deep_ep/buffers/ep.py:45-55`; consumed `v4_modules/mlp.py:113-136` | One int32 tensor replaces `m_indices`/sort enough to drive a grouped GEMM with no host sync. HIP-portable as-is. | S | TRANSFERS |
| Legacy **Triton** grouped bf16 GEMM with `m_indices`, incl. empty-expert zero-fill (`batch_id < 0`) | `vendor/DeepGEMM/deep_gemm/legacy/m_grouped_gemm.py:10-40` | The only grouped-GEMM kernel here that is already portable: plain `tl.dot`, no TMA/wgmma. Triton 3.8 lowers it to MFMA/WMMA. | M | TRANSFERS |
| `m_grouped_*_gemm_nt_masked` + host `expected_m` so M varies without re-capturing a graph | `vendor/DeepGEMM/csrc/apis/gemm.hpp:269-300`; `mlp.py:596-600` | Your decode case: 10 tokens over 512 experts. Removes a graph re-capture per batch size. | M | TRANSFERS-WITH-WORK |
| Blockwise FP8 scale *contract*: per-token 1×128 on A, 128N×128K on B (SM100: 32-granular packed ue8m0) | `csrc/utils/layout.hpp:88-95`; `sm90_fp8_gemm_1d2d.cuh:56` | Scales are separate tensors with their own ABI, so fp8 can be decompressed to bf16 outside the GEMM without touching the layout. | M | TRANSFERS |
| Per-op backend dirs + one dispatcher (`importlib` over `PLATFORM.tilelang_ops_backend`) | `layers/tilelang_ops/ops.py:15-22`; 30 files in `cuda/`, 30 in `asc/` | A `hip/` dir is additive, no call-site edits. **18 of 30 `cuda/` files already disable TMA lowering** (`cuda/fused_sample.py:13-17`) — plain SIMT, plausibly re-targetable. | M-L | TRANSFERS-WITH-WORK |
| Single-kernel fused sampler (softmax→filter→cumsum→multinomial, one launch) | `cuda/fused_sample.py:19-45` | At ~28 ms/token your sampling chain is 5-8 launches; fusing is a measurable decode win and it is TMA-free. | M | TRANSFERS-WITH-WORK |
| CPU-side zero-overhead decode metadata (pinned packing, `nextn` varlen) | `csrc/engine/zos.cpp:31,142,198,279` — 0 matches for `cuda` | Pure C++; compiles against HIP untouched. Kills the scheduler gap between graph replays. | S | TRANSFERS |
| Graph replay bucketed by batch size + per-graph-node event timing | `worker/cuda_graph_runner.py:100-137`; `csrc/graph_timer/graph_timer.hpp:15-31` | `hipGraph` has the same node/event model; per-op attribution for free. | M | TRANSFERS-WITH-WORK |
| `accelerator_kind()` three-way probe as the only hardware branch | `deep_cluster/hardware/device.py:18-23`; `platform.py:14-16` | Add `"amd"` + `PlatformEnum.ROCM` and the engine boots; `build.sh:5-11` is a one-line change. | S | TRANSFERS |
| Engram: 52B-class ngram table in a symmetric buffer, RDMA fetch, one wait hook per layer | `v4_modules/engram.py:164,195,214` | `initialize()` *moves* CPU weights into `EngramBuffer` and frees host storage (`engram.py:208-212`); nothing host-resident survives. | — | DOES-NOT-TRANSFER |
| DeepEP `dispatch_impl`/`combine_impl` device-side peer writes; "low-latency" single cooperative launch | `impls/ep/dispatch.cuh:31`; `combine.cuh:29`; `csrc/kernels/comm/context.cpp:126-164` | Needs `ncclGetLsaPointer` P2P and NCCL-Gin GDAKI. **Consumer RDNA3 has no peer DMA at all.** | — | DOES-NOT-TRANSFER |
| FlashMLA absorbed MLA + sparse index attention | `layers/attention_backend.py:2` — imported unconditionally at module scope | Wrong geometry for Qwen3.8 (GQA, head_dim 256, partial rotary 0.25) and SM90-only. | — | DOES-NOT-TRANSFER |

## Top 3 to port first

**1. The psum-expanded contract + the legacy Triton grouped GEMM (`ep.py:45-55`, `legacy/m_grouped_gemm.py:10-40`).**
Why it pays here: 70.4% of Flash-Next's weights are routed experts, and at decode the routed-expert read is 10 experts × 48 layers × 4.92 M params × 2 B = **4.72 GB/token**; split EP=2 that is ~2.4 GB per card, ~4 ms at 592 GiB/s. At M=1 per expert this is not a GEMM, it is bandwidth streaming — so the only thing the kernel must get right is layout, and DeepGEMM's layout is the one the quantizer, psum, and EPLB shuffle already assume. Adopt it before you touch Q4 weights.
First step: `tests/test_grouped_gemm_psum.py` — generate a psum for 512 experts/top-10 with `expert_alignment=128`, run the Triton kernel, compare against a per-expert `matmul` loop including zero-token experts. Then measure a `M=1`-per-expert specialization (batched GEMV, one block per expert); I expect it wins, and *that* kernel is yours, not theirs.

**2. The TileLang backend seam plus its 18 TMA-free kernels (`ops.py:15-22`, `cuda/*.py`).**
Why it pays here: the only place portability is structural rather than aspirational — one dispatcher, one dir per backend, and the kernels that matter at decode (sampler, norm, kv load/store, gather, seqlens) already opt out of TMA and warp specialization, precisely the two gfx1100 features you lack. Effort is "30 small kernels", not "an engine".
First step: add `tilelang_ops/hip/` with `norm_tl` and `load_paged_kv_cache_tl` only, set `tilelang_ops_backend="hip"`, and let `KernelUnavailableError` (`ops.py:25-32`) make the missing 28 fail loudly at import instead of silently. Correctness against the CUDA math, then speed.

**3. The CPU-side zero-overhead decode path (`csrc/engine/zos.cpp`, `cuda_graph_runner.py:100-137`).**
Why it pays here: the dense 27B ceiling is ~28 ms/token, so 1 ms of scheduler/launch gap is 3.5% of throughput — and 8 cores sit idle while the GPU streams weights. `zos.cpp` is 307 lines of pinned-buffer packing with **no CUDA API in it**.
First step: compile `prepare_decode_zos_inputs` against torch-ROCm untouched, unit-test psum/remap outputs on CPU, then wire `torch.cuda.graph` (ROCm maps it to `hipGraph`) with 4 batch buckets.

## What does NOT transfer and why

- **All of `vendor/DeepEP`.** EPv2.5 deleted NVSHMEM (`DeepEP/README.md:18`) for the NCCL device API + Gin/GDAKI RDMA: `ncclDevCommCreate(..., NCCL_GIN_TYPE_GDAKI)` (`context.cpp:134,164`), `ncclCommWindowRegister` (`:196-199`), peer pointers via `ncclGetLsaPointer` (`handle.cuh:66-92`). RCCL exposes none of it. The "fallback" is not one: when a peer is not NVLink-visible, `get_sym_ptr` returns `nullptr` and the op routes **over the RDMA NIC** (`handle.cuh:71-118`), and init hard-asserts Gin unless `EP_DISABLE_GIN=1`, which the README scopes to NVLink-only (`context.cpp:124-131`). No PCIe path, no shared-memory path, no `all_to_all` in the library (only the test oracle). **No peer DMA on consumer RDNA3** means host bounce at 14.2 GiB/s, and DeepEP's core trick — comm SMs writing into peer HBM while GEMM SMs run — has no substrate. Copy the psum contract, drop the transport.
- **MoE dispatch is not optional here.** `mlp.py:571-578` calls `self.dispatch()` unconditionally unless `use_mega_moe`, and `moe.py` (`FusedExperts:70`, `TopKGate:40`) is parameter containers only. No TP path, no `ep_size==1` skip anywhere in `deepinfer/`. You must write that seam.
- **All vendored kernels**: wgmma (`sm90_fp8_gemm_1d2d.cuh:313`), tcgen05+TMEM+UTCCP (`sm100_fp8_fp4_gemm_1d1d.cuh:18,49,187`), TMA descriptors (`jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:101-118`), 2-CTA clusters (`heuristics/sm90.hpp:71-74`, optional anyway). gfx1100 has none; fp8 must decompress to bf16, and DeepGEMM ships **no fp8→bf16 dequant kernel** (scale promotion happens inside the MMA loop, `sm90_fp8_gemm_1d2d.cuh:331-345`). Note the SM100 recipe is `(1,1,128)` packed ue8m0 (`layout.hpp:93-95`) — 32-granular, not the 128×128 block layout most people mean by "blockwise fp8".
- **Attention, KV layout, Engram, DSA/index attention**: all MLA- or V4-specific. Qwen3.8's 48 gated-DeltaNet layers (151 MB recurrent state/seq) have no analogue — grep finds no linear-attention, conv-state, or SSM code.
- **`flash-attn-4==4.0.0b34`** (`requirements-nvidia.txt:5`): no ROCm build; you need your own paged-attention kernel regardless.

## Claims I could not verify

- The pareto images (`doc/images/*.png`) — the only hard number is "4 SM100 GPUs for prefill and 64 for decode" (`README.md:83`). No benchmark run, no script outputs checked.
- `flash-attn-4` is a wheel, not vendored; its arch requirements are not establishable from this checkout.
- "Batch-invariant inference (NVIDIA only)" / deterministic sampling (`README.md:27-28`) — env vars noted, kernel-level determinism path not located.
- DSpark / confidence-scheduled verification: env vars exist (`environ.py:129-132`) and `mtp_varlen.hpp:19-30` handles varlen drafts, but the confidence scheduling logic I did not read.
- Ascend backend completeness — 30 `asc/` files mirror 30 `cuda/` ones and `setup.py:41-53` builds an `NpuExtension`, but file parity is not correctness and I did not build it.
- DeepGEMM's "SM90 NT only" claim — taken from `vendor/DeepGEMM/README.md:63`, not re-derived.

## Source map

- `README.md:35-36,83,27-28`; `requirements-nvidia.txt:1-5`; `requirements-ascend.txt:1-4`; `build.sh:5-11`; `setup.py:12,38-72`
- `deep_cluster/hardware/device.py:18-23,160-164`; `deepinfer/platform.py:14-16`
- `csrc/` 5,908 lines, **0 `__global__`**; `csrc/engine/zos.cpp:31,142,198,279`; `csrc/engine/mtp_varlen.hpp:19-30`
- `v4_modules/moe.py:24,40,70`; `v4_modules/mlp.py:113-136,184-188,461,469,493,501,571-578`; `v4_modules/engram.py:164,195,208-218`; `layers/attention_backend.py:2`
- `layers/tilelang_ops/ops.py:15-32`; `cuda/fused_sample.py:10-45`; `cuda/norm.py:11-33`
- `parallel_utils/userbuffer.py:63-95`; `eplb/shuffle.py:150,178`; `worker/cuda_graph_runner.py:100-137`; `csrc/graph_timer/graph_timer.hpp:15-31`
- `vendor/DeepGEMM/csrc/apis/gemm.hpp:269-300,567,635,870-918`; `csrc/utils/layout.hpp:88-95`; `deep_gemm/legacy/m_grouped_gemm.py:10-40`
- `vendor/DeepEP/README.md:18,29`; `deep_ep/buffers/ep.py:45-55`; `csrc/kernels/comm/context.cpp:124-137,164,196-199`; `deep_ep/include/deep_ep/comm/handle.cuh:66-118`; `impls/ep/dispatch.cuh:31`; `impls/ep/combine.cuh:29`; `csrc/kernels/ep/dispatch.hpp:124-131`
