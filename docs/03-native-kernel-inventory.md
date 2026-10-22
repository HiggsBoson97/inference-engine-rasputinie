# 03 — Native-kernel inventory

The contract for "100% gfx1100 native". Every row is a family of work that
currently runs code written for another architecture. A row is closed only when
its native path passes its test and its predecessor is **deleted**.

## What "native" means here

1. **No kernel in the measured path was written for another architecture.**
   Triton is allowed as a correctness reference and as a temporary scaffold,
   never as the shipped path. `torch` ops are allowed only where the whole
   operation is one launch and there is no measurable win from fusing it.
2. **The old path is deleted, not flagged.** A flag is how an engine rots: two
   half-tested paths, and nobody knows which one the number came from.
3. **Enabling changes are not ports.** The arch allowlist, `SGL_ROCM_WARP_SIZE`,
   C++20, the Quark import guard and the RCCL all-reduce are first-class code
   that stays. They are not the CDNA path kept alive; they are what makes the
   tree build and run here at all.

## The inventory

Ordered by expected gain, with the evidence we have today.

| # | Family | Today | Evidence | Native target | Test | Priority |
|---|---|---|---|---|---|---|
| 1 | Quantised GEMV for GGUF k/i-quants (dense MLP + attention projections) | nothing — no GGUF loader exists yet | IQ3_S is 5.9 GB/card at TP=2 → 9.5 ms floor; the five hot types are 94.5% of the bytes (`00-facts.md`) | HIP wave32, column-major, one-pass, dequant to bf16 in registers, WMMA above the GEMV crossover | bit-exact vs a torch dequant reference, then ≥60% of 619 GB/s | **P0** |
| 1b | FP8 GEMM (**reference only**, never a target) | Triton `_w8a8_block_fp8_matmul`, now with tuned tiles | 857 ms/token, 91.8% of decode, 17.7 GB/s vs 472 for bf16 hipBLASLt; no FP8 MMA on gfx1100 (`ADR 0003`) | none. Deleted when the GGUF path serves the model | — | — |
| 2 | KV cache dtype and paging | bf16, 45,727 tokens, 1.4 GB | 2.08 GB free per card → `max_running_requests=3` | q8 KV + paged layout, dequant fused into attention | logits within 1e-3 of bf16; tokens/session doubles | P1 |
| 3 | GDN linear-attention decode (48 of 64 layers) | Triton `chunk_gated_delta_rule_*`, `recompute_w_u_fwd_kernel` | 1-2 s JIT per kernel at first request; 151 MB state/seq | HIP recurrent kernel; state resident, O(1) per token | 32-token logit parity vs Triton; no per-token state rebuild | P1 |
| 4 | Full-attention decode (16 layers, head_dim 256, 6:1 GQA) | Triton attention backend | triton chosen after aiter proved MFMA-only (`19-aiter-gfx1100.md`) | HIP flash-decode for wave32, paged KV, no per-length specialisation | ≤1e-3 vs Triton, ≥1.15x | P2 |
| 5 | Graph capture | hipGraph via `torch.cuda.CUDAGraph` | **41 s** for bs=[1,2,3] | capture per layer, warm at init, bucket by batch | capture <5 s; replay bit-identical | P1 |
| 6 | Sampler | `sampling_backend: pytorch` | separate torch ops per token | one fused kernel (softmax→filter→cumsum→sample) | exact distribution match on fixed seeds | P2 |
| 7 | Fusions: RMSNorm+residual, RoPE, swish gate, SwiGLU | torch ops, several launches each | every launch is 20-30 µs on this stack | fused HIP kernels | ≤ launches, graph-safe | P2 |
| 8 | All-reduce (TP) | pynccl → `ncclAllReduce` on librccl | 20.65 µs per 5120-element reduce, 2.64 ms/token | own HIP ring reduce; PCIe-aware | ≤20 µs/token total | P3 |
| 9 | Speculation: MTP, then DFLASH block verify + top-k | in-tree, Triton + `torch.topk` | flashinfer top-k absent on ROCm; source says torch.topk halves throughput | HIP top-k over 248k vocab, fused to LM head | acceptance ≥2.8 at block 8 | P3 |
| 10 | HiCache host tier | disabled | `kvcacheio` refuses gfx1100 in the AOT build | HIP kvcacheio | host-tier KV at ~46 GB/s | P4 |
| 11 | Megakernel | not started | launch overhead after 1-10 | persistent per-layer kernel | only after 10 | P5 |

## What is out of scope, and why

- **MoE kernels.** The 27B is dense. DeepInfer's psum layout and every grouped
  expert kernel are irrelevant to this target.
- **FlashNext / ngram / tiered expert store.** Different model, dropped.
- **CUDA parity.** The NVIDIA branches stay behind dispatch; they are not our
  problem to keep working.

## Not yet measured, and therefore not in the table

- Whether the 857 ms is the fp8 GEMM itself or a per-step conversion around it.
  The profile (`20-decode-profile-27b.md`) decides which row 1 becomes: a kernel
  rewrite, or a deletion of a copy that should not happen.
- The achieved GB/s of an isolated fp8 GEMM at the model's shapes.
- The real acceptance rate of MTP and DFLASH on this box.
