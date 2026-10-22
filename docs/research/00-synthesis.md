# 00 — Synthesis of the four engine reports

> **Superseded in part.** The owner rejected the composition proposed below and
> chose to fork sglang instead (`docs/adr/0002-fork-sglang.md`). The six reports
> and the three convergent findings stand; the "composition proposal" section
> does not. Treat the reports as evidence about *techniques*, not about bases.


Compiled 2026-10-22 from `10-strata.md`, `11-freetoken.md`, `12-deepinfer.md`,
`13-tensorfold.md`. The sglang report (`14`) and the quantised-kernel report
(`15`) were still running when this was written; their conclusions are not
reflected here.

## The four are complementary, not competing

| Engine | Lang / license | 27B (dense `qwen3_5`) | Flash-Next (`qwen4_exp`) | gfx1100 | The asset only it has |
|---|---|---|---|---|---|
| Strata | C++20, MIT | not found in the report | **yes, headline model** | **validated** | three-tier expert store, router look-ahead, ngram PleReader, hipGraph per layer |
| FreeToken | Python + Triton, Apache-2.0 | no (it has `qwen3_5_moe`) | **yes** | experimental, gfx1100 listed | hybrid radix cache: KV pages **and** GDN state **and** SWA pages as tree currencies |
| TensorFold | Zig host + HIP C++, Apache-2.0 | **yes, it serves it** | not stated | **opt-in HIP backend, gfx1100 tuning tables** | column-major weights + `stream` GEMV tile + `dot2` DPP decode |
| DeepInfer | Python + CUDA, Apache-2.0 | no | no (V4-specific) | no (SM100 / Ascend) | psum MoE layout contract, legacy Triton grouped GEMM, TileLang backend seam |

No single engine is the answer. Each one owns a different third of the problem,
and all four licences permit reuse with attribution.

## Three findings that three independent codebases agree on

1. **Decode is launch-bound as well as bandwidth-bound.** Strata measured one
   hipGraph per layer at 1.585 ms against 2.393 ms for the same 43 nodes
   (**1.51x, pure launch overhead**). TensorFold captures a graph per round
   shape. FreeToken runs one batch ahead to hide launch and scheduling. This
   makes graph capture a phase-2 task and the megakernel a phase-8 one.
2. **The 27B is pinned to 592 GiB/s.** TensorFold's receipt of 12.87 tok/s on a
   gfx1151 at ~256 GB/s scales to ~30 tok/s at 592 GB/s, which is the same
   ceiling the arithmetic predicts. Two independent derivations, one number.
3. **The 151 MB per-sequence recurrent state must be a currency in the prefix
   cache.** FreeToken already models this: a radix node owns KV pages, a GDN
   snapshot and SWA pages, each with its own reference count. Without it a
   prefix hit on a hybrid model cannot be resumed, and the KV dedup work is a
   KV-only win on a model where KV is the smaller cost.

## Composition proposal

**Base: Strata** (C++20, HIP, Apache-free MIT, graphs and tiering already
there), with:

- **FreeToken as the executable reference** for the hybrid GDN radix cache and
  the model math. It is Python, so reading it is cheap; port the semantics, not
  the code.
- **TensorFold's HIP GEMV family** as the 27B decode kernels (column-major,
  stream tile, dot2-DPP). Apache-2.0: carry `LICENSE` and `NOTICE`, the kernel
  files have no per-file headers.
- **DeepInfer's psum layout** as the MoE contract, and nothing else from it.
- **sglang as the API and scheduler reference**, not as the base.

The alternative bases were considered: FreeToken (fastest iteration, but TP on
AMD is an admitted TODO and the decode path is Python), sglang (the serving
stack, but CDNA-oriented), and a thin engine from scratch (now clearly worse —
it would rebuild four things that already exist and are measured).

The counter-argument to Strata is honest: it has no confirmed dense `qwen3_5`
implementation, so the 27B model code is ours either way. TensorFold is the only
engine in the set confirmed to serve the 27B on gfx1100, which is why its kernels
are the largest single borrow.

## Decision this implies

Fork-versus-build was the wrong axis. The real decision is **which host
language the engine speaks** — C++20 (Strata) or Python/Triton (FreeToken) —
because that decides iteration speed against peak decode speed, and it decides
whether multi-user serving is the first milestone or the fifth. That question is
for the PLAN grill, with the user, not for this document.

## Addendum: reports 14 (sglang) and 15 (quantised kernels)

**sglang is the model reference, not the base.** Both architectures exist
upstream and are complete: `models/qwen3_5.py` has the gated-deltanet layer, the
LLLF hybrid wiring and the MTP head; `models/qwen4_exp.py` has the ngram/PLE
layers, the hyper-connections and its own MTP. That makes it the most complete
executable specification of our two models.

It is still not a base. Every ROCm architecture helper is gfx942 / gfx950 /
gfx1250, **no RDNA3 helper exists**, and the AOT kernel build hard-exits on
gfx1100 (`kernels/aot/setup_rocm.py:77-81`). The default attention backend on
HIP is `aiter`, whose non-CK path is opt-in and off by default. At M=1 with 512
experts the Triton MoE pads 10 real rows to about 160 — roughly **6% arithmetic
efficiency** — and no grouped GEMM exists. Decode graphs, the radix cache and
HiCache's L2/L3 are the parts that work or nearly work; HiCache's transfer ops
ship only in `sgl_kernel`, which refuses gfx1100.

So: do not fork sglang; port its model code as the specification, and write the
kernels. Its own top-3 gap list agrees with our composition — a batch-1
10-of-512 MoE kernel for wave32, a CK-free wave32 attention at head_dim 256 with
6:1 GQA, and a working `kvcacheio` for the host KV tier.

**GSQ-RCO needs no kernel.** It is a mixed-precision recipe, not a container
format: the published files are plain GGUF with unmodified type ids, and
`general.file_type` is a label, not a promise. The engine must dispatch per
tensor from the type field. Our own probe already proved the point without
knowing it: the file named `UD-Q4_K_XL` holds nine types at 5.05 bits/weight.

**First kernel: `gemv_k_q8_1`.** Q5_K times int8 activations, one code path
templated over Q5_K/Q4_K/Q6_K, which is 79% of the bytes this model reads per
token. The design is worked out at wave/block level with a falsifiable target:
480 GiB/s (81% of 592) and roughly 29-31 ms/token. Before writing it, one
30-minute `v_dot4_i32_i8` throughput microbenchmark decides whether the kernel
is bandwidth-bound or issue-bound; if the dot is quarter-rate on gfx11, the fix
is fewer dots per weight, not more parallelism.
