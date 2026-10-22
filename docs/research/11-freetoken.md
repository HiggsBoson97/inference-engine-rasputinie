# FreeToken — 546c5fe | Apache-2.0 | Python + Triton + CUDA/HIP C++

Read at `546c5fe2e4468e2f3f8a574c64021a47016c35f7` (main, 2026-10-21), clean tree. Vendored subtrees keep their own terms: `kernel/fla/` is MIT (Songlin Yang / vLLM PR #53906), `kernel/csrc/ggml/` is MIT — carry attribution if we lift them.

## What it is

A single-node MoE engine that assumes the model does not fit in VRAM: experts live in host RAM and stream into a global LRU GPU slot cache, one layer at a time, overlapped. It targets our exact architectures — `models/qwen3_5_moe/` (gated-deltanet hybrid) and `models/qwen4_exp/` (Qwen3.8-Flash-Next incl. its 47.7 GiB n-gram table). No spec decode: MTP is dropped at load (`qwen4_exp/weight.py:9,92`); `docs/models.md:146` — "DSpark speculative decoding is not served."

## Architecture in one text diagram

```
PrefillManager ─┐                       ┌─ Engine.forward: graph replay (engine.py:1227) | eager
 ChunkedReq     │ scheduler.py:870-875  │
 DecodeManager ─┘ prefill XOR decode,   │
 (decode.py:20)   never mixed, 8192 tok ▼
        ▼               CacheManager (scheduler/cache.py): page_table[req][tok] -> KV page
                         └─ prefix tree: radix|hybrid_radix|swa_radix (config.py:17)
                            ├─ KV pages, paged pool, page_size=1 (engine/config.py:87)
                            ├─ node.mamba_value -> LinearStatePool slot (GDN state)
                            └─ node.swa_*       -> SWA pool pages
 MoE layer ── OffloadMoeCache (offload_cache.py:121)
               ├─ GPU slot cache, global LRU, per-pool
               ├─ PCIe gather of misses (fast_index_copy)
               └─ CPU executor GEMV for the overflow (cpu/hybrid)
```

**(a) Scheduler.** Continuous batching, prefill-first, never mixed: `_schedule_next_batch` is `prefill or decode` (`scheduler.py:870-875`), so a decode step never rides with a prefill chunk, unlike vLLM. Chunked prefill is real: `ChunkedReq` suppresses sampling and decode admission (`prefill.py:31-42`); `PrefillAdder` admits against three currencies — KV pages, GDN state slots (1 live + 2 ping-pong, `prefill.py:91-98`), SWA pages (`prefill.py:100-112`). Overlap scheduling runs one batch ahead: launch N, drain N-1 while N runs (`scheduler.py:214-268`), with explicit cross-stream waits because the page table and GDN snapshots are rewritten between batches.

**(b) Larger than VRAM.** The strategy that works is *hybrid*, and it is arithmetic, not heuristics: `hybrid_fetch_fraction` sets, per MoE layer, the share of that step's expert *misses* fetched over PCIe, the CPU computing the rest, chosen as `pcie_bw / cpu_bw` so both finish together (`offload_cache.py:156-160`; per layer kind at `engine.py:913-934` from a cached `ft bench bw` profile). Prefill double-buffers whole layers so N+1's copy hides behind N's compute (`offload_cache.py:859-1050`), going serial — with a logged warning — when the slot cache cannot hold two whole layers. No mmap, no fault-driven paging: bytes come from pinned or OS-locked host banks, the latter synchronous and overlap-killing (`docs/models.md:72-85`). Multi-GPU is plain TP all-reduce (`distributed/impl.py:26-49`).

**(c) Prefix caching.** Prefix-addressed only: the key is the token id, or a tuple of `page_size` ids — no hashing, no content addressing, no dedup past a shared prefix (`radix_cache.py:265-268`). The multi-currency tree is the interesting part: a node may own KV pages, a GDN snapshot (`mamba_value`, `radix_cache.py:28-38`) and SWA pages, each with its own ref count and LRU, matched only to the deepest live snapshot boundary (`hybrid_radix_cache.py:43-48`). One semantic hook exists: when a sampled token equals the tokenizer's tool-call anchor id, GDN state is frozen there, so a rewritten echoed tool call forks instead of re-prefilling (`scheduler.py:376-380` → `cache.py:153-170`). One anchored position, not semantic matching.

**(d) Kernels and selection.** Three tiers: Triton by default (`kernel/triton/`, 36 modules; `kernel/fla/` vendored chunked delta-rule); C++/CUDA C for gather/scatter/memcpy and CPU MoE (`csrc/jit/*`, `csrc/cpu_moe/`, a patched ggml CUDA port); Gluon for NVFP4 on sm120. Call sites pick via `backend.py`: importable probes for flashinfer/sgl_kernel/vllm, each forced False on ROCm so Triton is taken (`backend.py:11-14,46-59`), plus closed arch-family gates (`utils/arch.py:84-100`). Compiled kernels go through tvm-ffi `load_jit`/`load_aot`, keyed by a name hash folding in arch, with a prebuilt `.so` wheel checked first (`kernel/utils.py:292-378`).

**(e) ROCm path.** Present, explicitly experimental: `docs/install_amd.md` scopes gfx1100–gfx1103/gfx1200/gfx1201 on ROCm 7.14. `is_rocm()` probes `torch.version.hip` (`backend.py:46`); `_rocm_arch_list()` targets one gfx arch (`kernel/utils.py:51-61`); `_hip_cflags` adds `--offload-arch` (`:91-94`, `TODO(ROCm)` on RDNA autotune); FLA swaps autotune grids for AMD (`fla/kda.py:36`); `gpu_sync_hip.cpp` supplies HIP stream-memory-op sync. CPU/hybrid MoE needs `--cuda-graph-max-bs 0` on ROCm 7.14 because graph nodes are added by hand. TP on AMD is an admitted gap: `pynccl.py:29` — "TODO(ROCm): NCCL -> RCCL migration".

**(f) Benchmarks.** `bench_decode_moe.py` measures bs=1 tok/s by spawning `ft serve` per backend and timing token arrivals over streamed `/v1/chat/completions` — end-to-end, including HTTP, tokenizer and sampling, not kernel time. `bench_offload_cache_copy.py` is a synthetic copy sweep. Steal `ft bench bw` (`moe/benchbw.py`): STREAM host bandwidth (`:195`), pinned H2D/D2H (`:267`), *actual* expert-gather bandwidth through `copy_missing` (`:472`), *actual* CPU GEMV bandwidth (`:561`), both overlapped (`:593`); `recommend()` says hybrid iff cpu_bw > 2×pcie_bw (`:653`), and rows persist per GPU UUID as JSON the engine reads for the fetch fraction.

## Techniques worth stealing

| technique | where (file:line) | what it buys on gfx1100 | effort | verdict |
|---|---|---|---|---|
| Hybrid GDN radix: state snapshots as second tree currency, COW, own LRU | `kvcache/hybrid_radix_cache.py:43-48`, `linear_state_pool.py:34-42` | 48 linear layers at 151 MB state/seq — without it a prefix hit cannot be resumed | M | TRANSFERS |
| Tool-call anchor freeze | `scheduler.py:376-380`, `cache.py:153-170` | fork a rewritten tool call instead of re-prefilling 100k tokens | S | TRANSFERS |
| Chunked prefill + 3-currency admission | `prefill.py:31-42,91-112` | long prompts without a whole-prompt KV reservation in 20 GiB | M | TRANSFERS |
| Vendored FLA chunked delta-rule, `is_amd` grids | `fla/kda.py:36`, `fla/utils.py:269`, `qwen3_5_moe/gdn_kernels.py:7` | ~1.4k lines of gated-delta-rule we would otherwise write | S | TRANSFERS |
| Pinned-UVA n-gram table + side-stream prefetch | `qwen4_exp/ple.py:120-160`, `ple_disk.py:1-5` | 47.7 GiB table (29% of Flash-Next) cannot fit; gather is latency-bound | M | TRANSFERS |
| Global LRU expert slot cache, pooled, hot-count split | `offload_cache.py:121-170`, `moe/pool_policy.py:split_slots` | ~72 GiB of routed experts into 20 GiB | M | TRANSFERS |
| Bandwidth-matched fetch fraction + double-buffered layer overlap | `offload_cache.py:156-160,859-1050`, `engine.py:913-934` | "fetch ~half, compute the rest"; chunk = max(copy, compute) | M | TRANSFERS-WITH-WORK |
| tvm-ffi JIT/AOT cache keyed by arch + prebuilt `.so` wheel | `kernel/utils.py:292-378`, `freetoken-kernel-cache/` | kills first-token compile stalls; gfx1100 path exists | M | TRANSFERS-WITH-WORK |
| One-batch-ahead overlap + padded graph replay | `scheduler.py:214-268`, `engine.py:57-60,1227-1229` | hides host scheduling and launch overhead; ROCm needs manual graph nodes | L | TRANSFERS-WITH-WORK |
| Sparse pools; NVFP4/Marlin/b12x/Gluon; PyNCCL TP | `kvcache/dsa_pool.py`, `nvfp4_moe.py`, `gluon/`, `pynccl.py:29` | nothing: no sparse indexer, no FP4 MMA, no working P2P | — | DOES-NOT-TRANSFER |

## Top 3 to port first

**1. Hybrid GDN radix + LinearStatePool + tool-call anchor.** Highest value per unit of work: pure Triton/torch, no new hardware feature, and our primary model is 48/64 linear-attention layers. Without it, prefix caching on Qwen3.8-27B is a KV-only win that cannot be resumed — 151 MB of recurrent state per sequence has no page in the tree. First step: port `LinearStatePool` (one slot per `table_idx`, conv + SSM tensors, fp32) and check identity against a from-scratch prefill on a 2-layer toy GDN model; then `mamba_value`, COW fork, deepest-live-snapshot truncation. The anchor is ~40 lines after that.

**2. `ft bench bw`, as our baseline-table generator.** It measures the five numbers that decide every remaining design choice and stores them per GPU UUID. Expect a surprise: their rule needs cpu_bw > 2×pcie_bw (`benchbw.py:653`), and 46.4 GiB/s host copy against PCIe 4.0 x16 pinned copy likely lands just under 2× — their rule would say `offload`, while the fraction formula says ~50/50. First step: one script emitting baseline JSON plus the ratio, not a verdict.

**3. Bandwidth-matched hybrid split + two-buffer prefill overlap.** Only Flash-Next needs it, but there it is the whole ballgame: ~72 GiB of routed experts into 20 GiB. Steal the *shape* of the decision, not the code. First step: a two-layer simulator taking `(expert_bytes_per_layer, pcie_bw, cpu_bw, chunk_tokens)` and predicting chunk time for serial / overlap / split; validate against a real 2-layer run. It also answers whether a 20 GiB card can seat two whole layers — the precondition for overlap.

## What does NOT transfer and why

- **TMA descriptors** — gated on `is_nvidia_hopper` + `FLA_USE_TMA` (`fla/utils.py:288-295`); RDNA3 has none, falls back to plain loads.
- **wgmma / FP8-FP4 MMA** — every `sm_90a`/`sm_100a`/`sm_120` path (`utils/arch.py:84-100`, `gluon/`, Marlin, b12x); gfx1100 has WMMA wave32 only, so quantized GEMMs must dequantize to bf16.
- **Clusters/DSMEM, NVLink, P2P** — no AMD equivalent; cross-GPU is a host bounce, so any design assuming cheap all-reduce must be re-costed.
- **flashinfer / sgl_kernel / vllm** — hard-disabled on ROCm (`backend.py:46-59`), making the less-tested Triton fallback our only path.
- **`cudaMemcpyBatchAsync`** — CUDA 13 driver API, conditional on a CUDA-specific alignment cliff they bisected (`offload_cache.py:17-22`); no HIP port.
- **Spec decode** — absent, MTP discarded; do not inherit the gap, both our targets ship one MTP layer.
- **TP as the scalability answer** — all-reduce with a PyNCCL fast path and no RCCL port; two cards, no P2P, so a dense 27B TP would live in the bounce.

## Claims I could not verify

- **"290B+ … at blistering interactive speeds"** (README): no tok/s figure in README or `docs/`; scripts exist, results are not in the repo.
- **"Semantic-Aware Caching / semantic anchor checkpoints"** (README): one anchor, keyed on a tokenizer token id at a position (`scheduler.py:376`, `cache.py:153`) — no semantic matching, no content addressing (`radix_cache.py:265-268`). Overstated.
- **"Elastic VRAM re-allocation … without restarts"**: plumbing exists (`offload_cache.py:686`, `scheduler.py:31`, executed only at an idle safe point), but reading a rebuild path is not running one.
- **"graph-compatible execution"**: true for decode (`engine.py:1227`), yet `install_amd.md` says CPU/hybrid MoE needs graphs off on ROCm 7.14.
- **GH200 gather 220→410 GB/s** (`docs/models.md:96`): not our hardware, not reproducible here.
- **arXiv 2608.16157**: not read; every number above comes from code in the checkout.
- **That any of it runs on gfx1100**: `install_amd.md` calls it experimental, gated on issue #541; I ran no GPU jobs.

## Source map

```
scheduler/scheduler.py:139-142, 214-268, 376-380, 870-875, 905
scheduler/prefill.py:31-42, 91-112, 151    scheduler/decode.py:20-40
scheduler/config.py:16-17                  engine/config.py:87
scheduler/cache.py:20-27, 153-170          kvcache/radix_cache.py:28-38, 265-268
kvcache/hybrid_radix_cache.py:43-48        kvcache/linear_state_pool.py:34-42
moe/offload_cache.py:17-22, 121-170, 686, 859-1050, 1102-1103
moe/benchbw.py:195, 267, 472, 561, 593, 653
engine/engine.py:57-60, 913-934, 1227-1229
kernel/backend.py:11-14, 46-59             kernel/utils.py:51-61, 91-94, 292-378
kernel/pynccl.py:29                        kernel/fla/utils.py:269, 288-295
kernel/fla/kda.py:36                       kernel/triton/attention.py:1051
models/qwen4_exp/ple.py:1-18, 120-160      models/qwen4_exp/weight.py:9, 92
models/qwen3_5_moe/gdn_kernels.py:7-45     utils/arch.py:84-100
docs/install_amd.md   docs/models.md:72-96, 146, 152   docs/performance/prefill.md
benchmarks/README.md  freetoken-kernel-cache/README.md  tests/ (175 files)
```
