# Strata — 011c9e600a93bbc9fd451c563501b5831c32bbca | MIT | C++20 / CUDA / HIP (+ optional SYCL, Metal)

## What it is

One C++20 engine that runs Qwen3.8-Flash-Next on consumer cards by treating VRAM, RAM and NVMe as one three-tier expert store, with the **host** deciding residency. CUDA is the reference path; HIP is the *same* `.cu` sources compiled with hipcc behind a header shim, and gfx1100 is one of two architectures the maintainers validate (`cmake/hip_backend.cmake:7-13`). Non-obvious decisions carry their own measured number in a comment — including the ones that failed.

## Architecture in one text diagram

```
host                                    GPU (per-layer hipGraph)
router l+1 on layer l's MoE input ─┐
 (RouterLookahead, worker thread)  │ predicts 10 experts
   ├→ warm(): pread/madvise ───────┼──→ [NVMe] O_DIRECT, io thread
   └→ ForesightSwap::predict ──────┼──→ [VRAM foresee slots]
                                   │
layer l: ids, weights ─────────────┴──→ residency table (host_res)
   ├─ hits ──────────────────────────→│ grouped expert kernel
   ├─ PCIe share ───────────────────→│ reads mapped host memory
   └─ misses ─→ CPU pool ───────────→│ + peer GPU
 between windows: adaptive tier swaps hottest misses in
```

Residency authority, in order: a **static profile** ranked by routing frequency fills VRAM slots at startup (`expert_cache.hpp:40-55`); the **host pool** re-decides per layer, per token, from that table plus a fitted PCIe cost model (`expert_source.hpp:552-555`); **ForesightSwap** keeps a small per-layer slot pool for just-missed experts (`foresight_swap.hpp:1-21`); the **adaptive tier** rewrites the table between windows from decayed routing counts (`expert_source.hpp:628-630`). Stated outright: "The host pool decides and the GPU follows the plan, so both always agree on who computes an expert" (`foresight_swap.hpp:7-8`).

## Techniques worth stealing

| technique | where (file:line) | what it buys on gfx1100 | effort | verdict |
|---|---|---|---|---|
| One hipGraph per layer | `session.hpp:127-138` | 1.585 ms vs 2.393 ms for the same 43-node block — **1.51x, pure launch overhead**. Decode is launch-bound, not bandwidth-bound (~0.028 ms/null launch) | S | **TRANSFERS** — hipGraph is native (`hip_compat/cuda_runtime.h:59-65`) |
| Router look-ahead, one layer ahead | `expert_source.hpp:296-362` | Layer l+1's router on layer l's MoE input on a worker thread → next 10 experts' bytes in flight during layer l | M | **TRANSFERS** — one BF16 matvec |
| Expert arena: anon, not mmap | `expert_source.hpp:1231-1245` | 34 GB experts + 26.8 GB table = 60.8 GB file-backed pages on 63 GB → **19 vs 42.8 GB/s** | M | **TRANSFERS** |
| PleReader: issue/collect, O_DIRECT | `ple_reader.hpp:1-16, 62-80` | 16 rows/token on 16 pages, known at the token → hidden in embedding+layer 0; 95 MB row cache serves **82%** | M | **TRANSFERS** |
| SSD keep-alive reads | `ple_reader.hpp:75-80`, `ple_reader.cpp:182-215` | Consumer NVMe idles after ~250 ms, then stalls 50-150 ms; one page per period fixes it | S | **TRANSFERS** |
| Foresight VRAM slots for misses | `foresight_swap.hpp:36-101` | A miss becomes a hit one window later; safety by window counter, not locks | M | **TRANSFERS** |
| Profile-ranked residency, no eviction | `expert_cache.hpp:45-55`; `expert_cache.cpp:647-658` | `h = 0.6447` held-out vs `0.4864` compulsory-miss. "full: no eviction, deliberately" | L | **TRANSFERS** |
| Scored prefetch admission | `expert_source.hpp:280-284, 1146-1161` | Stops speculation stealing drive bandwidth from demand; budget feeds back on used-share | M | **TRANSFERS** |
| Page-cache gate, 0.55/0.65 | `expert_source.hpp:994-1000` | Read-ahead off when RAM is short — our exact 60 GB / 4.4 GB/s case | S | **TRANSFERS** |
| Hyper-connection as one persistent kernel | `bench/results/2026-10-12-hc-persist-gfx1030/README.md` | Two grid barriers on an atomic counter, 80 blocks: **-4.4% to -6.6%** per window on gfx1030 | M | **TRANSFERS-WITH-WORK** — off by default, per-arch tuned |
| Programmatic dependent launch | `pdl.hpp:1-18, 50-51, 63` | Nothing: no-op on HIP (`kPdlPrefetch = false`) | — | **DOES-NOT-TRANSFER** |
| Two-card layer split, peer tier | `peer_experts.hpp:1-15, 59-60` | Runs `p2p = false` via mapped host memory, but our 14.2 GB/s bounce makes it a loss | L | **DOES-NOT-TRANSFER** here |

## Top 3 to port first

**1. hipGraph per layer. Effort S, pays on both models, today.** The engine's own note: "the engine is launch-bound, not bandwidth-bound" — every kernel in `gdn_layer` measured ~0.028 ms *including one over 48 elements*, and replaying the 43-node block as a graph gave 1.585 ms against 2.393 ms (`session.hpp:132-136`). Qwen3.8-27B at 16.35 GiB has a ~35 tok/s ceiling we will never approach if ~19 launches per block each cost 28 µs on the same ROCm stack. First step: capture the decode body as one `hipGraphExec` per layer with a stable `parts` pointer (`session.hpp:152-154` explains why the address must not move), then A/B against direct launches with a correctness diff before timing.

**2. Router look-ahead feeding both the file tier and a VRAM slot pool.** The only mechanism here that *hides* NVMe latency rather than measuring it, costing one BF16 matvec on a thread that already exists. It is what makes the hierarchy pay: the Flash-Next miss set is nearly independent across layers (`h_layer = 0.0456`, `expert_cache.hpp:16-17`), so nothing about layer l predicts layer l+1 — except layer l+1's own router. First step: port `RouterLookahead` alone and measure prediction hit-rate against actual misses.

**3. The PleReader pattern for the 28.8 GB ngram table.** The largest avoidable cost in our Flash-Next plan: 52B parameters we were treating as a 97 GiB BF16 problem is 28.8 GB of IQ4_NL rows read 16 at a time, never resident (`ngram.hpp:44-52`, `ple_reader.hpp:3`). The API shape is the trick: `issue()` when the token id is known, `collect()` before layer 1 (`ple_reader.hpp:7-12`), buying an embedding-plus-layer-0 window. First step: a standalone O_DIRECT reader with page dedupe and sorted submission, validated against `ref/ngram.py`.

The full tiering skeleton is highest-value overall but ranks fourth: L effort, and the 27B fits one card outright.

## What does NOT transfer and why

- **PDL / `griddepcontrol`** — sm_90+ only, compiled out on HIP (`pdl.hpp:17-18, 50-51`). AMD equivalent: hipGraph node dependencies. No launch-early primitive.
- **No TMA, wgmma, tcgen05 or mbarrier anywhere** — zero hits across every `.cu`/`.cuh`. Nothing to strip, but no answer for gfx1100's missing async bulk copy.
- **`mma.sync` / `cp.async`** — guarded per kernel, not hipified: `qsa_select.cu:174` reads `// AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)`.
- **`cudaLaunchKernelEx`** — used in `qsa_select.cu` and `sampler.cu`, **not** aliased in `hip_compat/`; both need their own path.
- **NVFP4 / FP8 MMA** — formats exist (`ngram.hpp:52`) but run as dequant-then-FMA. Correct for us: gfx1100 has no native FP8 MMA, so this is not a gap to close.
- **Cross-GPU streams** — HIP semantics carry, but at 14.2 GB/s streaming activations between cards is a loss.
- **`--pcie-frac` adaptation** — the 2x RX 7900 GRE report is the warning: a bridged card measured **3.1 GB/s** (`pcie_frac 0.08`) against **28.2 GB/s** on a clean x16 (`0.55`) (`bench/results/2026-10-06-community-2x-rx-7900-gre/README.md:9-12`). Their scheduler adapts to a bad link; ours has none.

## Claims I could not verify

- **The RX 6000 "engine package" is a README change only.** Commit `011c9e6` touches `README.md` alone, 3 insertions / 2 deletions — it routes installers, adds no code. gfx1030 support is older and files itself as **unvalidated** (`cmake/hip_backend.cmake:15`), because RDNA2 lacks `v_dot4_i32_iu8` and uses the signed `v_dot4_i32_i8` (`docs/AMD_HIP.md:677-681`).
- **The README speed table does not cover the headline model on our card.** `README.md:196-213` lists Q2_0/IQ2_XS/IQ3_XXS/IQ3_S/Coder on a 12 GB RTX 5070 and a 16 GB RX 9070 XT — no Flash-Next row, no gfx1100 row. The only Flash-Next-on-gfx1100 figure in the tree is a **community** run: RX 7900 XTX, IQ3_S, 262144 ctx, 8,243 experts / 15.6 GB cache, **70.5 tok/s decode**, 86.1% VRAM hit rate, 1,612 tok/s prefill at 38K (`bench/results/2026-10-06-community-rx-7900-xtx/README.md:83-84, :107`). A measurement, not a claim — and on 24 GB, so budget ~30% less cache at 19.98 GiB.
- **"1.6-1.8x sooner" from speculative decoding** (`README.md:353-355`) — no per-configuration ablation found; the XTX report's 63-80% draft acceptance is consistent but is not that claim.
- **HIP "completeness"** is a judgement: the same `.cu` sources and BLAS-switching CMake path build for HIP (`CMakeLists.txt:335, 377-421, 224-227`) and gfx1100 is validated, but an exhaustive audit of every `__HIPCC__` branch against a feature matrix was out of scope.

## Source map

- `cmake/hip_backend.cmake:7-15, 58-79`; `CMakeLists.txt:82, 224-227, 335-337, 377-421` — arch tiers, HIP BLAS discovery, shared kernel sources
- `include/strata/hip_compat/cuda_runtime.h:1-2, 59-65, 111, 138-143` — cuda*→hip* shim including graphs
- `include/strata/kernels/gfx_arch.hpp:23-26`; `pdl.hpp:1-18, 50-51, 63` — gfx1100 WMMA list; PDL no-op
- `src/kernels/cuda/qsa_select.cu:174` — per-kernel `__HIPCC__` branch
- `include/strata/core/expert_cache.hpp:11-17, 40-55`; `expert_cache.cpp:643-658` — hit-rate evidence, profile format, no eviction
- `include/strata/core/expert_source.hpp:280-284, 296-362, 367-390, 552-555, 628-630, 686-694, 771-808, 994-1000, 1146-1161, 1231-1245` — look-ahead, plan sink, adaptive tier, arena
- `include/strata/core/foresight_swap.hpp:1-21, 36-101`; `session.hpp:127-138, 152-154, 187-200` — foresee slots; graph per layer
- `include/strata/core/peer_experts.hpp:1-15, 59-60` — peer tier, p2p flag
- `include/strata/ngram/ple_reader.hpp:1-16, 27, 62-80`; `ple_reader.cpp:104-132, 182-215` — streaming, keep-alive
- `include/strata/kernels/ngram.hpp:44-52, 59` — table rows, per-format row bytes
- `include/strata/artifact/gguf_reader.hpp:6-19, 55` — GGUF v3 mmap reader, no ggml, type 42 = Q2_0
- `docs/AMD_HIP.md:1-18, 675-700` — backend scope, gfx1030 dp4a difference
- `bench/results/2026-10-06-community-rx-7900-xtx/README.md:83-84, 107` — Flash-Next on gfx1100
- `bench/results/2026-10-06-community-2x-rx-7900-gre/README.md:9-12, 16-18` — PCIe probe, 69.3 tok/s
- `bench/results/2026-10-12-hc-persist-gfx1030/README.md:6-20` — persistent HC kernel
- `README.md:191-213, 339-355` — speed tables, tier metaphor, spec-decode claim
