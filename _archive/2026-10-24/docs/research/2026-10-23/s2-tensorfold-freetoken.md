# S2 — TensorFold and FreeToken, read at the mechanism level

**Status:** research note for the PLAN loop. Not a decision record.
**Sources read (read-only, nothing modified):**
`development/_research/tensorfold` (7,002 files, Zig + Metal/HIP/CUDA kernels) and
`development/_research/freetoken` (836 files, Python + vendored ggml C++/CUDA).
**Scope:** the inference engine and its kernels in each tree. TensorFold's TypeScript app, its
`living/` test-time-training layer and its Metal-only Apple paths were skimmed only where they
explain a design choice that the HIP path inherits.
**Method:** every claim is traced to code, not to a README; `path:line` is relative to that repo's
root. Where a mechanism exists only in a comment, a test fixture or an external dependency, it says
so. `§5` of each repo is the honest-negative section.

Assumed box, used for all arithmetic: 2× RX 7900 XT (gfx1100, 19.98 GiB each, no peer DMA,
14.2 GiB/s cross-GPU), one dense 27B from GGUF IQ3_S, batch 1, 10.57 GiB read per token at a
measured 776 GB/s ⇒ **14.63 ms/token, 68.4 tok/s ceiling**. Every number below is measured against
that one line: `10.57 GiB ÷ 776 GB/s = 14.63 ms`.

---

## 0. Answer first

**Neither repo is an engine for our problem, but each contains one mechanism that is not available
anywhere else in either tree, and they are complementary.**

- **TensorFold's real asset for us is not its kernels — it is `backend/hip/engine/mtp.zig`: a
  working, graph-captured MTP chain drafter on the HIP path, with a *measured* rule for how deep to
  draft.** On a box where one forward pass costs one full 10.57 GiB sweep of HBM, spending 2.5
  tokens per sweep is the only lever that beats the memory wall by a factor rather than a percent
  (`§1.6`). Draft depth 3 at 70 % per-step acceptance is 2.53 tokens/forward; the weight bytes are
  paid once.
- **FreeToken's real asset is that it is the only ROCm-correct IQ3_S GEMV in either tree** — but it
  is not FreeToken's code: it is llama.cpp's `ggml-cuda` MMVQ/MMQ, vendored as source and compiled
  with `GGML_USE_HIP`, whose RDNA3 path is `__builtin_amdgcn_sudot4` (`freetoken/kernel/csrc/ggml/ggml-cuda/common.cuh:718-722`).
  Everything FreeToken itself differentiates on (expert slot cache, PCIe gather, CPU/GPU hybrid
  split) exists to serve MoE weights that do not fit in VRAM and is dead code for a dense 27B.
- **The two mechanisms do not overlap and do not conflict.** TensorFold cannot read our file (no
  k-quant or IQ path at all, `§1.5`); FreeToken throws away the MTP head it loads
  (`freetoken/models/qwen4_exp/weight.py:9`) and has no speculative decoding at all (`§2.2`).
  Taking TF's depth rule and FF's vec-dot ordering is the whole transfer.

---

## 1. TensorFold

### 1.1 What problem it actually solves, and the core trick

It is a **single-machine, single-model, many-shape family engine**: one Zig binary with Metal, HIP,
CUDA and XPU backends, one model family compiled in per path, tuned for a machine that holds the
whole model in VRAM or unified memory and serves a handful of streams. The core trick a kernel
engineer would recognise is the **descriptor-buffer round**: every per-layer and per-token
parameter of a decode round is written into one pinned-host/device plan buffer, and the launch
sequence depends only on a coarse `Shape`, so *one* captured HIP graph serves any set of streams,
any cache state, any layer count (`tensorfold/zig/src/families/qwen3_5/backend/hip/forward/plan.zig:1-2`,
buffer layout `:16-40`). The host loop walks layers issuing launches whose arguments never change
(`.../forward/forward.zig:52`).

The second half of the trick is **affine-only quantisation with the bias folded onto `sum(x)`**:
`y[m,n] = Σ_g ( scale[n,g]·Σ_k x·code + bias[n,g]·Σ_k x )`
(`tensorfold/zig/kernels/hip/quant/mlx.hpp:3`), so a group's contribution is one FMA after the dot
and one FMA after the row sum — never a per-element add (`.../quant/mlx_decoder.hpp:121-122`).

### 1.2 Scheduling and serving innovation

| Mechanism | What it actually does | Where |
|---|---|---|
| Lane rounds, not continuous batching | A round is 1..128 streams ("lanes"), each with its own KV window and exact width 1..16; all streams share one launch sequence, widths are exact (no padding to a bucket) | `.../hip/engine/hip_lanes.zig:174-196` |
| Graph per shape, LRU 64 | A launch sequence is captured the *second* time its shape is met and replayed after; 64 shapes kept, least-recently-used dropped | `tensorfold/zig/src/hip/graph_cache.zig:1-8`, capture in `.../hip/graph.zig` |
| Graphs split at collectives | A graph is cut at every collective and the pieces replayed in order on one stream — TP never needs cross-stream graph gymnastics | `tensorfold/zig/src/hip/graph_split.zig:26,43` |
| MTP chains batched across streams | One head forward per draft step for *all* chains in the round (grid z = jobs), one sync per round; a chain stops early on a confidence threshold | `.../hip/engine/mtp.zig:1`, `.../hip/engine/hip_lanes.zig:497-545` |
| Draft depth from measured cost | Verify windows of width 1..16 are *timed at load*; the head's per-step ms is timed too; a rule decides whether draft k pays for its row | `.../hip/engine/hip_costs.zig:15-25,72-100`; rule `tensorfold/zig/src/core/draft_depth.zig:55-95` |
| Chunked prefill with resumable snapshots | Prompt passes in 1024-row chunks, snapshots no closer than 256 tokens, cuts on page edges | `.../hip/engine/prefix.zig:5-13` |
| Token-granular prefix reuse | `prompt_radix` trie wired into the HIP engine's prompt path | `.../hip.zig:25`; grep confirms the radix module is the shared `engine_api.prompt_radix` |

The depth rule is the part worth reading twice. `Costs.measured` turns a table of
`verify[r]` ms into **marginal ms per row** (`tensorfold/zig/src/core/draft_depth.zig:11-47`), then:

```
keep(k, run)  ⇔  run^power  ≥  (tokens/ms) · row(k)        // verify draft k at all
more(k, run)  ⇔  run^power  ≥  (tokens/ms) · (row(k+1)+level) // draft one deeper
```

(`tensorfold/zig/src/core/draft_depth.zig:55-95`). `run` is the running confidence product, `row(k)`
the measured marginal cost of row k, `level` the measured cost of one more head step, and
`tokens/ms` is an EWMA of what recent rounds actually earned (`done`, `:88-95`). Note the design
comment: it follows the *model*, never a clock, "so every rank of a split model chooses alike"
(`:86-87`). Their own fixture data shows how non-linear this is: `verify[1]=9.85 ms`,
`verify[4]=16.92`, `verify[16]=32.08` (`:98`) — a 16-row window costs 3.26× a 1-row window on their
hardware, i.e. **verify is not free and the shape of that curve is the whole decision**. That curve
must be re-measured on our box (`§1.6`).

### 1.3 Memory-hierarchy innovation

- **Weights**: resident in VRAM, packed affine (2/3/4/5/6/8 bits, group 32/64/128 —
  `tensorfold/zig/src/hip/launches/affine.zig:84`, `.../affine.zig:89`). Never dequantised to a
  materialised tile: codes are unpacked into registers just before the dot
  (`.../quant/mlx_decoder.hpp:144-183`).
- **Activation x**: registers, one 32-element slice = 4×`u32x4` = 64 B per (row, chunk), re-permuted
  into code-pair order with `__builtin_amdgcn_perm` instead of a shared-memory round trip
  (`.../tiles/stream.hpp:22-24,126-135`).
- **Shared memory**: used for *row* staging and the 32×32 transpose, not for weights. The one-row
  tile transposes through padded `dots[4][32][33]`, the `+1` to kill bank conflicts
  (`.../tiles/dot2_tiles.hip:133`). The norm/residual tail stages a whole 8192-float row in LDS
  (`.../decode/decode.hip:104-140`, `kTailWidth = 8192`).
- **Host**: the *plan* buffer is pinned host + device copy (`forward/plan.zig:35-40`), so graph
  parameters are stable across replays. No weight streaming on the HIP path; `direct_io.zig`
  (O_DIRECT, 4096-aligned, `tensorfold/zig/src/core/direct_io.zig:1-30`) exists and is used by
  three CUDA-side loaders (`families/nemotron/cuda_source.zig:10`, `families/lfm2/weights.zig:11`,
  `families/drama3/cuda/stager.zig:4`) — not by the dense HIP path.
- **Prefix state**: page-granular KV snapshots with a 256-token minimum gap, 1024-row prompt chunks
  (`.../engine/prefix.zig:5-13`).

### 1.4 Kernel techniques worth stealing

1. **Launch amortisation by "sides": one launch, up to four products.** Products that share `x`, K,
   bit width and group are packed into one `StreamSides` struct (`4 × {words,scale,bias,out,n}` plus
   a 5-entry prefix sum, `.../quant/mlx_tiles.hpp:43-51`), and the grid's **x dimension is carved
   into contiguous block ranges per product** (`.../launches/affine.zig:324-342`); each block
   recovers which product it belongs to and binds its pointers
   (`.../tiles/stream.hpp:209-229`). This is how Q/K/V or gate/up cost *one* dispatch instead of
   two or three, with no extra stream and no second grid — the same currency as our "2–4 layers per
   launch", spent without needing a device-wide barrier.
2. **The decode stream tile** (the thing to beat): 128 threads = 4 wave32; grid
   `x = column blocks, y = m/R rows, z = plan items` (`.../launches/affine.zig:342`); with K ≥ 1024
   all 32 lanes of a wave split K (`lpc = 32`, `.../affine.zig:309-310`) and each lane owns a
   32-code chunk (`.../tiles/stream.hpp:63-71`); reduction is a `__shfl_xor` tree inside the lpc
   group, no LDS, no atomics (`:190-195`); column groups stagger their round-robin start by `col0>>3`
   to spread memory channels (`:158-165`); `R == 1` is double-buffered, `R > 1` hides latency with
   row dots instead (`:168-185`); the launcher picks the widest CB that still fills **3000 waves**
   (`.../launches/affine.zig:94,320`).
3. **Register-blocked rows, K in registers.** The 1–8-row tile stages `x` in LDS for *all* rows and
   keeps K in registers, so **K is read once for up to 8 rows** (`.../tiles/dot2_tiles.hip:23-47`),
   with exactly two fp32 partials per (group, row, column): the dot and `sum(x)` (`:27,92-109`).
   Accumulator budget is explicit: `CB·R ≤ 16` in the stream tile (`.../tiles/stream.hpp:18`),
   ≤16 code words per lane in the lanes tile (`.../dot2_tiles.hip:14-18`).
4. **Epilogue fusion.** gate|up + SiLU·up in the stream epilogue (`.../tiles/epilogue.hpp:74-85`);
   and `tf_tail` does residual + routed weighted sum + RMSNorm + scale + cast in **one launch, one
   block per row** (`.../decode/decode.hip:104-140`) — three launches and a global sync replaced by
   one, with the row kept in LDS.
5. **Unpack without arithmetic.** For bits ∈ {2,4,8} the two codes of a pair sit 16 bits apart, so
   one masked field yields both: `(w >> shift) & (((1<<BITS)-1) * 0x00010001u)`
   (`.../quant/mlx_decoder.hpp:168-171`); fp16 conversion by magic-number offsetting `|0x64006400`
   + `__hsub2` (`:176-177`); bf16 conversion by taking the top half of one float and the low half of
   the next (`:180-182`). Two codes, one instruction, no `cvt`.
6. **Matrix cores on gfx11 are bf16-only and wave32**: `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`
   gated to gfx11 (`.../common/wmma.hpp:22-27`, `.../common/arch.hpp:45-53`), used only above 16
   rows (`.../tuning/gfx1100.zon:12-17`). **No MFMA, no wave64, no `sdot4` anywhere in the tree** —
   the int8 dot path that IQ3_S wants is unused.

### 1.5 What it gets wrong, or assumes, that does not hold on our box

1. **It cannot read our file. At all.** There is no k-quant, no IQ path, no GGUF loader in the
   inference paths — `grep -ril "iq3|q4_k|k_quants|super_block"` over `*.zig/*.hpp/*.hip/*.cuh`
   returns **zero hits**, the only GGUF reader is `zig/src/xpu/loader.zig`, and the format enum the
   HIP backend registers is `{ mlx, dense }` (`tensorfold/zig/src/core/quant/types.zig:10`).
   The one bridge is an offline converter for GGUF **Q8_0** only
   (`tensorfold/tools/glm5_q8_0_gguf_to_mlx.py:1-8`).
2. **Their format choice is the opposite trade from ours.** The qualified width for their 27B is a
   uniform 4 bits, group 64 (`.../families/qwen27/widths.zig:12-13`). That is **4.5 bpw** (4 + one
   fp16 scale *and* one fp16 bias per 64 weights). Our IQ3_S is **3.4375 bpw** (110 B / 256 values,
   `freetoken/kernel/csrc/ggml/src/ggml-common.h:411-421`). Same 27e9 params:
   **15.19 GB vs 11.35 GB = 1.34× the bytes/token ⇒ a 51.1 tok/s ceiling instead of 68.4.**
   Re-quantising to *their* format costs us a third of our throughput; matching our byte count
   needs 3-bit affine g64 (3.5 bpw, 11.81 GB) and gives up IQ3_S's codebook quality. Either way the
   format is the product, not the kernel.
3. **The 27B family is Apple-only.** `families/qwen27/` imports Metal and drives the ANE
   (`.../families/qwen27/residency.zig:2-3`, `.../ane_policy.zig:13-27`); the HIP path serves
   `flashnext` and `qwen3_5` instead (`.../families/qwen3_5/hip.zig:1-27`). So "TensorFold has a
   Qwen 27B" is true only on a Mac.
4. **Its TP path has no answer to "no peer DMA".** The mailbox exchange — two ranks trading slots
   through **pinned host memory both GPUs map**, one kernel, `st_release_sys` after the data and
   `ld_acquire_sys` on the peer's flag, double-buffered by `seq & 1`
   (`tensorfold/zig/src/fabric/tp/mailbox.cu:6,311-338`) — is **CUDA-only** (`libcuda.so.1`,
   `cuMemHostRegister_v2`, `.../tp/mailbox.zig:44-60`) and `grep -rn mailbox zig/src/hip/` returns
   **nothing**. On AMD the TP path is RCCL or nothing (`.../hip/comm/rccl.zig:30`), and
   `hipDeviceEnablePeerAccess / CanAccessPeer / MemcpyPeer / hipIpc*` return **zero hits** in the
   whole tree: they never solved P2P on AMD, they delegated it.
5. **The tuning tables are hand-written thresholds, not measurements.** `gfx1100.zon` is integer
   `cost` ranks with shape predicates, and its own header admits "the numbers are the thresholds the
   launches kept before the registry; **a measured table replaces them**"
   (`tensorfold/zig/src/hip/tuning/gfx1100.zon:1-9`) — that replacement is not implemented. Only the
   *draft* cost table is genuinely measured at load.
6. **Two dead-end claims**: RDNA4's `wmma12` is named in caps with no kernel behind it
   (`.../launches/affine.zig:152` gates on wmma11), and `sdot4/sdot8` flags exist that no kernel
   consumes.
7. **`living/` and "sliding weights" are not inference.** They are test-time training against the
   served model (`tensorfold/packaging/omp/sliding-weights.ts:1`, `tensorfold/zig/src/living/*`).
   Irrelevant to tok/s; do not let the name suggest weight streaming.

### 1.6 The one idea most likely to move our tok/s

**MTP chain speculation with a measured marginal-row depth rule — i.e. buy tokens per weight
sweep, not bytes per token.**

*Mechanism.* The MTP head is a small transformer block that takes the last kept hidden row and
predicts the next token; it is run k times in a chain, each step one head forward over all chains in
the round (`.../hip/engine/mtp.zig:1,24` `max_depth = 3`). The drafted chain (plus the current
token) is fed to the main model as a **window of r rows in one forward pass**, and the accepted
prefix is kept. The decisive detail is that the main model's weight sweep is *row-independent*:
reading 10.57 GiB once serves r rows, so the marginal cost of row r is ALU and attention only. Which
is why the depth decision is expressed against measured marginal row cost, not against wall-clock
(`draft_depth.zig:55-95`). TF also cuts the draft head to the first 65536 ids
(`.../hip/engine/mtp.zig:37`, `draft_vocab`), so a draft step is a smaller read than the main model's
head.

*Expected effect on our box.*
- Cost side: 14.63 ms of memory per sweep. r = 4 rows adds ~4× the dot ALU: at 27e9 MAC/token,
  ≈1.05 ms/row fp32 at 25.8 TMAC/s, so ~4.2 ms — **but only if the unpack is amortised across rows**
  (`§1.4.3`: unpack once, dot r times). Unpack itself is the real ALU load: at ~4 int ops per weight
  it is ≈4.2 ms/token-row, i.e. **29 % of the memory time**, which is why row-blocking is mandatory
  and why the int8 dot path (`sdot4`) matters for r > 1.
- Gain side: at per-step acceptance p, tokens/forward = Σ_{i=0..k} p^i. p = 0.7, k = 3 ⇒ **2.53**;
  p = 0.8, k = 3 ⇒ 2.95. At p = 0.7 and r = 4 rows costing ~1.15× a single row's time, the net is
  **≈2.2× (68 → ~150 tok/s)**. At p = 0.5 it is 1.9 tokens/forward and still ~1.6×.
- Precondition that decides everything: **does our GGUF carry the MTP tensors?** FreeToken's loader
  drops them by name (`freetoken/models/qwen4_exp/weight.py:9`); TF's HIP path requires them
  (`.../hip/engine/engine.zig:199` allocates the head only when the checkpoint has it). If they are
  absent, the same machinery still works with a 0.5–1B draft model (reads ≈0.25–0.5 GB per draft
  step ≈ 3 % of a sweep for the whole chain) — but that is a design of ours, not of TF's.

*Honest caveat.* TF's own verify-cost curve is **not** flat: 16 rows cost 3.26× one row on their
hardware (`draft_depth.zig:98`). That hardware is compute-limited in a way ours is not, so our curve
should be far flatter — but "should be" is exactly the assumption that needs the 20-line measurement
in §5 before any kernel is written.

---

## 2. FreeToken

### 2.1 What problem it actually solves, and the core trick

A low-concurrency (`max_running_req = 4`, `freetoken/engine/config.py:28`) serving engine whose
differentiating bet is **model weights that do not fit in VRAM**: MoE expert banks live in pinned
host RAM and are streamed into a small GPU slot cache *inside a captured graph*, decided by a
single-CTA kernel that does hit/miss, LRU victim selection and in-place expert-id→slot rewriting in
one launch (`freetoken/moe/offload_kernels.py:291-467`). The second bet is orthogonal and portable:
a **token-granular radix prefix cache over a flat per-request page-table row**
(`freetoken/kvcache/radix_cache.py:126`, `freetoken/engine/engine.py:483-491`). Everything about
GGUF matmuls is not their work — it is llama.cpp's `ggml-cuda`, vendored and patched
(`freetoken/kernel/ggml.py:20-42`).

### 2.2 Scheduling and serving innovation

| Mechanism | What it actually does | Where |
|---|---|---|
| Strict prefill priority, one phase per forward | `_schedule_next_batch` is prefill-manager-or-decode-manager; **no mixed prefill+decode batch**, and the code admits a decode can starve behind a long chunked prefill | `freetoken/scheduler/scheduler.py:870-875`, `:580-583` |
| Chunked prefill with three reserved currencies | KV charged in whole pages, GDN state slots (1 live + 2 ping-pong), SWA pool tokens; check → lock → re-check | `freetoken/scheduler/prefill.py:43-112` |
| CPU/GPU overlap, not prefill/decode overlap | SGLang-style one-step lookahead: hold the previous batch's output while scheduling the next; two streams with explicit cross-waits both before and after | `freetoken/scheduler/scheduler.py:214-268` |
| KV: flat token indirection, `page_size = 1` | `int32[max_running_req+1, ceil(len/page)]` holding **raw token slot ids**, no block table; at page 1 the radix can share at single-token granularity | `freetoken/engine/engine.py:483-491`, `freetoken/engine/config.py:87` |
| Radix prefix cache | Key = raw token ids (not a hash), node payload = the canonical slot ids, ref-counted along the ancestor chain, **leaf-only cascading LRU** over a timestamp heap, and on commit the request's row is *re-pointed* at the tree's pages | `freetoken/kvcache/radix_cache.py:126-206,265-268`, `freetoken/scheduler/cache.py:302-330` |
| CUDA graphs | Capture decode only, bucket list `[1,2,4]+range(8,..,8)`, pad with a dummy request, all per-batch tensors in fixed buffers refreshed by `copy_from` | `freetoken/engine/graph.py:82-101,232-249` |
| Speculative decoding | **None.** No draft model, no EAGLE/medusa, no verify pass; a forward-compat `num_speculative_tokens=0` in a ring-sizing helper is the only trace, and an MTP head is *deliberately dropped at load* | `freetoken/models/qwen4_exp/weight.py:9`, `freetoken/kvcache/qsa_pool.py:49-51` |
| PD disaggregation | Does not exist (zero hits for `disagg`) | — |

### 2.3 Memory-hierarchy innovation

- **Two tiers only**: HBM, and pinned/registered host RAM. `mmap` appears at load time only; there
  is **no disk-to-GPU weight streaming during decode** (`freetoken/checkpoint/ftw.py:236-295`).
- **MoE slot cache**: `slot_for_id[L,E]`, `id_of_slot[slot]`, `usage[slot]` as a logical step;
  `id = layer*E + expert` makes "does this slot belong to this layer" a range check
  (`freetoken/moe/offload_cache.py:215-231`, `freetoken/moe/offload_kernels.py:271-273`). Eviction is
  `argmin(usage)` chosen **serially inside one CTA**, then the ids are rewritten in place so
  downstream GEMMs read slots (`offload_kernels.py:398-447`).
- **One fused copy kernel for every bank at once** — pointers and per-row byte counts passed as small
  device tensors built once at start-up — because the per-bank version was "6 banks × 36 layers =
  216 launches/decode step" (`freetoken/kernel/csrc/jit/fast_index_copy.cuh:501-509,539-560`).
- **Prefill streaming**: whole-layer double buffer on a second stream, with a D2D hit-gather and only
  misses crossing PCIe as one coalesced batch memcpy (`freetoken/moe/offload_cache.py:882-1035`).
- **Heterogeneous split** (the genuinely interesting one): the same layer runs on GPU (hits + ≤K
  fetched) and CPU (overflow, CPU GEMV) concurrently, with the split ratio computed from bandwidths
  **measured while the two contend**, `f = pcie_ov/(pcie_ov+cpu_ov)`, and `hybrid iff cpu_bw > 2 ×
  pcie_bw` (`freetoken/moe/benchbw.py:593-655`, `freetoken/moe/bench_profile.py:116-128`).
- **KV is not quantised**: the pool is model dtype, no `kv_cache_dtype` anywhere
  (`freetoken/kvcache/mha_pool.py:50-58`). At 64 layers × 8 KV heads × d128 fp16 that is
  **256 KiB/token = 8 GiB at 32k** — on a 19.98 GiB card this box is KV-limited before it is
  compute-limited.

### 2.4 Kernel techniques worth stealing

1. **IQ3_S stays integer until one float multiply.** Each 2-bit code plus its `qh` bit forms a
   9-bit index into `iq3s_grid[]` (pre-signed-packed 4×int8); signs are applied by
   `__vsub4(grid ^ signs, signs)`; the product accumulates by 4-lane dot-accumulate into an `int`;
   the per-32-value scale is an **integer** multiplier on the accumulator; only then one
   `d · act_d` float multiply (`freetoken/kernel/csrc/ggml/ggml-cuda/vecdotq.cuh:1211-1248`).
   No bf16 weight tile is ever materialised, no weight passes through LDS: **dequant is a table
   lookup, not an arithmetic chain**.
2. **This path really compiles and runs on gfx1100.** `ggml_cuda_dp4a` has a native RDNA3 branch —
   `__builtin_amdgcn_sudot4(true,a,true,b,c,false)` (`.../ggml-cuda/common.cuh:717-729`) — and the
   NVIDIA byte-SIMD ops IQ3_S needs are shimmed for HIP: `__vsub4` as 5 integer ops and
   `__vcmpne4` as 6 (`.../ggml-cuda/vendors/hip.h:279-312`). So the ordering transfers; the *cost* of
   the sign fix-up does not (see §2.5).
3. **One q8_1 activation quantisation shared by every segment of the same `x`.** A fused QKV or
   gate/up matrix is split into runs of equal ggml type; `x` is quantised **once** and threaded into
   every segment's vector kernel through a patched `src1_q8_1_pre` that skips the internal
   quantisation (`freetoken/layers/quantization/linear/gguf.py:22-45`,
   `freetoken/kernel/csrc/ggml/ggml-cuda/mmvq.cuh:11-14`). At batch 1 that removes one
   `quantize_row_q8_1` launch per extra segment.
4. **Fused gate/up + GLU epilogue** in the MMVQ/MMQ output stage
   (`freetoken/kernel/csrc/ggml/freetoken/ggml_ops.cu:311-356`).
5. **Tail padding as a property of the allocation.** MMQ tiles read up to 512 values past the last
   packed row; instead of hoping, `pad_rows()` appends whole zero rows at create time and a runtime
   `_check_mmq_tail` validates the extent before any MMQ launch
   (`freetoken/layers/quantization/ggml_types.py:47-56`, `freetoken/kernel/ggml.py:55-62`). This is
   the exact class of bug a hand-rolled quantised GEMM reimplementation ships.
6. **Dispatch by measured crossover, with per-arch tables**: on RDNA3 the MMVQ max-batch table caps
   **IQ3_S at 4 tokens** (`.../ggml-cuda/mmvq.cu:241-258`) — batch-1 decode takes the vector kernel
   by construction, and so does our verify window up to r = 4.
7. **Bandwidth measured under contention**, not in isolation (`freetoken/moe/benchbw.py:593-650`) —
   the methodology matters more than the result (see §5).

### 2.5 What it gets wrong, or assumes, that does not hold on our box

1. **Multi-GPU is NVIDIA-only in code**: `# TODO(ROCm): NCCL -> RCCL migration` at
   `freetoken/kernel/pynccl.py:30`, with `torch.distributed.all_reduce` as the AMD fallback
   (`freetoken/distributed/impl.py:25-31`). On ROCm this box gets the slowest possible collective and
   issues it **synchronously at the end of each row-parallel linear, with no overlap**
   (`freetoken/layers/linear.py:155,182`).
2. **The whole MoE apparatus is dead code for a dense model.** Slot cache, PCIe gather, hybrid
   split, expert banks: none of it is reachable from a dense 27B.
3. **The GGUF matmul is upstream code, HIP-targeted but not HIP-tuned by them.** Vendored
   `ggml-cuda` + `GGML_USE_HIP` (`freetoken/kernel/ggml.py:31-35`); the RDNA3/3.5 MMQ configs ship
   as data (`.../ggml-cuda/mmq-config-rdna3.cuh`, `mmq-config-rdna3-5.cuh`). Useful, verified, and
   *someone else's* kernel — read it, don't attribute a design to this repo.
4. **The byte-SIMD sign trick is not cheap on RDNA3.** `__vsub4` costs 5 ops and `__vcmpne4` 6 on HIP
   (`.../vendors/hip.h:279-312`) versus 1 each on NVIDIA. Per 4 weights that is ~11 extra ALU ops
   before the `sdot4`. The fix is to fold the sign into the table index: a 512-entry LUT
   `idx = code | (sign_bit << 9)` of pre-signed 4×int8 removes both shims entirely for 2 KiB of
   constant memory. *(This is my proposal, not their code; their source carries
   `// TODO: don't use lookup table for signs`, `vecdotq.cuh:1210`.)*
5. **Speculative decoding is absent and the MTP head is thrown away**
   (`freetoken/models/qwen4_exp/weight.py:9`) — the single cheapest multiplier on a bandwidth-bound
   box is deliberately not implemented here.
6. **`page_size = 1` costs 4 bytes per request per context token** in the page table
   (`freetoken/engine/engine.py:483-491`); negligible for 4 requests, and worth it, but it is a real
   design trade, not free.
7. **One documented mechanism is not in this checkout**: the default GPU slot-cache kernel delegates
   to `flashlib.kernels.slot_cache.lru_ensure`, an external pinned dependency
   (`freetoken/moe/offload_kernels.py:31-53`, `pyproject.toml:41`). The kernel that *is* here is the
   hybrid/capped-fetch variant.

### 2.6 The one idea most likely to move our tok/s

**Honest answer: nothing in FreeToken moves decode tok/s on our box, and it is worth saying why in
numbers rather than adjectives.**

Every byte FreeToken's own machinery saves is a byte of *expert* weights that do not fit; our model
is dense and fits in one card with ~9 GiB to spare. The decode rate is pinned by
`10.57 GiB / 776 GB/s = 14.63 ms`, and no scheduler, cache or copy engine touches that numerator or
denominator. Concretely:

- **TP=2 is arithmetic, not engineering, and the repo cannot build it on ROCm** (`pynccl.py:30`).
  Row-parallel weights halve per-card bytes to 5.29 GiB ⇒ 7.31 ms, but cost **2 all-reduces per
  layer = 128 collectives/token** of `[1, hidden]` (10 KiB at hidden 5120 bf16). The bytes are
  0.09 ms; the latency decides:

  | per-collective | comm ms/token | total ms | tok/s | speedup |
  |---|---|---|---|---|
  | 5 µs | 0.64 | 7.95 | 125.7 | 1.84× |
  | 10 µs | 1.28 | 8.59 | 116.4 | 1.70× |
  | 15 µs | 1.92 | 9.23 | 108.3 | 1.58× |
  | 25 µs | 3.20 | 10.51 | 95.1 | 1.39× |
  | 40 µs | 5.12 | 12.43 | 80.4 | 1.18× |
  | 60 µs | 7.68 | 14.99 | 66.7 | 0.98× |

  RCCL-on-PCIe small-message latency is in the 10–40 µs band, so **TP=2 through this repo's path is
  a 1.2–1.7× idea that decays to a loss at 60 µs**. It is only worth building together with a
  low-latency exchange, which is TensorFold's mailbox (`§3`).
- **The one transferable kernel ordering** (§2.4.1–2.4.2) buys a *few percent* at batch 1, not a
  factor: at 776 GB/s we are at 97 % of the card's 800 GB/s spec, so the memory pipeline is already
  saturated and the only thing left to fix is whether the unpack steals issue slots from loads.
  Where it does pay is **r > 1**: unpacking once per weight and dotting r rows is what makes a
  verify window cost ~1.1× a single row instead of 4× (this is the same conclusion TensorFold's
  register-blocked tiles reach from the other direction, `§1.4.3`).
- **What does transfer, and is worth the port, is not tok/s: it is TTFT.** The token-granular radix
  cache with the page-row re-point on commit (`freetoken/scheduler/cache.py:302-330`,
  `freetoken/kvcache/radix_cache.py:141-206,265-268`) turns a multi-turn turn into a re-prefill of
  the divergent suffix only, at the cost of one `int32` per request per context token. On a box
  whose prefill is compute-bound at batch 1, that is seconds per turn.

---

## 3. Cross-repo: the three things I would actually port, ranked

**P1 — Speculation with a measured depth rule (TensorFold, `§1.6`).** Expected **~1.6–2.2×** decode,
the only factor-level lever available without changing the quantisation. Port the *shape*, not the
code: (a) time `verify[r]` for r = 1..8 at load, on scratch caches, exactly as
`hip_costs.measure` does (`.../hip/engine/hip_costs.zig:72-100`); (b) time one draft step; (c) apply
`keep/more` against the measured marginal row (`draft_depth.zig:55-95`), following the *model* rather
than a clock so any future split stays consistent across ranks. Gate: does the GGUF have the MTP
tensors?

**P2 — The host-mapped mailbox for the residual/hidden-state crossing (TensorFold's CUDA idea,
`§1.5.4`).** TF's mechanism is exactly the one our topology needs: pinned host memory mapped into
both GPUs, slots indexed by `seq & 1`, producer stores the data then `st_release_sys` on the flag,
consumer spins `ld_acquire_sys`, all inside one kernel, one launch per exchange
(`.../tp/mailbox.cu:6,311-338`). It is written against `libcuda` (`.../tp/mailbox.zig:44-60`) and has
no HIP port — a HIP port is `hipHostRegister` + `__threadfence_system`/`ld_acquire_sys`-equivalent
(`s_waitcnt`/atomic) work, not a redesign. This is the difference between the 1.2× and the 1.8× row
of the TP table, and it composes with P1 — the two factors multiply because the collectives are paid
per *forward*, not per accepted token.

**P3 — IQ3_S integer-domain vec-dot with unpack-once/dot-r-rows (FreeToken's vendored kernel,
`§2.4`).** Expected **a few percent at r = 1, and it is the precondition for P1 being cheap at
r > 1**. Port the ordering (LUT grid → signed 4×int8 → `sdot4` → integer scale on the accumulator →
one float multiply, `vecdotq.cuh:1211-1248`) and drop the byte-SIMD sign shims in favour of a
512-entry pre-signed LUT (`§2.5.4`). On gfx1100 the dot is `__builtin_amdgcn_sudot4`
(`common.cuh:718-722`); nothing else in it is exotic.

**Composition, at p = 0.7, r = 4, 10 µs/exchange:** 14.63 ms sweep + ~1.15× row overhead + 1.28 ms
comm ≈ 18.1 ms per *forward* for 2.53 tokens ⇒ **~140 tok/s**, versus 68 today. That is the entire
achievable envelope; every remaining item in either repo is noise against it.

---

## 4. What I would not copy

- **TensorFold's affine format** (`4.5 bpw` at their qualified width, `§1.5.2`): +34 % bytes/token
  buys kernel simplicity we cannot afford at 776 GB/s. Also not `bits=3` as a byte-count match —
  3-bit affine is a quality regression against IQ3_S's codebook at the same bpw.
- **TensorFold's "living"/sliding-weights** feature: test-time training, not inference.
- **FreeToken's MoE machinery** in any form: slot cache, bank gather, hybrid CPU/GPU split, expert
  LRU. Dense model.
- **FreeToken's scheduler**: strict prefill priority with acknowledged decode starvation
  (`scheduler.py:580-583`) is a bug-shaped policy for interactive use, and its overlap is CPU-vs-GPU,
  which buys nothing at batch 1 where the GPU is the only worker.
- **FreeToken's `page_size = 1` page-table-per-token** unless we also build the radix cache that
  justifies it — the granularity is only worth its 4 B/token if something shares prefixes.
- **TensorFold's cost-rank tuning tables** as a pattern: they are unmeasured integers that admit so
  in their own header (`gfx1100.zon:9`). A `{shape → measured ms}` table with a staleness check (as
  `draft_depth.Costs` does) is the version worth having.

---

## 5. The 20-line measurement that decides P1, before any kernel exists

Neither repo can answer this for our box, and both would guess wrong (TF's own curve is compute-bound
in a way ours is not). On the existing 776 GB/s harness, with the real IQ3_S weights:

1. `verify[r]` for r = 1,2,4,8,16 — same 27B, same kernel, one variable. Report the **marginal** ms
   of row r, not the total. This is the number `draft_depth.keep` consumes.
2. `verify[4]` with and without **unpack-once/dot-4-rows** register blocking. If the second is not
   ~1.1× the first, the ALU is not amortised and P1's arithmetic collapses to TF's 1.7×-for-4-rows
   curve, which is barely worth it.
3. One collective: hidden-state all-reduce (10 KiB) round-trip latency, RCCL vs a two-kernel
   host-mapped exchange. This is the whole of P2's value; the table in `§2.6` says where the cliff is.
4. Concurrent-read bandwidth on both cards at once (the `benchbw.py` methodology, applied to GPUs
   instead of CPU/PCIe): two cards each pulling 776 GB/s independent streams is an assumption, not a
   fact, and TP=2's numerator depends on it entirely.

---

## Appendix — verification notes

Verified directly by reading the cited lines: TF's affine formula and unpack sequence
(`mlx.hpp:3`, `mlx_decoder.hpp:168-183`), the stream tile's reduction and stagger
(`stream.hpp:158-165,190-195`), the sides carve-up (`affine.zig:324-342`, `mlx_tiles.hpp:43-51`),
the tail fusion (`decode.hip:104-140`), graph cache LRU 64 (`graph_cache.zig:1-8`), RCCL-only HIP
collectives (`rccl.zig:30`), zero peer-access calls, zero k-quant/IQ hits, the `gfx1100.zon`
admission (`:9`), the MTP head and depth rule (`mtp.zig:1,24,41`, `hip_costs.zig:15-25,72-100`,
`draft_depth.zig:55-95`), and the mailbox protocol (`mailbox.cu:6,311-338`, `mailbox.zig:44-60`);
FF's IQ3_S block (`ggml-common.h:411-421`), the RDNA3 MMVQ cap (`mmvq.cu:241-258`), the vec-dot
(`vecdotq.cuh:1211-1248`), the HIP dp4a and `__vsub4` shims (`common.cuh:717-729`,
`vendors/hip.h:279-312`), the shared-q8_1 segment path (`linear/gguf.py:22-45`, `mmvq.cuh:11-14`),
tail padding (`ggml_types.py:47-56`), `page_size = 1` and the page table (`config.py:87`,
`engine.py:483-491`), the radix tree (`radix_cache.py:126-206`), the ROCm TP TODO (`pynccl.py:30`)
and the dropped MTP head (`models/qwen4_exp/weight.py:9`).

Read via a sub-agent and spot-checked here rather than line-by-line by me: TensorFold's GEMM/KP tile
internals (`gemm.hpp`, `gemm_kp.hpp`, `matrix_gemm.hpp`), the lanes-tile LDS fold
(`dot2_tiles.hip:92-122`), FreeToken's `mmq-load-tiles.cuh` and `offload_cache.py` copy pipeline.
Not read: TensorFold's Metal kernels, its TypeScript application, the XPU/CUDA families, and
FreeToken's DSA/DSV4/QSA sparse-attention variants and `fla/` GatedDeltaNet kernels — none are
reachable from a dense Qwen GGUF.

Arithmetic in this note was computed, not estimated in the head: 10.57 GiB / 776 GB/s = 14.63 ms;
IQ3_S 110 B / 256 values = 3.4375 bpw and our file implies 3.36 bpw over 27e9 params; affine
4 b/g64 = 4.5 bpw ⇒ 15.19 GB (1.34×) ⇒ 51.1 tok/s; affine 3 b/g64 = 3.5 bpw ⇒ 11.81 GB (1.04×);
TP=2 5.29 GiB/card ⇒ 7.31 ms; 128 collectives × latency; KV 64×8×128×2×2 B = 256 KiB/token = 8 GiB
at 32k; speculative `Σ p^i` at p ∈ {0.6,0.7,0.8}, k ∈ {2,3,4}.
