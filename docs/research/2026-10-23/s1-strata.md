# S1 — Strata, read at the mechanism level

**Status:** research note for the PLAN loop. Not a decision record.
**Source read:** `development/_research/strata` @ `011c9e6` (read-only; nothing modified).
**Scope of the read:** `include/strata/core/*.hpp`, `src/core/{verify,peer_experts,remote_experts,expert_source}.cpp`, `src/kernels/cuda/verify_kernels.cu`, `src/program/generate.cpp`, `include/strata/kernels/{pdl,s2_gemv,native_mmvq}.hpp`, `include/strata/program/batch_groups.hpp`, `docs/MULTI_GPU.md`.
**Method:** every claim below is traced to code, not to the README. Where the docs and the code disagree, the code wins and the delta is listed in §3.

---

## 0. Answer first

**The single most transferable idea is the shape of the stage boundary: one crossing per *token*, carrying a fixed-size residual, released by the producing GPU into mapped pinned host memory and acquired by the consuming GPU's *pre-launched* graph spinning on a flag word. The host never synchronises at the boundary.**

The consequence for our box, in one line: **on gfx1100 a PCIe boundary that the host mediates costs ~4.6 ms; the same boundary's bytes cost 6.7 µs.** Measured, same silicon, same model shape: `docs/MULTI_GPU.md:292-299` — 2× RX 7900 XTX, serial 53 tok/s → 70.1 tok/s with the flag. That 4.6 ms/token is 31 % of a token; 100 KiB of residual is 0.05 %. **The link is irrelevant. The sync is everything.**

What that buys and what it does not, since this is the part that matters for our build:

- It buys *parity*. 70.1 tok/s over two cards ≈ the ~70 tok/s the doc reports for one card. It does not make two cards faster than one for a model that already fits on one.
- What makes two cards beat one is §T3 (window-level speculation across the boundary), and on Strata's own rigs that measured +14 % (4 stages, `docs/MULTI_GPU.md:489-493`) to +16 % (`docs/MULTI_GPU.md:483-487`), with only **about a third of the guesses holding** (`docs/MULTI_GPU.md:491-492`).
- Our model's weights are **52× too large per token for the link** (794 GB/s would be needed to stream 10.57 GiB/token in 14.6 ms; we have 15.2 GB/s). Nothing that puts weights behind the link is fixable by tuning. That is the constraint that makes the residual-handoff architecture the *only* viable split, and it is why the idea transfers even though most of the repo does not.

---

## 1. What the repo actually is

Important for calibration, because the README reads like a general engine:

- **It is an MoE engine first.** The multi-GPU machinery (`PeerExperts`, `RemoteExperts`, `ForesightSwap`, `PcieModel`, expert tiers) exists to distribute *routed experts*, not dense weights. The layer split is real and separate, but it is the only part aimed at dense weights and it ships ~2 paragraphs of PR history per feature (e.g. `STRATA_STAGE_TRIM`, PR #639).
- **It has one target model family.** `include/strata/core/geometry.hpp:21-24` — defaults are Qwen3.8-Flash-Next (`n_embd=2560`, `n_layers=48`, `hc=4`, 512 SSM + QSA hybrid). Every kernel pins these at compile time and `static_assert`s against the geometry (`geometry.hpp:5-9`); another shape is *refused at load* (`geometry.hpp:8-9`). So this repo is not a general runtime — it is one model's engine with a config surface.
- **`sycl/`, `metal/`, `ref/` are ports**, largely mirrored copies with the same filenames. The multi-GPU logic I cite is CUDA/HIP-path only.
- **Tensor parallelism is explicitly rejected**, and I confirmed no all-reduce, no sharded projections, no TP path anywhere in `src/`. `docs/MULTI_GPU.md:9-10`: "This is pipeline (layer) parallelism, not tensor parallelism: a token crosses from one card to the next once per verify window (a few hundred KB through pinned RAM), not twice per layer."
- The unit of work is a **verify window** — a 1..8-token speculative batch (draft + verify), not a single token. `kVerifyMaxT = 8` (`include/strata/kernels/verify_kernels.hpp:21`). So "per token" in their numbers is really "per window", and the handoff amortises over up to 8 tokens. This is a free win we would also get from spec decode.

---

## 2. Technique inventory

Each entry: **name** — mechanism — constraint designed around — transfer verdict for 2×7900 XT, no P2P, dense 27B, batch-1.

### T1. Contiguous layer split with a residual-only handoff

**Where:** `include/strata/core/verify.hpp:166-194`; write side `src/core/verify.cpp:2154-2164`; read side `src/core/verify.cpp:1128-1137`; wiring `src/program/generate.cpp:8850-8851`, `8928-8929`.

**Mechanism.** Each `Verifier` owns a contiguous layer range `[lb_, le_)` of the *same* model. A stage that does not start at 0 skips embedding and instead copies its incoming state from `hand_in_`; a stage that does not end at `n_layers` skips the head and writes `hand_out_`. The payload is exactly three tensors per row, concatenated: the residual `R` (`hc × n_embd`), the last layer's pending write `bo` (`n_embd`), and the inject vector `inj` (`hc`). Size is a one-liner: `handoff_floats(g) = hc*n_embd + n_embd + hc` (`verify.hpp:194`).

The `bo`/`inj` part is the subtle bit and the reason the split is **bit-exact**: Strata's residual is a gated hyper-connection stream, so a layer's FFN write is *pending* and folded into the next layer's first read rather than added immediately. The split therefore cannot cut between "compute" and "add" — it hands the pending write across so the receiving stage folds it "exactly as the unsplit window does" (`verify.hpp:168-170`). Without that, a layer split of this architecture changes numerics at the seam every token.

**Constraint it is designed around.** No P2P, and a link ~2-3× slower than a good x16. `docs/MULTI_GPU.md:10-11`: "No NVLink or peer-to-peer access is needed; cards on x4 or x1 slots work." Confirmed in code: the buffers are `cudaHostAlloc(..., cudaHostAllocMapped | cudaHostAllocPortable)` (`generate.cpp:8896`) — pinned *host* memory both GPUs can see, so every byte is written to host RAM by the producer and read back out by the consumer. It is host-bounce by construction, which is exactly our topology.

**Transfer: yes, and it is the load-bearing idea.** Two caveats. (a) Our model has no pending-write gating contract unless we adopt one; if our residual is a plain pre-norm add, the handoff degenerates to `n_embd` floats and is *easier*. (b) The `hc=4` multiplicative factor is real: at `n_embd=5120` a plain residual handoff is 20 KiB/token, an `hc=4` one is 100 KiB/token. Both are noise against 14.6 ms — see §4.

### T2. GPU-published handoff: release/acquire through mapped host memory

**Where:** `verify.hpp:182-189` (API + the exact ordering contract); producer kernel `src/kernels/cuda/verify_kernels.cu:1204-1224`; consumer wait `verify_kernels.cu:1190-1202`; consumer call site `src/core/verify.cpp:1131-1137`; gating `src/core/verify.cpp:2678-2681`; opt-in wiring `src/program/generate.cpp:8903-8935`.

**Mechanism.** This is the actual trick, and it is a proper release/acquire, not a `cudaStreamSynchronize`:

1. The producing stage's graph ends not with a copy but with `handoff_publish_kernel` (`verify_kernels.cu:1204`). Each thread writes its slice into the mapped buffer with **`volatile` stores** (`:1208-1210`), then `__threadfence_system()` (`:1211`) — the store fence that makes the payload visible *to the host and the other GPU*, not just to the device.
2. A grid-wide `atomicAdd` on a mapped counter (`:1214`); the block that sees `gridDim.x - 1` resets the counter, fences again, and raises the flag word (`:1215-1221`). So the flag is raised by the GPU *after* all blocks' bytes are out, in one kernel, with no host involvement.
3. The consuming stage's graph was **already launched**; its first node is `wait_handoff_kernel` (`verify_kernels.cu:1190`), a one-thread spin on the mapped flag with `__threadfence_system()` on exit. Then three `copy_from_mapped` calls read `R`, `bo`, `inj` (`verify.cpp:1135-1137`).

The whole thing is conditional on the two stages being on *different* devices and sharing the same flag pointer (`verify.cpp:2678-2681`), i.e. it degrades correctly to the serial path when a split is faked onto one card.

**Constraint.** Host-sync latency on gfx1100/ROCm. The doc is honest about why this exists at all: "It takes one stream sync and one graph launch per window off the critical path" (`docs/MULTI_GPU.md:289-290`). And it is optional — `STRATA_SPLIT_OVERLAP`, default 0 (`generate.cpp:8906-8911`).

**Two details worth stealing.**
- **The wait is bounded.** `wait_handoff_kernel` takes a `timeout_ns`, and on expiry writes `err = 1` into a mapped word and *returns* rather than spinning forever (`verify_kernels.cu:1190-1202`). The window's remaining kernels still run on stale data so the graph completes and no stream is left hung, and the host then fails that window with a message naming the env var (`verify.cpp:2669-2676`, default 30000 ms, `docs/MULTI_GPU.md:305`). An unbounded GPU spin on a cross-device flag is a deadlock generator; this is the correct pattern and almost nobody does it.
- **A deliberate test hook for a lost publish:** `STRATA_TEST_HANDOFF_DROP=N` withholds the flag in window N (`verify_kernels.cu:1218`), so the timeout path is exercised in CI rather than discovered in production.

**Transfer: yes, verbatim.** This is the one piece I would port almost line for line. It is also *cheap*: the publish kernel is `min(64, ceil(n/256))` blocks × 256 threads (`verify_kernels.cu:1249`) — for 100 KiB that is 50 blocks, one launch.

**Honest limitation:** this is not a *free* overlap — the consumer spins, burning a CU. On NVIDIA they measured the second card at 94 % busy spinning with **no throughput gain** (`docs/MULTI_GPU.md:302-303`). It is a gfx1100-specific win, because RDNA3's launch+sync path is expensive in a way Ada's is not.

### T3. Window-level speculation across the boundary (the only path past one card)

**Where:** `include/strata/core/verify.hpp:263-308` (`pl_launch`/`service`/`done`/`prestage`); description `src/program/generate.cpp:14067-14074`; acceptance gate `generate.cpp:12928-12931`; snapshot machinery `generate.cpp:9110-9135`; speculative launch + commit `generate.cpp:14517-14544`; rollback `generate.cpp:14643`; VRAM cost `generate.cpp:4464-4472`.

**Mechanism.** `--pipeline-windows 2` keeps **two verifiers per stage** (one per window parity) each with its **own handoff buffer**, so stage 0 can run window K+1 while stage 1 still verifies window K (`include/strata/core/verify.hpp:263-270`, `generate.cpp:8860-8872`, `8983-8994`). Window K+1 is a *guess*: that window K is accepted whole, and that its bonus token is the drafter's.

- **Gate:** the guess is only launched if `p_on >= theta`, where `p_on` is the product of the drafter's per-token probabilities that K is accepted, and **`theta` defaults to 0.2** (`generate.cpp:12928-12931`). So it is not "always speculate" — it's a cheap EV calculation.
- **Pre-staging:** `prestage()` (`verify.hpp:295-298`, call at `generate.cpp:14484-14488`) computes positions and the PLE (n-gram embedding) rows for B *from tokens as they will be once A commits whole*, and prefetches their pages — so B's launch behind A is "only the graph launch".
- **Rollback:** because 48 of the layers are GatedDeltaNet, stage 0 carries **recurrent state**, which a speculative window mutates irreversibly. So before each speculative commit the stage copies its GDN state to a per-parity snapshot **on a side stream, ordered by events, never stalling the compute stream** (`generate.cpp:9110-9122`, `snap_take` at `14117-14127`). On a wrong guess the state comes back from the snapshot and the commit is **replayed with the real accepted count** (`generate.cpp:14070-14072`) — no recompute of the failed window; it is simply wasted.
- **Cost:** `kPipeWindowMib = 160` per card (second verifier + its graphs) and `kPipeSnapMib = 96` (two copies of ~15 GDN layers at ~3.27 MB each) — `generate.cpp:4465-4472`, formula `gdn_snapshot_bytes` at `4467-4470`.

**Constraint.** A boundary that cannot be removed, only filled. Filling it requires *work that is probably right* — i.e. it requires a drafter.

**Transfer: yes, and this is the only thing on the table that can make our 2 cards faster than our 1 card.** The arithmetic for us: two stages of 32 layers, each card owns ~5.3 GiB of the 10.57 GiB read, so a stage costs ~7.3 ms at 776 GB/s; a perfect 2-stage pipeline would emit a token every 7.3 ms = **137 tok/s, i.e. 2.0×**. That is the ceiling. Strata's measured realisation was +14 % to +32 % (`docs/MULTI_GPU.md:489-493`, `292-299`) because only ~1/3 of guesses held and a guessed window still costs ~15 ms against ~43 ms fresh.

**Costs we would pay, concretely:**
1. **A drafter is mandatory.** `--pipeline-windows` refuses to enable without `--mtp` and `--spec >= 2` and a `draft_vocab.bin` (`generate.cpp:4204-4219`, gate list `4224-4238`). Our IQ3_S 27B has no MTP head in the prompt's description, so we would need a draft model or a prompt-lookup drafter (they have `--suffix-draft` for the copy-heavy case, `generate.cpp:14732`, but the pipeline gate still wants MTP).
2. **Recurrent-state snapshot + rollback per front stage.** For us: 48 GDN layers split ~24/24, ~3-5 MB/layer ⇒ ~150-250 MB per card for the two parities. Fine in VRAM, but this is the hardest code in the repo — `snap_take` + event ordering + parity buffers + undo-commit-replay is where all the bugs live (they document a HIP clobbering bug that produced wrong tokens after a *correct* guess, `docs/MULTI_GPU.md:494-496`).
3. **Correctness discipline they needed and we would too:** `STRATA_PIPELINE_FORCE_MISS=1` forces every guess wrong to prove the rollback path is bit-identical (`docs/MULTI_GPU.md:492-493`); `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0` is the configuration in which they can claim bit-exactness against the serial loop.

### T4. Batch-group pipelining (real parallelism, no speculation — needs batch ≥ 2)

**Where:** `include/strata/program/batch_groups.hpp:1-48`; `include/strata/core/verify.hpp:228-243`; `src/core/verify.cpp:3790-3812`.

**Mechanism.** Instead of guessing, give each stage *different slots*. A batch window over slot group `g` is launched on one stage **with its commit right behind it on the same stream** (`verify.cpp:3797-3799`) — legal because a batch window keeps every row, so the commit needs no host decision. One host thread then serves the doorbell rings of every stage that has a window in flight, non-blocking (`batch_poll`, `verify.hpp:234-243`). Stage k runs group g while stage k+1 runs group g-1. Group count: `auto_groups` picks the largest divisor of the slot count that is ≤ number of stages (`batch_groups.hpp:41-48`), and since 0.1.41 a 2-stage split with `--batch >= 2` defaults to `auto` (`batch_groups.hpp:4-5`). Handoff rows are indexed by slot so groups never share a row (`generate.cpp:8888-8891`).

**Constraint.** The *serial* dependency a single conversation imposes. Batch concurrency removes it, so no speculation is needed.

**Transfer: yes, but it is not our case.** Our decode is batch-1-dominated. This is the design I would build if our target were 2-4 concurrent conversations: it is strictly more robust than T3 (no guesses, no rollback, no wasted work) and the doc even says the pipelined path beats serial MTP windows by 25-30 % on 2×A4000 / 2×3090 (`batch_groups.hpp:7-9`). Note it requires ≥ 2 slots *and* a layer split to engage at all (`auto_groups` returns 1 when `later_stages <= 0`).

### T5. Peer expert tier (MoE only) — the honest negative result

**Where:** `include/strata/core/peer_experts.hpp:1-15`; `src/core/peer_experts.cpp:124-200`; zero-copy variant `include/strata/core/remote_experts.hpp:84-86`.

**Mechanism.** Device 0 keeps dense weights, KV, draft layer, adaptive tier and verify graph. A second card holds a *second expert tier*. Each layer, the pool thread publishes which experts device 0 will compute; the peer computes its resident ones **with the same kernels and the same activations, so an expert is bit-identical on either card** (`peer_experts.hpp:11-12`), and the rows come back through the same mapped host rows. `RemoteExperts` adds a zero-copy variant: the helper GPU reads its input and writes its result **directly in pinned host memory** (`z_x_`, `z_out_`, `remote_experts.hpp:84-86` — "zero-copy: no input copy per layer"), so per-layer traffic collapses to the helper pulling its rows over PCIe inside the kernel.

**Constraint.** MoE expert capacity per card.

**Transfer: no.** Our model is dense; there are no experts to place. The retrievable idea is only the *pattern*: a second GPU as a compute tier over host-mapped buffers with no staging copies. Note the repo's own negative result — a 2080 Ti helper made decoding *slower*, and with a split it "measured slower again, as it did without a split: its per-layer round trip costs more than the CPU pool needs for those experts" (`docs/MULTI_GPU.md:390-392`). Per-layer cross-device work at batch 1 is a loss. That is evidence against any per-layer split for us.

### T6. P2P availability self-test (verify, don't ask the driver)

**Where:** `src/core/peer_experts.cpp:47-84` (`p2p_copies_data`), `140-159` (use + fallback).

**Mechanism.** `cudaDeviceCanAccessPeer` said yes, `cudaDeviceEnablePeerAccess` succeeded — and on a pair behind a host bridge `cudaMemcpyPeerAsync` "returned success having moved nothing: the peer's input rows were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN" (`peer_experts.cpp:47-50`). So they test: 16 K floats of a non-trivial pattern, copy each way, `memcmp` back; if it does not match they *disable* peer access and take the host route, printing one line (`:155-159`).

**Constraint.** Host-bridged and bifurcated PCIe topologies that lie about P2P.

**Transfer: yes, as a 40-line guard.** Our box reports no P2P and the measured cross-GPU copy equals host-bounce (14.2 GiB/s), which is consistent — but this is exactly the class of thing that silently corrupts output rather than failing loudly. Worth writing on day one, before any of the split code.

### T7. PCIe mode as a first-class scheduling choice (0/1/2), and the driver-lock lesson

**Where:** `include/strata/core/verify.hpp:333-337`; `src/program/generate.cpp:9104-9110`; per-layer decision in `src/core/verify.cpp:2072-2086`.

**Mechanism.** Three ways for a PCIe-resident blob to reach the GPU: **0** copy engine DMA into a staging buffer (best when the CPU is compute-bound), **1** the grouped kernel reads the mapped host arena *directly* (no staging at all), **2** a copy kernel stages it *inside* the graph (no host API call at all). Runnable per layer. The reason mode 2 became the default `auto` is the interesting part:

> "DMA has the host call `cudaMemcpyAsync` + `cudaLaunchHostFunc` inside a verify window while the GPU spins on the flag they raise; issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good. The copy kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S." (`generate.cpp:9106-9110`)

**Constraint.** The host thread is *inside* the GPU's dependency loop (it serves per-layer doorbells), so any host CUDA call it makes can deadlock against a driver lock the GPU's progress needs.

**Transfer: yes — the invariant, not the modes.** The invariant is: **while the GPU is spinning on a flag that only the host can raise, the host must not make CUDA calls.** Our design has the same shape (host serves the plan, GPU spins on the plan flag, `verify.cpp:2077-2086`). This is the single most expensive-to-debug failure mode in the whole design and it is a one-line rule.

### T8. Measure the link at startup and plan from it

**Where:** `src/program/generate.cpp:1988-2030` (`probe_pcie_h2d_gbps`), `3526-3540` (use).

**Mechanism.** 256 MiB pinned buffer, 4 timed bursts back to back with events bracketing each, one warmup copy, best-of; then `pcie_frac_for_gbps(bw, base)` maps the measured link to a default share of experts to stream. Their own note on gfx1100: ~28.4-28.5 GB/s probed (`generate.cpp:1962-1964`).

**Transfer: yes, cheaply, and we already have the number** (14.2 GiB/s cross-GPU). But note what it is *for*: sizing a bandwidth share. For us the relevant prediction is different — it is a **latency** measurement we need, not a bandwidth one. I would add a round-trip-flag latency probe (publish/observe 10 000 times through mapped memory) because §4 says latency is 700× more expensive per boundary than bytes, and we have no number for it on our box.

### T9. Runtime cost model for placement (PcieModel) + the overlap gate

**Where:** `include/strata/core/expert_source.hpp:398-470` (`PcieModel`, `pick` at `:467`); use `verify.hpp:344-353`; `AdaptGate` `verify.hpp:282-289`.

**Mechanism.** A recursive-least-squares fit with exponential forgetting (λ = 0.995, "~200 layers' memory: a few windows", `expert_source.hpp:411-413`) of two affine models: CPU-pool time as `a + b·n_cpu + c·n_pcie`, GPU time as `d + e·n_pcie`. Each window feeds real (count, time) samples; `pick()` then chooses the per-layer split. GPU-side timestamps come from spin kernels that stamp the plan's arrival and the wait's exit on the GPU clock (`wait_flag_ge_stamped`, `verify_kernels.cu:1227-1233`), so the model sees GPU time, not host-proxy time.

**Constraint.** The CPU/GPU/PCIe split point is unknowable statically and drifts with context, quant and cache state.

**Transfer: the *method*, not the model.** Fitting a cost model online from GPU-clock stamps and then *sizing the split* from it is exactly the right way to answer "how many layers on card 0" for our box — and the repo has a cautionary tale: on 2×RTX 3090 with a x16+x4 pair, link-miss scaling moved `auto` from K=27 to K=31 and *lost* 5-15 tok/s (`docs/MULTI_GPU.md:140-142`). A cost model that is 5 % wrong about a 4× link asymmetry costs more than it gains. Our two cards are symmetric, which makes the placement question much easier than theirs — equal split, decided offline, no adaptation needed.

### T10. Memory management

Four separate things, only one of which transfers.

**(a) Stage weight trimming — `STRATA_STAGE_TRIM`.** Each card keeps a *full* copy of the dense weights by default (~3.4 GB on their Coder) because that was simpler; the opt-in loads only the stage's own layers, and the freed VRAM goes to that card's expert cache. 2× MI50: 8 819 → 10 626 experts resident, decode 39.2 → 41.7 tok/s (`docs/MULTI_GPU.md:205-211`, `generate.cpp:3750-3755`). **Transfer: yes, mandatory for us.** If we split, each card must hold only its half — 5.3 GiB, not 10.57 GiB. This is a 5.3 GiB/VRAM decision that costs nothing but bookkeeping, and getting it wrong halves our KV budget.

**(b) Pinned host budget as a first-class resource — `pin_budget.hpp:1-20`.** The expert arena registers page-locked memory slice by slice until the driver refuses, which leaves *nothing* for the mapped buffers allocated afterwards, and the load then dies with "mapped staging allocation failed" on a box with 64 GB free (`pin_budget.hpp:1-8`). Fix: take two reserves (a soft one released after the arena is placed, a hard one held until the engine is up) *before* the arena pins anything, and shrink/skip rather than fail. **Transfer: yes, and it is directly relevant** — our handoff buffers and doorbell flags are exactly this class of allocation (mapped, portable, allocated late). On Linux the reserve is off by default because the NVIDIA driver pins past `RLIMIT_MEMLOCK` (`pin_budget.hpp:17-18`); **on ROCm/Linux I would not assume that** and would test `RLIMIT_MEMLOCK` explicitly.

**(c) Virtual-memory K/V growth — `vmm.hpp:1-5`.** An address range is reserved once and physical 2 MiB chunks are mapped/unmapped as the context grows, so every captured graph and stored pointer stays valid. **Transfer: yes if we want elastic context with captured graphs.** `cudaGraph` capture plus reallocation is otherwise incompatible; this is the standard workaround and it is 60 lines (`vmm.hpp:54-74`). Note: "Not under HIP" (`vmm.hpp:5`) — so on ROCm we would need `hipMemCreate`/`hipMemMap` equivalents and they may not behave.

**(d) Foresight swap slots — `foresight_swap.hpp:1-21`.** A few spare VRAM slots per layer, refilled on their own copy stream, usable by the plan only once the copy has landed; a slot is reused only after `depth` (default 4) windows that may have read it have finished; fed by `note_miss` and by a *router look-ahead* that predicts the next layer's experts from the previous layer's MoE input (`foresight_swap.hpp:4-17`, `78-84`). **Transfer: no** — it is per-layer MoE prefetch, and §T5's negative result applies. The *pattern* (spare slots + depth-based safe reuse + separate copy stream + "a copy that has not landed is simply not used") is generic and worth remembering for KV prefetch.

**What is streamed from host/NVMe:** only *experts*, as whole (layer, expert) blobs, through a file tier with `O_DIRECT`/`FILE_FLAG_NO_BUFFERING` and merged batched reads (`expert_source.hpp:855-914`). **Dense weights are never streamed** — they are fully resident per card. For our dense model there is nothing to stream and no prefetch policy to design. That closes item 3 of the brief: **the transferable memory technique is residency discipline, not streaming.**

### T11. Kernel-level tricks

| Trick | Where | What it does differently | Transfer |
|---|---|---|---|
| `handoff_publish` fused scatter + release | `verify_kernels.cu:1204-1224` | One kernel = gather 3 tensors from 3 sources into the mapped buffer + grid-wide completion count + release the flag. Replaces 3 copies + a host callback with 1 launch. | **Yes** — this is T2's engine. |
| Bounded spin with an error word | `verify_kernels.cu:1190-1202` | Spin on a mapped flag with a GPU-clock timeout, write `err=1` to mapped memory, return so the graph completes. | **Yes** — unchanged. |
| Wait folded into the next kernel | `src/kernels/cuda/elementwise.cu:244`; `verify_kernels.hpp:71-78` | The `wait_flag_ge` launch is fused into the following elementwise kernel ("143-amdk: the wait_flag_ge launch folded in"), so the wait costs no launch of its own. | **Yes**, trivially. |
| Device-planned layers (E-6) | `verify_kernels.cu:1262-…` (`resident_plan_kernel`); gate `verify.cpp:2054-2086` | The **GPU** decides, per layer, which experts it holds and which go to the CPU/PCIe, by reading a device copy of the residency table — so the layer needs **no host doorbell at all**. `wait_flag_ge_or(m_flag_, ring, skip_)` skips the wait for layers that do not need the host. | **Yes — conceptually the strongest kernel idea here.** It removes the host from the inner loop instead of making the host faster. |
| PDL (programmatic dependent launch) | `include/strata/kernels/pdl.hpp:1-18`, `62-96` | `griddepcontrol.wait/launch_dependents` on sm_90+ lets kernel N+1 start while N drains; only constant-weight loads happen before the wait, so numerics are unchanged. **`kPdlPrefetch = false` on HIP, `pdl_supported()` always false** (`pdl.hpp:30-32`, `51`). | **No.** CUDA sm_90+ only; no HIP equivalent. Do not budget for it. |
| GDN chunked recurrence / QSA kernels | `include/strata/kernels/gdn.hpp`, `native_gdn.hpp`, `qsa_*.hpp`, `src/kernels/gdn_chunk_parity.cpp` | Associative-scan formulation of the linear-attention recurrence so a prompt chunk is a few matmuls instead of a token loop; parity-tested bit-exactly against a reference oracle (`gdn_chunk_parity.cpp:201-212`). | **Yes, but out of scope here** — that is the single-card kernel story, not the interconnect story. Flagging it because our model is 48/64 GDN and this is where its prefill time will go. |

### T12. Prefill chunk pipeline with a calibrated b/a cost model

**Where:** `include/strata/prefill/prefill.hpp:92-123`; `docs/MULTI_GPU.md:311-344`.

**Mechanism.** A prompt read across stages is `sum_i (b + a·C_i) + (S-1)·max_i(b + a·C_i)` for chunks `C_i` over `S` stages, where `b` is the fixed per-chunk pass over each stage's streamed experts and `a` the per-token part. `pipeline_chunk()` picks the chunk count minimising that, given a b/a ratio **measured once by `--calibrate` and written to the config** — deliberately not learned online, because "that made a prompt's chunks, and so its output bits, depend on how busy the PC happened to be" (`docs/MULTI_GPU.md:325-327`). b/a varies an order of magnitude across rigs (700 on 4×GP100, ~5000-6000 on 2×3090 Ti) and a wrong one costs 19-36 % (`:324-326`).

**Transfer: partially.** The 4-stage chunk pipeline is prefill-only (decode has one token per window, nothing to chunk). But two ideas carry over: (a) **chunk unevenly on purpose** — equal chunks are only optimal when b/a = 0, and their table shows `--prefill auto` losing 22-36 % to a bad fixed 1024; (b) **calibrate the machine constant offline, never learn it from traffic**, so output bits stay reproducible. That second one is a policy we should adopt wholesale.

---

## 3. Where the docs and the code disagree (read this before trusting the README)

1. **Neither overlap feature is on by default on our silicon.** `STRATA_SPLIT_OVERLAP` is env-gated, default 0 (`generate.cpp:8906-8911`) and the doc says "opt-in" (`docs/MULTI_GPU.md:282`). `--pipeline-windows` auto-enables only on HIP **and** `gfx_arch_is_gfx12_wmma` (`generate.cpp:1941-1946`, `4216-4219`) — i.e. RDNA4 only, *not* gfx1100. So on a 7900 XT/XTX pair, **both of the interesting features are things the author measured and then chose not to enable by default.** The README's tone ("0.2.2", "stunning") is not the shipped default.
2. **`--pipeline-windows` requires the MTP drafter.** "it needs the MTP drafter (--mtp, --spec)" (`generate.cpp:4232`). The doc frames it as a scheduling choice; it is actually a *speculative decoding* feature that happens to be cross-device. Without a draft head there is no pipeline.
3. **The 2× gfx1100 result is a single config, 8 runs, one machine, and the "one card" column is `~70 tok/s`** (`docs/MULTI_GPU.md:292-299`). The headline is "+32 %", but the honest reading is "the split stops being a 24 % loss". I am using it as a measurement of *boundary cost*, which is what it really is.
4. **The 4× 7900 XT pipeline numbers (59.4 → 67.6) are on an MoE model with "every expert in VRAM"** (`docs/MULTI_GPU.md:489-493`). Part of that gain is each card holding more experts, not the window pipeline. Do not transfer +14 % to a dense model as a prediction.
5. **`docs/MULTI_GPU.md:39`** says a layer split makes prompts "about 20 % faster" — that is prefill, and it says nothing about decode. The decode claim elsewhere is "on par with the faster card alone" (`:417-418`), which is the honest one.
6. **The `--peer-device` path is a per-layer round trip and it lost.** Twice (`docs/MULTI_GPU.md:390-392`). Treat every per-layer cross-device design as guilty until measured.

---

## 4. The numbers for our box

Constants: 2× RX 7900 XT 19.98 GiB, no P2P, **14.2 GiB/s = 15.2 GB/s** cross-GPU, read wall **776 GB/s** per card, **10.57 GiB** of weights read per token, `n_embd = 5120` (from `research/03-bandwidth.md:31`, the `1×5120×17408` GEMM shape).

| Quantity | Value | Source |
|---|---|---|
| Handoff payload, `hc=4` (Strata's own residual form) | `4·5120 + 5120 + 4 = 25 604` floats = **100.0 KiB**/token | formula `verify.hpp:194` |
| Handoff payload, plain pre-norm residual (`hc=1`) | `5120 + 5120 + 1` ≈ **40 KiB**/token | same formula |
| Handoff buffer actually allocated | `8 rows × 100 KiB` = **800 KiB** (kVerifyMaxT = 8) | `generate.cpp:8892`, `verify_kernels.hpp:21` |
| Link time, 100 KiB one way | **6.7 µs** | 100 KiB ÷ 14.2 GiB/s |
| Link time, round trip (D2H by producer, H2D by consumer — no P2P) | **13.4 µs** | ×2 |
| Handoff as a fraction of a token | **0.05 %** of 14.6 ms | 10.57 GiB ÷ 776 GB/s |
| Cost of one *host-mediated* boundary, gfx1100 | **4.6 ms** = 31 % of a token | 1/53 − 1/70.1 s, `docs/MULTI_GPU.md:297` |
| **Latency : bytes, per boundary** | **~690 : 1** | 4.6 ms vs 6.7 µs |
| Weights that would have to cross the link per token to "stream" the model | 794 GB/s = **52× the link** | derived |
| Per-token time, one card | 14.6 ms ⇒ **68 tok/s** (776 GB/s); **~57 tok/s** at a realistic 650 GB/s end-to-end | derived; `03-bandwidth.md:20` |
| Per-token time, 2 stages, serial | 7.3 + 7.3 ms + boundary ⇒ **≈ one card, minus the boundary** | derived |
| Per-token time, 2 stages, perfect overlap | 7.3 ms ⇒ **137 tok/s = 2.0×** (ceiling) | derived |
| Realised, Strata, 2× gfx1100 | 53 → **70.1** tok/s (+32 %), one card ≈ 70 | `docs/MULTI_GPU.md:295-298` |
| Realised, Strata, 4 stages, 4× 7900 XT | 59.4 → 67.6 tok/s (+14 %), ~1/3 guesses held | `docs/MULTI_GPU.md:489-493` |
| Pipeline VRAM cost per card | **160 MiB** (second verifier) + **96 MiB** (two GDN snapshots) | `generate.cpp:4465-4466` |
| GDN snapshot per layer (their shape) | 3.27 MB | `generate.cpp:4467-4470`, arithmetic |

**The one relation that decides the architecture:** to feed a 10.57 GiB/token model across a 15.2 GB/s link you would need 794 GB/s — 52×. So weights *must* be stationary on the card that multiplies them, and the only thing that may cross is the residual. At 0.05 % of a token, the residual is not a budget item; it is noise. Per-layer handoff would be 64 × 100 KiB = 6.4 MiB = 0.43 ms = 2.9 % of a token — **still affordable in bytes, and still fatal, because it would add 64 host-mediated boundaries.** Byte count is not the design variable. Boundary count is, and then boundary *latency*.

---

## 5. The answer

### 5.1 The single most transferable idea

**Make the inter-card boundary a fixed-size, per-token residual that the producing GPU publishes into mapped pinned host memory under a release fence, and that the consuming GPU has already subscribed to — spinning on a bounded flag from a graph launched before the data existed. The host never synchronises there.**

This is `handoff_publish_kernel` + `wait_handoff_kernel` + the per-parity handoff buffers (`verify_kernels.cu:1190-1224`, `verify.hpp:182-194`, `generate.cpp:8896-8935`), applied to a resid structure that keeps the seam bit-exact (`verify.hpp:168-170`, `verify.cpp:2154-2164`).

Why this and not the layer split itself: the split is the *easy* part (contiguous ranges, one buffer, one `set_stage` call). The split alone is a **24 % loss** on gfx1100 (53 vs 70 tok/s, `docs/MULTI_GPU.md:297`). The flag is what converts the loss into parity. And the reason it matters rather than being a micro-optimisation is the 690:1 ratio in §4 — on this silicon the interconnect's *bandwidth* is a non-issue at any plausible handoff size, and its *synchronisation latency* is 31 % of a token. Every architectural decision follows from that one number.

### 5.2 What it would cost us

**Build cost (the honest part):**

| Item | Cost | Note |
|---|---|---|
| Handoff buffers | 2 × 800 KiB mapped pinned (±parity) | `generate.cpp:8892-8902` |
| Flag words | 64 B per boundary, mapped+portable | `generate.cpp:8918-8923` |
| Publish kernel | ~50 blocks × 256 threads, 1 launch/token | `verify_kernels.cu:1249` |
| Wait kernel | 1 thread spinning on 1 mapped word, **bounded** | `verify_kernels.cu:1190` |
| Second stream sync + graph launch | **removed from the critical path — this is the entire saving (4.6 ms/token)** | `docs/MULTI_GPU.md:289-290` |
| Stage weight trim | we must hold 5.3 GiB per card, not 10.57 | `docs/MULTI_GPU.md:205-211` |
| Correctness obligation | the seam must fold the pending write, or the split is not bit-exact; needs a `--split-device 0`-style A/B that runs both stages on one card and compares | `generate.cpp:8852` ("the A/B") |

**Benefit, priced:** +4.6 ms/token recovered = **+32 % decode on a split** (53 → 70.1 tok/s measured on our exact silicon, `docs/MULTI_GPU.md:295-298`). Against a one-card baseline of ~68-70 tok/s, this makes a split *free*. It does not make it *profitable*.

**And the decision that follows:** the model fits in one 19.98 GiB card (10.57 GiB + ~9 GiB for KV/activations). The repo's own answer for that case is `split_skip_if_fits` (`docs/MULTI_GPU.md:146-150`; `generate.cpp:4141-4145`): *don't split*, because an R9700 32 GB holding everything read 4K prompts at 1 776 tok/s instead of 1 244 and decoded at ~60 instead of ~51. So:

- **If the goal is single-stream decode latency: do not split.** One card, 68-73 tok/s, zero interconnect work. Copy T2/T7/T11's host-free disciplines for whatever *does* cross (KV, logits, prompts) and skip the rest.
- **If the goal is aggregate throughput at batch ≥ 2: replicate, don't split.** Two independent copies, one per card, zero bytes on the link — which is what Strata's app does (`docs/MULTI_GPU.md:14-20`: two models, one per card, ~60 tok/s each) and the literal version of "the constrained interconnect as an opportunity". We have the VRAM for it; a 10.57 GiB model twice is 21.1 GiB against 40 GiB.
- **If the goal is genuinely faster single-stream than one card can do: only T3 works**, and it needs a drafter, per-parity GDN snapshots with rollback, and it delivers +14-32 % against a 2.0× ceiling — with a third of guesses holding. That is the most expensive thing in the repo and the least proven.

**Costs of the thing I would actually copy (T2 + the invariants):** roughly 200 lines — publish kernel, bounded wait kernel, mapped buffer + flag allocation, the release/acquire ordering, and the rule "the host makes no CUDA call while a GPU spins on its flag" (T7, `generate.cpp:9106-9110`). It is cheap enough to build *even if we never split*, because the same primitive is what a host-free KV/logit path and a host-free drafter handoff need.

**The one number I would go and measure first, before writing any of it:** the round-trip flag latency on our pair (publish → observe, N=10 000, through mapped pinned memory). Every claim above assumes Strata's 4.6 ms is dominated by *launch and sync*, not by the flag crossing. If our mapped-memory round trip is itself ~1 ms, the whole calculus changes and the right answer becomes "two independent cards, no boundary at all". It is a 30-line benchmark on hardware we already have, and it is the only input in this note that is currently inferred rather than measured on our box.

---

## 6. What I would not copy

- **Tensor parallelism is absent.** They rejected it in one sentence (`docs/MULTI_GPU.md:9-10`) and shipped no code. The arithmetic for our box says TP is the one design with a clean 2.0× at batch 1 (5.3 GiB per card, ~7.3 ms, and a ring exchange of ~10 KB/layer × 64 = 640 KB/token each way = ~90 µs at 14.2 GiB/s, i.e. 0.6 % *if* the exchange hides under the next layer's compute). Strata provides **no evidence** either way, and its own per-layer-crossing negative results (§T5, twice) are a warning about the 128 synchronisation points, not about the bytes. If single-stream > 70 tok/s is the goal, this is the only route with a 2× ceiling and it needs to be evaluated on its own, not inferred from Strata.
- **The MoE machinery** (§T5, T9, T10d) — inapplicable to a dense model, and its per-layer pattern measured *slower* every time it was tried.
- **PDL** — CUDA sm_90+ only, hard no-op on HIP (`pdl.hpp:30-32`, `51`).
- **`--peer-device` / helper caches** — per-layer round trips; documented losses.
- **`cudaMemcpyBatchAsync`** (`dma_batch.hpp:1-15`) — CUDA 13+ only, and its stated benefit is coalescing WDDM interrupt delivery on Windows. Irrelevant on ROCm/Linux; the HIP path falls through to a plain per-blob loop (`dma_batch.hpp:96-103`).

---

## 7. Open questions for the plan

1. **Round-trip flag latency on our pair** — unmeasured, and it gates the design (§5.2). 30-line benchmark.
2. **Does our residual have a pending-write/inject structure?** If yes, the handoff is 100 KiB and needs the fold; if it is a plain pre-norm add, it is 40 KiB and `copy_from_mapped` alone suffices (`verify.cpp:2161-2163`). This decides how much of T1 we implement.
3. **Do we have (or will we train/borrow) a drafter?** Without one, T3 does not exist and cross-card speedup is capped at 1.0×.
4. **Is `hipMemCreate`/`hipMemMap` usable on ROCm 7 for the elastic K/V trick?** `vmm.hpp:5` says "Not under HIP" — untested on our driver.
5. **`RLIMIT_MEMLOCK` behaviour on ROCm/Linux** for late mapped allocations (§T10b) — the repo's reserve logic is Windows-only by default (`pin_budget.hpp:17-18`).
