# Donor transfer — what the reference engines give us, and what is dead on this card

Read 2026-10-23. Each row names the artifact actually read, the technique, and the gate that
decides whether it survives on **gfx1100 with no peer-to-peer DMA** — because a technique
that needs `cp.async`, tensor cores, or P2P is dead here regardless of how well it performs
on an H200.

Verdict vocabulary: **adopt** (applies, worth building), **adopt-later** (applies, not on the
critical path yet), **cross-check** (not a speed technique, but settles a fact),
**dead** (blocked by this hardware), **unread** (queued).

## strata

| artifact read | technique | verdict here | what it buys | what would confirm it |
|---|---|---|---|---|
| `bench/results/2026-10-01-community-2x-rtx-pro-4500/README.md` (launch line) | **`--layer-split 25`** — pipeline, not tensor, as the two-card strategy | **adopt** — no reduction primitive needed, 64 one-way handoffs instead of 128 all-reduces | 1.70× on a single stream at RCCL's 20.65 µs floor (arithmetic in `COST-LEDGER`) | the round-trip measurement, then a two-card step against one card |
| same | **`--kv k8v4`** — K at 8-bit, V at 4-bit, asymmetric | **adopt** | KV falls from 64 → 34 KiB/position bf16→q8, and further with V at 4-bit; moves boundary 1 out | a paged-attention kernel rate at q8 vs k8v4, checksum-gated |
| same | **`--spec 5 --spec-min-p 0.5`** — adaptive draft depth with a probability floor | **adopt-later** — needs the verify curve first | avoids drafting when acceptance is unlikely | `ms(k)` curve + acceptance per k |
| `bench/results/2026-09-27-cache-parity/`, `…-layer-split-limits/` | **KV streaming with a resident window** — 128K context, 32K resident, 96–97 % cache hits | **adopt-later** — capacity technique, not a rate technique | long context beyond VRAM without a 2× VRAM card | hit-rate and tok/s at a context exceeding VRAM |
| `…-layer-split-limits/` (negative result) | *"the helper card makes it slower — its per-layer round trip costs more than the CPU pool needs"* | **read as evidence** | this is the per-layer handoff cost that decides PP's viability | our own round trip |

## tensorfold

| artifact read | technique | verdict here | what it buys | what would confirm it |
|---|---|---|---|---|
| `zig/src/families/qwen3_5/cuda_cost_adapter.zig` (131 lines) | **measured verify-width cost table** — times the model at every batch width 1…`window_rows`, stores `ms[rows-1]`, errors on `IncompleteWindowMeasurements`, caches keyed by a hash of geometry + policy (`kv8`,`verify`,`window`) + drafter kind + drafter checkpoint | **adopt — highest-value single import in this survey** | turns draft depth from a guess into a lookup. `step_ms(k)` is non-linear, so `k` is unanswerable without it | build the curve on our kernel; it is measurement #1 in the ledger's queue |
| same file | `Kind = .dflash \| .mtp` as a first-class cost-model dimension | **adopt** — the MTP-vs-DFlash comparison already has a design, for our exact family | A3 stops being an open-ended study | acceptance at each drafter, same prompts |
| `zig/qwen35_cost_process.zig` | geometry constants for **qwen3_5**: hidden 5120, intermediate 17408, layers 64, vocab 248320, kv_heads 4 | **cross-check** | independently confirms our model geometry from a second source | already consistent with `01-models.md` and the inventory |
| same | `cuda_slide.zig`, `cuda_reuse.zig` — sliding context windows, kept prompt states for a shared cache | **adopt-later** | prompt reuse across requests; capacity | hit rate on a repeated-prefix workload |
| same | `laneFacts(window, verify)` with `batch_rows = verify` | **adopt** — the verification batch as an explicit dimension of the kernel's lane model | the kernel must be written to run at k verify rows, not batch 1 then adapted | kernel exists at k > 1 |

## sglang (vendored base)

| artifact read | technique | verdict here | what it buys | what would confirm it |
|---|---|---|---|---|
| `docs/research/18-tp-roccl.md` (measured on this box) | **RCCL all-reduce, 2 ranks, PCIe no P2P** | **cross-check** | 20.65 µs amortised at 5120 elems; a **~19.5 µs floor** up to 16 KiB, not tunable by `NCCL_PROTO`/`NCCL_ALGO`/`NCHANNELS`; `pynccl` is ~2× cheaper than `torch.distributed` | a custom mapped-flag protocol either beats 19.5 µs or it does not — that is the whole TP question |
| same | the TP budget built from 20.65 µs × 128 | **superseded framing** | DP=2 was chosen from aggregate throughput; `COST-LEDGER` re-opens it | round-trip measurement |

## exllamav3

| artifact read | technique | verdict here | what it buys | what would confirm it |
|---|---|---|---|---|
| `doc/optimize.md` | per-tensor sensitivity recipes: trace → error amplitudes → per-tensor sensitivity → compile a recipe → convert → evaluate | **cross-check only** | this is what RCO already did for our file, and the publisher ships the result as `rco-allocation.txt`. No speed technique | diff their recipe against the publisher's allocation — if they agree, the recipe is right twice |

## deepinfer

| artifact read | technique | verdict here | what it buys |
|---|---|---|---|
| index: `deep_cluster/`, DeepEP vendor, Ascend requirements | expert-parallel / MoE cluster serving | **dead for this project** | our 27B is dense and the inventory has no expert role (`03-native-kernel-inventory.md`) |

## freetoken

**unread** — queued. Its index shows an AMD install doc and a `freetoken-kernel-cache/`, so
it is the one donor that may already carry AMD-specific kernel work; that is the next read.

## The three techniques this survey says to build first

1. **The verify-width cost curve** (TensorFold) — it gates every speculation decision.
2. **Layer split as the two-card default** (strata) — it is what a shipping engine does on
   two dissimilar cards with no peer DMA, and it needs no reduction primitive.
3. **k8v4 asymmetric KV** (strata) — it is the only listed technique that moves the KV
   boundary instead of the weight rate.

None of the three is a kernel micro-optimisation. That is the finding: the remaining
large gains are in *how the engine is shaped*, not in how the dot product is coded.
