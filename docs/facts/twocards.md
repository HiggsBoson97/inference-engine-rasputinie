# facts — two cards

gfx1100 ×2, 40 GB of VRAM, joined by a link that is narrow and has no peer DMA. What each topology
costs, and what decides between them.

| fact | value | instrument | date | verified-by |
|---|---|---|---|---|
| Link | **PCIe 4.0 ×16, no peer-to-peer DMA** — cross-card traffic is host-mediated | `lspci`, `hipDeviceCanAccessPeer` | 2026-10-22 | `_archive/2026-10-24/docs/research/18-tp-roccl.md` |
| Link bandwidth | 32 GB/s theoretical, **~25 GB/s practical** = 2.1 % of the 1552 GB/s aggregate VRAM | ARITHMETIC | — | `facts/memory.md` + `18-tp-roccl.md` |
| RCCL all-reduce, 5120 elems | **20.65 µs** amortised (pynccl); **~19.5 µs floor** up to 16 KiB; not tunable by `NCCL_PROTO`/`NCCL_ALGO`/`NCHANNELS` | RCCL benchmark on this box | 2026-10-22 | `18-tp-roccl.md` §4.1 |
| `torch.distributed` | ~2× worse (41.48 µs) — never benchmark TP with it | same | 2026-10-22 | same |
| Host-mediated boundary | **690:1 latency-to-bytes** ratio | s1-strata | 2026-10-22 | `_archive/2026-10-24/docs/research/2026-10-23/s1-strata.md` |
| What TP actually moves | **nothing but partial sums**: 128 all-reduces × ~10 KiB = **1.3 MB = 52 µs of link bytes per token** | ARITHMETIC | — | `docs/plan/COST-LEDGER.md` |

## The three topologies, priced

Weights are **sharded permanently** — TP does not ship weights across the link, and neither does PP.

| topology | single stream | concurrency ≥ 2 | KV capacity (7 GiB pool) | cost |
|---|---|---|---|---|
| **TP=2**, 20.65 µs RTT | **1.48×** (15.1 → 10.2 ms/token) | 2× | ~6.5 × 32K sessions | 128 all-reduces per step |
| **TP=2**, 2 µs RTT | **1.93×** | 2× | ~6.5 | 128 × 2 µs = 0.26 ms |
| **PP=2**, 20.65 µs RTT | **≈1.70×** (64 handoffs, no reduction primitive) | needs pipelining | ~6.5 | pipelining bubbles at batch 1 |
| **DP=2** | **1.00×** — one card idles | 2× | ~7.5 (full model per card) | none |

**DP gives nothing on a single stream.** That is the correction to the earlier "DP=2 for IQ3_S"
decision, which priced only aggregate throughput.

Evidence that PP is the shipping answer on two dissimilar cards (`strata`, no NVLink):
layer split took decode from **66.6 → 75.8 tok/s**, and adding a third helper card made it
**slower** — *"its per-layer round trip costs more than the CPU pool needs"*. Both readings are in
`_archive/2026-10-24/docs/research/2026-10-23/s5-donor-transfer.md`.

## The gate

**A mapped-publish flag round trip, p99 over ≥1000 iterations, with a single-card loopback as the
floor.** 19.5 µs is the number to beat — it is RCCL's protocol floor, not the hardware's. Every
microsecond off it is ~0.13 ms off a TP step, because there are 128 of them.

`assumed:` that a hand-rolled flag protocol beats RCCL's floor. Unmeasured; that is the gate.
