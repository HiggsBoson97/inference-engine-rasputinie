# facts — memory

Every line is one measurement: value, instrument, shape, date, and where to re-run it.
`_archive/2026-10-24/…` paths are the pre-migration locations, preserved verbatim.

| fact | value | instrument | shape | date | verified-by |
|---|---|---|---|---|---|
| Read wall, one card | **776 GB/s** (max 777.4) | `bw2.cpp` streaming read, ≥3 ms kernels | 6 GiB, 4 passes | 2026-10-22 | `bench/gfx1100-probes/WALL-2026-10-22.md` |
| 619 GB/s is **retracted** | a 1.6 ms kernel artefact; never the wall | same harness, short kernel | — | 2026-10-22 | same file, correction banner |
| Copy (r+w counted) | 618 GB/s; non-temporal store 658 | `bw2.cpp` | 6 GiB | 2026-10-22 | same file |
| Aligned-stream control, today | 715.8 GB/s | `iq3s_valid` `wall_flat` | 1.0700 GiB, `GRIDMUL=8` | 2026-10-23 | `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-1.md` §5 |
| L2 size / rate | 96 MB / **1763.9 GB/s** | `iq3s_valid` `read`, 38.3 MB footprint, `rp=64` | 0.0357 GiB | 2026-10-23 | same, §5 |
| IQ3_S dequant-GEMV, VRAM | **521.3 GB/s** (67.2 % of wall) | `iq3s_valid` `full`, checksum-gated | 1.0700 GiB, `grid=336×256`, `rp=4` | 2026-10-23 | same, §5 |
| IQ3_S, memory-free ceiling | **645.1 GB/s** (83.1 %) | same kernel, L2-resident | 38.3 MB, `rp=64`, ≥3 ms | 2026-10-23 | same, §5 |
| IQ3_S read-only floor | 743.7 GB/s (95.8 %) | `k_read` | 1.0700 GiB | 2026-10-23 | same, §5 |
| Two-card link | PCIe 4.0 ×16, **no peer-to-peer DMA** | `lspci` + `hipDeviceCanAccessPeer` | 2 cards | 2026-10-22 | `_archive/2026-10-24/docs/research/18-tp-roccl.md` |
| All-reduce, 2 ranks, 5120 elems | **20.65 µs** amortised (pynccl) | RCCL benchmark | 10 KiB bf16 | 2026-10-22 | same |
| RCCL latency floor | **~19.5 µs** up to 16 KiB; not tunable by `NCCL_PROTO`/`ALGO`/`NCHANNELS` | four configs swept | — | 2026-10-22 | same |
| `torch.distributed` penalty | ~2× the pynccl cost (41.48 vs 20.65 µs) | same | — | 2026-10-22 | same |
| KV bytes per context position | **64 KiB bf16**, **34 KiB q8_0** (the f16 block scale costs 2 KiB) | ARITHMETIC from 16 full-attn layers × 4 KV heads × 256 dim × K,V | per position | 2026-10-22 | `_archive/2026-10-24/docs/research/2026-10-22/04-kv-cache.md` §1 |
| KV overtakes the weights at | **≈173 K resident tokens** (bf16); ≈326 K at q8 | `10.57 GiB ÷ KV bytes/position` | product `B×C` | 2026-10-23 | `docs/plan/COST-LEDGER.md` §"two boundaries" |
| GDN state per sequence | 159 MB FP32, read **and written** per token = 0.318 GB/token/seq | ARITHMETIC from the file | per sequence | 2026-10-22 | `docs/plan/REQUIREMENTS.md` R5 |
| GDN state overtakes the weights at | **≈36 concurrent sequences**, any context | `10.57 GiB ÷ 0.318 GB` | — | 2026-10-23 | `docs/plan/COST-LEDGER.md` |

## Terms, because one label caused a real error

- **VRAM** — the card's GDDR6, 19.98 GiB. Every wall measurement is here. It is *device* memory.
- **L2** — 96 MB on-die; the "memory-free" experiments fit a footprint inside it.
- **Host RAM** — 60 GiB DDR5. **Never in a timed path.** No spill, no managed allocation, no pageable
  buffer. An older report says "DRAM arm" where it means VRAM; that wording is the reason this note
  exists.

## `assumed:`

- That decode reads each weight **once** per step. Holds for a GEMV; unverified at a verify width
  above 8, where the kernel is already issue-bound (`facts/compute.md`).
