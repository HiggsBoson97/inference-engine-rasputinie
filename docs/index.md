# index — the reachable map

Every file in `docs/` is reachable from here. If a document is not linked below, it is either dead
or missing a citation.

## State

- [`../HANDOFF.md`](../HANDOFF.md) — the resume document for the sglang rework: concepts, bounds, the op contract, the unit sequence.
- [`status.md`](status.md) — the frontier unit, its proof, and the last landed state.
- [`plan/DEVELOPMENT-PLAN.md`](plan/DEVELOPMENT-PLAN.md) — every row a unit with start / goal / proof.

## Measured — `facts/`

| file | holds |
|---|---|
| [`facts/memory.md`](facts/memory.md) | the 776 GB/s wall, the 619 retraction, L2, the two boundaries, and the VRAM/L2/host-RAM terminology |
| [`facts/compute.md`](facts/compute.md) | CUs, clock, instruction rates, the 193-word body, issue occupancy — and the refuted `n = 1.57` model |
| [`facts/quant.md`](facts/quant.md) | geometry, the 11 types with byte shares and strides, per-type measured rates, the two ungated types, GSQ-RCO, the grids |
| [`facts/engine.md`](facts/engine.md) | the vendored sglang, the four bring-up blockers, MMVQ-vs-dequant, aiter's allow-lists, Triton's native status, `__dp4a` |
| [`facts/decoder.md`](facts/decoder.md) | the 47.1 tok/s estimate, launch/fusion, the verify curve `ms(k)`, and both drafters' cost |
| [`facts/twocards.md`](facts/twocards.md) | the link, RCCL's 19.5 µs floor, TP/PP/DP priced |
| [`facts/traps.md`](facts/traps.md) | the 18 traps that must not occupy `AGENTS.md`, each with its incident |
| `facts/instruments/` | the two calc scripts a fact cites as its instrument |

## Decided — `decisions/`

| id | decision |
|---|---|
| `D-01` | one served model: the published IQ3_S file with its MTP head; a drafter is not a second model |
| `D-02` | llama.cpp excluded |
| `D-03` | FP8 excluded as a format |
| `D-04` | fork sglang; the vendored tree *is* the engine |
| `D-05` | serve quantised as published, one internal layout, bf16 compute, no codebook expansion |
| `D-06` | records never targets; no shape is the target either |
| `D-07` | aiter's ASM/CK path excluded; Triton is the kernel backend |
| `D-08` | speculation is core; MTP and DFlash2 both in scope |
| `D-09` | topology is decided by the round trip, not by aggregate throughput |

## Plan

| file | holds |
|---|---|
| [`plan/REQUIREMENTS.md`](plan/REQUIREMENTS.md) | the hard constraints the decisions rest on |
| [`plan/RECORD.md`](plan/RECORD.md) | every measured rate, with provenance and superseded rows marked |
| [`plan/COST-LEDGER.md`](plan/COST-LEDGER.md) | each per-token term as a function of `(B, C)`, and the shape surface |
| [`plan/TENSOR-INVENTORY.md`](plan/TENSOR-INVENTORY.md) | the 72 `(role, type)` pairs and the byte budget |
| [`cleanup/MANIFEST-2026-10-24.md`](cleanup/MANIFEST-2026-10-24.md) | what moved here, what was archived, and why |

## Provenance

`_archive/2026-10-24/` holds the pre-migration tree verbatim: 58 files, including the 26 KB handoff,
the v10 control plane, 22 research lanes, 9 dispatch briefs and 6 hypothesis reports. Facts cite it
where the raw report is the evidence — **never cite it as current.**
