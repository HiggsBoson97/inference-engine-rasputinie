# rasputinie

A bespoke LLM inference engine for **one machine**: two AMD Radeon RX 7900 XT
cards, and one model, Qwen3.8-27B.

Not a portable fork. Not a support matrix. An engine whose every design decision
is derived from this hardware, measured on this hardware.

## The machine

| | |
|---|---|
| GPU | 2× AMD RX 7900 XT — gfx1100, RDNA3, 19.98 GiB each (40 GB total) |
| Compute | 84 CU, 5376 lanes, wave32. bf16 WMMA. **No** FP8 MMA, no MFMA, no TMA, no clusters |
| Memory | 800 GB/s spec · **619 GB/s measured read** (copy 595) |
| Interconnect | PCIe 4.0 ×16, **no peer DMA** — cross-GPU copy measured 14.2 GiB/s, identical to host-bounce |
| Model | Qwen3.8-27B: 64 layers — 48 GatedDeltaNet linear attention + 16 full attention, hidden 5120, MLP 17408, 4 KV heads × 256 |
| Weights | GGUF at 3.4–4.5 bits/weight **and** safetensors, both in scope |
| Compute precision | bf16 throughout |

## The governing constraint

**Decode at batch 1 is bandwidth-bound.** Bytes per token decide the ceiling;
the instruction set decides which bytes are affordable. Every optimisation is a
weight-byte reduction or a weight-byte speed-up. Bandwidth efficiency is the
objective — tok/s is a measurement of how well we are spending bandwidth, never
the goal itself.

The second constraint is the interconnect. With no peer DMA, the two cards are
two memory spaces joined by a link 42× slower than their own HBM. Data movement
between them is a cost to be designed around, not a tuning parameter.

## Phase

**PLAN.** Nothing is built until the plan gate passes. The working requirements
document is [docs/plan/REQUIREMENTS-DRAFT.md](docs/plan/REQUIREMENTS-DRAFT.md);
the measured physics is [docs/00-facts.md](docs/00-facts.md).

## Layout

```
engine/     pristine sglang at the pinned commit, vendored and to be reshaped in place
            (provenance: engine/UPSTREAM.md)
docs/       00-facts  measured physics        01-models  architecture and memory math
            03-native-kernel-inventory         adr/       accepted decisions
            research/  measurement-backed reports      plan/  the live plan
bench/      probes that regenerate every number in docs/
```

## Hard rules

1. **llama.cpp is excluded** — not a baseline, not a reference, no derived code.
   See `docs/decisions/0001-exclude-llama-cpp.md`.
2. **Every claim carries a measured number.** When a document and a measurement
   disagree, the measurement wins and the document gets a correction banner.
3. **This engine targets this machine.** Portability is not a goal; it is a
   non-goal that has already cost this project time.
