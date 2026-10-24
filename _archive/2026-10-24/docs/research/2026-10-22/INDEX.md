# Research, 2026-10-22 — seven lanes

Read in this order. **One correction governs the rest.**

## Correction first

Every derived ceiling in these documents was computed against a **619 GB/s** wall
that this project carried from its first session. That number is a kernel
artefact. The card reads at **776 GB/s** — verified twice on 2026-10-22, see
`../../bench/gfx1100-probes/WALL-2026-10-22.md`.

**Every absolute GB/s and tok/s figure below is therefore ~25 % low.** A fused
4-bit dequant-GEMV has been measured at 752 GB/s (97 % of the real wall) with
arithmetic verified to 2.6 × 10⁻⁷.

What still stands in these documents: every *relative* comparison, every
mechanism finding, every "X costs Y % more than Z", and every negative result.

## The documents

| file | answers |
|---|---|
| `01-languages.md` | What can we write gfx1100 kernels in? HIP C++ verdict, Triton's real status, CK's support, **and the definitive Rust answer** |
| `02-megakernels.md` | Megakernel vs persistent vs graph replay; the barrier economics; verdict: do not build one |
| `03-bandwidth.md` | **The wall correction.** What costs bandwidth and what is worth exactly zero; a measured GEMV design |
| `03b-gfx11-microarchitecture.md` | LDS, Infinity Cache, WGP structure — the facts behind the kernel decisions |
| `03c-gguf-quant-layouts.md` | The GGUF block formats as load patterns, not as specifications |
| `04-kv-cache.md` | q8 KV, FP32 recurrent state, what to reject and why; a quality gate costing ~2.5 h per candidate |
| `05-serving-knobs.md` | Every serving knob priced in bytes: batching, chunked prefill, prefix cache, speculation, graphs |
| `06-recent-research.md` | 2025–26 sweep, filtered for this box; what runtime adaptation is actually worth |
| `07-correctness.md` | How to prove a kernel and a model right: tolerances, two references, mutation testing |

## The five findings that change the plan

1. **The wall is 776 GB/s, not 619** (`03-bandwidth.md` §1, verified in `bench/`).
2. **A fused 4-bit GEMV already runs at 97 % of it**, so the single-card ceiling
   is ~66 tok/s weights-only and ~55 tok/s realistically — not 52.6/31.
3. **HIP C++ is the only viable kernel language here; Rust is a dead end for
   kernels** (`01-languages.md`).
4. **Do not build a megakernel or a run-scoped persistent kernel**
   (`02-megakernels.md`); a layer-scoped persistent kernel stays open.
5. **The 32-greedy-token arbiter is too weak to gate on** — it fails 28 % of the
   time at a 1 %/token error rate and passes kernels wrong in ways argmax cannot
   see (`07-correctness.md`).

## Provenance

Written by seven parallel research subagents on 2026-10-22, each with web access,
each required to tag claims `[measured-on-RDNA3]` / `[measured-elsewhere]` /
`[documented]` / `[claimed]`. The `03-*` lane compiled and ran HIP kernels on the
card; its two headline measurements were re-run by the orchestrator and
reproduced. Two claims in `06-recent-research.md` were withdrawn by their author
(SHEPHERD does not exist; a stream-K arXiv id did not resolve) — they are marked
as withdrawn in place.
