# Probe brief — H-10: the engine baseline, and the deletion map

```
STATION: probe (engine baseline; the turn from scratch kernels to the product)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin`, `~/development/inference-engine-rasputinie/engine` (the installed sglang) ; scratch `~/b1/`
TOOLSET: ssh, the venv at `~/venv/sglang-fork`, python, amd-smi, sudo -n
BUDGET: 110 tool calls; at ~90 % write out; unfinished items report as failed(budget)
ROUND: 1
CONTRACT: none — Discovery probe, but it produces the strip plan
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-10.md
```

## GOAL

Two things, in this order, and the first one needs no GPU:

1. **The deletion map.** What does a batch-1 decode of *this* model actually touch, which of
   those files are CUDA-only and unreachable on gfx1100, and what is the ordered list of what
   can be deleted from `engine/` without changing behaviour on this machine.
2. **The engine baseline.** Does the vendored sglang serve
   `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` on this card at all, at what tok/s, and **which dequant
   path does it take on ROCm**.

The baseline is the "before" for every optimisation the project will make. We have never had
it: the only engine number on record (857 ms/token) was a *different* checkpoint in FP8, a
format this project excludes.

## WHY THIS IS THE TURN

Every measurement in this project so far ran in `~/e1`, `~/a2`, `~/a9` — scratch harnesses
over 30 concatenated tensors. None exercised the layer sequence, the dispatch, the GDN
recurrence, the KV path, or two cards. The engine is the artifact; measure it in situ.

## THE FACT THAT DECIDES THE WORK

`engine/python/sglang/srt/layers/quantization/gguf.py` (1016 lines) already declares every
weight type in our file — `IQ1_M, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
IQ4_NL, Q4_K, Q5_K, Q6_K, Q8_0` (lines 155–174). But line 223 says: *"If there is no available
MMQ kernel, fallback to dequantize"*, and the fallback is
`dequantize_gguf_weight → ggml_dequantize → bf16 matmul`.

**On gfx1100 there is no MMQ kernel** (it is CUDA), so the likely path is: read the quantized
weight, dequantize it to bf16, then run a bf16 matmul. Against our measured fused `dp4a`
GEMV at 521 GB/s and a 776 GB/s wall, that is a different algorithm with different
arithmetic. **Confirm which path actually runs** — that is the single most valuable fact in
this probe, and it decides whether the first engine optimisation is "replace the dequant+GEMM
path with a fused dp4a GEMV" or something else.

## RULES

1. **`HIP_VISIBLE_DEVICES=1`** — PCI `07:00.0`, headless. HIP 0 / PCI `03:00.0` is the display
   card; a load there wedged gnome-shell once. Verify by readlink before launching.
2. **llama.cpp is excluded** — no clone, no read, no build, no derived code. Name it in your
   compliance line.
3. **The card may be busy.** Another probe may be running `~/a9/...iq3s_valid`. Before any
   timed run, check `pgrep -af iq3s_valid` and the VRAM table; if it is busy, **do phase 1 and
   wait**, re-checking every ~60 s for up to 30 minutes. A baseline measured against a
   contended card is worthless.
4. **Do not modify `engine/`** in this probe. You are mapping it, not editing it. All notes and
   logs in `~/b1/`.
5. **Do not install anything.** Use the venv as it is: `~/venv/sglang-fork` has sglang 0.5.9
   installed **editable from `engine/python`**, so the vendored tree *is* the running engine.
6. **Pin clocks** (`high` on 07:00.0) for the timed run, restore `auto`.
7. Kill by PID, never `pkill -f`.

## PHASE 1 — the deletion map (no GPU)

- **Does it import, and does the engine construct?** `import sglang`; then construct
  `sgl.Engine(model_path=<gguf>, ...)` far enough to load weights, on the headless card. If it
  fails, the traceback is the finding — report it verbatim with the file:line.
- **Which dequant path runs on ROCm.** Read `gguf.py:190-240` and trace the two branches. Then
  prove it at runtime: instrument by observation, not by editing `engine/` — e.g. run under
  `python -X importtime`, or inspect which module `ggml_dequantize` resolves to
  (`sgl_kernel` vs the pure-python `gguf` package) with a one-liner that imports it and prints
  `__module__`. **State which one it is, with the command's output.**
- **The reachable set.** What does a batch-1 decode of this model import and execute? Do it
  dynamically where you can (`sys.modules` after constructing the engine, filtered to
  `sglang.*`) rather than by static AST — a static graph over-counts. Report: files and LOC
  reachable, and how many of those import `cuda`, `CUTLASS`, `cute`, `flashinfer`, `sgl_kernel`
  at module scope.
- **The NVIDIA-only surface.** `engine/` holds 55 `.cu`, 263 `.cuh`, 44 `.cpp`, 87 `.h`, 7
  `.hip`, 7,882 `.py`. Produce an ordered deletion list: (a) never imported on any path from
  this model, (b) imported but only on CUDA-gated branches, (c) imported unconditionally.
  Category (a) is safe to delete; (b) needs the guard checked; (c) is the real work. Give file
  counts and LOC per category, not prose.
- **The launch line.** The exact command that would serve this GGUF on one card, with the
  flags that matter (`--load-format`, `--quantization`, `--mem-fraction-static`,
  `--context-length`, `--tp`, `--disable-cuda-graph` if needed). Cite the flags' definitions.

## PHASE 2 — the baseline (card free)

One run, short: serve the model, greedy, fixed prompt, 32 new tokens, `temperature 0`.
Report:
- **TTFT** and **decode tok/s** (at least 3 runs; median and spread).
- **VRAM** at steady state per card.
- **The printed quantisation/dequant configuration**, verbatim from the server log — this is
  what proves which path ran.
- **Whether the output is coherent**, with the generated text quoted. A fast garbage answer is
  not a baseline.
- **% of the 776 GB/s wall** implied by the decode rate, and the per-token ms.
- Where time goes, if cheaply obtainable: one `torch.profiler` run over 8 decode steps,
  reporting the top 10 kernels by self time. If profiling is expensive or breaks, skip it and
  say so — do not burn the budget on it.

## HOW TO VERIFY YOU DID IT RIGHT

- The engine's own log states the quantisation method it chose; quote it. A claim about the
  dequant path without that line is speculation.
- `ggml_dequantize.__module__` printed from a live import — this settles CUDA-vs-python.
- The generated text is coherent and quoted; a baseline that emits gibberish is a finding, not
  a number.
- The decode rate is reported with its shape (batch 1, context length, prompt length) — a tok/s
  without shape is not a record.
- The deletion map's categories sum to the file count you actually measured; state the count.
- If the card was contended, say so and mark the run `not-run` rather than reporting a
  depressed number.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-10.md`, then print it:

```
# H-10 — engine baseline and deletion map
## Question (can the vendored sglang serve this file here, at what rate, and what can go)
## What the engine is (version, editable install path, file counts by type)
## The dequant path on ROCm (the log line, the module, the branch taken)
## Phase 1: the deletion map (counts and LOC per category; the launch line)
## Phase 2: the baseline (TTFT, decode tok/s with median and spread, VRAM, coherent text)
## Where the baseline sits (per-token ms, % of wall, gap to our measured 521 GB/s fused GEMV)
## The first three things to change, ranked, each with the measurement that would prove it
## Compliance (llama.cpp excluded; engine/ unmodified; clocks restored)
## Receipts (absolute paths under ~/b1/)
```

Effort: **high**. This is the shift from scratch harnesses to the product, and its value is
that every later number is measured against it. Get the dequant path right and the baseline
honest; a wrong "before" number corrupts every "after".
