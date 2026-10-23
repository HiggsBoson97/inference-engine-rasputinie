# Research brief — drafters: MTP head vs DFlash2 on gfx1100

```
STATION: research probe (A3 pre-work: drafter selection)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: repo at /Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie (read-only) + the host's `engine/` tree
TOOLSET: read/search on the repo, ssh for the host tree, web fetch for the model card and upstream source
BUDGET: 70 tool calls; at ~90 % write out; unfinished items report as failed(budget)
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-5.md
```

## GOAL

Answer one question with evidence: **for this box, what does each candidate drafter cost
and what does it buy — the shipped MTP head, or the DFlash2 block-diffusion drafter?**

Deliver a decision table the plan can cite, plus a list of what is NVIDIA-only in the
DFlash2 path and what it would take to make it gfx1100-native. Do **not** implement
anything, and do **not** benchmark on the GPU — this lane is reading and reasoning only.

## CONTEXT

The project serves exactly one model, `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S
(3.44 bpw, 11.77 GB on disk, 10.57 GiB read per token), with its shipped MTP head
(`IQ3_S-mtp.gguf`, 12.12 GB). One card: 19.98 GiB VRAM, 776 GB/s measured read wall.

**Why this matters now.** Autoregressive decode costs one full weight sweep per token
(10.57 GiB → ~15 ms at a realistic rate → ~60 tok/s). Speculation replaces that with one
sweep per *verification step* and accepts several tokens per step, so the whole per-token
cost divides by the acceptance length. On the published numbers (H200, BF16 target,
7 draft tokens) MTP averages ~4.3 accepted tokens and DFlash2 ~4.8, but DFlash2's
throughput advantage (3.43× vs 2.59× at concurrency 1) is larger than the acceptance ratio
— because a block-diffusion drafter proposes the whole block in **one** forward pass
where MTP drafts autoregressively.

**Two facts about our situation that the published numbers do not cover:**
1. The drafter's own weight read is not free here. DFlash2 is a **2B-param BF16** model
   (~4 GB). Against a 10.57 GiB target sweep that is a ~35 % addition to every step unless
   the drafter is quantised. Our card has ~9.7 GB free after the target model.
2. The published benchmark is an H200 with FlashAttention 3. Our target is a gfx1100
   card with no FA3, no `cp.async`, and an AMD ROCm 10.1 toolchain. The acceptance
   distribution may transfer; the kernel stack will not.

## WHAT TO FIND OUT — each with a pointer, not a recollection

**A. What is actually in the vendored engine.** The repo's `engine/` is a vendored sglang
`d0a7565` (11 063 files), also present at `/home/mattia-bosi/development/inference-engine-rasputinie/engine`
on the host. Determine:
- Does it contain a DFLASH / DFlash2 speculative path at all? Which files, which
  `--speculative-algorithm` values are accepted, and is `DFLASH` among them?
- What does the MTP path look like (file, entry point, whether it is a separate head or a
  full draft model)?
- For every file on the DFLASH path: is it CUDA/Triton/FA3/CUTLASS-dependent? Name the
  exact imports and kernel files. This is the "is it AMD-native" question — answer it per
  file, not in aggregate.
- How much of `engine/` is reachable from the decode hot path at all? A rough count of
  files/imports that a batch-1 GEMV decode actually touches is enough to size the
  "strip sglang to the minimum" work; say what method you used.

**B. What DFlash2 actually is.** From the model card and the upstream repo
(`github.com/z-lab/dflash`, and the HF repo `z-lab/Qwen3.8-27B-DFlash2`):
- architecture (block-diffusion backbone, the "two-tap dynamic convolutions", the
  candidate-path selector), parameter count, dtype, file sizes;
- the exact draft-cost structure: how many forward passes per proposed block, what the
  selector costs, what `num_draft_tokens` / block size 8 means for verification;
- whether the drafter requires the target's hidden states or logits (this decides whether
  it can be bolted onto our engine without exposing internals);
- the losslessness claim and what it depends on (greedy match, distribution preservation).
- **Licence.** State it. A drafter we cannot legally embed changes the answer.

**C. The cost model, computed, not asserted.** Build the arithmetic for *this* box and put
it in the report:
- target sweep time at the measured 521 GB/s and at the 645 GB/s memory-free ceiling;
- draft pass time for DFlash2 at BF16 (~4 GB) and at a plausible 4–5 bpw quantisation
  (~1.0–1.25 GB), and for the MTP head (which is inside the 12.12 GB file — work out what
  the MTP tensors actually weigh);
- effective tok/s = accepted tokens per step ÷ (target sweep + draft cost), at MTP's and
  DFlash2's published acceptance lengths, and at a pessimistic half of each;
- VRAM left per card for KV at each choice, and what that implies for concurrent 32K
  sessions. The project's own figure: 159 MB of GDN state per sequence, 7 % of a 32K
  session.
State every input and its source; label the result ESTIMATE.

**D. What would have to be true for DFlash2 to win here.** A short list of falsifiable
conditions (acceptance ≥ X on our prompts, draft cost ≤ Y ms, VRAM ≤ Z), each with the
measurement that would settle it. This is the input to the A3 measurement design.

## RULES

1. **No GPU work.** Do not launch anything, do not set `HIP_VISIBLE_DEVICES`. Another
   probe holds the card.
2. **llama.cpp is excluded** — do not clone, read, build, or derive from it.
3. **Cite files by path and line**, and web claims by URL. A claim without a pointer does
   not enter the report.
4. Do not install anything on the host.
5. Distinguish **measured** (someone ran it), **published** (the card or a paper claims
   it), and **derived** (your arithmetic). The report is worthless if those blur.

## HOW TO VERIFY YOU DID IT RIGHT

- Every claim about `engine/` names a file that exists, and you quote the line.
- Every NVIDIA-only claim names the import or kernel file, not "it looks CUDA".
- The cost model's inputs each carry a source; recomputing it from your table reproduces
  your numbers.
- The licence is quoted, not summarised from memory.
- You did not touch the GPU: no VRAM change, no new dmesg amdgpu line.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-5.md`, then print it:

```
# H-5 — drafter selection: shipped MTP head vs DFlash2
## Question
## A. What is in the vendored engine (DFLASH path, MTP path, NVIDIA-only files, hot-path size)
## B. What DFlash2 is (architecture, draft cost structure, licence, inputs it needs)
## C. Cost model for this box (table: drafter, weight bytes, draft ms, acceptance, tok/s, VRAM left)
## D. Conditions for DFlash2 to win (falsifiable, each with its measurement)
## Recommendation (one paragraph, with the evidence that would change it)
## Compliance (no GPU; llama.cpp excluded; nothing installed)
## Receipts (paths and URLs)
```

Effort: **high**. This decides where the next several slices go. Read the actual files,
quote the actual licence, and do the arithmetic for a 776 GB/s card rather than repeating
an H200's numbers.
