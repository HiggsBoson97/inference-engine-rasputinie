# Probe brief — A2: in-situ rates for the four unmeasured quant types

```
STATION: probe (open item A2)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin`, card PCI 07:00.0 (`HIP_VISIBLE_DEVICES=1`) ; scratch `~/a4/`
TOOLSET: ssh, hipcc, llvm-objdump, amd-smi, python3, sudo -n
BUDGET: 120 tool calls; at ~90 % switch to write-out mode; unfinished items report as failed(budget), never omitted
ROUND: 1
CONTRACT: none — Discovery probe
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-4.md
```

## GOAL

Measure, in situ and checksum-gated, what each of the **four unmeasured quant types**
actually delivers on this card — the load floor first, then the dequant-dot rate where a
correct kernel can be gated. The project has measured only IQ3_S. 69 % of the file is
other types, and the whole-file estimate assumes they all run at 752 GB/s, which is an
assumption, not a measurement.

Types to measure, with their documented block strides (from `~/e1/mkmanifest.py`
`BLK_BYTES`, already validated structurally against the file):

| type | GGUF id | bytes / 256 weights | bits/weight |
|---|---:|---:|---:|
| IQ3_XXS | 18 | 98 | 3.06 |
| IQ2_S | 22 | 82 | 2.56 |
| IQ4_XS | 23 | 136 | 4.25 |
| Q4_K | 12 | 144 | 4.50 |

## CONTEXT

- The harness is `~/e1/iq3s_valid.cpp`, built as `~/e1/iq3s_valid`. Its `read` mode is
  **type-agnostic**: it walks block-strided rows and sums the raw bytes. Only the block
  stride is hardcoded (`#define BLK 110`). Generalising it to a runtime stride is a
  two-line change and gives you the load floor for every type at once.
- `~/e1/mkmanifest.py <gguf> <n> <out>` selects tensors and **validates
  `stride × nblocks == descriptor span` for every tensor it touches**. Extend its type
  filter rather than writing a new parser; the structural check is the thing that makes
  the manifest trustworthy, and trap 15 is that a GGUF parse is right when arithmetic
  says so.
- Measured reference points, all on this card, all in
  `docs/plan/RECORD.md`: wall 776 GB/s; IQ3_S `read` 717.7–743.7 GB/s (92.5–95.8 %);
  IQ3_S `full` 516.4–522.7 GB/s (66.6–67.4 %); clean-4-bit dequant-GEMV 752.5 GB/s.
- The two-bound model in `docs/plan/DESIGN-instruction-budget.md` is the frame: each type
  has its own instruction count per 32 weights and therefore its own memory-free ceiling
  and its own achieved rate. Report each type the same way if you can.

## RULES

1. **`HIP_VISIBLE_DEVICES=1`** — PCI `07:00.0`, headless. HIP 0 / PCI `03:00.0` is the
   display card; a load there wedged gnome-shell once. Verify by readlink before launching.
2. **llama.cpp is excluded** — no clone, no build, no source read, no derived code, not
   installed. Name it in your compliance line. Layout semantics come from
   `docs/research/2026-10-22/03c-gguf-quant-layouts.md` and from the data in the GGUF.
3. **Checksum-gate every row.** A probe without a data-dependent checksum once reported
   1784 GB/s (230 % of the wall) because the compiler hoisted the loads. A row above
   776 GB/s on a DRAM footprint is INVALID by definition.
4. **`GRIDMUL=8 BLK=256`** → `grid=336×256`. `GRIDMUL=4` silently halves the grid and
   reports 436 instead of 521. Check the harness's printed config line every run.
5. **Pin clocks** (`power_dpm_force_performance_level=high` on 07:00.0) and restore
   `auto` when done. **A kernel shorter than ~3 ms cannot be measured** — size the
   footprint and pass count so each timed region is ≥ 3 ms.
6. **Kill by PID**, never `pkill -f` (it matches your own ssh command line).
7. `~/e1/` is read-only to you: copy into `~/a4/`.
8. **No page-size or coalescing theories** — `read` is already at 92–96 % of the wall;
   the interesting question is what the *dequant* costs per type.

## WHAT TO COLLECT

**Phase 1 — the load floor (do this first, it is cheap and high-confidence).**
For each of the four types: build a manifest of real tensors of that type (≥ 1 GiB
footprint, `mkmanifest`'s structural check must print `0 mismatches`), run the
generic-stride `read` mode at `GRIDMUL=8`, 3 runs, report GB/s median and % of wall, with
the checksum verdict for every row.

**Phase 2 — the dequant-dot rate, where a correct kernel can be gated.**
For each type, write the dequant + dot kernel from the documented layout, with:
- an independent **fp64 CPU reference** in the same binary, compared per element
  (worst relative error reported, and a tolerance stated up front);
- the same data-dependent checksum gate;
- `v_dot4_i32_i8` for the dot where the type's weight layout allows it, and say what you
  used where it does not.
If a type's semantics cannot be pinned down from the documented layout, **stop on that
type, report the load floor, and say the dot rate is unmeasured** — do not guess a layout
and report a number from it. A wrong-but-fast kernel is worse than a missing row.

**Phase 3 — the instruction count.** Disassemble each dequant-dot body
(`llvm-objdump -d --mcpu=gfx1100`, delimit by the single backward branch as H-2 did) and
report **instruction words per 32 weights per lane**, split by role. That converts each
measured GB/s into an issue occupancy, which is what makes the rows comparable to IQ3_S's
193 words / 48.6 %.

## HOW TO VERIFY YOU DID IT RIGHT

- Every row's checksum reads `match`; every `read` row ≤ 776 GB/s.
- The manifest prints `0 mismatches` for the type it was built for, and the tensor count
  and footprint are stated.
- For every dot kernel: the fp64 comparison's worst relative error is reported, and the
  tolerance was stated before the run. `exp`-style losslessness is not required here; a
  bounded error is.
- A type you could not implement is reported as unmeasured with the reason — never as a
  plausible number.
- The printed `grid=336 x 256` appears in every log you cite.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-4.md` in the repo, then print it:

```
# H-4 — in-situ rates for IQ3_XXS, IQ2_S, IQ4_XS, Q4_K
## Question / hypothesis (from the ledger row A2)
## Configuration (card, clocks, GRIDMUL, footprints, manifests, pass counts)
## Phase 1 — load floor table (type, bytes/weight, GB/s, % wall, checksum verdict)
## Phase 2 — dequant-dot table (type, GB/s, % wall, fp64 worst rel err, tolerance, gate)
## Phase 3 — census table (type, instruction words / 32 weights, by role, occupancy)
## Verdict per type (measured / unmeasured-and-why; what its two-bound corner looks like)
## What this changes (which types the whole-file estimate must be re-derived from)
## Compliance (llama.cpp excluded; repo unmodified; clocks restored)
## Receipts (absolute paths under ~/a4/)
```

Effort: **high**. Four types, two phases each, every row gated. Phase 1 alone is a
complete answer to "does the layout penalise these types"; Phase 2 is what the plan
actually needs. Do not trade correctness for coverage: an unmeasured row with a reason
beats a wrong row.
