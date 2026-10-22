# 07 — Measuring correctness and quality

**Status:** research note, pre-SPEC. Everything here is either measured on this box (and labelled as such), quoted from a source with a URL, or derived arithmetically from a number in `REQUIREMENTS-DRAFT.md` (cited as `R##`).
**Question:** how does a bespoke engine prove that its kernels are right and its model is the model, cheaply enough that the proof runs on every slice?
**Short answer:** four gates at four levels, each with a tolerance derived from a measured noise floor rather than chosen by taste, and a reference built by *streaming one layer at a time* — because 27B in bf16 is 50.29 GiB and this box has 39.96 GiB of VRAM.

---

## 0. The verdict, in eight lines

1. **Dequant is exact arithmetic.** A dequant kernel gets **0 tolerance** — not 1e-2, not "close enough". Any non-zero difference is a bug, because the format is integers plus one or two multiplies and there is nothing to approximate.
2. **The GEMV's only legitimate freedom is accumulation order.** Measured here: 4.9e-7 relative. Gate at **1e-5 × RMS(y)** — 20× above the measured floor, 4×10⁴ below the quantisation noise it sits next to.
3. **Tight tolerances are only possible if the reference reproduces the kernel's precision pipeline.** A reference that keeps fp32 weights while the kernel rounds to bf16 differs by 6.3e-3 — 630× a 1e-5 gate. Specify the pipeline, then the gate.
4. **Model equivalence is a KL-and-agreement question, not a token-match question.** 32 greedy tokens is a smoke test: at a 1% per-token flip rate it fails 28% of the time, and it passes kernels that are wrong in ways argmax cannot see.
5. **The bf16 reference fits in 1.83 GiB**, because you never need the whole model resident — one decoder layer is 635 MB. Materialise the reference weights once (50 GiB on the 2 TB NVMe), hash them, stream them.
6. **A small proxy model of the same family is the cheap end-to-end oracle**; the 27B is the expensive one you run nightly, not per-commit.
7. **Task benchmarks cannot detect most engine bugs.** AIME25 has 30 problems: 30/30 has a 95% CI of [88.6, 100]. A three-point drop is noise. GPQA-D (198) and LCB v6 (1000+) are the graded sets; AIME is a smoke test.
8. **The suite must be tested.** Inject known faults into the reference dequant and require the suite to fail. A green suite that cannot fail is the most expensive artefact in the repo.

---

## 1. Four levels of "correct", and why they must not share a tolerance

| Level | Question | Oracle | What a failure means |
|---|---|---|---|
| **L0 block** | do these 110 bytes become the 256 floats the spec says? | fp64 scalar implementation of the format | the loader or the format understanding is wrong |
| **L1 kernel** | does the GEMV compute the dot product the dequant implies? | torch on the same dequantised weights | the kernel is wrong |
| **L2 layer** | does the layer compute the same function as the model code? | HF/transformers module, same weights, bf16 | fusion, layout, RoPE, mask, or state handling is wrong |
| **L3 model** | is this the same distribution? | teacher-forced logits vs a streamed reference | something upstream is wrong, or the quant is being measured instead of the engine |
| **L4 task** | is the model as good as the publisher's? | benchmark harness, paired statistics | the quantisation, the template, or the engine — in that order of likelihood |

The levels exist because **a failing tolerance at L3 tells you nothing about where the fault is.** A 1e-3 logit divergence is consistent with a wrong sign table (L0), a wrong tail (L1), a wrong mask (L2), and a wrong RoPE base (L2). Each level must be able to pass on its own, with the level beneath it held fixed. That is the whole argument for a pyramid rather than an end-to-end test.

Two reference flavours are needed at every level, and conflating them is the most common methodological error in this kind of project:

- **REF-A, matched precision.** Same dequant, same bf16 cast, same fp32 accumulation, different order. Answers *"is my kernel arithmetic the same arithmetic?"* Gate: tight, 1e-5.
- **REF-B, higher precision.** fp64 (or fp32 without the bf16 cast). Answers *"is the arithmetic right at all?"* Gate: loose, ~1e-2, and it is a **diagnostic**, not a gate — because REF-B's difference from the engine is dominated by bf16 rounding, which is not a bug.

Every tolerance below names which reference it is measured against.

---

## 2. The error budget, measured

Three noise sources sit between "the mathematically exact answer" and "what the GPU produced". Their magnitudes differ by five orders of magnitude, and the gate has to be placed in the gap.

Measured on this Mac (CPU emulation: bf16 inputs, fp32/fp64 accumulation, K=N=5120, weights N(0,1/√K), activations unit-RMS with 1% of channels at 10×). Script in Appendix A. These are *lower bounds* on GPU divergence, because a GPU adds tile-level ordering the emulation does not model.

| Source | Metric: max\|Δ\| / RMS(y) | Where it sits |
|---|---|---|
| Quantisation to 3-bit/32 (the format's own loss) | 8.4e-1 | — |
| Quantisation to 4-bit/32 | 3.5e-1 | — |
| bf16 rounding of inputs (both sides, if unmatched) | 9.5e-3 | 37× below 4-bit quant noise |
| bf16 cast of a dequantised weight, per element | 1.7e-3 RMS | 58× below 4-bit quant noise |
| **Accumulation order, fp32 accumulator, 40-way split-K** | **4.9e-7** | **2×10⁴ below bf16 rounding** |

Three consequences follow directly, and they are the spine of this document:

1. **The kernel gate lives at 1e-5**, in the gap between order noise (5e-7) and bf16 rounding (1e-2). A pass there cannot be explained by numerics; a fail there cannot be blamed on them.
2. **Any kernel that accumulates the K dimension in bf16 is disqualified**, not because it is slow but because it moves the floor from 5e-7 to 1e-2 and destroys the only cheap discriminator the project has. WMMA's fp32 accumulator is not an optimisation detail; it is what makes the test suite meaningful.
3. **The reference must cast to bf16 exactly where the kernel does.** Measured: reference-with-fp32-weights vs kernel-with-bf16-weights = 6.3e-3, which is 630× the gate. A test written that way can never pass at 1e-5, and the usual response — loosen the tolerance to 1e-2 — throws away the ability to detect a real bug. **Fix the pipeline, not the tolerance.**

The elementwise distribution matters too: the error is not uniform, so a single max is a poor statistic.

| Percentile of \|Δ\| (bf16 vs fp64, same GEMV) | value / RMS(y) | share of max |
|---|---|---|
| p50 | 1.6e-3 | 17% |
| p90 | 3.9e-3 | 41% |
| p99 | 6.3e-3 | 67% |
| p99.9 | 8.3e-3 | 88% |
| max | 9.5e-3 | 100% |

Reporting only the max means one outlier row decides the gate, and the max is the least stable statistic in the set. **Report p50, p99, p99.9, max, and the five worst rows by name.** A gate on p99.9 plus a separate, looser gate on max catches systematic error without becoming flaky.

---

## 3. L0 — block and dequant

### 3.1 What the formats actually are

The five types that carry 94.5% of the primary file's bytes (`R17`), with the block layouts that the tests must encode. Bytes per block and bpw are arithmetic from the layout, and they cross-check: `110 B / 256 w = 3.4375 bpw`, exactly the type's name.

| Type | Block | Bytes/block | Layout (from the third-party layout compendium, [dwarez](https://dwarez.dev/blog/gguf-quantization-bit-by-bit), which must be cross-checked against ISTA-DASLab's material before it becomes an oracle) | The bug it invites |
|---|---|---|---|---|
| **Q4_K** | 256 | 144 | fp16 `d` + fp16 `dmin` + 12 B of packed 6-bit scales/mins (8 sub-blocks of 32) + 128 B nibbles | 6-bit scale unpacking across byte boundaries; the min is *subtracted* |
| **IQ4_XS** | 256 | 136 | fp16 `d` + `scales_h` (2 high bits × 8) + `scales_l` (4 bits × 2/byte) + 128 B LUT nibbles | 6-bit scale split across two fields; 16-entry non-linear table indexing |
| **IQ3_S** | 256 | 110 | fp16 `d` + `qs[64]` (8 index bits per 4 weights) + `qh[8]` (9th bit) + `signs[32]` + `scales[4]` (4-bit × 8) | **the 9th index bit** lives in a separate array — the single highest-risk line in the project |
| **IQ3_XXS** | 256 | 98 | fp16 `d` + `qs[96]`: indices, then packed signs and scales | 7-bit parity signs shared between neighbours: one wrong bit flips 8 signs |
| **IQ2_S** | 256 | 82 | fp16 `d` + `qs[64]` + `qh[8]` + `scales[8]` (4-bit) | explicit sign plane + 2 extra index bits |
| **IQ1_M** (slow path) | 256 | 56 | fp16 scale packed into nibble bits, per-8 deltas | the scale is not where a scale normally is |

Two facts about all of them that the test design must respect:

- **Blocks run along rows**, and a row width must be divisible by 256; files silently fall back to another type per tensor when it is not (dwarez, §"rules of the game"). This is why per-tensor dispatch exists (`R18`) and why the type histogram test (§10, test 1) is the first test to write.
- **The reference for a block is a scalar fp64 function of the bytes.** Not a GPU kernel, not another engine. Written from the layout by someone who has not read the HIP kernel. `R42` requires a torch/HF oracle at the model level; at the block level the right oracle is stricter and cheaper: exact arithmetic in fp64.

**On llama.cpp (`R77`).** The project excludes it as baseline, reference, and reading material. Its CI's per-op, per-type backend-vs-CPU differential harness is the right *shape* for L1, and this document adopts the shape — but the harness must be written here, from the format specification, and no code or layout table may be copied from that tree. The layout table above comes from a third-party explainer and is flagged for cross-check for exactly this reason: it is evidence about the format, not a permitted source of implementation.

**An open tension the SPEC must resolve.** `R25` requires the kernels to be written "against the tensor-type block layouts in the specification", but the GGUF specification document defines the container and the type *identifiers*, not the byte layout of a `block_iq3_s`. The byte layouts live in a header inside the excluded tree. So either the SPEC names that header as the permitted specification (and `R77`'s "source is not read" is narrowed to "no kernel, loader or heuristic is derived from it"), or the project reconstructs the layouts from the publisher's material and the file itself. This is a planning decision, not a testing one — but the test oracle cannot be written until it is made, and every L0 test depends on it.

### 3.2 Tolerances

| Assertion | Tolerance | Why |
|---|---|---|
| Block byte size × n_blocks + tensor offset == tensor byte length | **exact** | catches a layout misunderstanding before any kernel exists |
| Dequant, per element, vs fp64 reference | **≤ 0.5 ULP of fp32**, i.e. correctly rounded | the arithmetic is exact; only rounding differs |
| Dequant, per element, kernel vs fp64, when both use the same formula order | **0 mismatches** | achievable; require it |
| Codes consumed by the kernel == codes in memory | **exact** (debug kernel dumps q, d, sc) | catches "right answer, wrong path", which passes today and fails on the next file |
| NaN/Inf in output | **zero** | a dequant of finite bytes is always finite |
| Re-run determinism | **bitwise identical** | any nondeterminism at L0 makes every golden test worthless |

The "0.5 ULP" line is the one people argue about. It is defensible because the alternative — a relative tolerance like 1e-2 — accepts a kernel that returns the adjacent quantisation level: a full grid step, which for weights near the bottom of a block's range is a ~100% relative error on that element. That is precisely the error the format was designed to bound, and a test that tolerates it is not testing the format. Dequant has no approximation in it; the tolerance should say so. If FMA contraction makes bit-exactness awkward, compile the kernel with `-ffp-contract=off` for the test build or write the multiply explicitly; do not widen the tolerance.

### 3.3 Edge cases that random data never reaches

- **Zero and denormal `d`.** fp16 has subnormals down to 6e-8. A block whose `d` is subnormal is legitimate in a real file. Document what the kernel does (flush to zero is acceptable if documented and matched by the reference) and test both.
- **Overflow.** Synthetic block with `d = 65504` (fp16 max) and all codes at maximum. In fp32 accumulation this is fine; in an fp16 accumulator it becomes `inf`. **This test detects an fp16 accumulator that no realistic tensor would expose.**
- **Outlier blocks.** Take the 0.1% of blocks with the largest max\|w\|/d ratio from the real file. Inside them, small weights legitimately quantise to 0. Assert: the reconstruction matches the codes exactly, and the *dot product* error for that row stays within gate. Do not assert per-element relative error on those weights — the denominator is ~0 and the metric is meaningless.
- **Sign patterns.** Set `signs` to a known alternating pattern and require the exact negation of the unsigned dequant. A sign-indexing bug then shows as an error of magnitude 2\|w\|, not as a small drift.
- **Tail shapes.** K not a multiple of 256 (the last partial block), N not a multiple of the tile, and a 1-row GEMV. On this model K ∈ {5120, 17408} and both divide 256, so **a tail bug will not appear on the primary file** — which is exactly why it must be tested synthetically before the MTP head or a different quant file arrives.

---

## 4. L1 — the GEMV and GEMM kernels

### 4.1 Metrics

For output vector `y` (M rows), reference `ŷ`, RMS over the tensor:

| Metric | Gate | Notes |
|---|---|---|
| `max abs(y-ŷ) / RMS(ŷ)` | ≤ 1e-5 | the primary gate |
| `mean abs / RMS` | ≤ 1e-6 | catches a small bias everywhere, which a max ignores |
| `1 - cos(y, ŷ)` | ≤ 1e-6 | scale-free; catches a rotated result that per-element gates can miss |
| per-row `max abs / RMS(row)` | ≤ 1e-4 for every row | a single bad row must not hide in a tensor average — this is the outlier-block test at kernel level |
| NaN/Inf | 0 | |
| rerun | bitwise identical | |
| batch invariance: row 0 at M=1 vs M=8 | ≤ 1e-6 | see §4.3 |

`torch.testing.assert_close` with `rtol=1e-5, atol=0` on the dequantised-scale tensor is a reasonable encoding of the first two rows; note that PyTorch's *default* bf16 tolerance is `rtol=1.6e-2, atol=1e-5` ([docs](https://docs.pytorch.org/docs/stable/testing.html)), which is 4.1 bf16 ulps and is **10³ too loose for this gate**. The framework default is calibrated for "any two reasonable bf16 implementations", not for "my kernel against itself". Cite the number, don't inherit it. Triton's own tests follow the same convention — `assert_close(..., atol=1e-2, rtol=0)` for a bf16 attention kernel ([tutorial](https://triton-lang.org/main/getting-started/examples/gluon/attention-forward.html)) — and that is the right convention for *two different implementations of one op*, which is exactly what a Triton fallback is (`R37`). It is the wrong convention for a kernel against the oracle it was derived from.

**What `opcheck` covers, and what it does not.** If the HIP kernel is registered as a torch custom op — which the fork needs for graph capture (`R34`, `R62`) — `torch.library.opcheck` validates the *contract*: schema, fake kernel, mutation and aliasing declarations, behaviour under `torch.compile`. Its documentation is explicit that it "is not a numerical correctness test" ([opcheck](https://docs.pytorch.org/docs/stable/library.html), [tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)). Run it anyway, on representative inputs per dtype/device/shape: it catches the class of bug where the kernel is numerically right and the wrapper is wrong — a mutating op declared functional, an output that aliases its input, a fake kernel with the wrong shape. Two failure surfaces, two tests.

### 4.2 The shape matrix

Twelve shapes per type for the per-commit gate — a covering set drawn from this grid, not the full cross product:

```
M ∈ {1, 2, 8, 64}          decode, small batch, prefill, chunked prefill
K ∈ {5120, 17408, 5121}    attention/MLP, and a deliberately ragged K
N ∈ {5120, 17408, 248320}  and the LM head, which is the largest single read
```

That is 12 per type and 72 cases in total; at ~10-30 ms each it is a few seconds of GPU time for the numerical checks, which is why it can run on every commit. The full cross product is 36 per type (216 cases) and belongs in the nightly job. The M sweep is the part that is not optional: the WMMA crossover (`R39`) means the M=1 path and the M=64 path are *different kernels*, and a suite that only tests M=64 ships an untested decode path.

### 4.3 Batch invariance is a correctness property, not a performance one

Thinking Machines Lab's analysis is the authoritative statement: kernels are usually run-to-run deterministic but not **batch-invariant**, so the same request returns different logits depending on how many other requests were in the batch, and that is why inference servers are not reproducible end-to-end ([He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). Their demonstration does 1-row vs full-matrix matmul of the same data and gets `max |Δ| = 1669.25` on the example tensor.

This project has two reasons to care more than a normal engine:

- Paged KV, continuous batching and chunked prefill (`R55`) mean the batch composition changes constantly, so a non-batch-invariant kernel makes **every** measurement unreproducible.
- Graph capture (`R62`, 41 s) fixes launch order but not reduction order; a captured graph replays the same kernels at whatever batch size it was captured for.

So: **the same row, computed at M=1 and at M=8, must agree to 1e-6.** This is cheap (two launches) and it is the only test that prevents the entire class of bugs the batch-invariance literature describes. It should also be asserted at the model level (§6.4), because a layer can be invariant while the batching logic in the scheduler is not.

If the project later adopts split-K for the M=1 GEMV — and the bandwidth argument in `R40`/`R44` suggests M=1 is where the effort goes — this test is what stops split-K from silently becoming a different numerical function per occupancy.

---

## 5. L2 — layers, attention, and the recurrent path

### 5.1 Per-op oracles

The model has four op families with non-trivial numerics; each gets its own oracle and its own tolerance, because they fail differently.

| Op | Reference | Tolerance | Why that number |
|---|---|---|---|
| RMSNorm | fp32 torch, same eps | rtol 1e-3 | a reduction over 5120 elements; fp32 accumulation makes it tight |
| RoPE | fp32 torch, same `rope_theta`, same dim-64 partial rotation | 1e-3 rel | partial rotary (64 of 256) is a classic mis-index; the test must include a position > 4096 to catch a wrong base |
| Gated attention | `torch.nn.functional.scaled_dot_product_attention` in fp32, then cast | 1e-2 rel on output, **1e-3 on the softmax denominator** | bf16 attention legitimately differs across implementations at the 1e-2 level; the normaliser is far more sensitive than the output and localises a mask bug |
| **Gated DeltaNet recurrence** | fp32 torch, layer by layer, state captured | 1e-3 rel per step; **drift gate over 8K tokens** | the state is a recurrence: error compounds with *time*, not just depth |

The GDN path deserves its own paragraph. 48 of 64 layers carry a 151 MB fp32 recurrent state (`R29`), and the state is a function of every token seen so far. Two consequences:

- **A per-token tolerance is insufficient.** The right assertion is on the *state trajectory*: capture the state after 128, 1024, 4096, 8192 tokens and compare each against the fp32 reference. A kernel that is 1e-4 wrong per step and unstable will look perfect at 128 tokens and be useless at 8K.
- **The failure signature is length-dependent, not depth-dependent.** If the engine passes at 512 tokens and fails at 8K, the fault is in the recurrence, the RoPE base, or the conv state — not in the GEMV. Put that mapping in the playbook (§11).

### 5.2 What "fused" changes

Each fusion (RMSNorm+quantise, RoPE+attention, dequant+matmul+silu) removes an intermediate that the reference materialises. Test the fused op against the *composed* reference, not against the reference of the last stage, and test the fusion boundary explicitly: dump the intermediate in a debug build and compare it to the reference's intermediate at 1e-5. Otherwise a fusion bug hides inside an acceptable end-to-end number.

---

## 6. L3 — model equivalence

### 6.1 The oracle hierarchy, strongest first

| # | Oracle | Cost | Sensitivity | Use |
|---|---|---|---|---|
| 1 | **Per-layer hidden-state equivalence**, teacher-forced | one streamed pass | catches the layer where it broke | primary diagnostic |
| 2 | **Full-logit KL + top-k agreement**, teacher-forced | one pass | catches anything that changes the distribution | **primary gate** |
| 3 | Teacher-forced perplexity on a fixed corpus | one pass | scalar, robust, insensitive | regression trend |
| 4 | 32-token greedy string match (`R73`) | seconds | flaky, insensitive | smoke test only |
| 5 | Task benchmarks | hours | quality, not correctness | §7 |

The ordering is by *sensitivity per unit cost after the reference exists*, not by cost. Once the reference logits are cached, #2 is a file comparison and #1 is a handful of matrix comparisons; #4 is only cheaper by seconds and is strictly worse.

### 6.2 Why greedy-token matching is the wrong gate

`R73` proposes 32 greedy tokens matching a torch reference exactly. It is a good smoke test and a bad gate, for three measurable reasons:

**It is flaky.** If a per-token flip probability of p arises from numerical noise, the probability that all 32 tokens match is (1-p)³²:

| p (flips/token) | P(32/32 match) |
|---|---|
| 0.1% | 0.968 |
| 0.5% | 0.852 |
| 1% | 0.725 |
| 2% | 0.524 |
| 5% | 0.194 |

A 1%-per-token divergence — entirely acceptable numerically, and expected if the reference and engine differ in batch size — fails the gate 28% of the time. A gate that fails a quarter of the time on correct code gets loosened or ignored within a week.

**It is insensitive.** A kernel that is wrong only on the tail rows, or wrong in channels that never dominate an argmax, passes 32 tokens and fails nothing. Quantisation error is ~10% of the logit vector's norm and still preserves argmax most of the time; that is the whole premise of 3.5 bpw. A test whose signal is "argmax unchanged" cannot distinguish a 10% error from a 1% one.

**It conflates one flip with a cascade.** Autoregressive generation is chaotic: one flipped token changes the context and every subsequent token. The test reports "mismatch at token 7" whether the model is subtly off or catastrophically broken.

Keep it as `make smoke` (30 s, useful when the GPU is misbehaving). Do not let it gate a merge.

### 6.3 Teacher forcing, and the metrics that replace token match

Teacher forcing fixes the token sequence: run both engine and reference on the same prompt *and* the same continuation, and compare the distribution at every position. Then a single numerical blip is one bad position, not a divergent trajectory, and the metrics are stable and additive.

| Metric | Gate (engine vs REF-A, same weights) | Reasoning |
|---|---|---|
| mean KL(ref‖engine), bits/token | **≤ 1e-3** | σ=1e-2 nats of logit noise gives 7e-5 bits; 1e-3 is 14× above that and 100× below the quantisation effect (~1e-2 bits) |
| p99.9 per-position KL | ≤ 1e-2 bits | catches a rare broken position that the mean hides |
| top-1 agreement | ≥ 99.9% over ≥50k positions | 50 flips in 50k; a 1% flip rate fails immediately |
| top-16 overlap | ≥ 99% per position | catches rank corruption below the top-1 |
| max\|Δlogit\| | ≤ 1e-2 nats | absolute sanity, independent of scale |
| teacher-forced PPL ratio | ≤ 1.001 | the scalar everyone will quote; tight because the weights are identical |
| per-layer 1−cos, final hidden state | ≤ 1e-5, monotone growth | a jump at one layer localises the fault to that layer |

Against REF-B (fp64) the same model gives 1e-2-level logit differences and is **not** a gate — it is the sanity check that the bf16 pipeline is doing what bf16 does.

For the quality question (engine-on-quant vs publisher-on-quant) the KL will be ~1e-2 to 1e-1 bits, dominated by the format. That is §7's problem, not this gate's.

### 6.4 Pitfalls, with the mitigation

| Pitfall | What actually goes wrong | Mitigation |
|---|---|---|
| **Reference nondeterminism** | torch/hipBLASLt picks a different reduction order per shape; the "reference" is a moving target | pin torch + ROCm versions in the golden manifest; generate goldens once, compare against the *file*, never re-run the reference in the test |
| **Batch-size dependence** | reference at batch 1 vs engine at batch 8 differ by 1e-2 for reasons that are not bugs (He, 2025) | run both at M=1 and at the same batch shape; assert batch invariance separately (§4.3) |
| **bf16 accumulation order** | different tile orders on two cards (`R3`: no P2P, host-bounce all-reduce) | TP=2 is a separate golden set; never compare a TP=2 run against a TP=1 golden |
| **Sampling temperature** | publisher numbers are at temp 1.0 ([Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)); a greedy run is a different experiment | fix the protocol per §7.2; never compare across temperatures |
| **Chat template / tokenizer** | silent, fluent, wrong: the model answers a differently-formatted prompt | byte-compare token ids for 200 prompts before any quality claim (§10, test 8) |
| **Thinking budget** | truncated reasoning tanks AIME and looks like a quant failure | record `max_tokens`, `reasoning_effort`, `preserve_thinking` per run |
| **KV/state reuse across tests** | a cached prefix or a stale recurrent state contaminates the next comparison | fresh state per test; assert state hash after reset |
| **Golden laundering** | someone regenerates the golden because the test "was wrong" | regeneration requires an explicit flag and a recorded reason (§9) |

The nondeterminism point is worth restating because it changes the test architecture: **the reference is run once and its outputs are frozen.** A test that re-runs the reference on every invocation is testing two moving objects against each other. Freeze, hash, compare.

---

## 7. L4 — quality against the publisher

### 7.1 The three-way comparison, not two-way

The publisher's quantised card claims AIME25 100.00, LCB v6 85.71, GPQA-D 89.39 at 3.50 bpw (`R28`). The base model's own card claims GPQA Diamond 89.2 and LiveCodeBench v6 90.3 ([Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)). Read together, those numbers say something the project should internalise before it runs a single benchmark:

**The quantisation already costs ~4.6 points on LCB v6 (90.3 → 85.71) and nothing measurable on GPQA-D (89.2 → 89.39).** So "matching the publisher" is a *different bar per benchmark*, and a single "we are within X of the publisher" statement is meaningless. The target is per-benchmark:

| Benchmark | bf16 publisher | quant publisher | our target | what a miss means |
|---|---|---|---|---|
| GPQA-D (198) | 89.2 | 89.39 | ≥ 87 (within noise) | quant path or engine is damaging knowledge retrieval |
| LCB v6 (~1055) | 90.3 | 85.71 | ≥ 84 | the 4.6-point gap is the format's; a further gap is ours |
| AIME25 (30) | not stated | 100.00 | 27/30 or better | see §7.3: this set cannot resolve small differences |

### 7.2 Fairness protocol

A benchmark number is a tuple, not a scalar. The tuple must be pinned:

1. **Harness and version.** lm-evaluation-harness (or the publisher's harness if named) at a commit, with the task definition and few-shot count recorded. The harness's own docs note the chat template is saved with results for reproducibility ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)).
2. **Prompt rendering.** The chat template from the model's own tokenizer config, byte-identical. Tested, not assumed (§10, test 8).
3. **Sampling.** Thinking mode: `temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1.0`; non-thinking: `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5` ([card](https://huggingface.co/Qwen/Qwen3.8-27B)). If the run is greedy, it is a **different experiment** and must be labelled as such.
4. **Reasoning budget.** `reasoning_effort` (`xhigh` default) and `max_tokens`. A truncated trace is indistinguishable from a worse model.
5. **Number of samples.** pass@1 averaged over k samples, or greedy — the publisher's k must be matched. Reasoning benchmarks move several points with k.
6. **Answer extraction.** A regex change can move AIME by 10 points. Pin the extractor's commit.
7. **Seed**, even though sampling at temp 1.0 makes exact reproduction impossible: record it so a rerun is comparable in distribution.

### 7.3 Statistics: what these sets can and cannot resolve

Wilson 95% intervals, computed (Appendix A):

| Result | 95% CI | width |
|---|---|---|
| AIME25 30/30 = 100% | [88.6, 100.0] | 11.4 pp |
| AIME25 26/30 = 86.7% | [70.3, 94.7] | 24.4 pp |
| GPQA-D 177/198 = 89.39% | [84.3, 93.0] | 8.6 pp |
| GPQA-D 170/198 = 85.86% | [80.3, 90.0] | 9.7 pp |
| LCB v6 904/1055 = 85.69% | [83.4, 87.7] | 4.2 pp |

Independent CIs overlap almost always, so the correct test is **paired**: same questions, same harness, compare per-question outcomes with McNemar's exact test. On a 30-question set, the minimum detectable effect (6 discordant questions all one way, p = 2/2⁶ = 0.031) is **6 problems = 20 percentage points**. On GPQA-D's 198, 6 flips give p = 0.031 and 7 give p = 0.016 — a 3-point difference is detectable. On LCB v6's ~1000, a 2-point difference is detectable.

**Operational conclusions:** AIME25 is a smoke test and a headline, never a discriminator. GPQA-D is the workhorse. LCB v6 is the sensitive one but the slowest and the most template-dependent (the 4.6-point quant gap says it is the benchmark most sensitive to weight precision). Report all three with CIs and the paired test; a bare point estimate is an opinion.

### 7.4 Broken, or merely mediocre?

The distinction is not a feeling; it has signatures. Ordered by how cheaply they can be checked:

| Signature | Broken | Mediocre |
|---|---|---|
| NaN/Inf in logits, KV, or recurrent state | always | never |
| Repetition loops / token salad at temp 0 | yes | no |
| Per-type error attribution: one GGUF type's GEMV 10-100× the others' error | yes | no — all types degrade together |
| Teacher-forced PPL vs **matched** reference | > 1.02 | ≤ 1.005 |
| KL vs matched reference | > 1e-2 bits/token | ≤ 1e-3 |
| Copy/count canary prompts (§7.5) | degraded at 200 tokens | intact |
| Degradation is uniform across categories | no — usually concentrated | yes |
| Benchmark delta is inside the paired CI | — | yes |

The decisive experiment, when the signatures disagree: **run the same engine with the dequantised bf16 weights loaded through the same loader.** If quality returns to publisher level, the fault is in the kernel or the quantised path. If it does not, the fault is in the loader (wrong type, wrong layout, wrong scale) or in the model code — and the L0/L1 tests will already have said which.

### 7.5 Canary prompts

Ten prompts that require no reasoning and are maximally sensitive to state corruption:

- copy a 200-token random string verbatim;
- count 1..100, then 100..1;
- repeat a 40-digit number, then reverse it;
- answer a 3-hop question whose facts are in the prompt 4K tokens back;
- continue a fixed 2K-token code file exactly.

A 3.5 bpw model should pass all of these nearly perfectly. A broken recurrent state, a wrong RoPE base, or a KV indexing bug fails them at length while leaving benchmark scores almost intact (because benchmarks mostly test short-context knowledge). They cost seconds and they localise the fault to "long context / state" rather than "knowledge".

---

## 8. The reference implementation on 40 GB

### 8.1 The arithmetic that rules out the obvious approach

| Quantity | Value |
|---|---|
| 27B at bf16 | **50.29 GiB** (54.0 GB) |
| VRAM, two cards | 39.96 GiB |
| Shortfall | **10.33 GiB** — bf16 does not fit, at any sharding |
| 27B at IQ3_S (3.5 bpw) | 11.00 GiB — fits one card, which is why the project exists |

So the reference cannot be "load the model in bf16 and run it". Options and verdicts:

| Option | Cost | Verdict |
|---|---|---|
| **Layer-by-layer streaming** (one decoder layer resident) | 635 MB weights + activations ≈ **1.83 GiB peak**; 11.4 s per pass from NVMe | **chosen** |
| CPU-only bf16 (60 GiB RAM holds 50.29) | 1.08 s/pass at RAM bandwidth *plus* compute: 54 GFLOP/token at CPU rates ⇒ minutes per 512-token prompt | block/tensor oracles only, not model-level |
| Smaller proxy model, same family | e.g. a 0.6-4B sibling: bf16 fits one card with room for KV | **chosen**, as the fast end-to-end oracle |
| Numerical double-check on random tensors only | trivial | necessary but insufficient — proves the arithmetic, not the plumbing |
| TP=2 bf16 across both cards | 50.29 GiB into 39.96 GiB | impossible |

### 8.2 The recommended stack: four references and a harness

The tiers are named `REF-0..REF-4` so they cannot be confused with the `R##` requirement ids.

```
REF-0  fp64 scalar dequant (numpy, CPU, ~200 lines, written from the spec)
      └─ oracle for L0. Runs on the Mac. Seconds. No GPU.

REF-1  torch layer oracle (bf16, GPU, one layer at a time, streamed from ref_bf16.safetensors)
      └─ oracle for L1/L2. Peak 1.83 GiB. One card (HIP_VISIBLE_DEVICES=1, R9).

REF-2  golden trajectory (generated ONCE by REF-1, frozen, hashed)
      └─ oracle for L3 in CI. 32 prompts x 3 lengths; per-layer hidden-state
         summaries + top-32 logits. ~50 MB. Comparison is a file diff.

REF-3  proxy model end-to-end (same family, bf16 and quantised, both fit one card)
      └─ catches plumbing/model-code bugs at 1/100th the cost of the 27B.

REF-4  task harness (publisher protocol, §7.2)
      └─ L4 only. Nightly/weekly. Never the first signal.
```

The key move is **REF-2: pay the 12 minutes once.** A full streamed pass over 64 prompts at NVMe speed is 12.2 minutes (64 × 11.4 s), and it produces an artefact that makes every subsequent model-level test a file comparison rather than a forward pass. Without it, every CI run pays 12 minutes and every sporadic numerical difference becomes a debate about whether the reference moved.

**Why layer streaming works and what it costs.** The largest layer is the MLP (267 M params, 498 MB bf16); attention projections add 137 MB. Holding one layer plus activations and a 248,320-wide logit slice peaks under 2 GiB, so the card is never the constraint — the NVMe is, at 11.4 s per full pass. That is the entire trick: **the model does not fit, but a layer does, and the reference is a composition of layers.**

**Materialise the reference weights once.** Dequantise the GGUF to `ref_bf16.safetensors` (50.29 GiB on the 2 TB drive), produced by **REF-0**, not by the engine's loader, and hash it. Then:

- REF-1 never re-derives weights, so a loader bug cannot contaminate the reference;
- "same weights" is a hash equality check, not an assumption;
- the engine's loader is tested *against* the reference, which is the correct direction of the arrow.

This is the single most important structural decision in this document, because it separates "the weights are right" from "the kernels are right". Every project that skips it spends weeks unable to tell the two apart.

### 8.3 Artifact sizing

Full fp32 logits for a 512-token prompt at a 248,320 vocabulary are 508 MB — too big to keep for 32 prompts. Store:

- **top-32 logits + log-sum-exp** per position: 512 × 33 × 8 B ≈ 135 KB per prompt, and enough to compute KL to within the top-32 mass (assert that the top-32 mass is > 0.999 for the prompts chosen; if not, store top-128).
- **per-layer hidden-state summary**: RMS, mean, and 64 fixed probe dimensions + a hash of the full tensor. A hash equality gives bit-exactness; the probes give a magnitude when it fails.
- **the full tensor for 2 prompts**, for debugging (508 MB × 2, on disk only).

Total: ~50 MB checked in, plus a few hundred MB of debug tensors outside git.

### 8.4 What the reference cannot prove

- It shares the model code with the engine (`R34` inherits model code), so a bug in the *shared* code is invisible to it. That is what the proxy model and the publisher benchmarks are for: they are independent of our code.
- It is bf16, so it cannot adjudicate a bf16-level numerical argument. REF-0 can.
- It is one sequence at a time, so it says nothing about batching. §4.3 covers that separately.

---

## 9. Regression gating: making the suite impossible to rot

### 9.1 The pyramid, with cadence and cost

| Gate | Contents | Time | Cadence | Blocks merge |
|---|---|---|---|---|
| **G0** `make check-blocks` | REF-0 vs golden blocks, all 6 types, edge cases | < 5 s, CPU | pre-commit | yes |
| **G1** `make check-kernels` | 72 shape cases, determinism, batch invariance, overflow | < 60 s, 1 GPU | every commit | yes |
| **G2** `make check-layers` | per-op + GDN trajectory | < 10 min | every PR | yes |
| **G3** `make check-model` | REF-2 golden trajectory: KL, agreement, hidden states | < 2 min (file diff) | every PR | yes |
| **G4** `make check-quality` | canaries + 3 benchmarks, paired stats | hours | nightly / weekly | no (records) |
| **G5** `make mutants` | inject faults, require the suite to fail | ~15 min | weekly | no (records) |

The ordering is chosen so that the fastest gate catches the most common fault. G0 catches format misunderstandings, which are the most common and the most expensive; G5 catches the case where the suite has quietly stopped testing anything.

### 9.2 Golden tensors

- **Blocks**: 4 real blocks per type from the primary file + 6 synthetic edge blocks (zero d, denormal d, max d, outlier, alternating signs, tail). Store input bytes, expected fp32 output, sha256 of both. Total < 1 MB. This is the highest-value checked-in artefact in the repo.
- **Kernel outputs**: for 3 of the 72 shape cases, store the full output tensor (5120 × 4 rows of fp32 = 80 KB each) and require **bitwise** equality on the same GPU/driver. Bitwise is legitimate here because the kernel is deterministic and the input is fixed; a tolerance is only needed when something is allowed to change.
- **Logits**: REF-2's top-32 + logZ, float32, with the manifest.

Every golden carries a manifest: git SHA of the engine, ROCm/hipcc version, torch version, GPU PCI bus id, the command that produced it, and the seed. A golden without a manifest is a rumour.

### 9.3 Golden hygiene

The failure mode is not a missing golden; it is a golden regenerated to match a bug. Mitigate structurally:

- `make golden` is a separate target that refuses to run without `--reason "..."`
- the reason is written into the manifest and into the commit message;
- a PR that changes a golden without changing the code that produced it fails review by rule;
- the golden diff is shown in CI output, not buried.

### 9.4 Property-based tests and fuzzing

Hypothesis (or a simple loop) over block payloads, asserting properties that hold for *all* inputs:

| Property | Assertion |
|---|---|
| Dequant is total | any payload of the correct block size with finite `d` produces finite output |
| Codes are respected | the set of reconstructed values is a subset of the format's representable set for that block |
| Sign symmetry | flipping all sign bits negates the unsigned reconstruction exactly |
| Scale monotonicity | doubling `d` doubles the reconstruction exactly (for type-0 formats) |
| Zero codes | a block whose codes are all zero reconstructs to exactly 0 (or exactly the min term, for type-1) |
| Idempotence | quantise(dequantise(x)) == the codes, for uniform formats only — **not** for lattice formats, and asserting it there would be a false test |

**Fuzzing the memory safety**: allocate each tensor buffer with 4 KB sentinels on both sides filled with a known pattern; after the kernel, assert the sentinels are unchanged and hash-equal. On ROCm there is no compute-sanitizer equivalent for race detection, so sentinels plus `rocgdb` on a failing case is the practical substitute. A one-byte over-read at a block boundary is invisible numerically on the primary file (K is a multiple of 256) and becomes a wrong answer on the next one.

### 9.5 Mutation testing: does the suite have teeth?

The suite is an artefact and needs its own test. Weekly, inject these mutants into the **reference** dequant (not the kernel — the reference is the oracle, and a suite that does not detect a broken oracle detects nothing):

1. drop the 9th index bit (`qh`) in IQ3_S;
2. swap the two 4-bit scales in a byte;
3. read `scales_l` where `scales_h` belongs in IQ4_XS;
4. omit the `dmin` subtraction in Q4_K;
5. off-by-one in the sign plane index for IQ2_S;
6. one entry of the 16-value LUT changed by 1;
7. `d` read as fp32 instead of fp16 (offset shift);
8. accumulation in fp16 instead of fp32;
9. dropped FMA (mul then add, rounding between);
10. tail row read from row 0 (off-by-one in the N tail).

Requirement: **every mutant is caught by G0 or G1.** If mutant 9 is not caught, the gate is not measuring accumulation; if mutant 5 is not caught, the sign test is decorative. This is the cheapest way to find out that a tolerance is 100× too loose, and it is the difference between a suite that proves something and a suite that runs.

### 9.6 The binding rule

`R36` says a kernel family's row closes only when its native path passes its test and its predecessor is deleted. `R41` says every kernel gets a correctness test before a speed test. `R65` says a number without a baseline is an opinion. The gate that makes all three mechanical:

> **A performance number is quoted only with the hash of the passing correctness run that produced it, in the same artifact.**

Concretely: the benchmark harness writes `{git_sha, kernel, shape, correctness_receipt_sha, gbps}`; the report generator refuses to emit a row whose receipt is missing or stale. That is a 50-line script and it is the only thing that prevents the project's headline claims from drifting away from its tests.

---

## 10. What to write first

Ordered by "what unblocks the most, soonest". Tests 1-4 do not need a GPU.

1. **`test_gguf_type_histogram`** — parse the primary file, assert the per-tensor type histogram and the total tensor byte length (`Σ blocks × bytes/block + header == file size`). *This is the single highest-value test in the project*: it validates the format understanding before any kernel exists, and it re-derives the numbers `R17` is built on. If it fails, everything downstream is measuring the wrong bytes.
2. **`dequant_ref.py` + `test_dequant_blocks`** — fp64 scalar dequant for the six types, against hand-computed blocks and the real file's first block per type. Write it from the layout, not from the kernel.
3. **`dump_ref_bf16.py`** — materialise `ref_bf16.safetensors` from the GGUF via test 2's dequant, hash every tensor, write the manifest. 50 GiB, one-time, off the critical path.
4. **`test_golden_blocks`** — the checked-in golden block set (real + synthetic edge cases), run by G0 in 5 s.
5. **`test_gemv_vs_torch`** — the 72-case per-commit matrix (216 nightly), REF-A semantics, gates from §4.1. First GPU test; this is the gate the first kernel must pass before it is allowed a bandwidth number (`R41`).
6. **`test_determinism_and_batch_invariance`** — rerun bitwise; M=1 vs M=8 row equality at 1e-6.
7. **`test_tail_and_overflow`** — K=5121, N=1, d=65504, denormal d, sentinel buffers.
8. **`test_tokenizer_template_parity`** — 200 prompts rendered by the HF tokenizer and by the engine, token ids byte-identical.
9. **`test_layer_equivalence`** — REF-1, per layer, hidden states and the GDN state trajectory at 128/1024/4096/8192 tokens.
10. **`test_golden_trajectory`** — REF-2, the L3 gate (KL, top-1, per-layer cosine).
11. **`test_canaries`** — the ten copy/count/long-recall prompts.
12. **`test_mutants`** — the ten mutants, wired into G5.

Tests 1-3 are a day's work and they make everything after them cheap. Tests 5-7 are the first kernel's gate. Tests 8-12 are the model gate; 8 in particular costs an hour and prevents the most embarrassing class of "quality" failure.

---

## 11. Debug playbook: symptom → cause

| Symptom | First suspect | Confirm with |
|---|---|---|
| error ≈ 2\|w\| on a subset of elements | sign indexing (IQ3_S `signs`, IQ3_XXS parity) | alternating-sign test (§3.3) |
| error grows with sub-block index within a block | 6-bit scale packing (Q4_K, IQ4_XS) | dump `sc` per sub-block, compare to REF-0 |
| error only for K not divisible by 256 | tail handling | K=5121 case |
| one tensor type bad, others fine | that kernel, not the loader | per-type attribution over the real file |
| all tensors of one type zero or garbage | `d` parsed at the wrong offset or dtype | first-block dump |
| error grows slowly over layers, no spike | accumulation precision, RoPE | REF-B comparison, per-layer cosine |
| error grows with token position, not depth | GDN recurrence, KV indexing, RoPE base | state trajectory at 4 lengths |
| fluent but wrong-format output | chat template / tokenizer | token id parity test |
| benchmark drop concentrated in code tasks | template, stop tokens, thinking budget | canaries + LCB rerun at matched protocol |
| results differ between batch sizes but not within | batch invariance | §4.3 |
| results differ run-to-run | atomics, uninitialised memory | rerun hash; sentinels |

Bisection: to localise a model-level fault to a layer in log₂(64) = 6 runs, substitute the reference hidden state at layer k into the engine and check whether the final logits recover. That is the fastest known method and it only works if the reference exposes per-layer states — which is why REF-1 captures them.

---

## 12. Threats to validity

- **The probe numbers are CPU emulation.** They bound the magnitudes and set the gates' order of magnitude; the GPU's actual split-K and tile ordering must be measured on the box before the 1e-5 gate is written into the SPEC. If the real floor is 1e-4, record why and set 1e-3 — but record it, because a tolerance with no stated floor is how projects end up at 1e-2 and lose the ability to see a bug.
- **The layout table is secondary.** It is consistent (bytes/block ⇒ bpw checks out to four decimals) but it is a third-party description; cross-check against the publisher's format material and the file's own header before it becomes an oracle (§3.1, open tension). A wrong oracle is worse than no oracle.
- **GPQA-D's n=198 is the standard diamond subset; take the denominator from the harness's copy, not from this document.** Every CI in §7.3 is computed from the stated n; if the harness ships a different count, recompute.
- **AIME25's publisher number (100.00) may not be a greedy pass@1.** If the protocol is unknown, the comparison is unknown. That is not a reason to skip the benchmark; it is a reason to report it as "not comparable" rather than as a delta.
- **The proxy model is a proxy.** It shares architecture and code, not scale. It catches plumbing, not numerical scaling; do not let it substitute for the 27B nightly.
- **No ROCm equivalent of compute-sanitizer** is assumed here. If one exists in ROCm 10.1, use it in place of the sentinel scheme for G1.

---

## Appendix A — the probe

Run on the Mac (Python 3.14, torch 2.12 CPU, numpy 2.2.6), 2026-10-22. Reproduces §2's table, §6.2's flip arithmetic and §7.3's intervals.

```python
import math, torch
from math import comb, sqrt
torch.manual_seed(0); torch.set_grad_enabled(False)
K = N = 5120

# realistic GEMV: gaussian weights, unit-RMS activations with 1% channels at 10x
w = torch.randn(N, K, dtype=torch.float64) / math.sqrt(K)
x = torch.randn(K, dtype=torch.float64)
sc = torch.ones(K, dtype=torch.float64); sc[torch.randperm(K)[:51]] = 10.0
x = x * sc
y64 = w @ x
wb, xb = w.to(torch.bfloat16), x.to(torch.bfloat16)
y_ref = wb.to(torch.float64) @ xb.to(torch.float64)          # single accumulation order

# 40-way split-K in fp32, then sum the partials: a different legal order
parts = [wb[:, i*128:(i+1)*128].to(torch.float32) @ xb[i*128:(i+1)*128].to(torch.float32)
         for i in range(40)]
y_ord = torch.stack(parts).sum(0).to(torch.float64)

def rel(a, b):  # max|d| / RMS(reference)
    return ((a - b).abs().max() / b.pow(2).mean().sqrt()).item()

print("bf16 rounding :", rel(y_ref, y64))   # ~9.5e-3
print("split-K order :", rel(y_ord, y_ref)) # ~4.9e-7

def qdq(w, block=32, bits=4):                # uniform block quant, as a noise stand-in
    W = w.reshape(-1, block)
    d = W.abs().amax(1, keepdim=True) / (2**(bits-1) - 1)
    q = torch.round(W/d).clamp(-(2**(bits-1)-1), 2**(bits-1)-1)
    return (q*d).reshape(w.shape)
for bits in (3, 4):
    print(f"{bits}-bit quant  :", rel(qdq(w, 32, bits) @ x, y64))  # 8.4e-1 / 3.5e-1

print("P(32/32 tokens match) =", [(p, round((1-p)**32, 3)) for p in (0.001, 0.005, 0.01, 0.02)])

def wilson(k, n, z=1.959964):
    ph = k/n; den = 1 + z*z/n
    c = (ph + z*z/(2*n))/den
    h = z*sqrt(ph*(1-ph)/n + z*z/(4*n*n))/den
    return c-h, c+h
for k, n in ((30,30), (26,30), (177,198), (170,198), (904,1055)):
    lo, hi = wilson(k, n); print(f"{k}/{n}: [{lo*100:.1f}, {hi*100:.1f}]")

for b in range(4, 9):   # exact McNemar, all discordant one way
    print(f"b={b}: p={2/2**b:.4f}")
```

**Memory and timing arithmetic** (§8), from the measured NVMe and RAM bandwidths in `R2`: 50.29 GiB / 4.4 GiB·s⁻¹ = 11.43 s per full pass; ×64 prompts = 12.2 min; one layer = 635 MB bf16; peak with activations ≈ 1.83 GiB.

---

## Sources

1. He, H. & Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, Sep 2025 — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (batch invariance, the 1669.25 Δ example, split-K as the cause).
2. PyTorch, `torch.testing` — default `rtol`/`atol` per dtype (bf16: 1.6e-2 / 1e-5) — https://docs.pytorch.org/docs/stable/testing.html
3. PyTorch, `torch.library.opcheck` — registration contract, explicitly *not* numerical correctness — https://docs.pytorch.org/docs/stable/library.html and https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html
4. PyTorch, *Reproducibility* — no guarantee across releases, commits, platforms, or CPU↔GPU — https://docs.pytorch.org/docs/stable/notes/randomness.html
5. Triton, attention tutorial — reference comparisons via `assert_close(..., atol=1e-2, rtol=0)` for bf16 attention — https://triton-lang.org/main/getting-started/examples/gluon/attention-forward.html
6. GGUF format specification — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
7. *GGUF quantization, bit by bit* — per-type block structs, bytes/block, bpw — https://dwarez.dev/blog/gguf-quantization-bit-by-bit
8. Frantar et al., *GPTQ*, ICLR 2023 — perplexity as the quantisation metric; 4-bit costs 0.03 PPL on OPT-175B, 3-bit RTN collapses — https://arxiv.org/abs/2210.17323
9. Rein et al., *GPQA* — 448 questions total; diamond subset used for leaderboards — https://arxiv.org/abs/2311.12022
10. Qwen, *Qwen3.8-27B* model card — architecture (64 layers, 16×3 GDN + 1× attention), sampling parameters, bf16 benchmark numbers (GPQA-D 89.2, LCB v6 90.3) — https://huggingface.co/Qwen/Qwen3.8-27B
11. AIME 2025 — 30 problems, integer answers — https://intuitionlabs.ai/articles/aime-2025-ai-benchmark-explained
12. LiveCodeBench v6 — >1000 problems, May 2023–Apr 2025 — https://www.vals.ai/benchmarks/lcb
13. EleutherAI, *lm-evaluation-harness* — harness/version/template as part of the result — https://github.com/EleutherAI/lm-evaluation-harness
14. Wikipedia, *bfloat16* — 8-bit significand, ε = 2⁻⁸ — https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
