# HANDOFF — rasputinie: the ground-up sglang rework

**As of 2026-10-24.** This is the resume document for the optimisation work. It carries the mental
model, the measured bounds, the decisions, what failed, and the exact rework. Numbers live in
`docs/facts/`, choices in `docs/decisions/`, units in `docs/plan/DEVELOPMENT-PLAN.md`, state in
`docs/status.md`.

**The one-line position:** the engine cannot serve this file on this card at all, for four located
reasons; every number we own says the hardware is fine and the software path is missing. So we stop
adapting sglang and rebuild its quantised-weight path for gfx1100.

---

## 1. The target

| | |
|---|---|
| Machine | 2× RX 7900 XT (gfx1100, RDNA3, 19.98 GiB each, PCIe 4.0 ×16, **no peer DMA**), 60 GiB DDR5 |
| Wall | **776 GB/s** measured streaming read, one card |
| Model | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S, 11.77 GB, **851 tensors, 11 quant types, 3.498 bpw**, 64 layers (48 GatedDeltaNet + 16 full attention), 10.57 GiB read per step |
| Objective | bandwidth efficiency — achieved GB/s against the wall, **per shape**. Records, never targets |
| Engine | vendored sglang 0.5.9 at `engine/`, installed **editable** — the tree *is* the running engine |

## 2. Concepts you must hold

**A decode step** reads every weight once (10.57 GiB) and produces one token. At 776 GB/s that is
**14.6 ms**; we measure 521 GB/s on the real kernel, so **~21 ms → 47 tok/s** for the whole file. Two
resource bounds sit within 6 % of each other: freeing memory gives +123.8 GB/s, removing instructions
gives +130.9. **Neither is saturated.** Any single-resource story is wrong.

**The file is mixed precision.** IQ3_S is 31 % of the bytes; IQ4_XS 25.7 %, IQ3_XXS 17.6 %, Q4_K
13.5 %, IQ2_S 6.8 %, and six more. Nine of the eleven types are **quantised** (three families:
standard, k-quant, i-quant); BF16 and F32 are not. **84 % of the bytes are i-quants**, whose dequant
needs **grid tables** that are not in the file.

**Two kernel shapes exist in the engine, not one.** `fused_mul_mat_gguf` in
`layers/quantization/gguf.py:203` branches on `x.shape[0]`:
- **`x.shape[0] ≤ mmvq_safe`** → `ggml_mul_mat_vec_a8(qweight, x, qtype, rows)` — one fused int8 GEMV.
  `mmvq_safe` is **8** for i-quants when the output dim exceeds 5120, else **16**; and **2**/**6** for
  k-quants. Decode is `x.shape[0] = 1`, below every variant, so **decode always takes MMVQ, for every
  quantised type**. This is the hot path.
- **above that** → k-quants take `ggml_mul_mat_a8`; **i-quants have no MMQ kernel** (there is a TODO
  saying so) and fall to `ggml_dequantize` + a bf16 matmul. That is the **verify/prefill** path.

**GatedDeltaNet vs full attention.** 48 layers carry a fixed recurrent state (159 MB/sequence, FP32,
read *and written* per token); 16 carry a KV cache (64 KiB/position bf16, 34 KiB q8). The state costs
0.318 GB/token at any context; KV grows with context. **KV overtakes the weights at ≈173 K resident
tokens** (`B × C`), the state at ≈36 sequences.

**Speculation.** Verify `k` tokens in one pass: the weight read amortises. Measured
`ms(k) = 1.764 + 0.588·k` on 1.07 GiB — **the marginal cost is flat, no knee**, because ~102 of the
body's 193 words are dequant that all `k` vectors share. So **verification does not limit depth**;
acceptance does, and the cost is the **drafter**: MTP costs 1.064 GB *per draft token* (it re-reads
0.666 GiB of LM head every pass), DFlash2 at 4.5 bpw costs 0.49 ms and is flat in block size.

**The shape surface.** No shape is the target. Every per-token term is a function of `(B, C)`
(`docs/plan/COST-LEDGER.md`). Report `(B, C)` with every rate or the rate is a slogan.

## 3. What we measured (headline — full tables in `docs/facts/`)

| | value | where |
|---|---:|---|
| IQ3_S dequant-GEMV, VRAM | **521.3 GB/s** (67.2 %) | `facts/memory.md` |
| same, memory-free | **645.1 GB/s** (83.1 %) | same |
| read-only floor | 743.7 GB/s (95.8 %) | same |
| IQ4_XS / Q4_K dot | 502.1 / 452.2 | `facts/quant.md` |
| IQ3_XXS / IQ2_S dot | **INCONCLUSIVE** (1.43 fp64 err; layout doesn't close) | same |
| whole-file decode | **47.1 tok/s** (21.18 ms/token) | `facts/decoder.md` |
| body | 193 words / 32 weights / lane; 48.6 % issue occupancy | `facts/compute.md` |
| verify curve | `1.764 + 0.588k` ms; marginal 5.40 ms/token whole-file | `facts/decoder.md` |
| drafter budget | 3.87 ms per drafted token (MTP uses 53 %, DFlash2 4.5 bpw 13 %) | same |
| RCCL all-reduce | 20.65 µs @10 KiB; **~19.5 µs floor**, not tunable | `facts/twocards.md` |
| engine size | 8,338 files measured; **5,363 dead (1.48 M LOC)**; 1,070 live (455 k) | `facts/engine.md` |

## 4. What we decided (full text in `docs/decisions/`)

`D-01` one served model, with a drafter admitted as *draft*, not a second model · `D-02` llama.cpp
excluded · `D-03` FP8 excluded · `D-04` fork sglang; the tree is the engine · `D-05` serve quantised
as published, one internal layout, bf16 compute, **no codebook expansion** (breaks even at 100.4 % of
the wall) · `D-06` records never targets; no shape is the target · `D-07` **aiter's ASM/CK excluded,
Triton is native and forced** · `D-08` speculation is core · `D-09` topology decided by the round
trip.

## 5. What we tried, and what it cost

Four beliefs were killed by measurement, and each is kept in the record because the *shape* of the
error recurs:

1. **"619 GB/s is the wall"** — a 1.6 ms kernel artefact. The wall is 776.
2. **"IQ3_S is instruction-issue-bound"** (`n ≈ 1.57` vs a 1.04 budget) — a **32× unit error**
   (wave vs lane instructions), and `n` was *solved from* the throughput: `809.08/516.4 = 1.5668`.
   Refuted by +23.7 % when memory is freed.
3. **"The other 69 % of the file runs at 752 GB/s"** — measured 502 and 452. The estimate fell 19 %.
4. **"DP=2 is right for this quant"** — DP gives **1.00× on a single stream**; the original answer
   priced aggregate throughput only.

Plus one instrument that does not work: **`rocprofv3` reads no instruction counter on this box**
(21 counters, canary with 200 000 known instructions, six read zero — `SQ_INSTS_VALU` among them).
The substitute that works is the **differential memory-pressure experiment**.

---

## 6. THE REWORK

### 6.1 What dies

| thing | why | how |
|---|---|---|
| **aiter** | MFMA asm on a card with no matrix cores; its two allow-lists disagree so it looks alive then asserts | **delete** from the reachable path — not a `try/except`. Start at `layers/quantization/quark/schemes/quark_w4a4_mxfp4.py:163` (module-scope import under `if _is_hip`), then any `is_hip → aiter` default |
| **the HIP default attention backend** | it resolves to aiter | force `--attention-backend triton` / the resolver; sglang's own `triton_backend.py` (118 KB, zero aiter refs) is already there |
| **the ROCm GGUF refusal** | `configs/model_config.py:2128-2133` rejects `gguf` because the wheel has no ops | replace with an **honest gate**: probe for *our* op module, allow only if importable |
| **the CUDA GGUF kernels** | `__dp4a` is **undeclared** for gfx1100; the C++ route is closed | do not port. Write the ops in Triton |
| **5,363 files / 1.48 M LOC** | dead on any path from this model | strip after the baseline exists (unit 3.1) |

### 6.2 What we write — the exact op contract

`fused_mul_mat_gguf` needs four names importable on HIP. Everything else is sglang plumbing that
already works:

| op | signature as called | scope for our file |
|---|---|---|
| `ggml_mul_mat_vec_a8` | `(qweight, x, qtype, rows)` → `[x.shape[0], rows]` | **the hot path — all 9 quantised types at decode** |
| `ggml_mul_mat_a8` | `(qweight, x, qtype, rows)` | Q2_K and Q4_K only (k-quants at ≥ mmvq_safe) |
| `ggml_dequantize` | `(qweight, qtype, *shape, dtype)` | the 7 i-quant types at ≥ mmvq_safe (verify/prefill) |
| `ggml_moe_a8`, `ggml_moe_a8_vec` | MoE only | **stub with `NotImplementedError`** — this model is dense |

`dequantize_gguf_weight` also uses `gguf.GGML_QUANT_SIZES[qtype]` for `(block_size, type_size)` —
that is the Python `gguf` package, installed, and a live dependency of the path. See §8.

**Where it plugs in:** `layers/quantization/gguf.py` has the import chain `_is_cuda → _is_musa →
_is_npu → else`. Add a `_is_hip` branch importing from a new module (e.g.
`sglang.srt.layers.quantization.gguf_hip`) that implements the four names in Triton.

### 6.3 The dispatch reality that sets the scope

```
x.shape[0] ≤ mmvq_safe  and  type ∈ MMVQ_QUANT_TYPES   →  ggml_mul_mat_vec_a8      (decode)
type ∈ MMQ_QUANT_TYPES  (standard + k-quant only)      →  ggml_mul_mat_a8          (verify, k-quants)
type ∈ DEQUANT_TYPES    (all quantised)                →  ggml_dequantize + bf16   (verify, i-quants)
type ∈ UNQUANTIZED      (BF16, F32)                    →  x @ w.T
```

`MMVQ_QUANT_TYPES = STANDARD | KQUANT | IMATRIX` — **every quantised type**. So the first op is the
one that matters; the other two are for the verify path and prefill, and they are *slower by design*
(dequantise + bf16 matmul), which is a known cost in the speculation budget.

### 6.4 The kernel to write (Triton, native on gfx1100)

Triton compiles straight to `amdgcn` + `hsaco` for gfx1100 (verified by running one: vector add, max
abs err 0.0). What it will **not** give you: VGPR allocation, instruction scheduling, delay slots,
load ordering. The hand-written asm kernel gets 521 GB/s using all four. **A Triton GEMV's rate is
unmeasured — that is the biggest unknown in the rework.**

Shape that fits the op: one program per output row-block; each lane walks its own contiguous 32-weight
block group; accumulate in int32; apply scale once per block; write one dot product. One Triton kernel
per quant *family* with the type as a `tl.constexpr`, so the block walk is compiled away — do not
dispatch per type inside the kernel (I-cache: interleaving types in one CU costs −10 %).

---

## 7. The unit sequence (proofs in `docs/plan/DEVELOPMENT-PLAN.md`)

| # | unit | why it is here |
|---|---|---|
| 0.1 ✅ | the recipe diff — `DONE`, 851/851 match, control fails | proves our inventory is the publisher's |
| **0.2** | **the i-quant grid tables** | gates 84 % of the bytes; from `IST-DASLab/GSQ` (Apache-2.0), never llama.cpp |
| **1.1** | remove aiter, force Triton | the engine cannot even import until this lands |
| **1.2** | one Triton `ggml_mul_mat_vec_a8` for all 9 quantised types — **correct, not fast** | the rework's first real kernel |
| **1.3** | wire it: the HIP branch + the honest gate | weights load |
| **1.4** | **the baseline** — 32 coherent tokens, TTFT, tok/s, VRAM, `(B, C)` | everything downstream is measured against this |
| 2.1–2.5 | per-type kernels by byte share (IQ4_XS, IQ3_XXS, Q4_K, IQ2_S, tail) | each replaces the generic path, each with a delta against 1.4 |
| 3.1–3.3 | strip · attention in situ (q8/k8v4) · speculation | the two large terms |
| 4.1–4.3 | round trip · two cards · prefill | the machine |

**Do not start 3.x before 1.4 exists.** A rate from a scratch kernel is a proxy; after 1.4 it is not
allowed as a proof.

## 8. Open decisions you must take, not assume

1. **The `gguf` Python package.** The engine reads GGUF through it (`GGML_QUANT_SIZES`, tensor names,
   `gguf_quant_weights_iterator`). Our older doctrine said *"the GGUF reader and every dequant kernel
   are ours"*; `D-02` excludes **llama.cpp**, and this package is the format reader, not the engine.
   **Recommendation:** keep it for metadata and block sizes, write every numeric kernel ourselves.
   But it touches an exclusion, so it is the owner's call.
2. **Two paths or one.** Keeping `ggml_mul_mat_a8`/`ggml_dequantize` means the verify path is a
   dequant+bf16 matmul — correct but slow, and it eats the speculation gain. Replacing them with a
   fused multi-row MMVQ is a *later* unit, not this one.
3. **The `engine/bringup` branch** at `bb39bf0f` holds two **unrun** commits from an interrupted
   probe (an aiter removal and an honest gate). Read them or delete them before writing new versions.

## 9. Problems we know we will hit

| # | problem | mitigation |
|---|---|---|
| 1 | **Triton may not reach 521 GB/s** — it controls neither scheduling nor VGPRs | measure unit 1.2 immediately against `facts/memory.md`; if it lands low, the per-type units are where the rate comes back, and ADR 0001 records TensorFold ships hand-written HIP GEMV for gfx1100 as the ceiling to borrow |
| 2 | **The grids are missing**, so every i-quant kernel is unverifiable | unit 0.2 first; it needs no GPU |
| 3 | **The verify path dequantises** → a bf16 matmul at k>16 | accept for now; it changes the speculation budget (`facts/decoder.md` says so) |
| 4 | **`qwen3_5` + GGUF on ROCm has never loaded** — arch support and the state cache are untested on HIP | unit 1.3 fails fast and names the blocker; that failure is the finding |
| 5 | **Triton compile stalls** the first requests (1–2 s, once 16.5 s) | warm up before timing |
| 6 | **Type-sorted dispatch** — interleaving types in one CU costs 10 % | one kernel per family with `tl.constexpr` type |
| 7 | **The LM head is 0.666 GiB Q4_K, read every token** (6 % of bytes) | it is in the per-type plan; do not special-case it |
| 8 | **Only one card is measured.** Two-card work is unit 4.x, gated on the round trip | do not design topology into kernels; keep the reduction a dispatch flag |

## 10. Verify you are in the right state

```bash
cd "$REPO" && git log --oneline -1 && git status --short        # clean
wc -l AGENTS.md                                                 # 54
find docs -type f | wc -l                                       # 26
ssh rasputin '~/venv/sglang-fork/bin/python -c "import sglang; print(sglang.__version__)"'   # 0.5.9
# unit 0.1's proof (needs the GGUF, so runs on the host):
scp bench/04_recipe_diff.py bench/manifests/rco-iq3s-allocation.txt rasputin:/tmp/ && \
ssh rasputin '~/venv/sglang-fork/bin/python /tmp/04_recipe_diff.py \
  ~/local-inference/models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf /tmp/rco-iq3s-allocation.txt'
# expect: matched 851/851 ... RESULT: PASS, exit 0
```

## 11. The rules that keep this from drifting

From `AGENTS.md` (15 invariants). The five that this rework will actually test:

- **Delete, do not guard.** A `try/except` around aiter keeps it alive and hides the next failure.
- **Absent hardware is not a software gap.** Never emulate MFMA or FP8; do write the kernel the
  hardware supports and the tree lacks.
- **Gate every probe.** A check that cannot fail is not a check — the two failed controls in
  `docs/status.md` are the standing example.
- **No shape is the target.** Report `(B, C)`; the variable is resident tokens `B × C`.
- **Measure in the engine.** Scratch harnesses may inform a unit; they cannot be its proof once 1.4
  exists.
