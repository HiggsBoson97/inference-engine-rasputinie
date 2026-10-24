# 18 — tensor parallelism (`--tp 2`) on gfx1100: does it work, and what does it cost

**Date:** 2026-10-22
**Host:** `rasputin`, ROCm 10.1 (`/opt/rocm`), torch 2.14.1+rocm7.2, venv `~/venv/sglang-fork`
**GPUs:** 2× RX 7900 XT `gfx1100`, 19.98 GiB each, **PCIe only** (no XGMI, no peer-to-peer DMA)
**Tree:** `engine/` @ the state left by report 16 (`0.5.9`, `sgl_kernel` 0.4.10 for gfx1100)
**Status:** **WORKS.** `--tp 2` serves and decodes correctly on both cards through RCCL. **No source change was required** — not one line inside `engine/`.

---

## 1. Verdict

| question | answer |
|---|---|
| does `--tp 2` work on gfx1100 today | **yes**, unmodified tree, `--attention-backend triton` |
| is it a config change, a patch, or a kernel project | **a config change.** The broken CDNA quickreduce kernel is never reached on this hardware — two independent gates keep it off (§2.3) |
| which transport serves the all-reduce | `librccl.so.1` (RCCL, reports NCCL-compat version **2.27.7**) via sglang's `PyNcclCommunicator` ctypes bindings; *not* `torch.distributed.all_reduce` |
| is the fallback clean or accidental | **both.** There is a real switch (`--disable-custom-all-reduce`). With it *off* the fallback is still correct but arrives via a swallowed `NameError` (§3.3) |
| measured 5120-element bf16 all-reduce | **20.65 µs** amortized (39.8 µs per-op with a sync each) = **0.50 GB/s** |
| measured peak all-reduce bandwidth | **12.91 GB/s** at 16 MiB — i.e. one PCIe 4.0 x16 link's payload rate |
| TP=1 vs TP=2 on Qwen3-0.6B, bs=1 | 103.4 → **129.8 tok/s** (1.26×), TPOT 9.61 → 7.56 ms |
| TP=1 vs TP=2 on Qwen3-0.6B, 32 concurrent | 1502 → **1166 tok/s** (0.78×) |

Correctness is byte-level: six greedy prompts produce **identical text and identical `finish_reason` under TP=1 and TP=2**, with logprob sums differing by at most 0.27 nats over 24 tokens (bf16 accumulation order).

Report 16 §7.3 ("any tp>1 run will fail or produce garbage all-reduce") is **wrong as a statement about this build** — the quickreduce symbols are indeed undefined for gfx11, but nothing on the TP=2 path ever calls them.

---

## 2. How sglang chooses its all-reduce

All paths below are `engine/python/sglang/srt/distributed/`.

### 2.1 The map

```
bootstrap.py:252          set_custom_all_reduce(not disable_custom_all_reduce)
      -> parallel_state.py:2332   global _ENABLE_CUSTOM_ALL_REDUCE   (default True, :2326)

initialize_model_parallel -> parallel_state.py:2681  _TP = init_model_parallel_group(..., group_name="tp")
        (use_custom_allreduce not passed => None => _ENABLE_CUSTOM_ALL_REDUCE, :2170-2171)
                         -> parallel_state.py:2793  _ATTN_TP (use_custom_allreduce=None, :2813)
                         -> parallel_state.py:2841  _SHARED_EXPERTS_TP (False)
                         -> parallel_state.py:2934  _MOE_TP ...

GroupCoordinator.__init__
  :470-473   on HIP, import QuickAllReduce + qr_rocm_arch_available
  :476-481   pynccl_comm = PyNcclCommunicator(...)        use_pynccl defaults True (:2180-2184)
  :515       if use_custom_allreduce and world_size > 1:
  :518         CAClass = dispatch_custom_allreduce(group, device)
  :519-523       self.ca_comm = CAClass(...)             <-- raises => ca_comm stays None
  :527-530     except Exception: logger.warning("Setup Custom allreduce failed with ...")
  :533-543     if qr_rocm_arch_available(): self.qr_comm = QuickAllReduce(...)
  :544-545   elif world_size > 1 and is_hip(): logger.info("[AR] All-reduce call path: NCCL (custom AR disabled)")

GroupCoordinator.all_reduce                                    (:693)
  :771-773   should_use_custom_allreduce = ca_comm and not disabled and ca_comm.should_custom_ar(x)
  :776-786   symmetric-memory / pynccl early path (only if is_symmetric_memory_enabled())
  :791-794   outplace_all_reduce_method = _resolve_outplace_all_reduce_method(...)   (:957)
  :801-803   else: inplace_all_reduce(x, group_name)           -> registered custom op
                    -> :1073 _all_reduce_in_place
                       :1076  if pynccl_comm is not None and not disabled:
                       :1077      pynccl_comm.all_reduce(x)          <-- the path taken here
                       :1085  else: torch.distributed.all_reduce(x, group=self.device_group)
```

Precedence inside `_resolve_outplace_all_reduce_method` (`:957-1005`), first match wins:

| order | line | backend | eligible on this box |
|---|---|---|---|
| 1 | :967-973 | `ca` (custom all-reduce / quickreduce) | **no** — `should_custom_ar()` False (§2.3) |
| 2 | :975-982 | `pcie_ipc` (FlashInfer PCIe-IPC) | no — `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE = EnvBool(False)` (`environ.py:1464`) |
| 3 | :983-989 | `qr` (QuickAllReduce) | **no** — `qr_comm` is None (§2.3) |
| 4 | :990-991 | `pymscclpp` | no — `_ENABLE_MSCCLPP = False` (:2327) |
| 5 | :992-998 | `torch_symm_mem` | no — `_ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = False` (:2328) |
| 6 | :999-1001 | `pynccl` | only under TC piecewise CUDA graph |
| — | :1002 | `None` → fall through to in-place | **yes** |

Result: `None` → `inplace_all_reduce` → `_all_reduce_in_place` → **`pynccl_comm.all_reduce()`**.
`pynccl_wrapper.py:56-60` picks the shared object by platform: `torch.version.hip is not None => "librccl.so.1"`. So on ROCm, PyNccl **is** RCCL, and `torch.distributed`'s `nccl` backend is RCCL too — same library, two different calling conventions.

### 2.2 Two all-reduces per layer, and what "TP" moves

The 27B-class shape is the one that matters: hidden 5120, bf16. Each decoder layer ends with a row-parallel `o_proj` all-reduce and a row-parallel `down_proj` all-reduce, so **2 × L all-reduces of 10 240 B per token** at batch 1.

### 2.3 Why the CDNA kernel is unreachable here — two independent gates

**Gate 1, the quickreduce arch check.** `device_communicators/quick_all_reduce.py:28-38`:

```python
def qr_rocm_arch_available():
    ...
    supported_archs = ["gfx94", "gfx95"]
    return any(gfx in gcn_arch for gfx in supported_archs)
```

`gfx1100` fails it, so `parallel_state.py:538` never constructs `QuickAllReduce` and `qr_comm` stays `None`. `quick_all_reduce.py:89-93` (`if not qr_rocm_arch_available(): return`) leaves `self.disabled = True` as set at `:88`. The `buffer_load_dwordx4` symbols report 16 §5 saw are simply never called.

**Gate 2, `full_nvlink`, which also gates the custom all-reduce.** `custom_all_reduce.py:276-281`:

```python
if _is_hip:
    if self.use_amd_deterministic_impl: return True
    if self.full_nvlink: return inp_size <= self.max_size
    return False
```

On HIP, `should_custom_ar()` returns **False for every tensor** unless the pair is 1-hop XGMI. `full_nvlink` comes from `custom_all_reduce_utils.py:354-371` (`is_full_nvlink`: `hops != 1 or type != 2 → False`). Measured on this pair, with amdsmi loaded (`PYTHONPATH=/opt/rocm/core-10.1/share/amd_smi`):

```
handles 3
link 0->1 {'hops': 2, 'type': 1}      # type 2 == XGMI; 1 == PCIe
```

`amd-smi topology` agrees — `LINK TYPE PCIE`, `HOPS 2`, `CACHE COHERENCY NC`, `DMA T`.

So even if `ca_comm` is constructed successfully, it is never selected. Note the trap at `custom_all_reduce_utils.py:492` and `:504`: the "more than two PCIe-only GPUs" bail-out is `world_size > 2`, and the P2P capability probe is guarded by `if not _is_hip` with the comment *"On AMD GPU, p2p is always enabled between XGMI connected GPUs"*. Both are CDNA/XGMI assumptions that are false on PCIe RDNA3 — `world_size == 2` therefore sails past both and reaches `return full_nvlink` (False). The only thing standing between a PCIe RDNA3 pair and the gfx9 assembly kernel is that final `should_custom_ar()` gate. **That is the one real fragility in this path.**

---

## 3. The fallback, and how to force it

### 3.1 The clean switch

`--disable-custom-all-reduce` (declared `srt/arg_groups/fields/exec_.py:691-696`, help text *"Disable the custom all-reduce kernel and fall back to NCCL."*) → `bootstrap.py:252` → `set_custom_all_reduce(False)`. **No code change needed.**

### 3.2 What it looks like on a clean run

```
$ HIP_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
    --attention-backend triton --tp 2 --disable-custom-all-reduce \
    --context-length 4096 --port 31012 --host 0.0.0.0
[TP0] [AR] All-reduce call path: NCCL (custom AR disabled)
[TP1] [AR] All-reduce call path: NCCL (custom AR disabled)
[TP0] sglang is using nccl==2.27.7
[TP1] [AR] All-reduce call path: NCCL (custom AR disabled)
[TP0] [AR] All-reduce call path: NCCL (custom AR disabled)
```
(4 lines: `tp` + `attention_tp` now join the non-TP groups that already took this branch.)

### 3.3 What happens *without* the switch (the default)

```
[TP0] [AR] All-reduce call path: NCCL (custom AR disabled)      <- non-TP groups only
[TP0] [AR] Aiter custom all-reduce not available; falling back to sglang CustomAllreduce. Details: No module named 'aiter'
[TP0] Setup Custom allreduce failed with name 'amdsmi_shut_down' is not defined. To silence this warning, specify --disable-custom-all-reduce explicitly.
```

The TP group *does* try, and fails — but for an **accidental** reason: `amdsmi` is not importable (report 16 §7.8), `custom_all_reduce_utils.py:39-50` swallows the `ImportError` and leaves the names unbound, and `is_full_nvlink` then raises `NameError` inside a decorator. `parallel_state.py:527-530` catches `Exception`, so `ca_comm` stays `None` and the run proceeds on RCCL.

I verified this is not load-bearing by making `amdsmi` importable without installing anything (`PYTHONPATH=/opt/rocm/core-10.1/share/amd_smi`). With it present: no `Failed to import amdsmi`, **no** `Setup Custom allreduce failed` — i.e. `CustomAllreduce.__init__` now *succeeds* and `ca_comm` is live — and the all-reduce still does not use it, because `should_custom_ar()` returns False (§2.3, Gate 2). Throughput is indistinguishable:

| TP=2 config | bs=1 tok/s | conc=32 tok/s |
|---|---|---|
| default (custom AR attempted, fails on `NameError`) | 129.84 | 1166.96 |
| `--disable-custom-all-reduce` | 129.81 | 1160.79 |
| default **with amdsmi present** (`ca_comm` constructed) | 129.90 | 1158.12 |

Three configs, same numbers within run-to-run noise. That is the empirical confirmation of the code reading in §2.3.

**Recommendation:** put `--disable-custom-all-reduce` in the rig anyway. It costs nothing, removes a `NameError` from every startup log, and makes the fallback intentional rather than dependent on a missing package.

---

## 4. RCCL all-reduce microbenchmark

Script: `bench/tp/rccl_ar_bench.py` (kept in both trees). Two processes, `mp.spawn`, `backend="nccl"` (= RCCL), rank *i* pinned to `cuda:i`, `HIP_VISIBLE_DEVICES=0,1`. It measures two paths on the same process group:

* `torch` — `torch.distributed.all_reduce`
* `pynccl` — `sglang...PyNcclCommunicator.all_reduce`, i.e. **exactly what the server does**

and asserts correctness (all-ranks-sum) before timing anything. *cold* = the very first call after `init_process_group`; *amort* = one event pair around 300 back-to-back calls; *med/min* = per-op with a sync each.

Transport, from `NCCL_DEBUG=INFO`:

```
NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
NCCL INFO Channel 00/0 : 0[3000] -> 1[7000] via P2P/IPC
NCCL INFO comm:..., nRanks:2, nNodes:1, coll channels:2 collnet channels:2, nvls channels:0, p2p channels:2
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
```

So RCCL picks direct `P2P/IPC` (2 channels) — the PCIe path works; it is just one 4.0 x16 lane's worth.

### 4.1 Results (bf16, default env, no NCCL_* tuning)

| elems | bytes | path | cold µs | amort µs | med µs | min µs | GB/s |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1024 | 2 048 | pynccl | 60.3 | 18.55 | 32.58 | 31.96 | 0.11 |
| 1024 | 2 048 | torch | 124.8 | 26.89 | 38.86 | 31.32 | 0.08 |
| 4096 | 8 192 | pynccl | 58.2 | 19.97 | 39.72 | 36.32 | 0.41 |
| 4096 | 8 192 | torch | 78.8 | 40.03 | 47.30 | 45.96 | 0.20 |
| **5120** | **10 240** | **pynccl** | **54.4** | **20.65** | **39.76** | **38.12** | **0.50** |
| 5120 | 10 240 | torch | 80.1 | 41.48 | 49.10 | 47.32 | 0.25 |
| **8192** | **16 384** | **pynccl** | **49.4** | **20.14** | **38.64** | **36.88** | **0.81** |
| 8192 | 16 384 | torch | 91.8 | 40.40 | 47.90 | 46.36 | 0.41 |
| 32768 | 65 536 | pynccl | 55.6 | 23.37 | 49.02 | 46.12 | 2.80 |
| 32768 | 65 536 | torch | 81.6 | 48.68 | 59.78 | 55.44 | 1.35 |
| 131072 | 262 144 | pynccl | 65.2 | 36.92 | 60.22 | 56.24 | 7.10 |
| 131072 | 262 144 | torch | 99.8 | 67.12 | 76.18 | 62.68 | 3.91 |
| 524288 | 1 048 576 | pynccl | 126.6 | 100.06 | 123.76 | 116.84 | 10.48 |
| 524288 | 1 048 576 | torch | 159.5 | 127.09 | 134.80 | 126.80 | 8.25 |
| 2097152 | 4 194 304 | pynccl | 640.7 | 339.66 | 367.98 | 360.60 | 12.35 |
| 2097152 | 4 194 304 | torch | 416.3 | 369.00 | 375.14 | 367.72 | 11.37 |
| 8388608 | 16 777 216 | pynccl | 1349.4 | 1299.73 | 1328.49 | 1320.69 | **12.91** |
| 8388608 | 16 777 216 | torch | 1444.0 | 1329.06 | 1344.01 | 1334.29 | 12.62 |

GB/s is `bytes / time` for one rank's payload; for a 2-rank all-reduce that equals the bus bandwidth (algbw × 2(n−1)/n, n=2). Both paths were verified to produce the correct sum at every size.

**Reading it:**

* **The 5120-element all-reduce costs 20.65 µs.** That is the number the 27B TP budget gets built from (§5.3).
* There is a hard **~19-20 µs floor** for anything up to ~16 KiB: the payload is nothing (0.8 µs of wire time at 12.9 GB/s), the cost is RCCL's latency-protocol handshake and cross-device polling. Per-op-with-a-sync costs another ~19 µs on top (CPU round trip) — the amortized figure is the GPU-side truth.
* The floor is **not** tunable by NCCL env. 5120/8192 amortized, four configs: `NCCL_PROTO=Simple` 25.14/19.44; `NCCL_PROTO=LL` 23.03/21.21; `NCCL_ALGO=Ring NCCL_PROTO=Simple` 22.03/19.50; `NCCL_MIN_NCHANNELS=8` 20.85/19.29. Nothing beats ~19.5 µs.
* `pynccl` is consistently **~2× cheaper than `torch.distributed.all_reduce`** at small sizes (20.65 vs 41.48 µs at 5120). That is the value of sglang bypassing torch's collective bookkeeping and issuing `ncclAllReduce` straight onto the current stream. It also means: **do not benchmark TP cost with `torch.distributed` — you will overstate it by 2×.**
* Peak 12.91 GB/s at 16 MiB matches one PCIe 4.0 x16 link. The earlier "14.2 GiB/s = host bounce" note in the handover is 15.2 GB/s in decimal — the same wire, measured differently; there is no P2P/DMA deficit to fix here, the link is simply saturated.

---

## 5. The real TP=2 test

### 5.1 Serving

```
$ HIP_VISIBLE_DEVICES=0,1 ~/venv/sglang-fork/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B --attention-backend triton \
    --tp 2 --context-length 4096 --port 31010 --host 0.0.0.0
[TP0] Load weight end. elapsed=0.78 s, type=Qwen3ForCausalLM, avail mem=18.98 GB, mem usage=0.62 GB.
[TP1] Load weight end. elapsed=0.80 s, type=Qwen3ForCausalLM, avail mem=18.98 GB, mem usage=0.62 GB.
[TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 286385, K size: 7.65 GB, V size: 7.65 GB
[TP0] Capture target decode CUDA graph end. bs=[1, 2, 4, 8], elapsed=4.86 s
[TP0] max_total_num_tokens=286385, chunked_prefill_size=2048, available_gpu_mem=3.31 GB
[2026-10-22 15:08:31] The server is fired up and ready to roll!
```

Per-rank weight memory halves (0.62 GB vs 1.21 GB at TP=1) and the KV pool doubles (286 385 vs 140 300 tokens). `"multimem all-gather disabled (no multicast for world_size=2)"` is expected on PCIe.

### 5.2 Both GPUs are actually used

`amd-smi metric --usage` sampled once a second during a 60-request bs=1 bench:

```
g0:24% | g1:24%
g0:33% | g1:33%
g0:100% | g1:100%
g0:98%  | g1:98%
... (60 consecutive samples at 99-100% on both) ...
g0:94% | g1:94%
```

Both cards hold ~17.1 GiB while serving (`USED_VRAM: 17497 MB` / `17410 MB`; sysfs `card1 18 347 200 512 B`, `card0 18 256 424 960 B`).

Worth noting: **both GPUs pinned at 100% for 7.5 ms/token of a 0.6 B model** is the all-reduce spin-wait, not useful work. See §7.1.

### 5.3 Is the output correct?

Six greedy prompts, `temperature=0, max_tokens=24, logprobs=1`, requested from a TP=1 server (`HIP_VISIBLE_DEVICES=1`) and a TP=2 server, diffed:

```
=== DIFF tp1 vs tp2 ===
6c6   <   "lp_sum": -8.89146      >   "lp_sum": -9.042154
13c13 <   "lp_sum": -1.664469     >   "lp_sum": -1.672053
20c20 <   "lp_sum": -12.183235    >   "lp_sum": -12.212866
27c27 <   "lp_sum": -12.885297    >   "lp_sum": -12.887114
34c34 <   "lp_sum": -12.595035    >   "lp_sum": -12.323397
41c41 <   "lp_sum": -4.704312     >   "lp_sum": -4.673869
```

Every `text` and every `finish_reason` line is **identical** — `diff` reports changes only on the logprob-sum lines. Max drift 0.272 nats over 24 tokens (~0.011 nats/token): the expected bf16 reduction-order difference from splitting matmuls across two cards. Greedy decoding is bit-stable.

Cross-check against report 16 §6(c): the same prompt under TP=2 gives
`' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'` — the exact string recorded there for TP=1.

### 5.4 TP=1 vs TP=2 measured

Same model, same `--attention-backend triton`, same machine, `bench_serving --dataset-name random --random-input-len 512 --random-output-len 256`. TP=1 ran on `HIP_VISIBLE_DEVICES=1` (the headless card) so the display card was free. Prompts run to EOS at ~130 output tokens, so `Output token throughput` is the comparable figure. Repeat-run spread is shown because the first run of every server was a warm-outlier.

| | bs=1 tok/s | TPOT ms | conc=32 tok/s | TPOT ms |
|---|---:|---:|---:|---:|
| **TP=1** (1× 7900 XT) | 103.37, 103.44 | 9.62, 9.61 | 1206\*, 1500.70, 1503.87 | 17.49\*, 16.14, 16.35 |
| **TP=2** (2× 7900 XT) | 129.84, 129.81, 129.90 | 7.57, 7.56, 7.56 | 983\*, 1166.96, 1164.36, 1167.24, 1160.79, 1158.12 | 23.15\*, 21.86, 21.96, 21.86, 22.03, 22.51 |
| **ratio** | **1.26×** | −21% | **0.78×** | +34% |

\* first run after server start; discarded (TTFT 700-670 ms vs 50-65 ms steady state).

**bs=1: TP=2 wins 1.26×.** **conc=32: TP=2 loses 22%.** That is the correct shape for this hardware and this model, and it is *not* a bandwidth win — see §7.2.

### 5.5 What TP=2 should do for the 27B, from measured numbers

Per decoder layer at batch 1: 2 all-reduces × 20.65 µs = **41.3 µs**. A 64-layer dense stack therefore pays **2.64 ms/token** of pure all-reduce.

Meanwhile the thing TP halves is the per-card weight read: 27B bf16 = 54 GB, so 27 GB per card. At ~650-700 GB/s achievable HBM read that is 39-42 ms/token. So:

* TP=2 all-reduce overhead ≈ **6%** of a bs=1 27B decode step.
* Expected speedup on the bandwidth-bound part ≈ **1.8-1.9×**, not 2×.

That is the case for TP=2 on this box: at batch 1 the 27B is weight-bandwidth-bound, so halving the bytes each card reads is worth far more than 2.64 ms of PCIe chatter. It is *not* an argument for TP=2 at high concurrency (§7.2). These are derived numbers from measured inputs, not measured 27B timings — the 27B was not loaded.

---

## 6. Corrections to report 16

1. **§2.1 has the display card wrong.** Report 16 says HIP device 1 = `card0` = "the display card" and cites `card0`'s 27 947 008 B as the display baseline. Measured now:

   ```
   /sys/class/drm/card0-HDMI-A-2/status: disconnected   (card0 = 07:00.0 = HIP device 1)
   /sys/class/drm/card1-HDMI-A-1/status: connected      (card1 = 03:00.0 = HIP device 0)
   ```
   The **display is on `card1` / `03:00.0` / HIP device 0**; `card0` / `07:00.0` / HIP device 1 is the headless card. The 27 947 008 B figure is the headless card's idle baseline. Idle figures are `card0` 27 947 008 B and `card1` 118 722 560 B (the latter is a kernel-side allocation, consistent with §6's post-run reading).
   Consequence: any future "use the headless card" instruction must say `HIP_VISIBLE_DEVICES=1`, and a TP=2 run necessarily puts compute on the display card. Both were idle (<120 MB, well under the 1 GiB stop-rule) for every run here, so TP=2 was safe; nothing else was running on either card.
2. **§7.3 is too pessimistic.** "Any tp>1 run will fail or produce garbage all-reduce" — it does not, for the reasons in §2.3. The symbol state is as described, but the path is gated off.
3. **§8.3's framing ("make the custom-all-reduce path refuse to initialise on RDNA") is still worth doing**, but for a different reason than corruption: today it initialises and then is silently never used, which is invisible. A one-line `is_full_nvlink`-style guard on RDNA3 would say so out loud.

---

## 7. What still breaks / what is fragile

### 7.1 The 0.6B decode is launch-bound, and the GPUs busy-spin

7.56 ms/token at bs=1 for a 1.2 GB model is ~5× worse than the HBM floor (1.5 ms). Both GPUs sit at 100% GFX activity while doing it. The all-reduce spin-wait is inside that, but so is per-kernel launch overhead on a 28-layer model with ~400 small kernels/token. This is a property of the fork on RDNA3 at batch 1, not of TP — but it makes the 0.6B a poor instrument for TP scaling, as expected.

### 7.2 TP=2 loses at high concurrency, and RCCL's 20 µs floor is why

At conc=32 the AR payload is 32 × 1024 = 64 KiB (23.4 µs) rather than 10 KiB (20.7 µs) — the latency floor barely moves, so 56 all-reduces/token costs ~1.3 ms of *serial* critical path that batching cannot hide, on top of the extra cross-card traffic. Measured penalty is 4.7 ms of TPOT (16.1 → 22.5), 3.5× my 1.3 ms estimate, so there is more than raw AR latency in it (the sharded attention path and the reduced per-card arithmetic intensity are also candidates — not decomposed here). Either way: **TP=2 on this PCIe pair is a batch-1 optimization, not a throughput optimization.**

### 7.3 Noise on every start, with the default flags

* `Setup Custom allreduce failed with name 'amdsmi_shut_down' is not defined` — a `NameError` from a swallowed `ImportError` (`custom_all_reduce_utils.py:39-50`), surfaced as a user-facing warning. Either install `amdsmi` or pass `--disable-custom-all-reduce`.
* `[AR] Aiter custom all-reduce not available; falling back to sglang CustomAllreduce` (`custom_all_reduce.py:398-401`) — fires on every ROCm start without `aiter`. Should be a debug log on an arch where `aiter` cannot exist.
* `[AR] All-reduce call path: NCCL (custom AR disabled)` is **misleading**: in a default TP=2 run it appears twice for the non-TP groups while the TP group is *also* falling back for a different reason. It cannot be used as evidence about the TP path. Compare 2 lines (default) vs 4 lines (`--disable-custom-all-reduce`) to see which groups took it.

### 7.4 `world_size == 2` on PCIe is the untested seam

`custom_all_reduce_utils.py:492` only rejects **>2** PCIe GPUs, and `:504` skips the P2P probe entirely on HIP. A 2-GPU PCIe AMD box reaches `should_custom_ar()`; only that final `full_nvlink` check keeps the gfx9 assembly off. If someone later relaxes `should_custom_ar()` (e.g. to enable the fused AR+RMSNorm path), gfx1100 will start executing `quick_all_reduce` with undefined memory ops. This is the one place in the reasoning that is not defended by a hard arch check.

### 7.5 Not tested

* The 27B itself (the numbers in §5.5 are derived, not measured).
* `--tp 2` with a model that has GQA head counts not divisible by 2, MoE, or FP8 — only `Qwen3ForCausalLM` dense bf16.
* `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` (the FlashInfer PCIe-IPC path, §2.1 row 2). On paper it is the interesting alternative for a host with no fabric — it is presumably designed for exactly this topology. Worth a follow-up: if it beats 20.65 µs at 10 KiB, TP=2 stops being a batch-1-only win.
* `NCCL_PROTO=LL` showed a better per-op median at 8192 (24.26 vs 38.64 µs) though the same amortized floor. Not pursued; no end-to-end gain demonstrated.

---

## 8. Repo state of this work

**Nothing inside `engine/` was modified.** No patch, no config swap beyond what report 16 left. The only changes are new files.

**Added (both trees):**

| file | what |
|---|---|
| `bench/tp/rccl_ar_bench.py` | the 2-process RCCL/pynccl all-reduce microbenchmark (§4) |
| `bench/tp/bench_endpoint.sh` | bench one endpoint at concurrency 1 and 32 |
| `docs/research/18-tp-roccl.md` | this report |

**Outside the fork (host only, declared):** `~/build/tp-bench/` (copies of the scripts), `~/build/tp-logs/{tp2-default,tp2-nocustomar,tp2-amdsmi,tp1-headless2,tp2-greedy}.log`, `/tmp/greedy.py`, `/tmp/greedy_tp{1,2}.json`, `/tmp/rccl_ar_bench.json`.

**Not touched:** `main`, git history, `pyproject.toml`, `sgl_kernel`, the venv, and any process not started here. Every server was `kill -TERM`'d; both cards returned to their idle baselines (27 947 008 B / 118 722 560 B) and no listeners remain.

---

## 9. Verdict, in one paragraph

**TP=2 on gfx1100 is a config change, not a patch and not a kernel project.** `--tp 2 --attention-backend triton` serves a real model on two 7900 XTs, uses both cards, and produces greedy text identical to TP=1; the CDNA `quickreduce` kernel that report 16 flagged as fatal is never reached, because `qr_rocm_arch_available()` admits only gfx94/gfx95 and, independently, `should_custom_ar()` returns False for every tensor on a non-XGMI AMD pair — both verified here, the second by loading `amdsmi` and reading `{'hops': 2, 'type': 1}` off the two GPUs. The transport that actually runs is `librccl.so.1` reached through sglang's own `PyNcclCommunicator` (not `torch.distributed`), and it is a 20.65 µs / 0.50 GB/s link for a 5120-element bf16 all-reduce with a hard ~19.5 µs latency floor that no `NCCL_*` setting moves, topping out at 12.91 GB/s on large messages — one saturated PCIe 4.0 x16 lane. That makes TP=2 a **1.26× win at batch 1** (103.4 → 129.8 tok/s on a 0.6 B model) and a **0.78× loss at 32 concurrent requests**, and it projects to roughly **1.8-1.9× and ~6% overhead** for a batch-1 27B decode, which is exactly the regime the owner wants it for. The work left is bookkeeping, not engineering: add `--disable-custom-all-reduce` to the rig to silence the accidental `amdsmi` `NameError` fallback, and add an explicit RDNA3 refusal to `should_custom_ar()` so that the only thing standing between gfx1100 and gfx9 assembly stops being one boolean.
