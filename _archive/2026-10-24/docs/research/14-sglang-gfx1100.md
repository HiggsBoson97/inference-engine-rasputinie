# sglang on gfx1100 — d0a75654368a470d8484e3ea45b06e500fee57af

Checkout: `development/_research/sglang` @ `d0a7565` (branch `main`, clean tree). Read-only.

## Verdict

**Build thin and borrow — do not fork.** Both target architectures already exist upstream (`qwen3_5`, `qwen4_exp`, MTP and the 52 B ngram table included), so models do not justify a fork.
ROCm support is CDNA3/CDNA4 + RDNA4 shaped: every arch helper is `gfx942` / `gfx95` / `gfx1250`, there is **no** RDNA3 helper, and the AOT kernel build hard-exits on gfx1100 (`python/sglang/kernels/aot/setup_rocm.py:77-81`).
Pin sglang as a reference dependency, patch the 10 sites below, and write the one kernel nothing upstream has: batch-1, 10-of-512 MoE for wave32.

## Backend selection map

| capability | entry point file:line | arch gate | gfx1100 status |
|---|---|---|---|
| platform identity | `python/sglang/srt/platforms/rocm.py:14-20` | none; device ops inherited from `CudaDeviceMixin` (torch.cuda shim) | WORKS |
| HIP test | `utils/common.py:141` | `torch.version.hip is not None` | WORKS |
| gfx95 / gfx942 / gfx1250 tests | `utils/common.py:1231-1240`, `:1244-1252`, `:1256-1264` | substring match on `gcnArchName` | all False on gfx1100 → every CDNA/RDNA4 branch dead |
| **RDNA3 / gfx11 test** | — | **does not exist** | **gap; 71 call sites of `is_gfx95_supported()`** |
| default attention backend | `arg_groups/model_override_base.py:326-327` | `elif get_platform().is_hip: return "aiter"` — arch-blind | NEEDS-WORK: gfx1100 defaults into CK |
| flashinfer availability | `utils/common.py:387-394` | `... and is_cuda()` | False on HIP; flashinfer excluded by construction |

ROCm env (`srt/environ.py`): `SGLANG_USE_AITER` :909 (False), `SGLANG_USE_AITER_UNIFIED_ATTN` :945 (False), `SGLANG_USE_AITER_MOE_GU_ITLV` :950 (True), `SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY` :954 (2), `SGLANG_ROCM_SMALLM_ROUTER/MOE/FP8_PROJ` :988-990 (all True, gfx950 kernels — reachable via `moe_runner/aiter.py:266` `smallm_moe_gfx950`), `SGLANG_AMD_USE_FLYDSL_MEGA_MOE` :1003 (False).

## Attention backends

Registry `layers/attention/attention_registry.py`; choices at `arg_groups/choices.py:69-99`. **There is no `ck` backend** — CK exists only inside `aiter` (`aiter_backend.py:1195`, `:4428`).

| backend | registry line | gate | gfx1100 |
|---|---|---|---|
| `triton` / `torch_native` | `:206-214`, `:217-221` | none | **WORKS** — `_is_gfx942` only pins MLA kv-splits (`triton_backend.py:67`, `:334-338`); the `sgl_kernel` import is guarded (`:70-71` `if _is_cuda:`) |
| `flex_attention` | `:224-228` | none | NEEDS-WORK — ROCm lowering unverified |
| `aiter` | `:127-131` | none in registry; default on HIP | **NEEDS-WORK** — see below |
| `wave` | `:134-138` | needs external `wave_lang` (`wave_backend.py:66-71`) | NEEDS-WORK — gfx11 target unverified |
| `flashinfer` | `:55-77` | `is_flashinfer_available()` → `is_cuda()` (`common.py:394`) | NVIDIA-only (unreachable) |
| `fa3` | `:238-251` | `assert (major == 8 and not mla) or major == 9` | NVIDIA sm80-90 |
| `fa4` | `:264-274` | same binding | NVIDIA-only |
| `flashmla` | `:231-235` | module-level `from sgl_kernel.flash_mla import ...` (`flashmla_backend.py:13`) | NVIDIA-only (import fails) |
| `trtllm_mha`/`trtllm_mla`/`cutedsl_mla`/`hpc_ops` | `:277-283`, `:94`, `:118`, `:286-300` | CUDA extensions; hpc_ops needs Hopper SM90 + `--page-size 64` (`choices.py:89`) | NVIDIA-only |
| `dsa`/`nsa`/`qsa`/`dsv4` | `:150-204` | gfx95/gfx1250 gates (`dsa_indexer.py:129`) | CDNA-ONLY (DeepSeek-class models) |
| `intel_amx`/`intel_xpu`/`ascend` | `:303`, `:570`, `:141` | other hardware | n/a |

The `aiter` default is the trap. Its Triton fallback is opt-in, never automatic: `aiter_backend.py:529-534` sets `use_triton_unified_attention = True` only for a sliding-window pool, else it reads `SGLANG_USE_AITER_UNIFIED_ATTN` (default False, `environ.py:945`). Everything else falls through to `paged_attention_ragged` / `mha_batch_prefill_func` (`aiter_backend.py:47-61`) — CK/ASM kernels built for gfx942/gfx950 (`:4428` "ck_tile paged batch_prefill"); fp8 prefill is gfx95-only (`:103-106`). Working gfx1100 config today: `--attention-backend triton` or `SGLANG_USE_AITER_UNIFIED_ATTN=1`.

Adjacent landmine: `layers/rotary_embedding/factory.py:75-86` `_aiter_rope_unsupported_arch()` returns true only for `gfx1250`, because aiter's rope kernels need `ck_tile` types that do not build there — gfx1100 has the same absence, unlisted.

Our shapes: head_dim 256, 24 q / 4 kv (GQA 6) — exactly on `mha_batch_prefill_func`'s `head_dim <= 256` assert (`aiter_backend.py:1195`); no CK MHA kernel covers 6:1 GQA on gfx1100.

## MoE backends

Selection is per quant-method: `fused_moe_triton/layer.py:557` → `quant_method.get_moe_runner_backend()`; unquantized resolves at `quantization/unquant.py:1122-1123` (`else: MoeRunnerBackend.TRITON`), with an aiter override at `:1126-1140` guarded by `_use_aiter` (`:70` = `SGLANG_USE_AITER and is_hip`).

| variant | entry | gate | gfx1100 |
|---|---|---|---|
| triton fused MoE | `moe_runner/runner.py:100`, kernel `moe_runner/triton.py:135` | none | **WORKS** (only arch-neutral path) |
| fused_moe_native | `fused_moe_native.py:21`, for `num_tokens == 1` at `unquant.py:1174-1183` | none | WORKS (reference speed) |
| aiter fused MoE (CK) | `moe_runner/aiter.py:168/202` → `aiter.fused_moe` | sglang's own refusal returns **None** on gfx1100: `unquant.py:1081-1082` `if not is_gfx95_supported(): return None` | **NEEDS-WORK** — selected, then fails inside aiter |
| flashinfer_trtllm/cutedsl/megamoe | `moe_runner/runner.py:125-147` | CUDA/NVFP4 | NVIDIA-only |
| cutlass_moe / cutlass_w4a8 | `cutlass_moe.py:28/43` | `assert ... is_sm100` (`:151`) | NVIDIA-only |
| mega_moe / sm90 / flydsl | `mega_moe.py:160-176` | `_is_hip and SGLANG_AMD_USE_FLYDSL_MEGA_MOE` (`mega_moe.py:49-52`, off by default) | CDNA-ONLY |
| ep_moe / dwdp / kt_ep | `ep_moe/layer.py:62`, `dwdp_manager.py:34`, `kt_ep_wrapper.py:117` | DeepGEMM branches `... and not _is_hip` (`ep_moe/layer.py:142`) | NEEDS-WORK (multi-GPU transport) |
| paged_experts | `paged_experts/method.py:38` | none; triton contract | WORKS by design |
| qwen4_decode | `moe/qwen4_decode.py:69`, gate `:52-62` | `get_device_capability()[0] != 10 → return None` | NEEDS-WORK — **exactly our shape** (512 experts, bf16), Blackwell-gated |

**Grouped GEMM at batch 1 / 10-of-512: no.** `grouped_gemm` appears only as CUTLASS int8 prose (`cutlass_w4a8_moe.py:71-95`). The Triton path sizes by *token* count, not routed tokens (`moe_runner/triton_utils/fused_moe.py:464`, `:474-483`); the config table is keyed by device name — all `NVIDIA_*` — so gfx1100 always lands on `get_default_config`, whose small-M branch (`fused_moe_triton_config.py:253-260`) yields `BLOCK_M 16 / N 32 / K 64 / GROUP_SIZE_M 1` whenever `M <= E`. With M=1, E=512 that always fires, rows are padded per-expert (`triton_utils/moe_align_block_size.py:32-53`), and 10 real rows become ~160 padded rows: ~6 % arithmetic efficiency, 10 blocks × 2 GEMMs.

## Graph capture

Decode graphs are on by default and arch-neutral: `model_executor/cuda_graph_config.py:149-151` (`decode: PhaseConfig(backend=Backend.FULL)`). Prefill graphs are not: `:119-123` `default_prefill_backend()` → `Backend.BREAKABLE if is_cuda() else Backend.DISABLED`.

Capture is plain torch: `runner/decode_cuda_graph_runner.py:1109-1172` (`graph_capture(stream=...)` + `backend.capture_session`); `runner_backend/full_cuda_graph_backend.py:182` `torch.cuda.CUDAGraph()`, `:198` `graph_ctx = self._device_module.graph`; `utils/common.py:632-636` maps `is_hip()` → `torch.cuda`, i.e. hipGraph. `is_hip` appears nowhere on the capture path except `eager_runner.py:64,74`.

**Decode capture WORKS on gfx1100** (untested for GDN/MTP); **prefill capture is DISABLED by default** and must be opted in.

## Radix cache and any tiered KV

- Tree: `mem_cache/radix_cache.py:250-278` (`children/key/value/lock_ref/last_access_time`); production default is `UnifiedRadixCache` (`mem_cache/registry.py:157`). Device dependence: none — int64 slot indices and Python dicts. **WORKS.**
- Hashing: not a bytes hash — the child dict key is a tuple of raw token ids at page granularity, namespaced by `extra_key`/`cache_salt` (`:222-247`). Chained SHA-256 page digests are separate, for L3 keys/KV events only (`mem_cache/utils.py:116-175`, `cpp_utils/hash_binding.cpp:66-83`).
- Refcount: `inc_lock_ref`/`dec_lock_ref` (`:582-616`) walk to root moving whole node segments between `evictable_size_`/`protected_size_` on the 0↔1 transition; eviction is a min-heap on `eviction_strategy.get_priority` (LRU default) over `evictable_leaves`, freeing page-aligned segments (`:552-580`).
- L2 host tier: `managers/cache_controller.py:295`, pools `mem_cache/pool_host/*`, driver `mem_cache/l2_transfer.py:47-58`; off by default (`arg_groups/fields/memory.py:98-101`). The ROCm code is deliberate: `memory_pool_host.py:44-50` imports `sgl_kernel.kvcacheio` under `if _is_cuda or _is_hip:`, `pool_host/mha.py:240-247` enables JIT HiCache kernels on HIP, `kernels/ops/kvcache/hicache.py:22-33` doubles the ROCm grid quota (`DEFAULT_BLOCK_QUOTA = 32 if _is_hip else 2`), `pool_host/common.py:285-327` uses `hipHostMalloc`.
- **Blocker:** those transfer symbols ship only in the `sgl_kernel` wheel, and the in-tree ROCm AOT build refuses gfx1100 — `kernels/aot/setup_rocm.py:77-81` `if amdgpu_target not in ["gfx942","gfx950","gfx1250"]: sys.exit(1)`, with `docker/rocm.Dockerfile:2-11` offering exactly those `GPU_ARCH` values. L2 = **NEEDS-WORK**: ROCm-ready code, non-RDNA3 build.
- L3 (`hicache_storage.py:176`; backends `storage/backend_factory.py:204-276`) is host-side and device-agnostic, unreachable without L2.

## Model support for qwen3_5 and qwen4_exp

**Both are supported upstream — the report's biggest finding.** Registration is `EntryClass` auto-discovery (`models/registry.py:19-33`), so class presence is the whole story.

qwen3_5: config `configs/qwen3_5.py:8,77` (`model_type = "qwen3_5"`), exported `configs/__init__.py:77`. Model `models/qwen3_5.py`: `Qwen3_5GatedDeltaNet` :358, `Qwen3_5LinearDecoderLayer` :1122, `Qwen3_5AttentionDecoderLayer` :1262, `Qwen3_5ForCausalLM` :1705, `Qwen3_5MoeForCausalLM` :2065, `Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)` :2374 (visual tower = Qwen3-VL ViT), `EntryClass` :3021; MTP `qwen3_5_mtp.py:94,506`. The LLLF hybrid is wired at `layers/attention/attention_registry.py:444-485` (`hybrid_gdn_config` → `GDNAttnBackend` + full-attn side backend).

qwen4_exp: config `configs/qwen4_exp.py:8,118` (text `:16`); model `models/qwen4_exp.py`: `Qwen4ExpNGramEmbedding` :553, `Qwen4ExpPinnedHostEmbedding` :894, `Qwen4ExpPLELayer` :1038, `Qwen4ExpLayerExtensionMixin` :1525 (hyper-connections), decoder layers :1743/:1788, `Qwen4ExpModel(Qwen3_5ForCausalLM)` :1982, `Qwen4ExpVLModel` :2153, `Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration)` :2195, `EntryClass` :2669; MTP `qwen4_exp_mtp.py:27,231`; expert mapping with the fused shared expert at `:2304-2320`.

HIP awareness inside both is real: `qwen3_5.py:1445` sends HIP to the Triton `fused_qk_gemma_rmsnorm` (`models/utils.py:600-640`), `:1602` `forward_prepare_fused_gate` for HIP, `:173-176` aiter/gfx950 small-M fp8 imports, `:1879` a ROCm packed-in-proj path — and `:166-167`: "qwen4_exp shares these classes, but the ROCm packed path is Qwen3.5-only", i.e. Flash-Next takes the generic path. GDN Triton kernels carry only empty-batch HIP guards, no arch gate (`kernels/ops/attention/triton_gdn_fused_proj.py:136,310,717`).

Nothing must be written for model *support*; what must be written is kernel quality — CK-free attention (or the unified path forced on) and the batch-1 MoE kernel.

## Top 10 patch sites

| file:line | what to change | why |
|---|---|---|
| `arg_groups/model_override_base.py:326-327` | arch-aware default; gfx11 → `triton` | every RDNA3 server otherwise boots into a CK backend |
| `utils/common.py:1231-1264` | add `is_rdna3_supported()` | 71 gfx95 sites treat gfx1100 as "no arch" |
| `layers/quantization/unquant.py:1081-1082` | return a real refusal reason on RDNA3 | `None` means "CK MoE is fine", so CK fused MoE is selected |
| `layers/attention/aiter_backend.py:529-534` | auto-enable `use_triton_unified_attention` on non-CDNA | the only non-CK aiter path is env-gated off |
| `layers/rotary_embedding/factory.py:75-86` | add `gfx11` to `_aiter_rope_unsupported_arch()` | same missing `ck_tile` as gfx1250 → aiter rope |
| `kernels/aot/setup_rocm.py:77-81` | admit `gfx1100` to the AOT target list | otherwise `sgl_kernel` cannot be built at all |
| `mem_cache/memory_pool_host.py:44-50` | torch/JIT fallback for `kvcacheio` | HiCache dead despite ROCm-ready code |
| `moe/.../fused_moe_triton_config.py:253-260` | gfx1100-keyed tile config | default small-M tile is a CUDA heuristic at ~6 % |
| `moe/qwen4_decode.py:52-62` | gate on "512 experts, bf16", not sm100 | our decode shape is CUDA-locked |
| `platforms/rocm.py:26-31` | make `hicache_device_kernels` conditional | the flag promises kernels gfx1100 cannot load |

## Top 3 highest-expected-gain changes

1. **Batch-1 10-of-512 fused/grouped MoE for wave32.** Every alternative is padded Triton (~6 %) or Blackwell/CK-gated; for Flash-Next this is most of decode time, and it is simultaneously required and absent upstream.
2. **Force and then accelerate a CK-free attention path.** Arch-aware defaults (patches 1/4/5) buy correctness; a wave32 flash-decoding kernel at head_dim 256 with 6:1 GQA buys speed, mostly for qwen3_5's 16 full-attention layers, which carry far more FLOPs per token than its 48 linear ones.
3. **Build `sgl_kernel` for gfx1100, or a torch fallback for `kvcacheio`.** Unlocks HiCache L2, turning the 46 GiB/s host pool into KV capacity — decisive at 19.98 GiB/card.

## Claims I could not verify

- Whether the installed aiter wheel contains gfx1100 CK/ASM binaries for `paged_attention_ragged` / `fused_moe`, or fails at call time. sglang checks nothing.
- Whether `kernels/jit/csrc/kvcacheio/hicache.cuh` nontemporal builtins compile under hipcc for wave32 gfx1100.
- Whether `wave_lang` targets gfx1100 (`wave_backend.py` has no arch check), or `flex_attention` lowers on ROCm 7.2.
- Whether any prebuilt ROCm `sglang-kernel` wheel targets gfx1100 (out-of-repo).
- Whether 180 B with ~4 B active is servable at all on 2×19.98 GiB — no weight-offload path was audited.
- MTP + decode-graph + GDN-replay interaction on ROCm; `attention_hook.py:394-409` covers ReplaySSM graph safety only in CUDA terms.

## Source map

Arch/platform `srt/utils/common.py` (141, 387-394, 632-636, 1231-1264) + `srt/platforms/rocm.py`; backend choice `arg_groups/{model_override_base.py:277-351,choices.py:69-99,attention_hook.py:280-409}`; attention `layers/attention/{attention_registry,triton_backend,aiter_backend,wave_backend}.py` + `linear/gdn_backend.py`; MoE `layers/moe/moe_runner/*` + `quantization/unquant.py:1072-1140` + `moe/qwen4_decode.py`; graph `model_executor/cuda_graph_config.py:90-235`; cache `mem_cache/{radix_cache,memory_pool_host}.py` + `pool_host/*`; models `configs/{qwen3_5,qwen4_exp}.py` + `models/{qwen3_5,qwen4_exp,*_mtp}.py`; build `kernels/aot/setup_rocm.py:55-95`, `docker/rocm.Dockerfile:2-40`.
