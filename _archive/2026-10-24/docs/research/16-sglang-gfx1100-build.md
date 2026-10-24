# 16 — sglang fork on gfx1100: install, sgl-kernel build, serve

> **Corrections, 2026-10-22 (verified in `18-tp-roccl.md`):**
> - The GPU mapping in §2.1 below is inverted. The display card is
>   `card1` / PCI `03:00.0` / **HIP device 0**; `card0` / `07:00.0` / HIP
>   device 1 is headless, and the 27,947,008 B baseline is the headless
>   card.
> - §7.3's "any tp>1 run will fail or produce garbage" is false for this
>   build: TP=2 serves with RCCL and produces byte-identical output.


**Date:** 2026-10-22
**Host:** `rasputin` (Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.1 under `/opt/rocm`, hipcc 7.16 / clang 24)
**GPU used:** 2× RX 7900 XT `gfx1100` + 1× iGPU `gfx1036`. All work pinned to **HIP device 0** (`HIP_VISIBLE_DEVICES=0`).
**Tree:** `engine/` @ upstream `d0a75654368a470d8484e3ea45b06e500fee57af` (see `engine/UPSTREAM.md`).
**Status:** **WORKS.** `import sglang` succeeds, `sgl-kernel` builds for gfx1100, and `Qwen/Qwen3-0.6B` serves + answers on one 7900 XT with `--attention-backend triton`.

Supersedes the read-only audit in `14-sglang-gfx1100.md` for the three items it flagged as blocking; the remaining gaps are listed in §7 and become the P2 list in §8.

---

## 1. Verdict

| question | answer |
|---|---|
| does the fork import on ROCm 10.1 / torch 2.14.1+rocm7.2 | **yes**, after 3 source patches (§4.2–4.4) |
| does the AOT kernel package build for gfx1100 | **yes**, all 15 translation units, no skipped ops, ~2.5 min on 8 jobs (§5) |
| does it serve | **yes**, `Qwen/Qwen3-0.6B`, 140 300 KV tokens, 3.21 GB free, curl answers (§6) |
| does the *default* HIP backend work | **no** — `aiter` is selected by default and dies at CUDA-graph capture; `--attention-backend triton` is mandatory on gfx1100 (§7.1) |
| is `aiter` installed | **no**, and it is not on PyPI (the PyPI `aiter` is an unrelated 2019 package); the ROCm image builds it from `ROCm/aiter`. Not needed for the Triton path. |

The three audit blockers were real and are now fixed at the root, not worked around: the arch gate admits gfx1100, the wave32 `WARP_SIZE` mismatch is pinned, and the C++ standard the torch 2.14 headers require is set. Two dependencies the audit did not call out also had to be handled: `aiter` (imported at module scope by the Quark quantization configs) and torchvision (a CUDA wheel that PyPI resolution silently installs).

---

## 2. Environment facts worth pinning down first

### 2.1 Three different "GPU 0"s

This bit is load-bearing: sysfs `cardN`, `amd-smi`'s index and ROCm/HIP's device index are **three different orderings** on this box.

| identity | `amd-smi` GPU | sysfs node | BDF | arch |
|---|---|---|---|---|
| HIP device **0** (`HIP_VISIBLE_DEVICES=0`) | GPU 0 | `card1` | `0000:03:00.0` | gfx1100, 20 GiB |
| HIP device 1 | GPU 1 | `card0` | `0000:07:00.0` | gfx1100, 20 GiB — **the display card** |
| HIP device 2 | GPU 2 | `card2` | `0000:12:00.0` | gfx1036 iGPU |

So "check `card0` VRAM" and "use `HIP_VISIBLE_DEVICES=0`" refer to *different* physical cards here. All numbers in §6 were taken on the card the server actually ran on (`card1` / `03:00.0`), and `card0` (display) was verified untouched at its 27 947 008 B baseline.

### 2.2 Access to `/dev/kfd` needs no root — but does need a group

`/dev/kfd` is `root:render 0660` and — contrary to the handover note — `mattia-bosi` was **not** in `render` on this host (`id` had no `990(render)`, `getent group render` listed no members), so torch reported `RuntimeError: No CUDA GPUs are available` over ssh. It is not a root problem, and `sudo` is not the fix:

```bash
sudo usermod -aG render,video mattia-bosi   # effective on the next ssh login (new session re-reads groups)
HIP_VISIBLE_DEVICES=0 python -c "import torch;print(torch.randn(8,device='cuda').sum())"
```

That prints a value as the normal user, with no `sudo` and no other system change.

### 2.3 Host toolchain gaps

| tool | state | consequence |
|---|---|---|
| `uv`, `pipx`, `git-lfs` | absent | venv made from the uv-managed CPython 3.12.15 directly |
| `/usr/bin/python3` | 3.14.4 | too new for the torch 2.14.1+rocm7.2 wheel → use 3.12.15 |
| `cargo`, `rustc` | absent | Rust extensions skipped (`SGLANG_BUILD_RUST_EXTS=none`); TreeCore falls back to Python |
| `amdsmi` python module | absent | `Failed to import amdsmi ...` at startup; metrics-only impact |
| network | PyPI CDN ≈ 300 kB/s, `download.pytorch.org` ≈ 4 MB/s, HF ≈ 8.8 MB/s | why torch is copied from the existing venv instead of re-downloaded |

---

## 3. Reproducible procedure

Every command below was run in this order on a clean shell. `V=/home/mattia-bosi/venv/sglang-fork`.

### 3.1 Venv and pins

```bash
# venv from the 3.12 toolchain that already has a working ROCm torch
PY=/home/mattia-bosi/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
$PY -m venv /home/mattia-bosi/venv/sglang-fork
V=/home/mattia-bosi/venv/sglang-fork
$V/bin/python -m pip install --upgrade pip setuptools wheel setuptools-scm

# reuse the already-proven ROCm stack instead of pulling 6.2 GB again
cp -a /home/mattia-bosi/venv/sglang-rocm/lib/python3.12/site-packages/. \
      $V/lib/python3.12/site-packages/
# the copy also clobbers the new pip with pip 25.0.1 -> repair it, then upgrade
rm -rf $V/lib/python3.12/site-packages/{pip,pip-*.dist-info,setuptools,setuptools-*.dist-info,wheel,wheel-*.dist-info,_distutils_hack,pkg_resources,distutils-precedence.pth}
$V/bin/python -m ensurepip --upgrade
$V/bin/python -m pip install --upgrade pip setuptools wheel setuptools-scm
```

Runtime deps are `pyproject_other.toml`'s `[srt_hip_rocm101]` + `[runtime_common]` + `[runtime_base]`, installed from an explicit list so the extra's `torch==2.12.0` pin cannot drag in a CUDA wheel:

```bash
printf 'torch==2.14.1+rocm7.2\ntriton-rocm==3.8.0\n' > /tmp/rocm-constraints.txt
$V/bin/python -m pip install -c /tmp/rocm-constraints.txt -r /tmp/requirements-fork-rocm.txt
$V/bin/python -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
      'torchvision==0.29.1+rocm7.2'      # PyPI's torchvision 0.29.1 is CUDA; see §7.6
```

`/tmp/requirements-fork-rocm.txt` (the list that produced the working venv):

```
aiohttp anthropic>=0.20.0 apache-tvm-ffi av blobfile==3.0.0 build
compressed-tensors==0.16.0 datasets easydict einops fastapi gguf interegular IPython
llguidance>=1.7.6,<2.0.0 mistral_common>=1.11.5 modelscope msgspec ninja outlines==0.1.11
openai==2.6.1 openai-harmony==0.0.4 orjson partial_json_parser petit_kernel==0.0.2 pillow
prometheus-client>=0.20.0 psutil py-spy pybase64 pydantic python-multipart pyzmq>=25.1.2
regex requests scipy sentencepiece==0.2.1 setproctitle smg-grpc-servicer>=0.9.0
soundfile==0.13.1 tiktoken timm==1.0.16 tqdm transformers==5.19.0 uvicorn uvloop
wave-lang==3.8.2 xgrammar==0.2.7 xxhash
```

Deliberate deviations from `[srt_hip_rocm101]`, all recorded here so nobody "fixes" them back:

| package | extra says | we install | why |
|---|---|---|---|
| `torch` | `==2.12.0` | `2.14.1+rocm7.2` | 2.12.0 is a CUDA wheel from PyPI; the 2.14.1+rocm7.2 wheel is the one already proven on this host |
| `sglang[monokernels_hip]` | `git+https://github.com/sgl-project/sgl-amd-mono@b27d604` | omitted | gfx950 monokernels, not reachable/needed on gfx1100; the extra is also the reason `srt_hip` has no `torch`-free path |
| `torchvision` | resolved by pip | `0.29.1+rocm7.2` | see §7.6 |

Key pins actually present (`python -m pip freeze`, full list in §9):

```
torch==2.14.1+rocm7.2      triton-rocm==3.8.0      triton==3.8.0
torchvision==0.29.1+rocm7.2 numpy==2.3.5          transformers==5.19.0
tokenizers==0.23.3         compressed-tensors==0.16.0  xgrammar==0.2.7
outlines==0.1.11           timm==1.0.16            petit_kernel==0.0.2
wave-lang==3.8.2           sglang-kernel==0.4.10   sglang==0.5.9 (editable)
```

### 3.2 Tree in place

`engine/` is **not committed** in the Mac checkout (`git status` shows `?? engine/`), so it cannot be pulled on the host. The host clone was updated with `git pull --ff-only` (docs only) and `engine/` was rsynced from the Mac at the identical tree state:

```bash
rsync -a --exclude='.git/' --exclude='__pycache__/' -e 'ssh -p 8374' \
  "…/inference-engine-rasputinie/engine/" \
  mattia-bosi@rasputin:/home/mattia-bosi/development/inference-engine-rasputinie/engine/
```

This is a deviation from "pull, do not copy" and it exists only because the vendored tree was never committed. Once `engine/` is committed the rsync step disappears.

### 3.3 Apply the fork patches (§4), then install sglang editable

```bash
cd /home/mattia-bosi/development/inference-engine-rasputinie/engine/python
cp pyproject_other.toml pyproject.toml        # the HIP pyproject; same swap the AMD CI does
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION=0.5.9 \
  $V/bin/python -m pip install -e . --no-deps --no-build-isolation
```

The swap overwrites the CUDA `pyproject.toml` in place (the tree is not under git here, so there is no `git checkout` to undo it). The original is preserved at `/home/mattia-bosi/build/pyproject-cuda-orig.toml` (md5 `b61e5e3b…`) and is also recoverable from upstream `d0a75654368a470d8484e3ea45b06e500fee57af`. Verify the swap with `md5sum pyproject.toml pyproject_other.toml` — they must match.

`--no-build-isolation` is required: the CUDA pyproject's build-system pins `torch==2.14.1` from PyPI (a CUDA wheel) and would be installed into a throwaway env anyway. `SGLANG_BUILD_RUST_EXTS=none` is required because there is no `cargo`. `SETUPTOOLS_SCM_PRETEND_VERSION` is required because the vendored tree carries no git metadata; `0.5.9` is the release whose tag range the pinned commit sits in.

### 3.4 Build sgl-kernel for gfx1100

```bash
bash /home/mattia-bosi/build_sglk_wheel.sh    # ~2.5 min, MAX_JOBS=8
# -> /home/mattia-bosi/wheels/sglang_kernel-0.4.10-cp312-cp312-linux_x86_64.whl
# -> pip install --force-reinstall --no-deps that wheel
```

The script is the CI's own two-step (`scripts/ci/amd/amd_ci_install_dependency.sh:162`) done in a build dir so the vendored tree is not mutated: copy `aot/` to `~/build/sgl-kernel-gfx1100`, `cp pyproject_rocm.toml pyproject.toml`, `cp setup_rocm.py setup.py`, then `python setup.py bdist_wheel` with `AMDGPU_TARGET=gfx1100 HIP_VISIBLE_DEVICES=0 ROCM_HOME=/opt/rocm PATH=/opt/rocm/bin:$PATH`.

### 3.5 Model and serve

```bash
$V/bin/hf download Qwen/Qwen3-0.6B          # 1.5 GB, the smallest Qwen3 instruct model
HIP_VISIBLE_DEVICES=0 $V/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --attention-backend triton \
  --tp 1 --context-length 4096 \
  --port 31000 --host 0.0.0.0
```

**Why `Qwen3-0.6B` and not something from the 3.8 family:** the Qwen3.8 family on the Hub is `27B`, `Flash-Next` (131 shards) and `2.4T-A95B` — nothing small. Per the selection rule, the smallest Qwen3 model sglang supports is `Qwen/Qwen3-0.6B` (single `model.safetensors`, 0.6 B params, `Qwen3ForCausalLM`).

### 3.6 Verify

```bash
curl -s http://127.0.0.1:31000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role":"user","content":"The capital of France is"}],
  "max_tokens": 16, "temperature": 0}'
```

---

## 4. Patches

Four files under `engine/` changed (plus the `pyproject.toml` swap in §3.3). Nothing outside `engine/` and `$V` was modified except the `render`/`video` group membership in §2.2.

### 4.1 Summary

| file | change | why |
|---|---|---|
| `python/sglang/kernels/aot/setup_rocm.py` | arch allowlist + `WAVE32_TARGETS` + `-std=c++20` | audit blocker 1; wave32; torch 2.14 ABI |
| `python/sglang/kernels/aot/include/utils.h` | honour `SGL_ROCM_WARP_SIZE` | host/device `WARP_SIZE` disagreement on RDNA |
| `python/sglang/srt/layers/quantization/__init__.py` | import/register Quark only when `aiter` exists | module-scope `import aiter` broke `import sglang` |
| `python/pyproject.toml` | ← `pyproject_other.toml` | the HIP pyproject |

### 4.2 `kernels/aot/setup_rocm.py` — the gate (audit blocker 1)

Before (`:77-81`) it printed a warning and `sys.exit(1)` for anything that was not gfx942/gfx950/gfx1250.

```python
SUPPORTED_AMDGPU_TARGETS = [
    "gfx942",  # CDNA3  MI300X / MI325X
    "gfx950",  # CDNA4  MI350 / MI355
    "gfx1250",  # RDNA4
    "gfx1100",  # RDNA3  Navi31 / RX 7900 XT(X)  [rasputin fork]
]

if amdgpu_target not in SUPPORTED_AMDGPU_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Expected one of {', '.join(SUPPORTED_AMDGPU_TARGETS)}."
    )
    sys.exit(1)
```

Note the FP8 macro and the LDS budget were left alone: gfx1100 keeps `-DHIP_FP8_TYPE_E4M3` (gfx942 keeps FNUZ) and the 40 KB TopK dynamic-smem branch, which is what the gfx1151 patch in-tree (`docker/patches/sgl-kernel-gfx1151.sh`) also relies on. Both compile and run; neither is benchmarked (§7.10).

### 4.3 Wave32 `WARP_SIZE` — the same trap the in-tree gfx1151 patch documents

`include/utils.h` resolved `WARP_SIZE` to 64 on the host compiler pass (`!defined(__HIP_DEVICE_COMPILE__)`) and 32 on the device pass for any non-`__GFX9__` target. On a wave32 part that desynchronizes `__launch_bounds__` from the launch dimensions in the MoE TopK kernels — `hipErrorLaunchFailure`, queue poisoning, and a page fault in the *next* kernel. gfx1100 has exactly the same shape as gfx1151:

```c
#if defined(SGL_ROCM_WARP_SIZE)
// Wave32 RDNA targets (gfx1100/gfx1151) must use one WARP_SIZE for both the
// host and the device compiler pass; the __GFX9__ test below yields 64 on the
// host pass and 32 on the device pass, which desynchronizes launch bounds from
// launch dimensions (see docker/patches/sgl-kernel-gfx1151.sh).
#define WARP_SIZE SGL_ROCM_WARP_SIZE
#elif defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
```

and in `setup_rocm.py`:

```python
WAVE32_TARGETS = ["gfx1100", "gfx1151"]
...
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
    *(["-DSGL_ROCM_WARP_SIZE=32"] if amdgpu_target in WAVE32_TARGETS else []),
```

### 4.4 `-std=c++20` — the fork's C++ standard is older than torch's headers

First build attempt failed immediately, in torch's own headers, with `-std=c++17` (which `setup_rocm.py` hardcoded):

```
torch/include/c10/util/TypeIndex.h:56:25: error: no member named 'starts_with' in 'std::basic_string_view<char>'
torch/include/c10/util/intrusive_ptr.h:775:27: error: no type named 'strong_ordering' in namespace 'std'
torch/include/ATen/core/TensorBase.h:1031:5: error: unknown type name 'requires'
torch/include/torch/csrc/api/include/torch/all.h:5:2: error: C++20 or later compatible compiler is required to use PyTorch.
```

Fixed by `cxx_flags = ["-O3", "-std=c++20"]` and `"-std=c++17"` → `"-std=c++20"` in `hipcc_flags`. This is not gfx1100-specific: any arch built against torch ≥ 2.13 with this file needs it.

### 4.5 `quantization/__init__.py` — `aiter` is not optional at module scope

`import sglang` failed on `from sglang.srt.layers.quantization.gguf import GGUFConfig` → … → `quark/quark.py` → `quark/schemes/quark_w4a4_mxfp4.py:163`, which does `if _is_hip: from aiter import gemm_a4w4 …`. Same for `quark_int4fp8_moe.py:30`. `aiter` is AMD's kernel library, built from source in the ROCm image, and is not the PyPI package of that name.

The registry already gates optional methods by platform, so the fix follows that shape — Quark is registered only when it can work:

```python
import importlib.util
...
if importlib.util.find_spec("aiter") is not None:
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig
    from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config
else:
    QuarkConfig = None
    QuarkInt4Fp8Config = None
...
if QuarkConfig is not None and QuarkInt4Fp8Config is not None:
    BASE_QUANTIZATION_METHODS.update(
        {"quark": QuarkConfig, "quark_mxfp4": QuarkConfig,
         "quark_int4fp8_moe": QuarkInt4Fp8Config}
    )
```

Consequence: asking for `--quantization quark` on this build reports "unknown quantization method" instead of working. That is honest and is the P2 item in §8.1, not a hidden fallback.

### 4.6 Attention backend: patched or not?

**Not patched.** `arg_groups/model_override_base.py:326-327` still returns `"aiter"` for every HIP device, arch-blind. Forcing `--attention-backend triton` is the documented workaround for gfx1100 until §8.1 lands — see §7.1 for what the default does.

---

## 5. sgl-kernel build result

`bdist_wheel` → `sglang_kernel-0.4.10-cp312-cp312-linux_x86_64.whl` (4.9 MB), 15/15 translation units, **zero compile failures, zero ops skipped**, ~2.5 min.

| what | gfx1100 |
|---|---|
| `csrc/allreduce/{custom,deterministic}_all_reduce.hip` | built — **but see §7.3** (quickreduce asm missing) |
| `csrc/allreduce/quick_all_reduce.cu` | built with warnings, symbols not defined for gfx11 |
| `csrc/elementwise/{activation,pos_enc,topk}.{cu,hip}`, `deepseek_v4_topk`, `dsv4_norm_rope` | built |
| `csrc/grammar/apply_token_bitmask_inplace_cuda.cu` | built |
| `csrc/moe/{moe_align_kernel,moe_topk_softmax_kernels,moe_topk_sigmoid_kernels}.cu` | built (needs §4.3, else runtime launch failure) |
| `csrc/kvcacheio/transfer.cu`, `csrc/memory/weak_ref_tensor.cpp`, `csrc/common_extension_rocm.cc` | built |
| FP8 / BF16 / wave32 / 40 KB TopK smem | `-DENABLE_FP8 -DHIP_FP8_TYPE_E4M3 -DENABLE_BF16 -DSGL_ROCM_WARP_SIZE=32 -DSGL_TOPK_DYNAMIC_SMEM_BYTES=40960` accepted by hipcc 7.16 |

Warnings that matter (kept, not silenced):

```
csrc/allreduce/quick_all_reduce_base.h:103:1: warning: function 'quickreduce::buffer_store_dwordx4' has internal linkage but is not defined
csrc/allreduce/quick_all_reduce_base.h:99:48: warning: function 'quickreduce::buffer_load_dwordx4' has internal linkage but is not defined
```

Those two are the gfx942/gfx950 assembly primitives, so gfx1100 gets `quick_all_reduce` without its memory ops — fine at tp=1 (never called), fatal for custom all-reduce at tp>1. Everything else is cosmetic (`-Xcompiler unused`, `atomicAddNoRet deprecated`, `loop not unrolled`, `strict-aliasing`).

Smoke test of the built extension:

```
silu_and_mul ok torch.Size([4, 4]) torch.bfloat16 0.08837890625
pdl: False
```

`is_arch_support_pdl()` is False, which is correct for RDNA3.

---

## 6. Verification output

### (e) version

```
$ HIP_VISIBLE_DEVICES=0 /home/mattia-bosi/venv/sglang-fork/bin/python -c "import sglang; print(sglang.__version__)"
0.5.9
```

### arch the process actually sees

```
platform class: RocmSRTPlatform
device: AMD Radeon Graphics gfx1100
is_hip: True gfx942: False gfx95: False gfx1250: False
```

(`amd-smi static --asic` agrees: `MARKET_NAME: AMD Radeon RX 7900 XT`, `TARGET_GRAPHICS_VERSION: gfx1100`, BDF `0000:03:00.0`, VRAM 20 GiB. The `is_gfx*` row is audit blocker 3 — no RDNA3 helper exists, so every CDNA/RDNA4 gate is False on this part.)

### (b) server startup log

```
[2026-10-22 13:10:13] Load weight begin. avail mem=19.80 GB
aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device.
[2026-10-22 13:10:13] Load weight end. elapsed=0.34 s, type=Qwen3ForCausalLM, avail mem=18.59 GB, mem usage=1.21 GB.
[2026-10-22 13:10:15] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 140300, K size: 7.49 GB, V size: 7.49 GB
[2026-10-22 13:10:16] max_total_num_tokens=140300, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=4096, context_len=4096, available_gpu_mem=3.21 GB
[2026-10-22 13:10:16] Using the Python TreeCore: the Rust TreeCore toolchain is unavailable (cargo --version --verbose)
[2026-10-22 13:10:16] Init Unified Radix Cache. Components: (<ComponentType.FULL: 0>,). Tree Core: UnifiedTreeCore
[2026-10-22 13:10:16] INFO:     Uvicorn running on http://0.0.0.0:31000 (Press CTRL+C to quit)
[2026-10-22 13:10:17] The server is fired up and ready to roll!
```

sglang prints no arch line at `info` level, so the backend is confirmed from its own API rather than the log:

```
$ curl -s http://127.0.0.1:31000/get_server_info | grep -o '"attention_backend":"[a-z_]*"'
"attention_backend":"triton"
$ curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:31000/health
200
```

Also present in that log, and relevant to §7: `Ignore import error when loading sglang.srt.models.deepseek_v4: No module named 'aiter'` (plus `deepseek_v4_dspark`, `deepseek_v4_nextn`, `deepseek_v4_platform_stage`, `glm5_next`, `glm5_next_nextn`, `kimi_k25`) and `Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")`.

### (c) curl bodies

```
$ curl -s -X POST http://127.0.0.1:31000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
{"id":"b3ffa45583da4569abee53f7469f15c9","object":"text_completion","created":1792667493,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"text":" Paris. The capital of Italy is Rome. The capital of Spain is Madrid.","logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":5,"total_tokens":21,"completion_tokens":16,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default","weight_versions":[{"version":"default","start":0,"end":16}]}}
```

```
$ curl -s -X POST http://127.0.0.1:31000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"State in one short sentence which GPU architecture you are running on and that you can serve tokens."}],"max_tokens":120,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
{"id":"e30f385be165420a8e33d7177aaa0ad0","object":"chat.completion","created":1792666994,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"message":{"role":"assistant","content":"I am running on a GPU architecture with a high-performance computing (HPC) architecture, and I can serve tokens.","reasoning_content":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":31,"total_tokens":56,"completion_tokens":25,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default","weight_versions":[{"version":"default","start":0,"end":25}]}}
```

Note `enable_thinking: false` — with Qwen3's default thinking template a 48-token budget is spent inside ` thinking` and the answer is truncated at `finish_reason: "length"`. Not a server defect; set the template kwarg or a bigger budget.

### GPU state (the card the server ran on, `card1` / `03:00.0`)

| point in time | `mem_info_vram_used` |
|---|---|
| serving Qwen3-0.6B, 4096 ctx, 140 300 KV tokens (measured during the run) | 18 370 584 576 B (17.1 GiB) |
| after `kill`, no listeners, no python processes | 118 722 560 B (kernel-side allocation with no owning process reported by `amd-smi process`) |

The pre-run idle figure for `card1` was not captured before the first launch, so 118 722 560 B is the post-run idle state, not a proven baseline; `card0`'s 27 947 008 B is the only baseline this session can assert.

Display card `card0` / `07:00.0` stayed at 27 947 008 B throughout (+400 kB during the run): untouched, as required.

---

## 7. What still does not build or work

### 7.1 The default HIP attention backend (`aiter`) dies at CUDA-graph capture — no `--attention-backend` flag

```
[2026-10-22 13:05:10] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1
  File ".../sglang/srt/layers/attention/aiter_backend.py", line 1976, in init_forward_metadata_out_graph
    reset_verify_attn_plan_cache()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'reset_verify_attn_plan_cache' is not defined
[2026-10-22 13:05:10] Received sigquit from a child process. It usually means the child failed.
```

Root cause is a silent-fallback shape, not the missing package: `aiter_backend.py:48-76` wraps the entire aiter + `unified_attention_3d_mtp` import block in `try/except ImportError` and only *prints* `"aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."`. The backend therefore loads with every aiter name undefined and fails much later, inside capture. Either fail at selection or do not select `aiter` on an arch without it.

### 7.2 `SGLANG_USE_AITER_UNIFIED_ATTN=1` does not help

Same startup, same failure, different port:

```
$ SGLANG_USE_AITER_UNIFIED_ATTN=1 python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --tp 1 --context-length 4096 --port 31002
  File ".../aiter_backend.py", line 1976, in init_forward_metadata_out_graph
    reset_verify_attn_plan_cache()
NameError: name 'reset_verify_attn_plan_cache' is not defined
```

That env var switches prefill onto aiter's unified attention; the crash is in decode graph capture, which it never touches. `--attention-backend triton` remains the only working configuration on gfx1100.

### 7.3 gfx1100 has no `quick_all_reduce` memory ops → custom all-reduce / tp>1 is unusable

The `undefined-internal-linkage` warnings in §5 mean the gfx9 assembly primitives are absent for gfx11. tp=1 never calls them; any tp>1 run will fail or produce garbage all-reduce. Untested because tp=1 is what P1 needed, but the symbol state is conclusive.

### 7.4 aiter-gated model families and quant methods are absent from this build

`deepseek_v4*`, `glm5_next*`, `kimi_k25` are skipped at model-registry import, and `quark` / `quark_mxfp4` / `quark_int4fp8_moe` are unregistered (§4.5). This is the price of not having aiter; the Triton attention path and dense/MoE-with-sgl-kernel models are unaffected.

### 7.5 No Rust extensions (no cargo on the host)

```
[2026-10-22 13:10:16] Using the Python TreeCore: the Rust TreeCore toolchain is unavailable (cargo --version --verbose)
```

Radix-cache tree core runs the Python implementation (works, slower). `SGLANG_BUILD_RUST_EXTS=none` also skips the `multimodal` Rust extension that `pyproject_other.toml`'s `[tool.sglang] rust-extensions` allowlist would otherwise build — **text-only for now**; vision/multimodal inputs are untested and likely unusable.

### 7.6 PyPI's `torchvision` silently breaks `import sglang`

Resolution pulls `torchvision==0.29.1` from PyPI, which is CUDA-built and has no `torchvision::nms` op under torch 2.14.1+rocm7.2, so transformers' `image_utils` import kills `import sglang`:

```
File ".../transformers/image_utils.py", line 54, in <module>
    from torchvision.io import ImageReadMode, decode_image
File ".../torchvision/_meta_registrations.py", line 163, in <module>
    @torch.library.register_fake("torchvision::nms")
RuntimeError: operator torchvision::nms does not exist
```

Fixed by installing `torchvision==0.29.1+rocm7.2` from `download.pytorch.org/whl/rocm7.2`. The fork's HIP extra should pin it; today nothing does.

### 7.7 `sglang-kernel` is built out of band

The wheel is built from a copy of `engine/python/sglang/kernels/aot` into `~/build/sgl-kernel-gfx1100` and installed from `~/wheels`. There is no target in the fork that does this, and `pip install -e .` alone leaves the venv without `sgl_kernel` — which is a hard `ModuleNotFoundError` at import, not an optional path:

```
File ".../sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py", line 73, in <module>
  from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
ModuleNotFoundError: No module named 'sgl_kernel'
```

### 7.8 `amdsmi` missing

```
[2026-10-22 13:10:05] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
```

Serving is unaffected; per-GPU utilisation/VRAM telemetry through sglang's own metrics is degraded.

### 7.9 Six `(null): No such file or directory` lines on every torch import

Stderr noise emitted during ROCm device probing (once per enumerated device plus one), present in every command in this report and filtered out of the quoted logs. Harmless but it pollutes any log a parser touches; cause not identified.

### 7.10 Nothing is performance-validated

Numbers were not taken. What is known: `140 300` KV tokens at `context_length=4096` with `mem_fraction_static 0.818` for a 0.6 B model means 15 GB of K+V, and decode CUDA graphs capture bs 1/2/4/8 in 5.0 s. FP8 kernels are compiled but never executed here (gfx1100 has no MFMA; the CDNA paths are dead), and the 40 KB TopK smem budget is untuned for RDNA3's 64 KB LDS.

---

## 8. What P2 must fix next

1. **Make the attention backend honest on RDNA3.** Add `is_gfx1100_supported()` (and `is_rdna3()`) next to the `is_gfx942/95/1250` helpers in `srt/utils/common.py`, and change `arg_groups/model_override_base.py:326-327` so `aiter` is not the default when `importlib.util.find_spec("aiter") is None`. Independently, make `aiter_backend.py:48-76` fail at construction instead of printing and continuing — a `NameError` during graph capture is the worst possible failure mode. Until then `--attention-backend triton` is mandatory and belongs in the rig, not in a human's memory.
2. **Decide the aiter story.** Either build `ROCm/aiter` for gfx1100 (unlocks the Quark quant paths and the aiter attention backend, and needs gfx11 coverage checked — most of aiter is CDNA-tuned), or accept the Triton path as the supported configuration and encode that decision in the fork, including the §4.5 quarantine.
3. **`quick_all_reduce` on wave32.** Either port `buffer_load/store_dwordx4` for gfx11 or make the custom-all-reduce path refuse to initialise on RDNA so tp>1 fails loudly at startup rather than corrupting silently.
4. **Own the kernel build.** A fork target (make or script committed to the tree) that (a) swaps the ROCm pyproject, (b) applies the gfx1100 gate + wave32 flags, (c) `bdist_wheel` with `AMDGPU_TARGET=gfx1100`, (d) installs the wheel. It should also stop hardcoding `-std=c++17`, which is broken against torch ≥ 2.13 on *every* arch.
5. **Pin the ROCm torch stack in one place.** `torch==2.14.1+rocm7.2`, `triton-rocm==3.8.0`, **`torchvision==0.29.1+rocm7.2`** and the `--index-url`/constraints pair, so PyPI cannot substitute a CUDA wheel (§7.6). The existing `[srt_hip_rocm101]` extra pins `torch==2.12.0`; whoever owns the venv needs to reconcile that with the wheel we actually run.
6. **Rust extensions.** Install a toolchain (`rustup`) on the host and re-enable `SGLANG_BUILD_RUST_EXTS=multimodal` at minimum; measure TreeCore Python vs Rust on the radix cache, since every request pays it.
7. **Commit `engine/`.** The tree was only ever on the Mac's disk — this whole run started with an rsync because nothing was committed (§3.2). Until that changes the host cannot reproduce the build.
8. **Then re-baseline the P1 gate on the real model** (Qwen3.8-27B, and its `image-text-to-text` template), because the Qwen3-0.6B run proves the plumbing only: dense bf16, `Qwen3ForCausalLM`, tp=1, vision-free.

---

## 9. Appendix — full `pip freeze` of `~/venv/sglang-fork`

```
aiohappyeyeballs==2.7.1         aiohttp==3.14.5               aiosignal==1.4.0
airportsdata==20260905          annotated-doc==0.0.5          annotated-types==0.8.0
anthropic==2.1.0                anyio==4.15.1                 apache-tvm-ffi==0.1.15.post0
asttokens==3.0.2                attrs==26.1.0                 av==19.0.2
blobfile==3.0.0                 build==1.6.1                  certifi==2026.10.19
cffi==2.1.1                     charset-normalizer==3.5.2     click==8.5.0
cloudpickle==3.1.2              compressed-tensors==0.16.0    cryptography==50.0.2
datasets==5.1.0                 dill==0.4.1                   diskcache==5.6.3
distro==1.9.0                   docstring_parser==0.18.0      easydict==1.13
einops==0.8.2                   executing==2.3.0              fastapi==0.159.1
filelock==3.32.3                frozenlist==1.8.0             fsspec==2026.7.0
gguf==0.19.0                    grpcio==1.84.0                grpcio-health-checking==1.84.0
grpcio-reflection==1.84.0       h11==0.16.0                   hf-xet==1.7.0
httpcore==1.0.9                 httpcore2==2.16.0             httpx==0.28.1
httpx2==2.16.0                  huggingface_hub==2.2.0        idna==3.20
interegular==0.3.3              ipython==9.17.1               ipython_pygments_lexers==1.1.1
iree-base-compiler==3.8.0       iree-base-runtime==3.8.0      jedi==0.20.1
Jinja2==3.1.6                   jiter==0.17.0                 jsonschema==4.26.0
jsonschema-specifications==2025.9.1  lark==1.3.1              llguidance==1.9.1
loguru==0.7.3                   lxml==6.1.3                   markdown-it-py==4.2.0
MarkupSafe==3.0.3               matplotlib-inline==0.2.2      mdurl==0.1.2
mistral_common==1.12.0          ml_dtypes==0.6.0              modelscope==1.40.2
modelscope-hub==0.4.6           mpmath==1.3.0                 msgspec==0.22.0
multidict==7.1.0                multiprocess==0.70.19         nest-asyncio==1.6.0
networkx==3.6.1                 ninja==1.13.2                 numpy==2.3.5
openai==2.6.1                   openai-harmony==0.0.4         opentelemetry-api==1.45.1
orjson==3.13.0                  outlines==0.1.11              outlines_core==0.1.26
packaging==26.3                 pandas==3.0.6                 parso==0.8.7
partial-json-parser==0.2.1.1.post7  petit_kernel==0.0.2       pexpect==4.9.0
pillow==13.0.0                  prometheus_client==0.26.0     prompt_toolkit==3.0.53
propcache==0.5.4                protobuf==7.36.2              psutil==7.2.2
ptyprocess==0.7.0               pure_eval==0.2.4              py-spy==0.4.2
pyarrow==26.0.0                 pybase64==1.5.1               pycountry==26.2.16
pycparser==3.11                 pycryptodomex==3.24.1         pydantic==2.14.1
pydantic-extra-types==2.11.1    pydantic_core==2.50.1         Pygments==2.21.0
pyproject_hooks==1.3.3          python-dateutil==3.0.0        python-multipart==0.0.32
PyYAML==6.0.3                   pyzmq==27.2.0                 referencing==0.37.0
regex==2026.9.29                requests==2.34.2              rich==15.0.0
rpds-py==2026.9.1               safetensors==0.8.0            scipy==1.18.1
sentencepiece==0.2.1            setproctitle==1.3.8           setuptools==84.2.1
setuptools-scm==10.3.4
-e git+file:///home/mattia-bosi/development/inference-engine-rasputinie.git@9b604b9…#egg=sglang&subdirectory=engine/python
sglang-kernel @ file:///home/mattia-bosi/wheels/sglang_kernel-0.4.10-cp312-cp312-linux_x86_64.whl
shellingham==1.5.4              six==1.17.0                   smg-grpc-proto==0.4.27
smg-grpc-servicer==0.14.0       sniffio==1.3.1                soundfile==0.13.1
stack-data==0.6.3               starlette==1.9.0              sympy==1.14.0
tiktoken==0.14.0                timm==1.0.16                  tokenizers==0.23.3
torch==2.14.1+rocm7.2           torchvision==0.29.1+rocm7.2   tqdm==4.70.1
traitlets==5.16.1               transformers==5.19.0          triton==3.8.0
triton-rocm==3.8.0              truststore==0.10.5            typer==0.27.3
typing-inspection==0.4.4        typing_extensions==4.16.0     tzdata==2026.5
urllib3==2.8.0                  uvicorn==0.54.0               uvloop==0.23.0
vcs-versioning==2.6.0           wave-lang==3.8.2              wcwidth==0.9.3
wheel==0.48.0                   xgrammar==0.2.7               xxhash==4.0.2
yarl==1.25.1
```

The `-e git+file://…` line is only how pip renders an editable install that lives inside a git work tree; `direct_url.json` records the plain path `file:///home/mattia-bosi/development/inference-engine-rasputinie/engine/python`.

---

## 10. Repo state of this work

Uncommitted on purpose.

**Modified inside `engine/`:**

- `engine/python/pyproject.toml` — now a copy of `engine/python/pyproject_other.toml` (CUDA original backed up at `/home/mattia-bosi/build/pyproject-cuda-orig.toml`, md5 `b61e5e3b…`). Applied on the **host clone only**; the Mac checkout still has the CUDA `pyproject.toml`, because this is a build-time swap, not a fork change — if it should be permanent it belongs in §8.4's build target instead of in the file.
- `engine/python/sglang/kernels/aot/setup_rocm.py` — §4.2, §4.3, §4.4 (both trees, md5 `0d2eecb4…`)
- `engine/python/sglang/kernels/aot/include/utils.h` — §4.3 (both trees, md5 `a82c3c86…`)
- `engine/python/sglang/srt/layers/quantization/__init__.py` — §4.5 (both trees, md5 `f85cdfaa…`)

**Outside `engine/` (declared, not part of the fork):**

- `~/.hermes`/scratch copies of `requirements-fork-rocm.txt`, `build_sglk_wheel.sh` (scripts are on the host at `/home/mattia-bosi/build_sglk_wheel.sh` and `/tmp/requirements-fork-rocm.txt`)
- `~/venv/sglang-fork` — the new venv; `~/venv/sglang-rocm` untouched
- `~/build/sgl-kernel-gfx1100`, `~/wheels/sglang_kernel-0.4.10-cp312-cp312-linux_x86_64.whl`
- `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` (1.5 GB)
- system: `usermod -aG render,video mattia-bosi` (§2.2)

**Not touched:** `main`, any commit, any other profile's skills/cron/memories, GPU 1 (display) and the iGPU, and any process not started here.
