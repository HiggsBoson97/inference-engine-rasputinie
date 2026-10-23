# Probe brief — H-11: engine bring-up, and the dp4a shim

```
STATION: probe (engine bring-up; the first change to `engine/`)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin`, working copy of `~/development/inference-engine-rasputinie` ; branch `engine/bringup` ; scratch `~/b2/`
TOOLSET: ssh, git, hipcc, python, the venv `~/venv/sglang-fork`
BUDGET: 110 tool calls
ROUND: 1
CONTRACT: none
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-11.md
```

## GOAL

Get the engine from "cannot serve this file on ROCm" to the **smallest true next blocker**,
and settle whether the missing GGUF kernels can compile for gfx1100 at all.

`H-10` established four ordered blockers (all verified by the orchestrator):
1. `aiter` imported at module scope under `if _is_hip` (`quark/schemes/quark_w4a4_mxfp4.py:163`),
   not installed → `import sglang.srt.configs.model_config` fails.
2. `ModelConfig` hard-refuses GGUF on ROCm (`configs/model_config.py:2128-2133`).
3. On HIP, `layers/quantization/gguf.py` has no import branch: the chain is `_is_cuda` →
   `_is_musa` → `_is_npu` → `else`, so the five `ggml_*` names are undefined.
4. The ops are absent from the ROCm wheel: 0 of 47 `sgl_kernel` ops are ggml,
   `nm common_ops.so` has 0 ggml symbols, and `csrc/quantization/gguf/gguf_kernel.cu` is not
   in `setup_rocm.py`'s source list.

## THE SHORTCUT WORTH TRYING

The CUDA kernel source does **not** compile for gfx1100 because `__dp4a` is *undeclared* in
this toolchain — verified: `error: use of undeclared identifier '__dp4a'`. There are ~45
`__dp4a` call sites in `csrc/quantization/gguf/vecdotq.cuh`.

**Do not rewrite the call sites.** Define `__dp4a` itself, as a HIP shim header with the two
overloads the CUDA code expects, implemented with inline-asm `v_dot4_i32_i8`. The project
already proved that instruction works: `bench/gfx1100-probes/iq3s_valid.cpp` uses it as inline
asm and measured 521 GB/s. Approximate shape:

```cpp
// dp4a_hip.h  — include before vecdotq.cuh
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
__device__ __forceinline__ int __dp4a(const char4 a, const char4 b, int c) {
  int r; asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}
__device__ __forceinline__ int __dp4a(int a, int b, int c) { /* same, bit-cast */ }
#endif
```

Read `bench/gfx1100-probes/iq3s_valid.cpp` for the exact asm form and operand constraints that
this project has already validated — do not invent the syntax.

## WHAT TO DO, IN ORDER

**1. Fix 1 and 2 on a branch.** `git checkout -b engine/bringup`. Guard the `aiter` import so a
missing package does not break `model_config` (a lazy import or a `try/except ImportError` that
only fails when that quantisation scheme is actually used). Lift the ROCm GGUF refusal in a way
that is explicit and reversible — a clearly-commented allow for gguf when the ops are present,
not a blanket deletion. Commit both, separately, with messages that say why.

**2. Prove the import, then prove the config.** `python -c "import sglang.srt.configs.model_config"`
must succeed. Then construct a `ModelConfig` for
`~/local-inference/models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` far enough to see the next
blocker, and report the **exact** error and its file:line. That error is the deliverable.

**3. The compile test — the real question.** Build the dp4a shim and attempt:

```
hipcc --offload-arch=gfx1100 -O3 -c csrc/quantization/gguf/gguf_kernel.cu \
  -I csrc -I csrc/quantization/gguf -I <whatever else it needs> -include dp4a_hip.h
```

Report the **complete error list, verbatim**, deduplicated and counted by error class (missing
header, missing intrinsic, wrong type, arch-specific). That list is the port's work plan and is
worth more than any partial build. If it compiles, say so and report the object file's size.

**4. If it compiles**, add `gguf_kernel.cu` to `setup_rocm.py`'s source list on the branch and
report what the build would need — **do not run the full wheel build** in this probe.

## RULES

1. **Work on a branch, never `main`.** Commit per change. Leave the repo pushable: `git status`
   clean at the end, branch pushed to `origin`.
2. **Do not touch `bench/` or `docs/`** except your own report file.
3. **No GPU work is required.** Compilation and imports are CPU. If you do launch anything,
   `HIP_VISIBLE_DEVICES=1` only, but prefer not to — another probe may hold the card.
4. **llama.cpp excluded** — no clone, no read, no derived code. The `v_dot4_i32_i8` asm comes
   from this project's own harness.
5. **Never invent an error.** If a command fails, quote it. If a step is not run, say `not-run`
   and why.
6. Restore anything you change outside the branch; leave the working tree clean.

## HOW TO VERIFY YOU DID IT RIGHT

- `import sglang.srt.configs.model_config` succeeds, and you show the command and its exit code.
- The `ModelConfig` failure (or success) is quoted with file:line — a paraphrase is not evidence.
- The compile test's errors are quoted verbatim and classified; a summary without the text does
  not count.
- `git log --oneline engine/bringup` shows one commit per fix, each with a reason; `git status`
  is clean.
- If the shim compiles, the object exists and you give its size and the exact command.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-11.md`, then print it:

```
# H-11 — engine bring-up and the dp4a shim
## Question and why it is the next step
## Branch and commits (sha, message, one line of why each)
## Import result (command, exit code, output)
## ModelConfig result (command, error verbatim with file:line, or the constructed config)
## The compile test (command, full error list classified and counted, or success + object size)
## What the error list implies (the port's work items, ordered by class)
## What I did not do (and why)
## Compliance (llama.cpp excluded; GPU untouched; tree clean)
## Receipts (branch, shas, paths under ~/b2/)
```

Effort: **high**, and precision matters more than coverage. One fix at a time, one error quoted
at a time. The deliverable is an exact work list, not a working engine.
