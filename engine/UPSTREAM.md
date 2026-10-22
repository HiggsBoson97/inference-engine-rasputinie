# Provenance

| | |
|---|---|
| Upstream | https://github.com/sgl-project/sglang |
| Pinned commit | `d0a75654368a470d8484e3ea45b06e500fee57af` |
| Vendored | 2026-10-22 |
| Licence | Apache-2.0 (see `LICENSE` in this tree) |

## How this tree is maintained

This is a **vendored snapshot, not a submodule**. The engine is reshaped in
place: we delete what this machine cannot use and rewrite what it needs.

- **Deletions are deliberate commits**, each with the measurement or capability
  argument that justifies it. Nothing is deleted "because it looks unused".
- **Upstream is a reading reference, not a rebase target.** There is no merge
  path back to sglang and no intention to preserve one.
- **Every file upstream ships is committed here.** If a file exists in the
  pinned upstream tree and not in this repo, that is a defect, not a cleanup.

## Two facts that shaped this import

1. Previous iteration's ignore rules silently dropped 836 files from this tree —
   including all of `python/sglang/srt/models/` — producing a repo that could
   not import the engine it was supposed to contain. The root `.gitignore` now
   anchors every directory rule, and `.mod` files are no longer excluded.
   Verification: `git ls-files engine | wc -l` must equal the upstream tracked
   count (11,062).
2. GGUF quantisation has **no compute path on ROCm** in this tree.
   `python/sglang/srt/layers/quantization/gguf.py:46-70` imports the ggml
   kernels only under `_is_cuda` and `_is_musa`; the `else` branch warns only
   `if not _is_hip`, so on HIP it neither imports them nor warns, and
   `fused_mul_mat_gguf` raises `NameError` on the first linear layer. The real
   kernels live in `kernels/aot/csrc/quantization/gguf/` — pure CUDA, no HIP
   guards — and `python/pyproject.toml` excludes `kernels/aot*` from the wheel.
   There is nothing here to port. There is something to write.
