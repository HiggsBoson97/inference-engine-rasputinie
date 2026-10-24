# D-04 — Fork sglang; the vendored tree is the engine

- **Decided:** the engine is a fork of sglang, vendored at `engine/` and installed **editable from
  `engine/python`** into `~/venv/sglang-fork`. There is no second package and no out-of-tree kernel
  directory: a change to the engine is a change to that tree.
- **Rejected:** a thin engine composed around sglang (the abandoned `rspt/`, `kernels/`, `tests/`
  layout) — it duplicated the scheduler and the model loader while owning neither; and a from-scratch
  engine — the loader, the scheduler and the paged-KV plumbing are not where the performance is.
- **Why it binds:** it makes "work on the engine" mean *edit `engine/`*, which is what the owner
  directed. It also means sglang's assumptions become ours to delete, not to accommodate (`D-07`).
- **Evidence:** `docs/adr/0002-fork-sglang.md`; `pip list` shows sglang 0.5.9 pointing at
  `engine/python`.
- **Cited by:** every unit that touches a kernel or a loader.
