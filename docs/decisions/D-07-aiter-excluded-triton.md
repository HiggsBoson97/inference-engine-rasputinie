# D-07 — aiter's ASM/CK path is excluded; Triton is the kernel backend

- **Decided:** do not depend on `aiter` on this card. Delete it from the reachable path rather than
  guarding its import (`AGENTS.md` principle 3). Force the attention backend to **Triton** explicitly,
  because the HIP default resolves to aiter.
- **Rejected:** patching aiter for gfx1100 — its paged-decode kernel is hand-written **MFMA**
  assembly and **RDNA3 has no matrix cores**, so "porting" it means writing a different kernel from
  scratch; and a `try/except` around its import — it keeps a dead dependency alive and hides the
  next failure.
- **Why it binds:** two facts make it non-obvious. First, aiter's **two allow-lists disagree** —
  `jit/core.py:456-477` accepts `gfx1100` by name while `csrc/cpp_itfs/utils.py:131-147` accepts only
  CDNA — so aiter looks enabled and then asserts. Second, **its Triton subtree does work** on
  gfx1100, which means the useful half of aiter is Triton anyway.
- **Consequence:** Triton emits native gfx1100 ISA (`amdgcn` + `hsaco`, verified by running a
  kernel), so `REQUIREMENTS.md` S4's "native hot path" is satisfied. What Triton does not give is
  *control* — no VGPR allocation, scheduling, delay slots or load ordering. **The open question is
  whether a Triton GEMV reaches the 521 GB/s the hand-written asm kernel reaches; it is unmeasured.**
- **Evidence:** `facts/engine.md` (the allow-list table, the Triton smoke test);
  `_archive/2026-10-24/docs/research/19-aiter-gfx1100.md`.
- **Cited by:** `AGENTS.md` invariant; the bring-up units.
