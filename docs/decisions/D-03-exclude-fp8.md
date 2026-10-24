# D-03 — FP8 is excluded as a format

- **Decided:** no FP8 checkpoint, no FP8 reference run, no FP8 kernel. Weights are quantised; compute
  is bf16. **RDNA3 has no FP8 hardware**, so there is nothing to emulate.
- **Rejected:** FP8 as a fast path — the measured run was **857 ms/token, 2.7 % of the wall**, and it
  is evidence *for* the exclusion, never a baseline.
- **Why it binds:** it removes an entire class of "can we just use fp8?" from every design
  conversation, and it is the first application of `AGENTS.md` principle 4 (absent hardware is not a
  software gap).
- **Evidence:** `docs/adr/0004-exclude-fp8.md`; `_archive/2026-10-24/docs/research/20-decode-profile-27b.md`
  for the 857 ms/token run.
- **Cited by:** `D-05`; `facts/engine.md` (aiter's FP8 paths are excluded by design).
