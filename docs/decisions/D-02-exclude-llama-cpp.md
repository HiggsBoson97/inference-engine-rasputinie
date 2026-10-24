# D-02 — llama.cpp is excluded

- **Decided:** llama.cpp is **not a baseline, not a reference, contributes no code, no layout, no
  constant, and is not installed** on the host. Name it in every brief; a worker that does not know
  it will reach for the obvious reference implementation.
- **Rejected:** using it as a correctness oracle — its GGUF reader and dequant kernels are exactly
  the work this project exists to do on gfx1100, and borrowing them would make the engine a port.
- **Why it binds:** it is the reason the GGUF layout work is derived from the file plus the format
  documentation, and the reason a *legitimate* grid-table source had to be found elsewhere
  (`IST-DASLab/GSQ`, Apache-2.0 — see `facts/quant.md`).
- **Evidence:** `docs/decisions/0001-exclude-llama-cpp.md` (pre-migration); the incident where a
  research subagent cloned and built it anyway and it had to be deleted.
- **Cited by:** `D-05`; every kernel unit's brief.
