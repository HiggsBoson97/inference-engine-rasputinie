# D-05 — Serve the file as published; one internal layout; bf16 compute

Three decisions that always travel together, because a kernel cannot satisfy one without the others.

- **Decided, weights:** the weights stay **quantised as published**. No offline conversion, no
  prepared artifact, no second format on disk.
- **Decided, layout:** kernels see **one internal layout**; front-ends (GGUF now, a quantised
  safetensors container later) normalise into it. A kernel does not know where its bytes came from.
- **Decided, compute:** bf16. No FP8 arithmetic (`D-03`), no int8 GEMM path unless a measurement
  justifies it.
- **Decided, packaging:** **do not expand the codebooks.** The lossless 4-bit expansion needs
  **100.4 % of the 776 GB/s wall** to break even (1.5091× the bytes for 1.197× the speed), and even
  with the sign path made free it is a 0.994 tie.
- **Rejected:** expansion (above); a 4.1875 bpw signed-nibble repack — faster, but it is a lossy
  re-quantisation, not a repack of these bytes, so it violates `D-01`; and per-type special-cased
  layouts, which would multiply the kernel set by the number of front-ends.
- **Evidence:** `docs/adr/0003-representation-doctrine.md`; `docs/plan/DECISION-packaging-iq3s.md`;
  the break-even arithmetic in `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-4.md`
  and the E1 report.
- **Cited by:** every kernel unit; `facts/quant.md` carries the rates the arithmetic rests on.
