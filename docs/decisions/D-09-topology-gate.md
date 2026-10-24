# D-09 — Topology is decided by the round trip, not by aggregate throughput

- **Decided:** the two-card question stays **open** until a mapped-publish flag round trip is
  measured (p99, ≥1000 iterations, single-card loopback as the floor). The number to beat is
  **RCCL's ~19.5 µs floor**, which is a protocol floor, not the hardware's.
- **Rejected:** the previous answer, "DP=2 for IQ3_S" — it priced **aggregate** throughput, and DP=2
  gives **1.00× on a single stream** because one card idles. It is right only at concurrency ≥ 2.
  Also rejected: choosing TP on the published 20.65 µs figure — that figure is RCCL's handshake cost,
  so it cannot settle what a hand-rolled protocol would do.
- **Why it binds:** the cost is `128 × RTT` for TP and `64 × RTT` for PP. At 20.65 µs that is TP
  **1.48×** / PP **≈1.70×** on a single stream; at 2 µs, TP **1.93×**. The choice flips on one number.
- **What is *not* in question:** weights are **sharded, never moved**. What crosses the link is
  partial sums — 1.3 MB per token, 52 µs of bytes. The link's bandwidth is irrelevant; its latency
  is everything.
- **Evidence:** `facts/twocards.md`; `_archive/2026-10-24/docs/research/18-tp-roccl.md` §4.1.
- **Cited by:** the topology unit; `AGENTS.md` principle 6.
