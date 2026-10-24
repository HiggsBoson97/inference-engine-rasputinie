# Probe brief — A4 gate: two-card round-trip latency

```
STATION: probe (open item A4, remaining gate)
SKILL: fullstack-software-development
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/discovery-records.md
ARTIFACT: host `rasputin`, **both** RX 7900 XT (PCI 07:00.0 and 03:00.0) ; scratch `~/a5/`
TOOLSET: ssh, hipcc, amd-smi, python3, sudo -n
BUDGET: 80 tool calls
ROUND: 1
CONTRACT: none — Discovery probe
REPORT: <repo>/docs/agents/production/run-artifacts/hypotheses/H-6.md
```

## GOAL

Measure the **round-trip latency of a 10 KiB exchange between the two 7900 XT cards**, by the
cheapest path available on this box, and convert it into the number that decides the topology:
`128 × RTT` per token.

This is the last gate on A4. Everything else about the two-card question is already priced;
only the latency is a guess, and it is the guess that decides TP vs DP.

## WHY THIS NUMBER, AND NOT BANDWIDTH

Tensor parallelism shards the weights permanently — each card owns its half of every tensor,
and nothing large crosses the link. What crosses is the partial sums: **128 all-reduces of
~10 KiB per token = 1.3 MB, which is 52 µs of link time at 25 GB/s.** The bytes are
irrelevant. The cost is `128 × round-trip latency`:

| RTT | TP=2 step (from a 15.1 ms sweep) | speedup |
|---|---|---|
| 20.65 µs (the recorded guess) | 10.2 ms | 1.48× |
| 10 µs | 8.8 ms | 1.71× |
| 5 µs | 8.2 ms | 1.84× |
| 2 µs | 7.8 ms | 1.93× |

DP=2 gives 1.00× on a single stream at every RTT. So a single microsecond here is worth
~0.13 ms per token, and the whole TP-vs-DP decision rides on this measurement.

## WHAT TO MEASURE

**1. What the hardware actually allows.** `hipDeviceCanAccessPeer` in both directions; whether
`hipEnablePeerAccess` succeeds; whether the cards share a PCIe root complex
(`lspci -tv`, `nvidia`-style topology is not available, so use `lspci` + `/sys/bus/pci`).
The project's baseline says **no peer DMA** — verify it, because if P2P works the answer
changes completely.

**2. Round-trip latency, three ways, so the number is not an artefact of one API:**
- **(a) mapped pinned memory, GPU-publish / pre-launched-acquire**: `hipHostMalloc` with
  `hipHostMallocMapped`, both cards map it; card A writes a flag from a kernel, card B spins
  on it in a pre-launched kernel and writes an ack, card A spins for the ack. Time with
  `clock64()` inside the kernels and cross-check against wall time over many iterations.
- **(b) host-mediated**: `hipMemcpyAsync` device→host pinned, host writes, host→device, stream
  sync, measured over N iterations — the naive path, for comparison.
- **(c) loopback floor**: the same flag protocol between two queues on **one** card, which
  bounds how much of the measured time is PCIe and how much is the protocol itself.

**3. Size scaling.** Repeat (a) at 10 KiB and at 80 KiB (batch 8 of a 10 KiB all-reduce), so
the report can say where bandwidth starts to matter: 10 KiB at 25 GB/s is 0.4 µs, so a 10 KiB
RTT dominated by latency and an 80 KiB one should not be 8× slower.

**4. Jitter.** Report median, p99 and max over ≥ 1000 round trips. A mean hides the tail, and
a TP step waits for the slowest of 128 exchanges.

## RULES

1. **The display card is involved and that is a known hazard.** HIP 0 / PCI `03:00.0` drives
   gnome-shell; a *load* there once caused a `ring gfx_0.0.0` timeout. This probe is a 4-byte
   flag protocol, not a load — keep it that way: no large allocations, no compute kernels, no
   memory copies beyond the flag and the two sizes in §3. If the desktop shows any distress
   (`dmesg` amdgpu ring timeout), stop and report it.
2. **Verify the card mapping by readlink before you launch.** HIP 0 = amd-smi 0 = sysfs card1
   = PCI 03:00.0 = display; HIP 1 = amd-smi 1 = sysfs card0 = PCI 07:00.0 = headless.
3. **llama.cpp is excluded** — no clone, no read, no derived code.
4. **Checksum/consistency gate**: the ack payload must carry a value derived from the sent
   payload (not a constant), so a stale flag cannot be counted as a round trip. Report the
   mismatch count; a nonzero mismatch count invalidates the run.
5. Pin clocks to `high` on both cards and restore `auto`. Idle memory clock is 96 MHz and
   costs ~3 ms of wake-up on the first launch — warm up before timing.
6. Kill by PID, never `pkill -f`.
7. Nothing in the repo is modified; all work in `~/a5/`.

## HOW TO VERIFY YOU DID IT RIGHT

- The loopback floor (c) is strictly lower than the cross-card number (a). If it is not, the
  protocol is measuring itself and the cross-card number is meaningless — say so.
- The 80 KiB RTT is not ~8× the 10 KiB RTT; if it is, you are measuring bandwidth-limited
  copies, not latency, and should say which path that was.
- The payload-derived ack matched on every counted iteration.
- `dmesg` has no new amdgpu error line; VRAM returns to 26 / 88 / 15 MiB after the run.
- The three numbers are reported side by side, never just the best one.

## REPORT SHAPE

Write `docs/agents/production/run-artifacts/hypotheses/H-6.md`, then print it:

```
# H-6 — two-card round-trip latency
## Question (A4's remaining gate)
## Topology as the hardware reports it (P2P capable? same root complex? links)
## Method (a/b/c, iteration counts, warm-up, clock state)
## Results (table: path, size, median, p99, max, mismatch count)
## Implied cost at 128 all-reduces (TP) and 64 handoffs (PP), over the measured RTT
## Verdict (TP vs PP vs DP for single-stream and for concurrency >= 2, with the number.
##          PP needs no reduction primitive at all — 64 one-way layer handoffs instead of
##          128 all-reduces — so at batch 1 it may beat TP while being simpler to build;
##          say so if the arithmetic supports it, and say what it costs at batch >= 2.)
## Compliance (display-card caution observed; llama.cpp excluded; clocks restored)
## Receipts (absolute paths under ~/a5/)
```

Effort: **medium-high**. One protocol, three paths, enough iterations for a p99. The value is
in the loopback floor and the size scaling, because those are what tell a reader whether the
number is PCIe or the measurement apparatus.
