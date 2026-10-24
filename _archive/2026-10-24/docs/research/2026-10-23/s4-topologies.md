# S4 — The full space of two-GPU strategies over a 14.2 GiB/s link

**Status:** research note for the PLAN loop, 2026-10-23. Supersedes nothing; it
adds the topologies `docs/01-models.md` never enumerated. The existing plan
considers exactly two: **TP=2** and **DP=2**. There are at least nine distinct
mechanisms, and one of them (pipeline parallelism) is not a variant of either.

**Question asked:** given that cross-GPU copy measures 14.2 GiB/s — the same as
host bounce, so there is no P2P DMA — what two-GPU strategies exist for one
model, what does each move per token, and which survive a byte budget?

**Companion notes:** `03-bandwidth.md` (the 776 GB/s read wall),
`05-serving-knobs.md` (§8 ranks TP2 third), `06-recent-research.md` §b.4
(prefill/decode disaggregation), `seed/docs/research/18-tp-roccl.md` (the
measured all-reduce latency this note leans on).

**Tags.** `[M]` measured on this box. `[D]` derived here from measured inputs.
`[V]` verified in a primary source, URL given. `[U]` unverified or a user report.
`[H]` hypothesis with a named measurement that would settle it.

---

**One-paragraph answer.** The link is not a bandwidth problem. 14.2 GiB/s is
**2% of one card's 776 GB/s**, and it can carry 0.208 GiB inside the 14.63 ms a
single card needs to produce one token. No strategy in this document moves more
than 1.25 MiB per token except the ones that move weights, and those move 10.57
GiB — **5,089% of the per-step budget** — which is why weight streaming loses by
a factor of 51 no matter where the weights come from. What actually costs money
is not bytes but **the count of serialized hops**: TP=2 moves 1.25 MiB per token
(0.59% of budget) but pays 128 synchronous all-reduces at 20.65 µs each, which is
**2.64 ms or 27% of the step** — the transfer itself needs 86 µs. TP=2 is
therefore 1.47×, not the 2.0× that halving the bytes promises, and even a
perfect all-reduce only lifts it to 1.81×. The strategy the
plan has never considered, **PP=2 (32 layers per card)**, moves **10 KiB per
token** — one handoff, 0.67 µs of link time, 0.0046% of budget — and reaches
**136.7 tok/s aggregate, identical to DP=2, at half the weight footprint per
card**. It has one hard precondition: **at least two requests in flight**. At
batch 1 PP buys exactly nothing, because latency can only be divided by
parallelism *inside* a layer, and PP has none. That is the structural fact that
splits the whole design space: **TP buys latency, PP buys throughput, and on this
box DP is a special case of PP with a 2× memory penalty.** The second card's
memory is worthless as a store — anything on it returns over the same link that
reaches host DRAM, which is 3× larger — so card 1 is only ever useful as compute.

---

## 0. Three laws that decide everything below

**L1 — The budget is 0.208 GiB per token.** One card needs 14.63 ms to read
10.57 GiB at 776 GB/s. In that window the link can carry
`14.2 GiB/s × 14.63 ms = 0.208 GiB`. Any split that moves more than a few MiB
per token is not "a bit slower", it is off by orders of magnitude. Activations
(10 KiB) are 0.0046% of budget. Weights (10.57 GiB) are 5,089% of it. There is
no middle ground to design in.

**L2 — Hops cost more than bytes.** A 10 KiB all-reduce moves 0.67 µs of traffic
and takes 20.65 µs of wall time `[M]` (measured, 18-tp-roccl §5.5). The link is
**31× less efficient than its own bandwidth** at this transfer size, because a
collective is a synchronization, not a stream. So the figure of merit is
`hops × 20 µs`, not `bytes ÷ 14.2 GiB/s`. TP=2 has 128 hops; PP=2 has 1.
This is the single most under-used number in the existing plan: `docs/00-facts.md`
prices TP=2's all-reduce at "86 us per token", which is the *bytes* answer and
understates the *latency* answer by 31×.

**L3 — Card 1's memory is strictly dominated by host DRAM as a store.** Anything
parked on card 1 must come back over the *same* 14.2 GiB/s link that connects it
to host DRAM, so a restore costs the same either way (**159 MB = 10.4 ms** both
ways) — while card 1 offers 20 GiB against the host's 60, and occupies the card
that could otherwise be computing. **Host DRAM is the tier-2 store, always.** The
one case where card 1 wins is data the card consumes locally and never ships
back. Corollary, counterintuitive enough to state plainly: **the second card
should never hold anything that has to return over the link.**

---

## 1. Every strategy, priced

`W = 10.57 GiB` per token (IQ3_S, parent's figure; `11.35 GB`). One card:
14.63 ms = 68.4 tok/s `[D]`. "Interval" is the token-completion period of the
whole box; "lat" is the single-stream time from step start to token.

| # | Strategy | Bytes/token | % of 14.63 ms step | Interval | Aggregate tok/s | Latency ms |
|---|---|---|---|---|---|---|
| 1 | One card (`ATOM`) | 0 | 0 | 14.63 | 68.4 | 14.63 |
| 2 | **DP=2**, two replicas | **0** | 0 | 14.63 | **136.7** | 14.63 |
| 3 | **TP=2**, RCCL all-reduce | 1.25 MiB | 0.59 | 9.96 | 100.4 | **9.96** |
| 4 | TP=2, hand-written reduce | 1.25 MiB | 0.59 | 8.08 | 123.8 | **8.08** |
| 5 | **PP=2**, ≥2 streams | **10 KiB** | 0.0046 | 7.31 | **136.7** | 14.63 |
| 6 | PP=2, one stream | 10 KiB | 0.0046 | 14.64 | 68.3 | 14.64 |
| 7 | Hybrid: TP the MLP only | 1.25 MiB | 0.59 | 12.02 | 83.2 | 12.02 |
| 8 | Hybrid: TP attention only | 320 KiB | 0.15 | 14.77 | 67.7 | 14.77 |
| 9 | Quant doesn't fit: head on card 1 | 10 KiB | 0.0046 | 26.98 | 37.1 | 26.98 |
| 10 | Stream weights from host DRAM (over PCIe) | 10.57 GiB | **5,089** | 744.4 | 1.3 | 744.4 |
| 11 | Stream weights from NVMe | 10.57 GiB | 5,089 | 2030 | 0.5 | 2030 |
| 12 | PP + stream each stage from host | 5.29 GiB | 2,545 | 372.2 | 2.7 | 372.2 |
| 13 | GPU+CPU split (llama.cpp `-ngl`), BF16 | 31.8 GiB on CPU | — | 685 | 1.5 | 685 |
| 14 | KV offload to card 1 (32K) | 2.00 GiB | 963 | — | — | +140.8 |
| 15 | Card 1 as drafter / prefix prefill | ~10 KiB | ~0 | — | ≤1.5× | — |

Rows 3–8 assume 776 GB/s on both cards (the kernel wall, `03-bandwidth.md`), so
they are ceilings; at the engine-realistic 650 GB/s every tok/s figure falls by
~17% and the *rankings do not change*. Row 10 and "weights resident on card 1"
are the same number: both cross the same link.

---

## 2. Pipeline parallelism (PP=2) — the mechanism the plan is missing

**Mechanism.** Card 0 holds layers 0–31 plus the embedding; card 1 holds layers
32–63 plus the LM head. A token enters card 0, is pushed through 32 layers, and
its final residual (the hidden state, **5120 × bf16 = 10,240 B = 10 KiB**) is
handed to card 1 over the link; card 1 runs the remaining 32 layers and samples.
No collectives, no reduction, no replication, no sharding inside any layer.

### 2.1 The activation transfer is nothing

`10,240 B ÷ 14.2 GiB/s = 0.67 µs` of link time per token. Even at a pessimistic
100 µs for a host-bounced 10 KiB send plus a flag spin, the handoff is **0.7% of
a step**. Compare TP=2's 128 collectives: **2.64 ms, 27%** `[M]`,`[D]`. The ratio
of link *overhead* is 128 hops versus 1.

### 2.2 The bubble is real and it is fatal at batch 1

A two-stage pipeline cannot overlap anything within one sequence: stage 1 cannot
start token *n* until stage 0 finishes token *n*. With one request in flight,
`step = 2 × t_stage = 14.63 ms` — **exactly one card's speed, no gain, and one
extra hop of latency**. Rows 5 and 6 of the table are the same topology and
differ by 2× purely on concurrency.

This is not a tuning artefact, it is structural:

> **A decode step's latency can only be divided by parallelism that exists inside
> a layer.** PP has none, so at batch 1 PP cannot be faster than one card, ever.
> TP has it, so TP is the only topology that beats one card on single-stream
> latency.

The condition for PP is therefore **≥2 concurrent requests**, which this box
has by design (`MAP.md`: "serves several long-context sessions at once"; the KV
note budgets 6.5 sessions/card at 32K). At ≥2 streams both stages are busy every
slot and the box emits **one token per 7.31 ms = 136.7 tok/s**, with each stream
still seeing 14.63 ms/token — the same per-stream latency DP delivers.

### 2.3 Stage imbalance: the 48-vs-16 worry dissolves, and here is the number

The question was whether "48 of 64 layers are a different kind of layer than the
other 16" breaks the split. It does not, because of a coincidence in the config
that is worth writing down:

- full attention layer ≈ 104.9 M params (q 5120×6144, k/v 5120×1024, o
  6144×5120, output gate 5120×6144)
- gated-deltanet layer ≈ 104.9 M params (q/k 5120×2048, v 5120×6144,
  o 6144×5120, gate 5120×6144)
- MLP = 267.4 M params, in **every** layer

So **an attention layer and a linear-attention layer are the same size to within
rounding** `[D]`, and the pattern is `LLLF × 16`, i.e. periodic with period 4
across exactly 64 layers. **The split at layer 32 lands on a pattern boundary and
gives 24 GDN + 8 attention on each card — a 1.000 ratio, in bytes and in op
count.** Both stages also carry exactly 8 of the 16 KV-bearing layers, so KV
splits 8/8 and the 159 MB recurrent state splits 79.5/79.5 MB.

The only asymmetry is the **LM head**: 248,320 × 5120 at Q6_K = 0.97 GiB, i.e.
9% of the model, landing entirely on stage 1. Byte-balancing by layer count then
gives:

| split point | stage 0 | stage 1 | imbalance |
|---|---|---|---|
| layer 28 | 4.82 GiB | 6.53 GiB | **35.4%** |
| layer 32 | 5.44 GiB | 5.91 GiB | **8.6%** |
| layer 32 + head halved across cards | 5.40 GiB | 5.40 GiB | 0.0% |

Splitting the head by vocabulary (124,160 logits per card, then a top-k merge —
one extra 10 KiB hop, 0.67 µs) removes the imbalance completely. Without it, 8.6%
is the price, and it costs 8.6% of throughput: 125 tok/s instead of 136.7.

**Verdict.** PP=2 is the only strategy that uses both cards' bandwidth with
essentially zero link cost, and this model's periodic layer pattern makes it
*exactly* balanced for free — most models do not offer that. Its precondition is
concurrency ≥2. `[H]` The one measurement that could still hurt it: if a GDN
layer is much slower in *time* than an attention layer despite equal bytes (the
48-head sequential recurrence is the suspect), stage time is set by the slower
op class — but both stages hold 24 GDN + 8 attention, so the split stays balanced
even then. The real risk is that *both* stages become recurrence-bound and the
776 GB/s assumption fails. Measure: time layers 0–31 and 32–63 separately on one
card. **This is a 20-minute experiment and it is the gate on the whole option.**

### 2.4 PP also fixes prefill, which is TP's worst moment

`docs/00-facts.md` already records that TP=2 makes 4K prefill PCIe-bound: 4096
tokens × 1.25 MiB = **5.37 GB, 352 ms** `[D]`. PP's prefill traffic is
4096 × 10 KiB = **41.9 MB, 2.75 ms** — **128× cheaper** — because the handoff is
per *chunk*, not per layer, and chunked prefill (64–256 tokens, already ranked
cheap in `05-serving-knobs.md`) pipelines the two stages against each other.

---

## 3. Hybrid TP/DP splits

The generic hybrid is: **split the layers whose bytes repay the all-reduces,
replicate the rest.** Since both cards must then hold a full copy of the
replicated part, it is a three-way trade between bytes-per-card, hops, and
speed. Two configurations matter:

**(a) TP the MLP only** — the MLP is 71.8% of the model's bytes `[D]`
(267.4 M × 64 layers), attention 7.0%, GDN 21.1%. Splitting only the dominant
term: per-card bytes fall to 6.77 GiB from 10.57, but **all 128 all-reduces are
still paid** (the residual stream is still sharded, so both attention and MLP
outputs need reduction). Result: **12.02 ms, 83.2 tok/s.** It buys 1.22× while
still spending 2.64 ms of all-reduce, 22% of the resulting step. Dominated by
plain TP=2 (9.96 ms) and by PP=2 (136.7 aggregate).

**(b) TP attention only** — the attention layers are just **7%** of the bytes, so
splitting them saves 0.37 GiB per card: **14.77 ms, 67.7 tok/s**, i.e. *slower*
than one card once the 32 all-reduces are added. This is the clearest possible
statement that **layer class matters less than byte share**: the 16 "different"
layers are the cheap ones.

**(c) Asymmetric TP ratio** (card 0 takes 60% of every layer's output channels)
saves nothing on two identical cards and still pays 2 hops per layer.

**(d) Per-phase topology switch** (TP for prefill, DP or PP for decode) requires
reshaping the weight placement between phases: each card must acquire the other's
half, **5.29 GiB = 382 ms at 14.2 GiB/s**. There is no faster path — a host
master copy still crosses the same PCIe at 14.2 GiB/s, so it is 382 ms per card,
764 ms if serialized. That is **26–52 decode steps of dead time per switch**. Only
viable if phases last tens of seconds; not worth the second code path.

**(e) TP inside a PP stage** needs 4 GPUs. Not available.

**Verdict.** Every hybrid is a point on a straight line between TP and DP that
strictly loses to one of its endpoints. The reason is now obvious: the hops are a
fixed cost per layer regardless of how much you split, so you should either split
*everything* (TP, minimise latency) or split *between* layers (PP, minimise
hops). Splitting *some* operators gets the hop cost of TP and the byte count of
DP.

---

## 4. The decoy: "put the bit that doesn't fit on the other card"

When a quant exceeds one card by a little — Q6_K is 20.47 GiB against 19.98 GiB
usable — the tempting move is to keep 63 layers resident and park the LM head
(0.97 GiB) on card 1. It works, and it is the worst option on the table:

- card 0 reads 19.50 GiB/token at 776 GB/s = **26.98 ms**
- card 1 reads 0.97 GiB = 1.25 ms, then waits
- one 10 KiB hop per token

**37.1 tok/s, card 1 idle 95% of the time.** PP=2 at the same quant puts 10.23
GiB on each card and delivers **70.6 tok/s aggregate** (28.3 ms latency) — 1.9×
the throughput for the same memory. A split that does not balance the *work* is
not a split; it is offloading with extra steps.

---

## 5. Weight streaming from host RAM or NVMe

Every byte of every weight crosses the PCIe link once per token: **10.57 GiB**,
or **5,089% of the 0.208 GiB step budget**. The ratio is the whole answer:

| where the weights are | path into the card | measured bandwidth | ms/token | tok/s | vs on-card |
|---|---|---|---|---|---|
| on-card HBM | — | 776 GB/s | 14.63 | 68.4 | 1× |
| host DRAM | **PCIe, 14.2 GiB/s** | 14.2 GiB/s | **744** | **1.3** | **51× worse** |
| NVMe, 4 streams | PCIe | 5.59 GB/s | 2030 | 0.5 | 139× worse |

`776 GB/s ÷ 14.2 GiB/s = 51`. Streaming is not a 20% tax, it is a 51× tax. Note
that the host's own 46.4 GiB/s DDR5 copy bandwidth is **irrelevant here**: it is
a CPU-side number, and a GPU reading host-resident weights still crosses PCIe at
14.2 GiB/s. The 46.4 only applies if the *CPU* does the multiply — which is the
next strategy.

**Is there any regime where it wins?** Two candidates, both fail on this box:

1. *The model does not fit one card.* PP=2 covers everything up to
   **39.96 GiB** of weights — Q8_0 (27.05) fits comfortably, and only BF16
   (51.77) does not. BF16 is unservable at any speed on this box: streamed over
   PCIe it is **3.65 s/token = 0.27 tok/s**, and the CPU-split variant is 1.5
   tok/s (§5.1). Q8_0 under PP runs at **53.4 tok/s**. The "doesn't fit" regime
   is already served; the "bigger than two cards" regime is not servable at all,
   so streaming has no territory left.
2. *The link is free and the model is used at very low duty cycle.* Then so is a
   second card, and PP still wins.

**Verdict: never.** This matches `05-serving-knobs.md` §8's existing "weight
streaming — reject", and now has the number attached: 5,089% of budget.

### 5.1 The variant that actually ships: split the model across GPU and CPU

llama.cpp's `-ngl` puts the first N layers on the card and the rest in host RAM,
computed by the CPU. It is the same economics one tier down, and it deserves a
line because it is the only "streaming" shape with a working implementation:

- The CPU half is read from DDR5 at 46.4 GiB/s, but **not** through PCIe — the
  CPU computes locally, so the PCIe cost is only the activation handoff (10 KiB,
  one per token, in each direction).
- So the constraint moves to CPU FLOPs. BF16 (51.77 GiB) leaves **31.8 GiB** on
  the CPU, and the read alone is **34.1 GB ÷ 49.8 GB/s = 685 ms**, before any
  arithmetic. **≈1.5 tok/s.**
- For a quant that fits one card (IQ3_S), the split buys nothing at all — it
  just moves bytes onto a slower compute device.

**Verdict: no, and for the same reason as GPU streaming — the offloaded tier is
always ≥16× slower per byte than the card, and it must be read every token.**
Where it *does* earn its place is as a **capacity** fallback for a single
oversized request (BF16, or a 256K session that will not fit), where the
alternative is refusing the request. It is a correctness escape hatch, not a
performance option.

**The literature agrees, and it is the strongest single data point in this note.**
Petals ran BLOOM-176B (8-bit, 176 GB) four ways `[V]`
([arXiv:2209.01188](https://arxiv.org/abs/2209.01188), Table 3):

| method | link | steps/s |
|---|---|---|
| pipeline over 3×A100, 1 Gbit/s (0.125 GB/s) | 0.125 GB/s | **1.71** |
| pipeline over 3×A100, 100 Mbit/s, 100 ms RTT | 0.0125 GB/s | 1.23 |
| pipeline, 14 real servers across Europe/NA | ~0.1 Gbit/s | 0.83 |
| **parameter offloading, 1×A100 @ PCIe4 ×16** | **32 GB/s** | **0.18** |

Moving activations beat moving weights by **9.5×** while using a link **256×
slower**. That is L1 and L2 measured on someone else's hardware, and it is the
reason to take PP seriously rather than as a novelty.

---

## 6. Activation and KV offload in the other direction

Keeping everything on card 0 and using card 1 as a KV store sounds like the
cheapest possible use of the second card. It is the most expensive: **the KV of
an active session is read in full on every token**, so offloading fraction *f*
of it adds `f × KV × (1/14.2 − 1/776)` per token.

| context | KV/session (bf16) | read on-card | read over link | penalty | share of a 1-card step |
|---|---|---|---|---|---|
| 4 K | 0.25 GiB | 0.35 ms | 17.6 ms | 51× | 2% |
| 32 K | 2.00 GiB | 2.77 ms | 140.8 ms | 51× | 19% |
| 128 K | 8.00 GiB | 11.07 ms | 563 ms | 51× | 76% |
| 256 K | 16.00 GiB | 22.14 ms | 1127 ms | 51× | 151% |

At 32K you can afford to offload **0.7% of one session's KV** (216 tokens)
before the penalty reaches a 1 ms step. The scheme is dead for active KV. Note
also how big the *on-card* KV term already is: 19% of a step at 32K, 76% at 128K
— that, not the weights, is the reason `04-kv-cache.md` wants q8 KV.

Where the second card *could* have helped — holding the KV of **idle** sessions —
host DRAM wins on capacity and on not occupying a card: 60 GiB against 20, and
crossing the link to restore costs **the same 10.4 ms either way** (a 159 MB
recurrent-state checkpoint is 10.4 ms from host RAM *or* from card 1, because both
cross PCIe at 14.2 GiB/s), against **445 ms to re-prefill the 2,048 tokens it
replaces** — a 43× win. That is a real argument for prefix caching, and it is an
argument for *host RAM*, not for card 1. `[D]`

**Verdict.** Card 1 as a KV/expert/spill store: dead on bandwidth and dominated
on capacity. Card 1 as compute: the only thing it is uniquely good at.

---

## 7. Using card 1 for something orthogonal

| use | mechanism | buys | cost | verdict |
|---|---|---|---|---|
| **Second stream (DP)** | full replica per card | **2.0× aggregate**, 0 bytes | 2× weight footprint | the baseline; nothing beats it on aggregate for free |
| **Speculative drafter** | 0.6–1.5B draft model on card 1, target verifies on card 0 | 1.3–2.5× on latency, *if* draft and verify overlap | draft must fit in the verify window | **marginal** — see below |
| **Prefix prefill** | card 1 prefills turn *n+1* while card 0 decodes turn *n* | removes prefill interference | needs a second 10.57 GiB replica; 1 GiB KV handoff = 70 ms `[D]` | already priced in `06-recent-research.md` §b.4 as a trade, not a win |
| **State/checkpoint bank** | 159 MB checkpoints on card 1 | 10.4 ms restore | host DRAM also 10.4 ms, but 3× larger and leaves the card free | dominated by L3 |
| **Keep the display card free** | run everything on HIP device 1 | — | — | device 1 is the headless card `[M]`; free hygiene |

The drafter deserves the arithmetic, because it is the one orthogonal use that
could plausibly pay:

- **Verify is sublinear in tokens.** Measured: a 5120×17408 bf16 GEMM at M=1
  takes 0.405 ms, and at **M=8 takes 0.295 ms** — *fewer* milliseconds for eight
  rows than for one, because the weight read is amortised over a better-shaped
  kernel `[M]` (`00-facts.md`). So verifying 4–8 candidate tokens costs ≈0.73× a
  single-token pass, not 4–8×.
- **The draft must therefore hide inside ~10.7 ms.** A 0.6B drafter at 1.2 GB of
  weights per token is 1.55 ms/token at 776 GB/s, but drafting is autoregressive:
  4 tokens = 4 sequential passes ≈ 6.2 ms of card-1 time — it fits. Measured on
  this box, though, a 0.6B model under sglang runs at **7.56 ms/token**
  (launch-bound: ~400 kernels per token), so 4 draft tokens = 30 ms and it does
  **not** fit `[M]`. The drafter only works if its engine is as lean as the
  target's.
- **And the model ships an MTP head**, which gives 1.3–1.8× with no second card
  and no link at all (`05-serving-knobs.md` ranks it #2).

**Verdict.** Every orthogonal use delivers ≤1.5× on one stream and costs a card
that DP or PP would have turned into 2.0× aggregate. **Orthogonal use of card 1
is dominated by simply staging it.** The exception is a latency-SLO-driven
deployment serving *one* user, where MTP + a card-1 drafter is the only way to
push single-stream latency below 9.96 ms.

---

## 8. The literature on single-model inference over a slow link

| system | what it does | link assumed | measured result |
|---|---|---|---|
| **Petals** ([arXiv:2209.01188](https://arxiv.org/abs/2209.01188), ACL'23 demo) | layer-range pipeline across volunteer servers; client keeps embeddings | 100 Mbit/s – 1 Gbit/s, 5–100 ms RTT | **1.71 steps/s** for BLOOM-176B on 3×A100 at 1 Gbit/s; **0.83 steps/s** on 14 real servers across Europe/North America; **9.5× faster than PCIe offloading at 256× less bandwidth** (Table 3). NeurIPS'23 extension reports Llama-2-70B and BLOOM-176B "up to 10× faster than offloading" ([OpenReview](https://openreview.net/forum?id=XmN7ZNbUAe)) `[V]` |
| **SWARM parallelism** ([arXiv:2301.11913](https://arxiv.org/abs/2301.11913), ICML'23) | pipeline + dynamic layer reassignment over heterogeneous, preemptible, failing nodes; the analysis shows comms per device *fall* as hidden dim grows | **<200 Mbit/s** | trained a **1B-parameter model (≈13B before factorisation) on preemptible T4s at under 200 Mbit/s** `[V]`. Training, not inference, but it is the canonical proof that pipeline stages tolerate a link three orders of magnitude worse than ours |
| **Helix** ([arXiv:2406.01566](https://arxiv.org/abs/2406.01566), ASPLOS'25) | places layer ranges and routes requests by solving a max-flow / MILP over a graph whose **edges are network capacities** | heterogeneous, 24–42 GPU nodes | **up to 3.3× serving throughput, −66% prompt latency, −24% decode latency** vs prior placement `[V]`. The −24% on decode is the direct measure of what good stage placement is worth on a slow link; the −66% on prompt says most of the win is prefill, which is exactly where our TP=2 pays 352 ms |
| **AlpaServe** ([arXiv:2302.11665](https://arxiv.org/abs/2302.11665), OSDI'23) | chooses intra-op (TP) vs inter-op (PP) vs replication per model, as a function of load | intra-node NVLink-class, but the *trade-off* is topology-independent | **up to 10× request rate, or 6× more burstiness, within SLO for >99% of requests** `[V]`. Its framing is ours: inter-op/PP wins on aggregate throughput, intra-op/TP wins on latency, replication wins when load is low — and the choice flips with concurrency |
| llama.cpp RPC ([discussion #9136](https://github.com/ggml-org/llama.cpp/discussions/9136)) | layer-split across machines over TCP, i.e. PP with a network instead of PCIe | Ethernet | a user reports **~3 tok/s** with the link carrying only 30–50 Mbit/s `[U]` — a practical floor for PP where the handoff is a TCP round trip rather than shared-memory IPC |

**BigModel** was named in the brief; I could not verify any system paper by that
name — the term only resolves to Zhipu's API brand. The four systems above carry
the argument, and the two that matter for our decision are Petals (activations
are ~10⁷× cheaper than weights per token, measured) and AlpaServe (the
TP/PP/replica choice is a function of concurrency, which is precisely the axis
our plan ignores).

**What the literature does *not* give us:** every one of these systems assumes
≥3 stages or ≥4 nodes, where the bubble amortises and the topology is chosen per
request. At **two** stages the bubble is 50% at batch 1 and the topology must be
chosen per *deployment*, not per request. That is why the answer here is a single
number — the concurrency you expect — and not a scheduling algorithm.

---

## 9. Verdicts

### 9.0 The same comparison across quant levels

Every topology recomputed for each available file, same 776 GB/s, at ≥2 streams
`[D]`. "—" means the weights do not fit the topology's per-card budget.

| quant | GiB | 1 card | DP=2 | TP=2 | TP=2 + 6 µs AR | **PP=2** |
|---|---|---|---|---|---|---|
| IQ3_S | 10.57 | 68.4 | **136.7** | 100.4 | 123.8 | **136.7** |
| Q4_K_XL | 16.35 | 44.2 | **88.4** | 71.7 | 82.8 | **88.4** |
| Q6_K | 20.47 | 35.3 | — | 59.5 | 67.0 | **70.6** |
| Q8_0 | 27.05 | 26.7 | — | 46.8 | 51.3 | **53.4** |
| BF16 | 51.77 | 14.0 | — | — | — | — |

Two things fall out that reorganise the decision:

1. **Wherever DP is legal, PP is numerically identical** (136.7 vs 136.7, 88.4
   vs 88.4). Both divide the same total bytes by the same two cards. They are
   not two performance options; they are one performance point with two memory
   footprints and two isolation properties. Choose on those, not on tok/s.
2. **Wherever DP is illegal, PP beats TP by 19%** (70.6 vs 59.5, 53.4 vs 46.8) —
   the all-reduce's 27% of step time, minus PP's one hop. And the whole Q6_K /
   Q8_0 row is a *quality* purchase paid in throughput: Q8_0 on two cards (53.4)
   is still slower than IQ3_S on one card (68.4). The quant decision dominates
   the topology decision, which is why `03-bandwidth.md` §7 is more important
   than this note.

### 9.1 Verdict table

| strategy | mechanism | bytes/token | % of budget | verdict for this box |
|---|---|---|---|---|
| DP=2 | 2 replicas | 0 | 0 | **run it today** for IQ3_S; 136.7 aggregate, zero risk, zero code |
| **PP=2** | 32 layers/card, 1 handoff | **10 KiB** | **0.0046%** | **the missing option.** Equal to DP on tok/s, half the weights per card, 128× cheaper prefill. Needs ≥2 concurrent. Gate: per-stage timing |
| TP=2 | split every layer, 128 AR | 1.25 MiB | 0.59% | only for single-stream latency (9.96 ms); 1.47× not 2.0× |
| TP=2 + hand AR | replace RCCL's 20.65 µs reduce | 1.25 MiB | 0.59% | **8.08 ms, 1.81×** — one experiment, biggest single-topology gain available |
| Hybrid TP-MLP | split the 72% | 1.25 MiB | 0.59% | 83.2 tok/s: dominated by both endpoints |
| Hybrid TP-attn | split the 7% | 320 KiB | 0.15% | 67.7 tok/s: slower than one card |
| Head-on-card-1 | offload the overflow | 10 KiB | 0.0046% | 37.1 tok/s with card 1 95% idle: a decoy |
| Weight streaming | weights cross PCIe per token | 10.57 GiB | **5,089%** | **never** — 51× tax; PP covers the fit problem |
| KV offload to card 1 | active KV lives across the link | 2.00 GiB @32K | 963% | **never** — 51× on every token |
| Card 1 as store | idle KV / checkpoints | — | — | restores at the same 10.4 ms as host DRAM, but 3× smaller and occupies a card |
| Card 1 orthogonal | drafter / prefill | ~0 | ~0 | dominated by staging, except at P=1 |

### Ranked recommendation

1. **Do not change DP=2 for IQ3_S.** 136.7 tok/s aggregate, zero link traffic,
   zero new code. The existing rule in `MAP.md` is right for the primary target.
2. **Adopt PP=2 as the answer for any quant that does not fit one card** —
   Q6_K (70.6 tok/s) and Q8_0 (53.4 tok/s). Same fit envelope as TP=2
   (39.96 GiB), **19% more aggregate** than TP=2, no all-reduce, and
   prefill traffic falls from 352 ms to 2.75 ms. This is the change to make.
3. **Replace TP=2 with a hand-written 2-GPU reduce before choosing TP at all.**
   20.65 µs for a 10 KiB collective is a software floor, not a link property
   (the bytes need 0.67 µs). At 6 µs, TP=2 goes from 100.4 to 123.8 tok/s and
   from 1.47× to 1.81× over one card. `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` is
   the existing untested flag for exactly this.
4. **Prefer PP=2 over DP=2 when memory binds** — same throughput, 5.3 GiB more
   per card for KV, which is **4.4 → 6.8 sessions at 32K bf16** (2.16 GiB each
   including the 159 MB state) or **8.1 → 12.7 at 32K q8**.
5. **Card 1 as a drafter or prefill host: do not build.** ≤1.5× on one stream,
   against 2.0× for staging it, and MTP already collects 1.3–1.8× on card 0.
6. **Never stream weights and never offload active KV.** 5,089% and 963% of the
   budget. These are not close calls.

### The one measurement that would change the ranking

**Time layers 0–31 and layers 32–63 separately, one card, batch 1, at IQ3_S.**
Everything above assumes 776 GB/s applies to both halves. If the 48 gated-deltanet
recurrences are latency-bound rather than bandwidth-bound — the 857 ms/token
anomaly in `00-facts.md` is still unexplained — then a PP stage containing 24 of
them costs more wall time than its bytes predict. The split stays *balanced*
either way (both stages hold 24 GDN + 8 attention), so PP degrades gracefully
toward "both stages equally slow" rather than becoming lopsided; but the absolute
136.7 tok/s moves. Twenty minutes on the existing harness settles it.

---

## Appendix — arithmetic

Reproduce with `research/s4-topologies-calc.py`.

```
GiB = 2^30;  W = 10.57 GiB = 11.35 GB;  B = 776 GB/s;  L = 14.2 GiB/s
t1  = W/B                    = 14.63 ms          -> 68.4 tok/s
budget = L * t1              = 0.208 GiB
hidden = 5120*2              = 10 240 B          -> 0.67 us of link
TP  step = 64*(W/64/2/B + 2*20.65us) = 9.96 ms    -> 100.4 tok/s  (AR = 27%)
TP' step = 64*(W/64/2/B + 2*6us)     = 8.08 ms    -> 123.8 tok/s
PP  t_stage = (W/2)/B        = 7.31 ms            -> 136.7 tok/s at >=2 streams
DP  interval = 14.63 ms, agg = 2/14.63 = 136.7 tok/s
link : HBM = 14.2/776 = 1:51   (a weight stream is 51x slower than on-card)
the 14.2 GiB/s is the same whether the far end is host DRAM or the other card
```
