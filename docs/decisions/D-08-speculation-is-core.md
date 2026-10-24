# D-08 — Speculation is core; both drafters are in scope

- **Decided:** the decode path is built to accept a draft and verify it **from the first milestone**,
  not retrofitted. The shipped **MTP head** is the baseline drafter; **DFlash2** is a first-class
  candidate, not a later option. A3 decides which ships first.
- **Rejected:** drafting at depth 1 "until later" — the verify path is a shape, not a flag, and
  retrofitting a block-shaped verify path costs a rewrite of the batch dimension; and treating the
  drafter's weight read as free — it is 13–53 % of the budget depending on which drafter.
- **Why it binds:** verification is **flat at the margin** (`ms(k) = 1.764 + 0.588k`, no knee), so
  depth is limited by **acceptance**, and the real cost sits in the **drafter**. A drafter that
  cannot produce a token inside the `p × 5.40 ≈ 3.87 ms` budget is not worth drafting with.
- **What the numbers say:** MTP costs **1.064 GB per draft token** (it re-reads 0.666 GiB of LM head
  every pass) = 53 % of that budget. DFlash2 at 4.5 bpw costs **0.49 ms** = 13 %, and is flat in
  block size. Both clear the budget; DFlash2 clears it comfortably.
- **Evidence:** `facts/decoder.md`; `_archive/2026-10-24/docs/agents/production/run-artifacts/hypotheses/H-5.md`
  and `H-9.md`.
- **Cited by:** the speculation unit; `D-01`'s drafter amendment.
