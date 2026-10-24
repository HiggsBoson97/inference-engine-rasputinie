# D-06 — Records, never targets; and no shape is the target either

- **Decided:** no number is a goal. The objective is **bandwidth efficiency** — achieved GB/s against
  the measured 776 GB/s wall, per shape — and the discipline is to break the standing record every
  time and explain the shape. A fixed tok/s or GB/s target is a false objective: it licenses leaving
  performance on the table the moment the number is met.
- **Decided:** **no shape is the target either.** "B sessions at C context" is a record to keep, not
  an operating point to optimise for, unless the owner names it. The load-bearing variable is
  **resident tokens `B × C`** (`facts/memory.md`).
- **Rejected:** a concurrency/session target — explicitly rejected by the owner as a goal-shape; and
  a per-shape "operating point" — a whole analysis was once built around `8 × 32 K`, a shape the
  owner never named, and it would have tuned the plan to a point on a surface.
- **Why it binds:** it is the rule that stops the project optimising for the wrong thing, which has
  already happened once.
- **Evidence:** `docs/plan/RECORD.md` (the mechanism), `docs/plan/COST-LEDGER.md` (the surface),
  `docs/plan/REQUIREMENTS.md` H6/H7.
- **Cited by:** every unit that reports a rate; `AGENTS.md` invariants.
