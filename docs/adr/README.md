# Architecture Decision Records

ADRs live here: numbered sequentially from `0001`, one decision per file named
`NNNN-<slug>.md`, with a `Status:` line first (proposed, accepted, superseded,
rejected) followed by Context, Decision and Consequences — the shape
`docs/decisions/0001` already uses. They record the hard choices that were
expensive to reverse and that a future slice must not silently contradict:
data-placement policy, the GGUF and kernel surface, TP-versus-DP per model, the
dedup scope, the multi-user memory model. A decision still being explored belongs
in `docs/agents/domain/decisions/<chapter>.md`, which carries confidence and
rejected alternatives; the ADR is written once it settles. Numbers come from
`docs/00-facts.md` and `bench/results/`, never from a specification. The pre-loop
records in `docs/decisions/` — `0001`, the exclusion decision — predate this
directory and stay where `HANDOFF.md` and MAP.md (removed) point at them; new decisions
land here.
