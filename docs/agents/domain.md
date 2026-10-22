# Domain docs

**Single-context repo:** one `docs/adr/`, one decision record, one `HANDOFF.md`.
No per-package glossaries, no GLOSSARY-MAP.md (removed).

## File structure — the layout that exists

```
/
├── HANDOFF.md                     ← status, read order, baseline, traps
├── AGENTS.md                      ← operating rules for this repo
├── docs/
│   ├── 00-facts.md                ← measured physics; every number derives from it
│   ├── 01-models.md               ← the one model, its architecture and memory
│   ├── 03-native-kernel-inventory.md
│   ├── adr/                       ← numbered decisions, status line first
│   ├── decisions/                 ← pre-loop records; 0001 is authoritative
│   ├── plan/                      ← REQUIREMENTS, RECORD, the structural decisions,
│   │                                the tensor inventory
│   └── research/                  ← measurement reports; 2026-10-22/ is current
├── engine/                        ← vendored sglang, reshaped in place
└── bench/                         ← probes and GGUF manifests
```

**`rspt/`, `kernels/` and `tests/` do not exist.** They belong to an abandoned
thin-engine composition (ADR 0001, superseded). All engine code lives in
`engine/`; there is no second package and no out-of-tree kernel directory.

## What does not exist yet

- **GLOSSARY.md (removed)** — the project has a dense vocabulary (GDN, dispatch key, the
  record, footprint cliff) that will need one glossary. `mattpocock-domain-
  modeling` creates it when the SPEC is written, not before.
- **`docs/agents/domain/decisions/`** — the high-fidelity decision record. It is
  what a SPEC is synthesised from, and it exists because ADRs are lossy: a
  one-paragraph ADR cannot carry confidence, rejected alternatives, or a
  dependency between two decisions.

Missing domain docs are a **persistence gap, not a blank slate.** Before
treating the repo as fresh, search the git history, `HANDOFF.md` and
`docs/plan/RECORD.md` for prior decisions on the same subject. Prior decisions
are historical context, not automatically accepted — re-evaluate each against
the current measurements.

## The three artifacts, and why there are three

| Artifact | Answers | Owner |
|---|---|---|
| GLOSSARY.md (removed) | What does this word mean here? | `mattpocock-domain-modeling` |
| `docs/adr/` | Was a hard choice made, and why? | `mattpocock-domain-modeling` |
| `docs/agents/domain/decisions/` | What did we choose, how sure are we, what does it rest on? | `mattpocock-domain-modeling` |

A glossary term is never restated as a decision; an ADR is a pointer from a
decision, never a copy of it.

## Use the project's vocabulary

When your output names a domain concept — an issue title, a slice objective, a
hypothesis, a test name — use the term the project uses. It is `recurrent state`,
not "SSM cache"; `the record`, not "the leaderboard"; `footprint cliff`, not
"launch overhead". DP and TP are not exchangeable words.

If the concept you need has no name yet, that is a signal: either you are
inventing language the project does not use (reconsider), or there is a real gap
(note it for `mattpocock-domain-modeling`).

## Flag conflicts

If your output contradicts an existing ADR or a pre-loop decision record, surface
it explicitly instead of silently overriding it:

> _Contradicts ADR 0004 (FP8 stays excluded), but worth reopening because…_

When it contradicts a decision in `docs/agents/domain/decisions/`, name the
decision's ID and check its `Depends on` entries to see what else the change
would move.
