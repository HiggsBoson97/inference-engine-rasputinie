# Domain Docs

How the skills consume this repo's domain documentation when exploring the
codebase. **Single-context**: one root `GLOSSARY.md`, one `docs/adr/`. There is
no `GLOSSARY-MAP.md` and no per-package glossary — the engine is one context.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root — the vocabulary this project actually
  uses, in this project's sense.
- **`docs/adr/`** — numbered decisions, one per file, status line first. Read
  the ADRs that touch the area you are about to work in.
- **`docs/decisions/`** — the pre-loop decision records. `0001` is the
  exclusion decision and is still authoritative; `HANDOFF.md` and `MAP.md`
  point at it there. New ADRs land in `docs/adr/`, not here.
- **`docs/agents/domain/decisions/<chapter>.md`** — the decision record. Read
  the chapters relevant to the area, then the ADRs they point at.

If any of these do not exist yet, **proceed silently**. Do not flag the absence
and do not propose creating them upfront: `mattpocock-domain-modeling` (reached
via `grill-with-docs`, and by the fullstack PLAN stage) creates them lazily when
a term or a decision actually gets resolved.

## The three artifacts, and why there are three

| Artifact | Answers | Owner |
|---|---|---|
| `GLOSSARY.md` | What does this word mean here? | `mattpocock-domain-modeling` |
| `docs/adr/` | Was a hard choice made, and why? | `mattpocock-domain-modeling` |
| `docs/agents/domain/decisions/` | What did we choose, how sure are we, and what does it rest on? | `mattpocock-domain-modeling` |

The first two are terse by design and are the durable, human-facing surface. The
third is the high-fidelity record a SPEC is synthesised from, and it exists
because the first two are lossy: a one-paragraph ADR cannot carry confidence,
rejected alternatives, or a dependency between two decisions. A SPEC written
from ADRs alone drifts from what was actually agreed.

They are not duplicates. A glossary term is never restated as a decision; an ADR
is a pointer from a decision, never a copy of it.

## File structure

```
/
├── GLOSSARY.md
├── docs/
│   ├── adr/
│   │   ├── README.md
│   │   └── 0001-<decision>.md
│   ├── decisions/                 ← pre-loop records; 0001 is still authoritative
│   └── agents/
│       └── domain/
│           └── decisions/
│               └── <chapter>.md
├── kernels/                       ← HIP kernels for gfx1100
├── rspt/                          ← the engine package
└── tests/                         ← correctness tests against torch references
```

## Use the glossary's vocabulary

When your output names a domain concept — in an issue title, a slice objective, a
refactor proposal, a hypothesis, a test name — use the term as `GLOSSARY.md`
defines it. Do not drift to synonyms the glossary avoids: it is `recurrent
state`, not "SSM cache"; it is `bandwidth ceiling`, not "roofline"; DP and TP are
not exchangeable words.

If the concept you need is not in the glossary yet, that is a signal: either you
are inventing language the project does not use (reconsider), or there is a real
gap (note it for `mattpocock-domain-modeling`).

## Flag conflicts

If your output contradicts an existing ADR or a pre-loop decision record, surface
it explicitly instead of silently overriding it:

> _Contradicts ADR-0001 (the excluded engine stays excluded), but worth reopening
> because…_

When it contradicts a decision in `docs/agents/domain/decisions/`, name the
decision's ID and check its `Depends on` entries to see what else the change
would move.
