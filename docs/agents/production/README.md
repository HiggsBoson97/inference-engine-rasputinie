# production/

The apparatus the fullstack loop uses on this repo: `test-rig.md` (the rig — the
instruments, the gate set, the correctness-versus-speed ordering, written against
`rasputin` and the sglang ROCm venv), `lessons-learned.md` (cross-run memory: escapes,
standing rules, the outcome ledger), `hypothesis-ledger.md` (Discovery's ledger, open
until every row closes), and `run-artifacts/` (the receipts and the reports a GATE
comment points at).

`test-rig.md` and `lessons-learned.md` are **targets, not files.** PLAN writes the rig as
its last act, from the skill's rig schema; LEARN creates the lessons file after the first
promotion. Nothing here is authored by hand at priming time, and an empty file in either
place would claim work the loop has not done.

`run-artifacts/` already exists, and fills as the loop runs — `briefs/` holds the dispatch
briefs now; the other directories appear with their first file:

```
run-artifacts/
├── briefs/           ← every dispatch brief, written before the dispatch
├── hypotheses/       ← H-<n>.md evidence reports
├── reviews/          ← <slice>-r<n>.md, the reviewer's return, committed by the coder
├── test-<slice>.md   ← the rig report at CODE
├── integrity-<slice>.md
└── final-<slice>.md
```

A coder or reviewer reads only what its brief points it at; the orchestrator owns every
write here that a worker does not commit itself.
