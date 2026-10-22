# Control-plane protocol — GitHub Issues as the loop's state machine

> Canonical home of the issue schema, the derivation contract, and the dispatch
> contract for **this repo**. Paths are already resolved for
> `rasputinie`; the grammar is unchanged from
> `fullstack-software-development/references/control-plane-protocol.md`.

GitHub Issues is the control plane: a slice's issue body is its static contract,
its comment thread is the position ledger, labels are the stage vocabulary. **The
orchestrator operates it alone** — it derives each stage, dispatches the coder
unit, and posts every comment from a presence-checked report; workers never write
to it.

## Project configuration (settled at priming, 2026-10-22)

| Setting | Value |
|---|---|
| Repo | `HiggsBoson97/inference-engine-rasputinie` (**public** since 2026-10-22), remote `github` |
| Issue tracker | GitHub Issues via `gh` — see `docs/agents/issue-tracker.md` |
| Slots | **2** — two slices, each a coder and its reviewer, four workers in flight |
| Labels | `station:code`, `station:merged`, `escalated`, `blocked-spec` (closed set) |
| Branch protection on `main` | **available, not yet enabled** |

**Branch protection became available on 2026-10-22**, when the repo was made
public. It must be enabled **before the first dispatch**: require a pull request,
name the deploy-integrity check as a required check, and set
`enforce_admins: true`. Until it is enabled — and the fact must be recorded here
when it is — the gate layer is **local discipline only**: `main` moves only
through an orchestrator promotion after a FINAL exercise PASS, because the loop
keeps that rule, not because GitHub enforces it.

## Where the configuration lives

One prefix, one standard. No worker guesses which prefix a project uses.

```
docs/agents/
├── issue-tracker.md          ← where issues live, and the gh commands for them
├── domain.md                 ← where GLOSSARY.md (removed), the ADRs and the decision record live
├── control-plane.md          ← this file: the loop grammar
├── production/
│   ├── test-rig.md           ← the apparatus PLAN writes last; coder self-test and reviewer read it
│   ├── lessons-learned.md    ← cross-run memory
│   ├── run-artifacts/        ← receipts and dispatch briefs; GATE comments point here
│   │   └── briefs/
│   └── issues.md             ← the control plane when a project has no GitHub remote (unused here)
└── invariants/               ← project rules the coder loop must not violate
```

The repo's operating rules live in the root `AGENTS.md` (hardware, the two hard
rules, sync topology, measurement discipline), not in `docs/agents/agents.md`.

## Issue body schema (static contract — never edited after creation)

```markdown
## Objective
<one sentence: what this slice delivers>

## Repo / Worktree
- repo: /Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie
- branch: slice/<slice-id> (cut from main at dispatch by the orchestrator)

## Acceptance Criteria
- <AC, gradeable, one line each>

## Gates
<ordered stage list — CODE is the only stage inside the coder loop>

## Routing
<stage=guide-path pairs, relative to the skill's references/ — e.g. "CODE=coder/coder-guide.md">

## Decisions
- <every decision this slice depends on — the coder loop receives nothing else>

## Risk hypotheses (<=2 predicted failure trajectories)
- <trigger -> changed surface -> observable consequence>   (or "none identified")

## Persisted data (conditional — required when the slice changes what data saved by a previous version means)
- <migration for that data | documented no-op proof | none>
```

Rules:

- The body carries **shape, never state**. No stage, no status — the stage lives
  in labels plus comments.
- `Gates` lists the stages this slice passes, in order; the stage vocabulary is
  closed — CODE is the only stage inside the coder loop.
- `Routing` is the stage→guide-path map; a guide path is relative to
  `fullstack-software-development/references/`.
- One issue per slice. Slice dependencies are recorded in `Blocked by` (ordering
  constraint) — the orchestrator sequences slot dispatches; there is no parent
  auto-promotion to emulate.
- **Risk hypotheses are hypotheses, never findings:** authored at PLAN from the
  rig heuristics and the slice's touched surfaces; the coder's self-test walks
  each to confirmed or refuted; the reviewer disposes them through its findings;
  an undisposed plausible trajectory blocks promotion.
- **Persisted data** is required for any slice that changes what data saved by a
  previous version means — here that includes a GGUF file that already exists on
  disk, a `bench/results/*.json` schema, or a KV/state checkpoint layout. Missing
  on such a slice → the issue is not created. A non-`none` declaration routes the
  boot-on-persisted-state check into the slice's self-test.
- At PLAN, before issue creation, run the coherence scan — an authoring error is
  caught there, not later.

## Label vocabulary (closed set)

| Label | Meaning | Set by |
|---|---|---|
| `station:code` | slice open — in the coder loop, or passed and awaiting promotion | orchestrator at dispatch, and after any FAIL |
| `station:merged` | final gate passed and promoted to main | orchestrator at promotion close |
| `escalated` | needs a user decision | orchestrator, with a comment naming the decision |
| `blocked-spec` | spec gap discovered | orchestrator, with the gap evidence |

A closed slice carries `station:merged`; every open slice carries `station:code` —
the comment tail is the position of record (pipeline lines take no label). A
label that contradicts the tail is an anomaly: fix the ledger before any
dispatch.

## Thread grammar (position ledger — append-only)

**The orchestrator posts every comment on a slice issue**, written from the
presence-checked report — a claim without evidence lands nowhere:

```
GATE|<stage>|PASS|<one-line evidence pointer>
GATE|<stage>|FAIL|<ranked findings, one line each>
->GATE:<next-stage>       (posted together with the PASS line, where the Gates list names a successor)
```

- Rejection: post `GATE|CODE|FAIL|<findings>`, set `station:code`, re-dispatch the
  coder loop with the findings stamped into the brief.
- Approval: post `GATE|CODE|PASS|<evidence>`; the slice awaits promotion. **The
  successor — where one exists — is the next entry in the issue's `Gates` list:
  read it, never recall it.** A posted successor the list does not name is an
  anomaly: fix the ledger with a corrective comment naming the right successor
  before any dispatch continues.
- A coder loop claiming done before the final PASS is rejected; the comment
  records the violation; the next derivation uses the comment tail, never the
  claim.

## Pipeline-level gates (pipeline issue — one per project)

Pipeline lines live on the project's pipeline issue, posted by the orchestrator;
they take no label and do not enter slice derivation:

| Line | When |
|---|---|
| `GATE|PLAN|PASS|<rig path>` | PLAN loop close (once) |
| `SLOT|<slice-id>|OPEN|<stage>` / `SLOT|<slice-id>|CLOSE|<stage>|<verdict>` / `SLOT|-|IDLE|<reason>` | every coder-unit dispatch, loop return, or idle period |
| `GATE|PROMOTE|PASS|<old-sha..new-sha>` | after each promotion merge |
| `GATE|FINAL|PASS|<report path>` | after each promotion's exercise |

Promotions refuse a project whose `GATE|PLAN|PASS` line is missing; each
promotion requires the FINAL exercise of its own candidate.

A FINAL FAIL routes by class: slice defects reopen the slice (re-dispatch the
coder subagent with the findings stamped into a fresh brief), rig gaps re-enter
the PLAN loop, a broken or missing instrument holds and escalates.

## Derivation contract (normative)

Before dispatching anything on a slice, the orchestrator computes the stage from
the issue's comment tail:

```
stage =
  first Gates entry                        if no GATE comment exists
  CODE                                     if last GATE comment is GATE|X|FAIL
  successor(X) in Gates                    if last is GATE|X|PASS and X is not final
  promotion-ready                          if last is the final Gates entry PASS
```

- **Closure precondition (hard rule):** an issue closes only via the promotion
  close (`station:merged`).
- **Failure routing (canonical).** Every FAIL routes by class — every other
  mention points here:

  | Failure | Routes to |
  |---|---|
  | Slice defect (reviewer, closing, or FINAL findings) | reopen the slice (`station:code`), re-dispatch the coder loop with the findings stamped into a fresh brief |
  | Rig gap | re-enter the PLAN loop, targeted |
  | Broken or missing instrument | hold + escalate, naming the exact decision needed |
  | GitHub-side failure (auth, rate limit, conflict) | retry with backoff; if it persists, hold + escalate naming the operation |

  An all-rig-gap FAIL re-enters PLAN; a mixed FAIL routes to CODE for its slice
  findings, and the rig-gap subset separately re-enters PLAN.
- **Marker obligation (hard rule):** every stage run produces exactly one GATE
  comment, posted by the orchestrator from a presence-checked report. The comment
  tail is the position of record; labels mirror it.
- **Caps (canonical home — every other mention points here).** Two caps bound
  cycles on one slice, both counted from the issue's comment history, never from
  recollection:
  - **Refire cap:** 3 identical FAIL verdicts on one gate = forced re-orient
    (replan the brief, never a 4th identical dispatch).
  - **Round cap:** 5 fix cycles on one slice, regardless of verdict variety =
    forced re-orient (replan the brief or escalate).

## The coder dispatch (the coder-unit contract)

The coder unit is a **first-layer subagent** the orchestrator dispatches with
`delegate_task`. The dispatch goal is SHORT and names the brief file as
authoritative; the brief file — written with `mattpocock-handoff` before the
dispatch, under `docs/agents/production/run-artifacts/briefs/` — carries the full
contract. There is no standing goal, no terminal session, no completion marker:
the coder runs to its report and returns it to the orchestrator, who
presence-checks it against evidence.

**The brief file — required contents, in this order:**

```
STATION: CODE (the coder loop)
SKILL: fullstack-software-development
GUIDE: <absolute path to the skill root>/references/coder/coder-guide.md
WORKTREE: <absolute path>   BRANCH: slice/<id>   BASE SHA: <sha>
TOOLSET: <the rig's declared instruments>   (workers hold no control-plane write access)
BATCHING: <verbatim — the standing block below>
PROCESSES: <verbatim — the standing block below>
BUDGET: <tool-call budget; at ~90% switch to write-out mode; over-budget items report as failed(budget), never omitted>
ACS: <the issue's AC list, verbatim>
DECISIONS: <the issue's Decisions block>
RISK-HYPOTHESES: <the issue's risk hypotheses, verbatim>
FINDINGS-SO-FAR: <on fix loops, the ranked findings being addressed>
PERSISTED-DATA: <the issue's ## Persisted data block, verbatim ("none" if none)>
JOURNEYS: <orchestrator-enumerated journeys for this slice, if any — floor, never ceiling; name for each journey which stage owns it>
REPORT SHAPE: <the coder's report shape, below>
```

**The two standing blocks, verbatim:**

```
BATCHING: independent reads, searches, fetches and read-only commands go in ONE turn — they run concurrently, and a turn costs one whole model request; only a call that needs a result you do not have yet waits for the next turn
PROCESSES: join or kill every process you start before your final report; a started process whose result was never read is a named gap, never assumed successful
```

The two blocks are neither optional nor paraphrased: a worker cannot infer them —
a dispatch starts in a fresh context with no memory and no inherited identity.
Keep the field order exactly as listed: the brief's opening is the dispatch
prefix, and a byte-stable prefix is what the local runtime's cache reuses.

Rules that bind the dispatch:

- **The opening trio — every dispatch, every depth (hard rule).** The brief
  begins with three lines: **who the worker is** (its `STATION:` line names it),
  **which skill to load** (`fullstack-software-development`), and **the absolute
  path to its own guide**. From those three lines alone the worker knows its
  role, its skill, and the guide that defines its behaviour.
- **The brief never tells the worker how to do its job** — which tests to run,
  which checks to perform, how to read. That is the guides' business. The brief
  states the task, the evidence standard, and the report shape; nothing else.
- **The branch moves under the coder between rounds; a review window freezes it.**
  The coder commits as it works; when a review round opens, the round's tip is
  frozen — no commits while the reviewer reads — and the round's findings,
  receipts, and verdict bind to that SHA. Commits land between rounds.
- **Findings-so-far (fix loops) ride in the brief** — the ranked findings being
  addressed, so the coder addresses exactly those and nothing else.
- **Fix loops re-dispatch the same way** — a fresh coder subagent with the
  findings-so-far stamped into a fresh brief. There is no session to resume: the
  worktree and the issue thread are the state.

**The report shape the brief demands** (the coder's report; the orchestrator
presence-checks it):

```
GATE|CODE|PASS|<summary line>
- built: <one line per AC — done / how>
- self-test: <journeys driven, real entry point named; evidence pointers; negative control result>
- review: <rounds; findings fixed/refuted, pointers>
- closing: <VERIFIED; report path>
- receipts: <committed paths on the slice branch, force-added past any ignore rule>
- changed: <file list>
- commits: <shas on the slice branch>
- PR: <number, url>
- effects: <observable surfaces changed — env/config/route/schema — one line each, or "none">
FRAMEWORK-REPORT
- worked / broke / improve / other
```

On a failed close or an unfinished loop the tail is `->GATE:CODE` (fix loops) or
`BLOCKED: <decision needed>` instead of the PASS block.

**The reviewer the coder dispatches** follows the same contract one depth down.
The coder writes the reviewer's brief with `mattpocock-handoff`; it opens with the
trio (its role: adversarial-code-reviewer; the skill to load; its guide:
`<skill root>/references/coder/adversarial-code-reviewer/adversarial-code-reviewer-guide.md`).
The brief carries only the slice's facts — the ACs, Decisions and Persisted-data
verbatim, the coder's report, the worktree, the rig pointer — and nothing that
shapes the verdict. The coder never biases it. The reviewer returns VERIFIED /
CHANGES_REQUESTED / BLOCKED with findings and evidence; the coder commits the
return verbatim with its fix round, in the same commit. The reviewer never writes
the branch.

## Dispatch contract (the primer rule — any orchestrator-dispatched worker)

Every orchestrator dispatch (the FINAL exercise and any bake-off candidate)
carries, self-contained:

```
STATION: <station name>
SKILL: fullstack-software-development
GUIDE: <path to the worker's guide, relative to the skill root>
ARTIFACT: <worktree + BRANCH: slice/<id>, or — FINAL — the deployed candidate URL (no worktree, no branch)>
TOOLSET: <the rig's declared instruments>   (workers hold no control-plane write access)
BATCHING: <verbatim — the standing block above>
PROCESSES: <verbatim — the standing block above>
ACS / DECISIONS / RISK-HYPOTHESES / FINDINGS-SO-FAR / PERSISTED-DATA: <as the worker needs>
JOURNEYS: <for FINAL, the candidate's touched surfaces enumerated from the slice issue — floor, never ceiling>
REPORT SHAPE: <the worker's GATE line + evidence pointers; FINAL: GATE|FINAL|PASS|<report path> or GATE|FINAL|FAIL|<ranked findings>, followed by FRAMEWORK-REPORT>
```

A FINAL dispatch is fresh context, never a coder or reviewer of this slice; its
guide is `references/orchestrator/final-exercise-guide.md`.

Workers know nothing else. They never self-route, never post to GitHub, never
choose their role — the orchestrator posts all comments and owns every stage
transition.

## GitHub verb map (orchestrator)

| Situation | Operation |
|---|---|
| PLAN creates a slice | `gh issue create -R HiggsBoson97/inference-engine-rasputinie --title "SLICE <id>: <objective>" --body-file <schema> --label station:code` |
| Stage passed / failed | `gh issue comment <n> --body "<GATE line>"` + `gh issue edit <n> --add-label/--remove-label` |
| Coder-unit dispatch | `delegate_task` to the coder subagent, the brief file on disk; `SLOT` line on the pipeline issue |
| Stalled coder loop | stop the delegation, re-dispatch fresh with the last GATE line stamped into a fresh brief |
| Escalation | label `escalated` + decision comment |
| Promotion | local merge of slice branch into `main` (no branch protection — the gate is discipline), push to `origin`; deploy + integrity check; FINAL exercise; `GATE\|MERGE\|PASS` + `station:merged` + `gh issue close`; `GATE\|PROMOTE\|PASS` + `GATE\|FINAL\|PASS` on the pipeline issue |
| Pipeline view | `gh issue list --label station:code` (and the other labels) — one query per stage |

## Evidence conventions

- Every claim in a worker report carries an evidence pointer (file:line, command
  output, artifact path). The orchestrator presence-checks the report before
  posting; a report missing a required section goes back, never posted.
- **Findings travel in full — a count never stands in for the list.** Every
  report that carries findings carries each one with its pointer; counts are
  headings, never substitutes.
- **The closing check owns number truth.** The coder loop's closing pass reads
  the machine-readable rows (the results file, receipt paths, SHAs); the
  orchestrator cites the source row and never re-derives a number itself. A green
  console line proves only that the build ran.
- **Receipts ride in the same commit as the code they evidence**, on the slice
  branch, force-added past any repo-level ignore rule. A receipt that exists only
  in the main checkout, or only where an ignore rule swallowed it, is not a
  receipt.
- A speed number without its baseline in the same artifact is an opinion, not
  evidence; `bench/results/` is the schema for both.
- Run artifacts live in `docs/agents/production/run-artifacts/`; the GATE comment
  points at them.
- GitHub comments are immutable history — the audit trail needs no separate
  ledger.

## Restart protocol (per slice)

1. Reconcile the pipeline issue, the slice thread, and `git log` — the comment
   tail is the position of record; never re-fire from memory.
2. A stalled coder loop is a failed dispatch: stop the delegation, then
   re-dispatch a fresh coder subagent with the ledger's findings-so-far stamped
   into a fresh brief.
3. Caps count from comment history.
4. The worktree state is ground truth for what was built; the thread is ground
   truth for what was judged.
