# Control plane — GitHub Issues as the loop's state machine

One issue per slice: the body is the contract, the comment tail is the position, labels
are a view of the tail. **The orchestrator operates the plane alone** — it derives the
stage, dispatches every worker, and posts every comment from a presence-checked report.
Workers never write to it.

**This file is a projection, not the grammar.** It carries this repo's configuration and
absolute paths; the normative rules — the envelope, the grammar lines, the cap, the
failure responses — live in the skill's reference files and are cited here, never
restated in a repo-local variant. Where this file and the skill disagree, the skill wins
and this file is the defect.

Skill root, and the base of every `references/…` path below:

```
/Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development
```

Pinned: `fullstack-software-development` **v10.0.0**. The orchestrator owns this plane;
its own instructions are `references/orchestrator.md`.

## Project configuration

| Setting | Value |
|---|---|
| Repo | `HiggsBoson97/inference-engine-rasputinie` — **public** since 2026-10-22 |
| Remotes | `origin` = bare repo on host `rasputin`; `github` = the public repo |
| Mac checkout | `/Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie` |
| Host checkout | `/home/mattia-bosi/development/inference-engine-rasputinie` |
| Issue tracker | GitHub Issues; `gh` for every operation — `docs/agents/issue-tracker.md` |
| Slots | **2** — two coder slots; each carries a coder and its reviewer, four workers in flight |
| Labels | `station:code`, `station:merged`, `escalated`, `blocked-spec` (closed set) |
| Branch protection on `main` | **available since 2026-10-22, not yet enabled** |
| Rig | `docs/agents/production/test-rig.md` — a target; PLAN writes it last |
| Lessons | `docs/agents/production/lessons-learned.md` — a target; LEARN creates it |

Branch protection became available when the repo went public. Enable it **before the
first dispatch**: PR required, the deploy-integrity check named as a required check,
`enforce_admins: true`, then record the change here. Until it is enabled the gate layer
is **local discipline only** — `main` moves only through a promotion merge that stands on
a FINAL PASS, because the loop keeps that rule, not because GitHub enforces it.
`references/promote.md` treats absent protection as a refusal, so the first dispatch
after enabling is also the first that can merge.

## Where the configuration lives

One prefix, one standard. No worker guesses which prefix a project uses.

```
docs/agents/
├── control-plane.md          ← this file: the loop's configuration and grammar
├── issue-tracker.md          ← the transport: gh commands, label set, protection state
├── domain.md                 ← the ADRs and the decision record (`D-<n>` ids)
├── invariants/               ← rules the coder loop must not violate
└── production/
    ├── test-rig.md           ← TARGET: PLAN's last act, written from `references/rig.md`
    ├── lessons-learned.md    ← TARGET: created by LEARN from `references/lessons.md`
    ├── hypothesis-ledger.md  ← Discovery's ledger (`references/discovery-records.md`)
    └── run-artifacts/
        ├── briefs/           ← every dispatch brief
        ├── hypotheses/       ← H-<n>.md evidence reports
        ├── reviews/          ← <slice>-r<n>.md, the coder-committed returns
        ├── test-<slice>.md   ← the rig report a GATE comment cites
        ├── integrity-<slice>.md
        └── final-<slice>.md
```

The rig and the lessons file are **targets, not placeholders**: nothing is authored by
hand at priming time. PLAN writes the rig as its last act and a dispatch before it exists
is not a dispatch; LEARN appends the first lessons line after the first promotion. An
empty file in either place is a lie about what the loop has done. The directories under
`run-artifacts/` appear with their first file; `briefs/` is populated now.

There is no `issues.md` here: this project has a GitHub remote, so the GitHub-less
transport in `references/protocol.md` §Contract versions does not apply. The repo's
operating rules live in the root `AGENTS.md`: hardware, the two hard rules, sync
topology, measurement discipline.

## The issue body is the contract

Authored at PLAN, immutable afterwards.

```markdown
## Objective
<one sentence: what this slice delivers>

## Repo / Worktree
- repo: /Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie
- branch: slice/<slice-id> (cut from main at dispatch)

## Acceptance Criteria
- <gradeable, one line each>

## Gates
CODE -> FINAL -> MERGE

## Decisions
- <the D- ids this slice rests on>

## Seams
- <the public boundaries this slice is proven at>

## Required Properties
- <the invariants that must still hold afterwards>

## Disqualifying Conditions
- <the finding that voids the slice outright>

## Risk tier
standard | high (<reason, or the one-way-door catch-all>)

## Risk hypotheses
- <trigger -> surface -> observable consequence>   (or "none identified")

## Persisted data
- <migration | documented no-op proof | none>

## Blocked by
- <slice ids, or "none">
```

- The body carries **shape, never state**. No stage, no status — the stage lives in the
  tail, the label mirrors it.
- `Seams`, `Required Properties`, and `Disqualifying Conditions` are **mandatory**. A
  missing one is a **human escalation**, never a default, never filled from the plan's
  prose. The plan judge fails the plan gate on any of the seven blocks missing
  (`references/plan-judge.md`, checklist item 5).
- `Seams` is agreed with the human before authoring: the module boundaries this work is
  proven at, a short list, not a test plan.
- `Risk tier` is `standard` or `high`. `high` requires the promotion approval, names its
  reason, and takes the one-way-door catch-all when the slice cannot be undone. The
  high-risk floor is auth/identity, payments, secrets, CI/CD and branch protection,
  schema and data deletion, deploy/infra, public API contracts, crypto — here that
  includes the GGUF reader contract and anything that changes on-disk weight layout.
- `Gates` is the ordered stage list; v10 defines one, `CODE -> FINAL -> MERGE`. CODE is
  the only stage inside the coder loop. A successor is **read from this list, never
  recalled**.
- **Risk hypotheses are hypotheses, never findings**: authored at PLAN from the rig
  heuristics and the slice's touched surfaces; the coder's self-test walks each to
  confirmed or refuted; the reviewer disposes them through its findings.
- **Persisted data is always present**: `none`, a migration, or a documented no-op proof.
  For this repo that means a GGUF file already on disk, a `bench/results/*.json` schema,
  or a KV/state checkpoint layout. A non-`none` declaration puts the
  boot-on-persisted-state check into the slice's self-test.
- One issue per slice. `Blocked by` carries the ordering constraint; the orchestrator
  sequences the dispatch, and there is no parent auto-promotion. A slice with an
  unpromoted blocker waits.
- `D-` ids come from `docs/agents/domain.md`'s decision record. The issue cites ids and
  never restates their content.

## Contract versions

The body is immutable, so its identity travels as a version. Every brief carries
`CONTRACT v<n> sha256:<hash>`; every worker report echoes the version it read; the
orchestrator recomputes the hash at presence check.

- The hash is SHA-256 over the canonical contract bytes: **the issue body exactly as
  stored, then the latest `CONTRACT-AMENDMENT` block exactly as posted, then one trailing
  newline.** That is the text the brief points at.
- A worker that cannot read the contract reports `BLOCKED`. It never reconstructs the
  contract from the envelope.
- A mismatch, or an unreadable contract, is a **failed dispatch** — never a report.
- A change is a **new version**: post `CONTRACT-AMENDMENT v<n+1> sha256:<hash>` carrying
  the changed block. Never edit the body, never edit an amendment.
- An amendment that changes an AC, a seam, a required property, the risk tier, or the
  slice's scope is a re-entry and needs the human re-approval `references/re-entry.md`
  defines.

## Labels are a view

The closed set is `station:code`, `station:merged`, `escalated`, `blocked-spec`. The
comment tail is authoritative; a label that contradicts it is an anomaly, fixed before
any dispatch. **Labels never select work**: selection reads open issues and derives the
stage from the tail. Every label write is read back.

| Label | Set | Clear |
|---|---|---|
| `station:code` | at dispatch | at `GATE\|CODE\|PASS`; a passed slice awaits promotion unlabeled |
| `station:merged` | at `GATE\|MERGE\|PASS` | never |
| `escalated` | at an escalation | when the decision is recorded |
| `blocked-spec` | at an architectural re-entry | at `RESUMED`, or a superseding PLAN line naming the slice |

No fifth stage label, and the stock GitHub labels (`bug`, `documentation`, …) carry no
loop meaning. Posting a pipeline line takes no label.

## The thread is the position

Append-only. The orchestrator posts every comment from a presence-checked report, so a
claim without evidence lands nowhere.

```
GATE|<stage>|PASS|<evidence pointer>
GATE|<stage>|FAIL|<ranked findings>
GATE|CODE|BLOCKED|<decision needed>
->GATE:<next-stage>
APPROVAL|PLAN|<artifact path>
APPROVAL|PROMOTE|<slice>|<artifact path>
ESCALATED|<decision needed>
PAUSED|<re-entry ref>
RESUMED
```

- `<stage>` is `CODE`, `FINAL`, or `MERGE`. One GATE line per round of a stage.
- `->GATE:<next-stage>` is a separate line in the same comment as its PASS, and the
  successor comes from the issue's `Gates` list. A posted successor the list does not
  name is an anomaly: correct it with a comment before any dispatch continues.
- A FAIL inherits no successor. `GATE|CODE|BLOCKED` is a **first-class line** with no
  successor, and it routes to escalate, not to STALL.
- A worker's `BLOCKED` is a report status, not a comment: the orchestrator records
  `ESCALATED` (a decision is needed) or `PAUSED` (an architectural re-entry). Only CODE
  has a `BLOCKED` line.
- `ESCALATED` and `PAUSED` block dispatch until `RESUMED`; the last GATE line remains the
  position throughout.
- Comments are append-only: a correction is a new comment naming the one it corrects.
  Never edit or delete a line.
- A coder claiming done before the final PASS is rejected; the next derivation uses the
  tail, never the claim.

## Pipeline-level lines

The pipeline issue is the project's single issue titled `PIPELINE`; its transport is
`docs/agents/issue-tracker.md`. It carries project facts only — a slice's position stays
on its own thread, and pipeline lines enter no slice derivation.

| Line | When |
|---|---|
| `APPROVAL\|PLAN\|<artifact path>` | the human approves the SPEC and the slice list |
| `GATE\|PLAN\|PASS\|<rig path>\|<approval ref>` | Discovery close; again on a superseding re-entry (the ref is the re-approval, or "none — unchanged") |
| `IDLE\|<reason>` | a slot with no ready slice |
| `STALL\|<slice>\|<reason>` | a dispatch failed to return; posted before the re-dispatch |
| `GATE\|PROMOTE\|PASS\|<old..new sha>` | after a promotion merge |

**There is no slot-granularity line and no pipeline-level FINAL line.** Slot occupancy is
derived, not posted: the coder slots count worktrees whose slice has no `GATE|CODE|PASS`;
the promotion lane counts open PRs whose slice has `GATE|CODE|PASS` and no
`GATE|MERGE|PASS`. A count that disagrees with the tails is an anomaly.

## Derivation

Read the slice's tail, the PR head, the worktree, and the pipeline's newest line.

```
first entry in Gates                 if no GATE line exists and no PR is open
successor(X) in Gates                if the last GATE is X|PASS and X is not MERGE
closed                               if the last GATE is MERGE|PASS
the failure response table           if the last GATE is X|FAIL or CODE|BLOCKED
```

- A missing GATE line is CODE only when the worktree is freshly cut and no PR exists.
  An open PR with no GATE line is an anomaly: **STOP**, never default to CODE.
- A comment read older than the pipeline's newest line is stale: **STOP**.
- Labels are not an input.
- The derivation is computed from the thread every time. There is no cached stage and no
  remembered round.

## Failure responses

Five responses, and one precondition. **STOP/RECONCILE is a precondition, not a
response**: a diverged `main`, a foreign checkout, a stale read, a label/tail
contradiction, or an occupancy anomaly stops the transition until it is reconciled.

| Response | When | Evidence obligation |
|---|---|---|
| **OWNER RETRY** | a slice defect, a merge conflict, or a stalled dispatch | same role, same gate, typed findings or a conflict report; a retry does not reset the cap. A maintenance retry is typed: no label churn, no review round, and any head change voids FINAL |
| **RE-PLAN** | a rig gap, a wrong decomposition, a cap hit, a finding that indicts a plan premise | targeted Discovery, one judge, a superseding `GATE\|PLAN\|PASS`; human re-approval when scope, public contract, or risk tier changes |
| **ESCALATE** | a broken or undeclared instrument, an environment block, a scope conflict, a repeated failure | the exact decision, the options, the evidence; a hold (pending check) is distinct from a decision |
| **MECHANICAL RETRY** | auth, rate limit, a transient network failure | same actor, same operation, backoff; no dispatch, no GATE line |
| **RECORD** | a defect found after the merge | the outcome ledger row names the stage that should have caught it |

- A `BLOCKED` return routes to ESCALATE. It is never a stall and never a FAIL.
- A mixed FAIL can need two responses at once: the slice findings to OWNER RETRY, the rig
  gap to RE-PLAN.
- Repetition escalates: two strikes of any shape, two identical stalls, or any failure on
  a degraded lane.
- A failed presence check is a failed dispatch: post `STALL|<slice>|<reason>` and dispatch
  a fresh coder from `references/redispatch-brief.md`. Nobody takes over the work or runs
  one step of it.

## The review cap

**One cap: three review rounds per cycle.**

- A round is one `GATE|CODE|` comment on the issue, including a `BLOCKED` round.
- A `STALL` or a re-dispatch adds no round. A retry never resets the count.
- The counter is the issue's `GATE|CODE|` lines — not a file count, not recollection.
- A fourth round is a **re-plan or an escalate**, never a reset and never an archive.
- The brief carries `REVIEW-CAP: 3` and the reviewer's brief carries `ROUND: n/3`. They
  name the same limit; `references/review-contract.md` owns the round, the return and
  the pin.

## Slots and occupancy

Two slots. A slot is occupied from dispatch until `GATE|CODE|PASS`; a passed slice awaits
promotion in the promotion lane, not in a coder slot. A ready slice plus an empty slot is
a dispatch; no ready slice plus an empty slot is one `IDLE|<reason>` line. Occupancy is
derived from worktrees plus PRs, split by the tail — never from a posted counter.

## The dispatch envelope

One contract for every worker (`references/protocol.md` §Dispatch). The filled envelopes
are `references/templates/coder-brief.md`, `references/templates/reviewer-brief.md`, and
`references/templates/final-brief.md`; the stall envelope is
`references/redispatch-brief.md`.

```
STATION: <role>
SKILL: fullstack-software-development
GUIDE: <absolute path to the role's guide>
SLICE: <slice-id>
ARTIFACT: <worktree + branch, or the deployed candidate for FINAL>
CONTRACT: v<n> sha256:<hash> — the issue body plus any amendments
RIG: /Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie/docs/agents/production/test-rig.md
TOOLSET: <the instruments this role may run>
BATCHING: <the protocol's standing block, verbatim>
PROCESSES: <the protocol's standing block, verbatim>
BUDGET: <tool-call budget; at ~90 % switch to write-out mode; unfinished items report as failed(budget)>
ROUND: <n>
FINDINGS-SO-FAR: <ranked findings, or "none">
ACS / DECISIONS / SEAMS / REQUIRED-PROPERTIES / DISQUALIFYING-CONDITIONS / RISK-HYPOTHESES / PERSISTED-DATA: <the issue's blocks, as the role needs>
REPORT SHAPE: <the role's shape>
```

- **The opening trio is a hard rule**: `STATION:` names the role, `SKILL:` names
  `fullstack-software-development`, `GUIDE:` is the **absolute** path to that role's
  guide. From those three lines a worker knows its role, its skill, and the guide that
  defines its behaviour. `BATCHING` and `PROCESSES` are the protocol's blocks, copied
  verbatim — a worker cannot infer them.
- **Coder extras:** `WORKTREE`, `BRANCH`, `BASE SHA` (the work baseline), `DIFF BASE`
  (the slice's original branch point, never moves), `ROUND: n`, `REVIEW-CAP: 3`. Guide:
  `references/coder.md`.
- **Reviewer extras:** the coder's round report **verbatim**, `DIFF BASE`,
  `PRIOR-RETURN: <path or "none — round 1">`, `ROUND: n/3`. Guide: `references/reviewer.md`.
- **FINAL extras:** the deployed `ARTIFACT`, the `INTEGRITY` record path, the `SURFACES`
  floor, and the issue's ACs and `PERSISTED-DATA` inline. FINAL has no worktree and no
  branch. Guide: `references/final.md`.
- The brief carries a **projection** of the issue plus the contract version. It is not a
  bare pointer: a worker that cannot read the issue reconstructs nothing and reports
  `BLOCKED`.
- A worker report travels as **content, never a pointer** — the coder's round report and
  the reviewer's return are the artifacts under test.
- Depth: the coder at depth 1, its reviewer at depth 2, FINAL in a fresh context that is
  never a coder or reviewer of this slice.
- Every brief is written to `docs/agents/production/run-artifacts/briefs/` before the
  dispatch, and the dispatch goal names the brief as authoritative.

## Promotion

The orchestrator alone moves `main`, and only on a FINAL PASS. Green CI means the gates
ran; it is never merge authorization. The sequence, the PR reads and the conflict rule are
`references/promote.md`.

**One promotion SHA:** the last `VERIFIED` return's `PIN` == the PR head at merge == the
SHA the candidate was built from == the SHA in the merge record. Any commit after that
PIN voids the review and the FINAL exercise together, and both re-run on the new head. A
merge record whose SHA does not match is refused.

**Two records, or the promotion is refused:**

1. Slice issue: `GATE|MERGE|PASS|<sha>|<build ts>|<env ok>|<FINAL report path>|<approval ref or "none">` + `station:merged` + close.
2. Pipeline issue: `GATE|PROMOTE|PASS|<old..new sha>`.

Run artifacts the records cite:

| Path | Written by |
|---|---|
| `docs/agents/production/run-artifacts/integrity-<slice>.md` | the promotion's deploy-integrity pass |
| `docs/agents/production/run-artifacts/final-<slice>.md` | the FINAL exercise |
| `docs/agents/production/run-artifacts/reviews/<slice>-r<n>.md` | the coder, committed verbatim |
| `docs/agents/production/run-artifacts/test-<slice>.md` | the rig report at CODE |

`Risk tier: high` puts a recorded `APPROVAL|PROMOTE|<slice>|<artifact path>` before the
merge; a missing approval does not reach it. The merge is the last act: squash, delete the
branch, then read it back on GitHub — a merge that cannot be seen did not happen. `--auto`
and `--admin` are forbidden, and `main` is never pushed directly.

## Presence check before posting

Read one pointer per required section; never re-run the work.

1. Every section of the role's report shape is present and non-empty. A round report is
   checked for its sections and its return; the open-PR check belongs to the final CODE
   PASS.
2. The reviewer's committed return exists at its path, and its `ROUND` and `PIN` match the
   report.
3. The report echoes the brief's contract version; `TEST-DIFF` is present; no P0/P1 is
   open; the cap is not exceeded.
4. `changed` equals `git diff --name-only <DIFF BASE>..HEAD`.

A failed check is a failed dispatch: `STALL`, then a fresh coder. A report is never
returned for repair and never posted unverified. Findings travel in full — a count never
stands for the list. The reviewer's committed return owns number truth; the orchestrator
cites it and never re-derives a number.

## Re-entry

A finding that contradicts a Discovery decision, a SPEC line, a seam, a gate, or a public
contract is a Discovery error, not a slice defect. `references/re-entry.md` owns the
procedure.

- **`local`** — the slice absorbs it without changing the SPEC, a seam, a gate, or a public
  contract. Fixed in the coder's fix round; the reviewer's round is its approval; no plan
  gate, no new `GATE|PLAN|PASS`.
- **`architectural`** — anything else. Post `PAUSED|<re-entry ref>` on the slice thread
  before the targeted pass; post `RESUMED` when the superseding `GATE|PLAN|PASS` lands.
  Disagreement is `architectural`: the conservative class wins.
- The targeted pass updates only the affected artifacts, reruns the affected hypothesis
  tests, takes one scoped judge, and posts a superseding `GATE|PLAN|PASS` whose approval
  ref is the re-approval or "none — unchanged".
- A contract change inside a re-entry is a `CONTRACT-AMENDMENT`, never an edit, and the
  human re-approves only when scope, a public contract, or the risk tier changes.
- An architectural re-entry never posts `GATE|CODE|FAIL`: the slice is paused, not failed.
  An architectural re-entry without a targeted `APPROVED` leaves the slice `BLOCKED`.

## LEARN

After a promotion, append the outcome row and the `FRICTION:` lines to
`docs/agents/production/lessons-learned.md` — the project instance of
`references/lessons.md`, which owns the schema, the durable bar, and the outcome ledger.
The orchestrator writes it, from worker reports; workers only report. Append-only, and
reworked in place rather than overwritten. A lesson recurring across three slices compiles
into a standing rule in `AGENTS.md`, citing the proving slices.

## GitHub verb map

`gh` for every issue operation; run it in the clone and it infers the repo.

| Situation | Operation |
|---|---|
| PLAN creates a slice | `gh issue create --title "SLICE <id>: <objective>" --body-file <path> --label station:code` |
| Stage passed / failed | `gh issue comment <n> --body "<GATE line>"`, then `gh issue edit <n> --add-label/--remove-label`, then read the label back |
| Escalation | label `escalated` + `ESCALATED|<decision needed>` |
| PR state | `gh pr view <n> --json state,mergeable,reviewDecision,headRefOid`, then `gh pr checks <n>` |
| Protection | `gh api repos/HiggsBoson97/inference-engine-rasputinie/branches/main/protection --jq '.required_status_checks.contexts'` |
| Promotion | `gh pr merge <n> --squash --delete-branch`, then read back `mergedAt,mergeCommit`; the two records follow |
| Close | only through the promotion close: `station:merged` + `GATE|MERGE|PASS` + `gh issue close` |

Create, read, list, comment, label and close are `docs/agents/issue-tracker.md`'s job;
this table is only the loop-specific calls.

## Evidence

Every claim in a worker report carries a pointer: file:line, command output, or artifact
path. A receipt is a committed file under `docs/agents/production/run-artifacts/`, riding
in the same commit as the code it evidences on the slice branch — force-added past any
ignore rule. A receipt that exists only in the main checkout is not a receipt.

A number without its baseline in the same artifact is an opinion; `bench/results/` is the
schema for both. A green console line proves only that the build ran — counts come from
the machine-readable output. GitHub comments are immutable history, so the thread is the
audit trail and there is no separate ledger.
