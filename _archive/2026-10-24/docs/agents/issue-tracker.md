# Issue tracker: GitHub

Slice issues, specs and research tickets for this repo live as GitHub issues on
`HiggsBoson97/inference-engine-rasputinie` (remote `github`, **public**). Use the
`gh` CLI for every operation; run it inside the clone and it infers the repo.

## Conventions

- **Create an issue**: `gh issue create --title "SLICE <id>: <objective>" --body-file <path>` — write the body to a file under `docs/agents/production/run-artifacts/briefs/` and pass the path, never a shell heredoc that mangles backticks.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments with `jq`, and read the labels too.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with the `--label` and `--state` filters the stage needs.
- **Comment on an issue** (the position ledger): `gh issue comment <number> --body "<GATE line>"`.
- **Apply / remove labels**: `gh issue edit <number> --add-label "station:code" --remove-label "station:merged"`.
- **Close**: `gh issue close <number> --comment "..."` — and only through a promotion close, see `docs/agents/control-plane.md`.

GitHub shares one number space across issues and pull requests, so a bare `#42`
may be either: resolve with `gh pr view 42`, then fall back to
`gh issue view 42`.

## Label vocabulary (closed set)

Created on the repo; do not invent a fifth stage label.

| Label | Set | Clear |
|---|---|---|
| `station:code` | at dispatch | at `GATE\|CODE\|PASS`; a passed slice awaits promotion unlabeled |
| `station:merged` | at `GATE\|MERGE\|PASS` | never |
| `escalated` | at an escalation, with a comment naming the decision | when the decision is recorded |
| `blocked-spec` | at an architectural re-entry, with the gap evidence | at `RESUMED`, or a superseding PLAN line naming the slice |

The labels are a **view of the comment tail**, never the position itself: the tail
decides, a contradicting label is an anomaly fixed before any dispatch, selection reads
open issues rather than labels, and every label write is read back. The rules are in
`docs/agents/control-plane.md`.

The stock GitHub labels (`bug`, `documentation`, `enhancement`, …) exist too and
may be added for humans browsing the repo; they carry no loop meaning and the
derivation contract ignores them.

## Branch protection: available, not yet enabled

`main` **can** be protected since 2026-10-22, when the repo was made public.
Before the first dispatch, enable: PR required, the deploy-integrity check named
as a required check, `enforce_admins: true`, then record the change here.

Until that is done, the gate layer is **local discipline only**: `main` moves only
through an orchestrator promotion after a FINAL exercise PASS; that is a rule the
loop keeps, not a rule GitHub enforces.

## When a skill says "publish to the issue tracker"

Create a GitHub issue on this repo, labelled `station:code`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments` and read the comment tail — that is the
position of record.

## Wayfinding operations

This repo keeps no in-file wayfinder map: a map lives as a GitHub issue from the start.
When the work outgrows one sitting and Discovery's fog test sends it to
`mattpocock-wayfinder`, publish it GitHub-native (`.scratch`-free):

- **Map**: a single issue labelled `wayfinder:map` holding Notes / Decisions-so-far / Fog. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues are not enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, assign it to the driving dev.
- **Blocking**: GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/HiggsBoson97/inference-engine-rasputinie/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/.../issues/<n> --jq .id`), not the `#number` and not the `node_id`. GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Fallback: a `Blocked by: #<n>, #<n>` line at the top of the child body.
- **Frontier query**: list the map's open children, drop any with an open blocker or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a pointer to the map's Decisions-so-far.
