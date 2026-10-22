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

| Label | Meaning | Set by |
|---|---|---|
| `station:code` | slice open — in the coder loop, or passed and awaiting promotion | orchestrator at dispatch, and after any FAIL |
| `station:merged` | final gate passed and promoted to `main` | orchestrator at promotion close |
| `escalated` | needs a user decision | orchestrator, with a comment naming the decision |
| `blocked-spec` | a spec gap was discovered | orchestrator, with the gap evidence |

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

This repo's map is in-file: [MAP.md (removed)](../../MAP.md (removed)), a wayfinder map whose
tickets are decisions rather than build steps. Resolve a ticket by appending to
its `## Decisions so far` table in that same file, in the same commit as the
evidence. Publish a map as an issue (`.scratch`-free, GitHub-native) only when
the work outgrows one file; then:

- **Map**: a single issue labelled `wayfinder:map` holding Notes / Decisions-so-far / Fog. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues are not enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, assign it to the driving dev.
- **Blocking**: GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/HiggsBoson97/inference-engine-rasputinie/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/.../issues/<n> --jq .id`), not the `#number` and not the `node_id`. GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Fallback: a `Blocked by: #<n>, #<n>` line at the top of the child body.
- **Frontier query**: list the map's open children, drop any with an open blocker or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a pointer to the map's Decisions-so-far.
