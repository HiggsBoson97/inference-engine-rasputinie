# Work brief — bring the repo's fullstack machinery to skill v10.0.0

```
STATION: machinery maintainer (docs-only)
SKILL: fullstack-software-development  (v10.0.0 — the source of truth)
GUIDE: /Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development/references/protocol.md
ARTIFACT: /Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie
SCOPE: docs/agents/** only
BUDGET: 60 tool calls
```

## GOAL

The repo's local control-plane documents were written against an older version of the
`fullstack-software-development` skill. Rewrite them so a fresh orchestrator reading
**only** `docs/agents/control-plane.md` plus the skill's own reference files sees one
consistent contract, with no path, grammar line, or cap that the skill no longer has.

The repo files are a **projection**: they carry this repo's configuration and its
absolute paths, and they point at the skill's references for the normative grammar. They
must not fork the grammar, and they must not contradict it.

## SOURCE OF TRUTH — read all of these first

Skill root: `/Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development`

Read, in this order: `SKILL.md`, `references/protocol.md`, `references/orchestrator.md`,
`references/discovery.md`, `references/discovery-records.md`, `references/plan-judge.md`,
`references/re-entry.md`, `references/rig.md`, `references/promote.md`,
`references/final.md`, `references/review-contract.md`, `references/coder.md`,
`references/reviewer.md`, `references/lessons.md`, `references/redispatch-brief.md`,
`references/templates/coder-brief.md`, `references/templates/reviewer-brief.md`,
`references/templates/final-brief.md`.

## CURRENT STATE — the known deltas

`docs/agents/control-plane.md` (361 lines) is the file with the damage. It currently:

1. Cites `references/control-plane-protocol.md` — that file no longer exists; the
   canonical grammar is `references/protocol.md`.
2. Cites role guides at paths that no longer exist:
   `references/coder/coder-guide.md`, `references/coder/adversarial-code-reviewer/adversarial-code-reviewer-guide.md`,
   `references/orchestrator/plan-guide.md`, `references/orchestrator/final-exercise-guide.md`.
   The v10 files are flat: `references/coder.md`, `references/reviewer.md`,
   `references/orchestrator.md`, `references/discovery.md`, `references/plan-judge.md`,
   `references/final.md`.
3. Carries an issue-body schema missing `Seams`, `Required Properties`,
   `Disqualifying Conditions`, `Risk tier`, and carrying a `Routing` block that v10 does
   not have. v10's `Gates` list is `CODE -> FINAL -> MERGE`.
4. Has no contract-version mechanism. v10 requires `CONTRACT v<n> sha256:<hash>` in every
   brief and every report echo, with the hash taken over the canonical bytes (issue body,
   then the latest `CONTRACT-AMENDMENT` block, then one trailing newline), and amendments
   posted as a new version, never an edit.
5. Posts `SLOT|…` pipeline lines and a pipeline-level `GATE|FINAL|PASS`. v10 says
   explicitly: **there is no `SLOT` line and no pipeline-level FINAL line.** The pipeline
   lines are `APPROVAL|PLAN`, `GATE|PLAN|PASS|<rig path>|<approval ref>`, `IDLE|<reason>`,
   `STALL|<slice>|<reason>`, `GATE|PROMOTE|PASS|<old..new sha>`.
6. Has two caps (refire 3 / round 5). v10 has **one**: three review rounds per cycle,
   counted from the issue's `GATE|CODE|` lines, a `BLOCKED` round counts, a retry or
   re-dispatch adds none, and a fourth is a re-plan or an escalate.
7. Routes failures by class only. v10 has five responses — **OWNER RETRY, RE-PLAN,
   ESCALATE, MECHANICAL RETRY, RECORD** — plus STOP/RECONCILE as a precondition, and
   `GATE|CODE|BLOCKED` as a first-class line that routes to escalate.
8. Has no `ESCALATED` / `PAUSED` / `RESUMED` lines, no re-entry (`references/re-entry.md`)
   procedure, no presence-check list, no LEARN step pointing at `references/lessons.md`
   and `docs/agents/production/lessons-learned.md`, no one-promotion-SHA rule, and no
   run-artifact paths (`final-<slice>.md`, `integrity-<slice>.md`).
9. Describes the dispatch envelope in an older shape. v10's envelope and its coder /
   reviewer / FINAL extras are in `references/protocol.md` §Dispatch; the filled examples
   are the three templates. The brief's opening trio (station, skill, absolute guide path)
   is a hard rule.

`docs/agents/issue-tracker.md` also still points at `MAP.md (removed)` as "this repo's
map" — dead prose about a file that was deleted. Remove that section's dead references
and keep the parts that are true (gh commands, label closed set, branch-protection state).

`docs/agents/production/README.md` says `run-artifacts/` "is created when the first
receipt lands" — `run-artifacts/briefs/` and `run-artifacts/hypotheses/` now exist.

## RULES

1. **Do not invent grammar.** Every normative line, cap, label behaviour, and failure
   response you write must be traceable to a file under the skill's `references/`. If the
   skill is silent, say the repo does not decide it and point at the skill; do not fill
   the gap with a plausible rule.
2. **Preserve every repo-specific fact**, even where the skill knows nothing about it:
   - repo `HiggsBoson97/inference-engine-rasputinie`, public since 2026-10-22, remote `github`
   - remotes: `origin` = bare repo on host `rasputin`, `github` = public
   - Mac checkout `/Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie`
   - host checkout `/home/mattia-bosi/development/inference-engine-rasputinie`
   - two slots, four workers in flight
   - labels `station:code`, `station:merged`, `escalated`, `blocked-spec` (closed set)
   - branch protection on `main`: available since the repo went public, **not yet
     enabled**; until it is, the gate layer is local discipline only
   - `gh` for every issue operation; `docs/agents/production/issues.md` is the
     GitHub-less fallback and is unused here
3. **Do not create** `test-rig.md`, `lessons-learned.md`, or any rig content. The rig is
   PLAN's last act and is written from `references/rig.md`; the machinery doc must say so
   and must not pretend it exists. Do not create an empty one.
4. **Do not touch** `HANDOFF.md`, `AGENTS.md`, `docs/plan/**`, `docs/adr/**`,
   `docs/research/**`, `bench/**`, `engine/**`, or anything outside `docs/agents/`.
   `AGENTS.md` §"Build loop" points at `docs/agents/control-plane.md`; if you find a
   contradiction between the two, report it in your summary rather than editing AGENTS.md.
5. Keep the file's existing voice: dense, declarative, tables where the skill uses tables,
   no motivational prose. Aim for one screen of configuration plus the normative sections
   — the old file's length came partly from restating things the skill already owns;
   prefer a pointer over a paraphrase for anything a worker never reads.
6. No date-stamped filenames, no tombstones, no "previously said X" notes.

## DELIVERABLES

1. `docs/agents/control-plane.md` — rewritten. Must contain, in a sensible order:
   project configuration table; where the configuration lives (the `docs/agents/` tree as
   it exists, plus the target entries the rig and lessons will fill); the issue-body schema
   (v10, with every mandatory block and the "missing one is a human escalation" rule);
   contract versions and the amendment rule; labels as a view, with the closed set and the
   "labels never select work" rule; the thread grammar incl. `BLOCKED`/`ESCALATED`/
   `PAUSED`/`RESUMED`; pipeline-level lines; derivation (v10's five-line tree); the five
   failure responses + the STOP/RECONCILE precondition; the one review cap; slots and
   occupancy; the dispatch envelope and its extras, pointing at the three templates; the
   one-promotion-SHA rule and the two merge records; run-artifact paths; the presence-check
   list; re-entry; LEARN. Point at the skill's reference files for anything normative and
   long — do not reproduce a full guide.
2. `docs/agents/issue-tracker.md` — surgical edits only: remove the dead `MAP.md` material,
   keep everything true, and make the label table agree with control-plane.md.
3. `docs/agents/production/README.md` — correct the `run-artifacts/` sentence.
4. Nothing else.

## HOW TO VERIFY YOU DID IT RIGHT — run these before you report

```bash
SKILL=/Users/mattiabosi/.hermes/profiles/fullstack-software/skills/software-development/fullstack-software-development
REPO="/Volumes/Crucial X9 Pro 2TB/Personal/tailscale/rasputin/development/inference-engine-rasputinie"

# 1. every skill path named in the repo's docs/agents resolves
grep -oh "references/[A-Za-z0-9_./-]*\.md" "$REPO"/docs/agents/*.md | sort -u |
  while read -r p; do [ -f "$SKILL/$p" ] || echo "DEAD PATH: $p"; done

# 2. no stale names survive anywhere under docs/agents
grep -rn -E "control-plane-protocol|coder-guide|adversarial-code-reviewer-guide|plan-guide|final-exercise-guide|SLOT\||MAP\.md" "$REPO"/docs/agents || echo "clean"

# 3. the caps and responses match the skill
grep -n "three review rounds" "$SKILL/references/protocol.md"
grep -n -E "OWNER RETRY|RE-PLAN|ESCALATE|MECHANICAL RETRY|RECORD" "$SKILL/references/protocol.md" | head

# 4. the issue schema block-lists match, block for block
diff <(grep -E "^## " "$SKILL/references/protocol.md" | sed -n '1,20p') \
     <(grep -E "^## " "$REPO/docs/agents/control-plane.md")
```

Check 4 is advisory, not literal — the repo file is a projection, not a copy. What it must
catch is a **missing mandatory block** (`Seams`, `Required Properties`,
`Disqualifying Conditions`, `Risk tier`) or an **invented one** (`Routing`).

Also confirm by reading, not grepping: a reader who opens control-plane.md and nothing
else can answer — what stage is this slice at, what may I dispatch, what do I post, when
may `main` move, and how many review rounds are left. If any of those four is not
answerable from the file, the rewrite failed.

## REPORT SHAPE

```
# Machinery sync — repo docs/agents → skill v10.0.0
## Changed (file: what, and why in one line)
## Deltas removed (the old line / path, and what replaced it)
## Preserved verbatim (the repo-specific facts, listed)
## Verification (each of the four checks, with its actual output)
## Contradictions found but not edited (AGENTS.md or elsewhere — with the pointer)
```

Effort: **medium**. One careful read of the skill, one rewrite, then the checks. Do not
expand scope: if the skill and the repo disagree about something outside `docs/agents/`,
report it, do not fix it.
