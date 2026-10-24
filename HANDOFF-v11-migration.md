# Handoff — v11 migration

**As of 2026-10-24.** Read order, state, verification, open items. No narrative here: the reasoning
is in `docs/decisions/`, the numbers in `docs/facts/`, the history in `_archive/2026-10-24/`.

## Read order

1. `AGENTS.md` — 15 invariants, the commands, the map. 54 lines.
2. `docs/status.md` — the frontier unit, its proof, the last landed state.
3. `docs/plan/DEVELOPMENT-PLAN.md` — Phase 0 order is declared on line 8.
4. `docs/index.md` — everything else, reachable.

## State

- **Process:** v11. Units with a proof; nine states; knowledge split measured / decided / assumed.
- **Landed:** unit **0.1**, the recipe diff — `DONE`, 851/851 tensors match, control fails as required.
- **Frontier:** unit **0.2**, the i-quant grid tables. Not started. Gates 84 % of the bytes.
- **Tree:** `docs/` 26 files / 196 KB (was 65 / 1.0 MB). 58 files archived. `AGENTS.md` 189 → 54 lines.
- **Machine:** `~/a1 a2 a4 a9 b1 e2` → `~/archive/2026-10-24/` (18 MB). `~/e1` **kept** — it is the
  harness `AGENTS.md` names, not scratch. `local-inference`, `venv`, `wheels` untouched.
- **Branch:** `engine/bringup` @ `bb39bf0f`, 2 commits, **unrun** — an interrupted bring-up. Unrelated
  to this migration; review or delete before building on it.

## Verify the state

```bash
cd "$REPO"
git log --oneline -1 && git status --short                  # expect: clean
wc -l AGENTS.md                                             # expect: 54
find docs -type f | wc -l                                   # expect: 26
python3 -c "import ast; ast.parse(open('bench/04_recipe_diff.py').read()); print('bench parses')"
# the unit-0.1 proof needs the GGUF, so it runs on the host:
#   scp bench/04_recipe_diff.py bench/manifests/rco-iq3s-allocation.txt rasputin:/tmp/ && \
#   ssh rasputin '~/venv/sglang-fork/bin/python /tmp/04_recipe_diff.py \
#     ~/local-inference/models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf /tmp/rco-iq3s-allocation.txt'
#   expect: "matched 851/851 ... RESULT: PASS", exit 0
grep -rho "_archive/2026-10-24/[A-Za-z0-9_./-]*" docs | sort -u | while read -r p; do
  [ -e "$p" ] || echo "DANGLING: $p"; done                  # expect: no output
ssh rasputin '~/venv/sglang-fork/bin/python -c "import sglang; print(sglang.__version__)"'
```

## Open — each names the check that closes it

| open | closes with |
|---|---|
| The grids are not in the tree, so i-quant dequant is unverified (84 % of bytes) | unit 0.2: one block per type bit-exact against the publisher's GSQ reference |
| A Triton GEMV's rate is unknown | the first per-type Triton kernel vs the 521.3 GB/s in `facts/memory.md` |
| Acceptance may not transfer to a 3.44-bpw target | teacher-forced replay at 3.0 / 4.25 / 5.5 / 8.0 bpw of drafter |
| Topology undecided (`D-09`) | mapped-publish round trip, p99, vs RCCL's 19.5 µs |
| The attention path has no rate | a paged decode-attention kernel over a context ladder, q8 and k8v4 |
| Prefill TTFT unmeasured | `2 × 26.9e9 × tokens` FLOPs vs measured achievable FLOPS |
| The 54-line `AGENTS.md` has not been cold-tested | a fresh session reproduces the three numbers in `docs/status.md` from the tree alone |
