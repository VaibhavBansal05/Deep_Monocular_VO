## What this changes

<!-- One or two sentences. What and why, not a file list. -->

## Checklist

- [ ] `pytest -q` passes locally (run from `src/`)
- [ ] `ruff check mvo scripts tests` is clean
- [ ] **`reports/CONTRIBUTIONS.md` updated in this PR** — not retroactively later
- [ ] If I adapted external code: attribution header added **and** a row in `THIRD_PARTY.md`
- [ ] If I touched `mvo/common/`: I explained why below, since it changes everyone's results

## Touching the shared base?

<!-- Delete this section if you only changed your own model folder.
     mvo/common/ holds the frozen contracts (pose convention, batch dict, split,
     metrics schema). Changing them invalidates results the others have already
     produced, so say what changed and who needs to re-run. -->

## Evidence

<!-- Loss curve, metrics.json diff, trajectory plot, or test output.
     Screenshots are fine. This is what the Interim marks are awarded on. -->
