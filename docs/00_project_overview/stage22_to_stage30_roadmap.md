# Stage22 to Stage30 Roadmap

**Status**: Planning baseline
**Prepared on**: 2026-05-31
**Starting point**: `stage21-a-fixed-failure-pool-v1.0`
**Current engineering focus**: Stage22 multi-pool holdout preparation

## Purpose

This roadmap extends the current Stage21 validation track into a Stage30
governed production baseline. It is a planning and readiness document only:
it does not change Stage20 behavior, Stage21-A frozen artifacts, or the
approved Stage21-B scope.

The Stage30 goal is to freeze a governed production baseline after the
offline evolution loop has passed fixed-pool, rotating-pool, long-horizon,
rollback, observability, operator, and release-candidate gates.

## Baseline Chain

```text
Stage20 Baseline v1.0
  -> Stage20-R revalidated baseline
  -> Stage21-A fixed failure pool baseline
  -> Stage21-B rotating failure pool baseline
  -> Stage22 multi-pool baseline
  -> Stage23 learned-prior validation baseline
  -> Stage24 adversarial stress baseline
  -> Stage25 promotion and rollback governance baseline
  -> Stage26 observability and audit baseline
  -> Stage27 integration and operator workflow baseline
  -> Stage28 production shadow baseline
  -> Stage29 release candidate baseline
  -> Stage30 governed production baseline
```

## Stage Roadmap

| Stage | Theme | Primary question | Freeze output |
|---|---|---|---|
| Stage21-B | Rotating failure pools | Does the Stage20 loop remain stable across controlled pool variation? | `stage21-b-rotating-failure-pools-v1.0` |
| Stage22 | Multi-pool holdout validation | Does the loop generalize across independent failure pools and holdouts? | Multi-pool stability baseline |
| Stage23 | Learned prior validation | Can empirical Bayesian priors improve diagnosis without weakening safety? | Learned-prior validation baseline |
| Stage24 | Adversarial and stress horizon | Does the loop hold under adversarial, retrieval-heavy, multiturn, and safety-heavy stress? | Stress resilience baseline |
| Stage25 | Promotion and rollback governance | Can candidate fixes be promoted or rolled back through explicit governance gates? | Promotion governance baseline |
| Stage26 | Observability and audit cockpit | Are metrics, lineage, drift, rollback, and freeze evidence observable and auditable? | Audit readiness baseline |
| Stage27 | Integration and operator workflows | Are CLI/API/operator workflows deterministic, documented, and safe by default? | Operator readiness baseline |
| Stage28 | Production shadow validation | Does the governed loop behave correctly in shadow/canary mode without uncontrolled learning? | Shadow validation baseline |
| Stage29 | Release candidate freeze | Are all prior baselines, reports, tests, and runbooks complete for Stage30 entry? | Stage30 release candidate |
| Stage30 | Governed production baseline | Is the system ready to freeze as the governed production baseline? | `stage30-governed-production-baseline-v1.0` |

## Stage21-B to Stage22 Handoff

Stage22 entry is open after Stage21-B commit, tag, and push. The minimum
handoff package from Stage21-B is:

- `stage21_b_freeze_ready=true`
- `completed_cycles=10`
- `cycle_pass_count=10`
- at least 3 distinct failure pool hashes
- no critical drift
- rollback-chain verification passed
- full test suite passed
- Stage21-B tag pushed

Current Stage21-B evidence:

- `stage21_b_freeze_ready=true`
- `completed_cycles=10`
- `cycle_pass_count=10`
- `distinct_failure_pool_count=10`
- `pool_rotation_count=9`
- `cross_pool_regression_drop=0.0`
- `drift_status=no critical drift`
- `rollback_chain_passed=true`
- full test suite passed: 107 passed

Stage22 should then extend validation from rotating pools in one controlled
schedule to independent multi-pool holdouts with stricter cross-pool evidence.

## Stage30 Direction

Stage30 is not a new theory stage. It is the final governance freeze after the
system has demonstrated that the Stage20 offline evolution loop can be operated
repeatedly, observed, audited, promoted, and rolled back under bounded rules.

Stage30 must preserve these constraints:

- No mutation of frozen Stage20, Stage20-R, or Stage21-A artifacts.
- No uncontrolled online self-training.
- No automatic production writes outside approved promotion gates.
- No learned priors until Stage23 has passed explicit safety validation.
- Every stage must produce JSON and Markdown evidence under `reports/`.

## Planning Status

| Area | Status |
|---|---|
| Stage21-A fixed-pool baseline | Frozen |
| Stage21-B rotating-pool validation | Complete |
| Stage22 preparation | Prepared |
| Stage22-30 roadmap | Prepared |
| Stage30 completion definition | Prepared |
| Stage30 implementation | Not started |
| Stage30 freeze | Blocked until Stage21-B through Stage29 pass |
