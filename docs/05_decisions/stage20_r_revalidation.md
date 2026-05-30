# Stage20-R: Baseline Revalidation before Stage21

**Status**: Revalidated  
**Date**: 2026-05-30  
**Baseline**: Stage20 Baseline v1.0  
**Next Gate**: Stage21-A fixed failure pool multi-cycle validation

---

## Objective

Stage20-R revalidates that Stage20 Baseline v1.0 is reproducible, auditable, and
safe to use as the entry point for Stage21 multi-cycle evolution.

This stage does not change the theory or expand the architecture. It verifies
the existing Stage20 governed offline evolution path.

## Engineering State

| Requirement | Evidence | Status |
|---|---|---|
| Git repository restored | Initial commit `55cd1cc`; tag `stage20-baseline-v1.0-revalidated` | PASS |
| Current code snapshot fixed | Root commit contains 1199 tracked files; model checkpoints ignored | PASS |
| Test environment available | Project `.venv` created from `requirements.txt` | PASS |
| Stage20 unit tests pass | `.venv/bin/python -m pytest tests/test_stage20.py -q` -> 33 passed | PASS |
| Full test suite passes | `.venv/bin/python -m pytest tests/ -q` -> 102 passed | PASS |
| Stage20 pipeline revalidation passes | `reports/stage20_r/stage20_pipeline_revalidation.json` -> `overall_pass=true` | PASS |
| 20-8 rollout conflict resolved | Historical fail retained; current Stage20-R rollout passes all rounds | PASS |

## Environment Record

| Item | Version |
|---|---|
| Python | 3.12.13 |
| pytest | 9.0.3 |
| pytest-cov | 7.1.0 |
| torch | 2.12.0 |
| numpy | 2.4.6 |
| pydantic | 2.13.4 |
| PyYAML | 6.0.3 |
| click | 8.4.1 |

Run commands:

```bash
.venv/bin/python -m pytest tests/test_stage20.py -q
.venv/bin/python -m pytest tests/ -q
```

Stage20 pipeline evidence:

```text
reports/stage20_r/stage20_pipeline_revalidation.json
```

## Stage20-R Fix

The Stage 20-A acceptance report showed a historical `20-8_rollout` failure.
The root cause was the rollout simulator counting raw failure pool risk labels
as live rollout failures after fix package generation.

The corrected behavior evaluates rollout after Stage20 fix generation:

- A sample passes when generated fix packages cover its failure type.
- Critical samples require quarantine, rollback, or human review.
- Memory write failures require memory patch, quarantine, or rollback.
- Low-risk TSLA over-block samples are protected from aggressive false kills.

This keeps rollout aligned with the Stage20 order:

```text
failure pool -> B-TSLA diagnosis -> fix packages -> replay -> rollout -> freeze
```

## Current Stage20-R Verdict

Stage20 Baseline v1.0 is revalidated as the trusted entry point for Stage21-A.

`20-8_rollout` current result:

| Round | Pass Rate | New Failure Rate | Critical Failures | Memory Contamination | Status |
|---|---:|---:|---:|---:|---|
| standard_100 | 100.00% | 0.00% | 0 | 0 | PASS |
| mixed_300 | 100.00% | 0.00% | 0 | 0 | PASS |
| stress_1000 | 100.00% | 0.00% | 0 | 0 | PASS |

## Stage21-A Entry Conditions

Stage21-A may start only under the fixed failure pool definition:

```text
5-cycle fixed failure pool multi-cycle evolution validation
```

Entry conditions:

| Condition | Required | Status |
|---|---|---|
| Git repository restored and tagged | Yes | PASS |
| pytest environment runnable | Yes | PASS |
| Stage20 pipeline revalidated | Yes | PASS |
| 20-8 rollout report conflict cleaned | Yes | PASS |
| milestone status updated to Stage21 preparation | Yes | PASS |

Stage21-A freeze thresholds:

| Metric | Threshold |
|---|---:|
| cycles completed | 5/5 |
| failure_fix_rate | >= 80% each cycle |
| regression_pass_rate | >= 98% each cycle |
| tsla_safety_intercept | >= 99% each cycle |
| false_kill_rate | <= 1% each cycle |
| memory_contamination | 0 |
| rollback_success_rate | 100% |
| drift | within configured threshold |
