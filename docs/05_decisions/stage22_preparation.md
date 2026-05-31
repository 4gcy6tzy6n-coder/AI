# Stage22 Preparation: Independent Multi-Pool Holdout Validation

**Status**: Prepared after Stage21-B freeze
**Prepared on**: 2026-05-31
**Starting baseline**: `stage21-b-rotating-failure-pools-v1.0`
**Starting commit**: `9241a14 stage21-b: validate rotating failure pool stability`

## Decision

Stage22 starts the Stage22-30 governance chain by moving beyond Stage21-B
controlled pool rotation. Stage22 must validate independent multi-pool holdouts:
failure pools should be built as separate holdout sets, not only as sequential
mutations of one base pool.

The purpose is to prove that the governed Stage20 offline evolution loop can
generalize across independently constructed failure pools while preserving
regression protection, TSLA safety interception, memory contamination controls,
drift limits, and rollback-chain integrity.

## Stage21-B Handoff Evidence

| Requirement | Evidence | Status |
|---|---|---|
| Stage21-B commit | `9241a14` | PASS |
| Stage21-B tag | `stage21-b-rotating-failure-pools-v1.0` | PASS |
| completed_cycles | `10` in `reports/stage21_b/stage21_b_results.json` | PASS |
| cycle_pass_count | `10` in `reports/stage21_b/stage21_b_results.json` | PASS |
| distinct_failure_pool_count | `10` | PASS |
| pool_rotation_count | `9` | PASS |
| cross_pool_regression_drop | `0.0` | PASS |
| overall_pass | `true` | PASS |
| stage21_b_freeze_ready | `true` | PASS |
| drift_status | `no critical drift` | PASS |
| rollback_chain_passed | `true` | PASS |
| full tests | 107 passed | PASS |

## Stage22 Objective

Validate that the Stage20 governed offline evolution pipeline remains stable
across independent holdout pools and not only across Stage21-B rotations.

## Scope

Stage22 includes:

- at least 4 independent holdout pools;
- 12 total validation cycles;
- fixed regression and stress sets for comparability;
- per-pool and cross-pool drift analysis;
- holdout lineage that records pool construction source and sample signatures;
- per-cycle artifacts under `reports/stage22/cycle_01..cycle_12/`;
- final reports under `reports/stage22/`;
- rollback-chain verification after all cycles.

Stage22 excludes:

- learned Bayesian priors;
- Stage23 prior training or CPT updates;
- production promotion automation;
- changes to frozen Stage20, Stage21-A, or Stage21-B artifacts.

## Proposed Entry Point

```text
stage22/run_stage22.py
```

Recommended behavior:

- Build fixed regression and stress sets once.
- Build at least 4 independent 20-sample holdout pools.
- Run 12 cycles, assigning 3 cycles per holdout pool.
- Record `holdout_pool_id`, `holdout_pool_hash`, ordered sample signatures, and lineage.
- Write `reports/stage22/stage22_results.json`.
- Write `reports/stage22/STAGE22_FINAL_REPORT.md`.

## Acceptance Rules

Stage22 passes only if all are true:

| Metric | Required |
|---|---:|
| completed_cycles | 12/12 |
| cycle_pass_count | 12/12 |
| independent_holdout_pool_count | >= 4 |
| cycles_per_holdout_pool | >= 3 |
| failure_fix_rate | >= 80% each cycle |
| regression_pass_rate | >= 98% each cycle |
| tsla_safety_intercept | >= 99% each cycle |
| false_kill_rate | <= 1% each cycle |
| memory_contamination | 0 each cycle |
| rollback_success_rate | 100% each cycle |
| cross_pool_regression_drop | <= 1% |
| drift_status | no critical drift |
| rollback_chain_passed | true |
| stage22_freeze_ready | true |

## Required Tests

Add `tests/test_stage22.py` covering:

- 12 cycles complete.
- `cycle_pass_count == 12`.
- at least 4 independent holdout pool hashes are produced.
- each holdout pool runs at least 3 cycles.
- regression and stress sets remain fixed across cycles.
- per-cycle artifacts are written under a temporary Stage22 output directory.
- Stage20, Stage21-A, and Stage21-B frozen artifacts are not modified.
- rollback chain report is generated and passes.
- synthetic cross-pool regression degradation causes `overall_pass=False`.
- synthetic memory contamination causes `overall_pass=False`.

Required validation commands:

```bash
.venv/bin/python -m pytest tests/test_stage20.py tests/test_stage21_a.py tests/test_stage21_b.py tests/test_stage22.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m stage22.run_stage22
```

## Expected Artifacts

```text
reports/stage22/STAGE22_FINAL_REPORT.md
reports/stage22/STAGE22_MULTI_POOL_REPORT.md
reports/stage22/STAGE22_ROLLBACK_CHAIN_REPORT.json
reports/stage22/stage22_results.json
reports/stage22/fixed_regression_set.jsonl
reports/stage22/fixed_stress_set.jsonl
reports/stage22/holdout_pool_lineage.json
reports/stage22/cycle_01/ ... cycle_12/
```

## Freeze Rule

Stage22 may be frozen only after:

- all acceptance rules pass;
- the full test suite passes;
- `stage22_results.json` includes `stage22_freeze_ready=true`;
- a tag is created from the passing commit:

```bash
git tag stage22-multi-pool-holdout-v1.0
```

## Do Not Start Stage23 Yet

Stage23 learned-prior validation remains blocked until Stage22 proves
independent multi-pool stability without learned priors.
