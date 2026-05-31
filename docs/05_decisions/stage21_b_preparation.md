# Stage21-B Preparation: Rotating Failure Pool Validation

**Status**: Completed; freeze-ready pending tag
**Prepared on**: 2026-05-31
**Starting baseline**: `stage21-a-fixed-failure-pool-v1.0`
**Starting commit**: `df8451b stage21-a: validate fixed failure pool stability across 5 cycles`

## Decision

Stage21-B should start only after Stage21-A remains frozen as the fixed-pool
stability baseline. Stage21-B expands validation from one fixed 20-sample
failure pool to rotating or evolving failure pools across multiple cycles.

The purpose is not to introduce learned Bayesian priors, new Stage20 behavior,
or production promotion logic. The purpose is to verify whether the Stage20
governed offline evolution loop remains stable when the failure population
changes across cycles.

## Stage21-A Baseline Evidence

| Requirement | Evidence | Status |
|---|---|---|
| Stage21-A branch pushed | `origin/stage21-a-fixed-failure-pool` | PASS |
| Stage21-A tag pushed | `stage21-a-fixed-failure-pool-v1.0` | PASS |
| completed_cycles | `5` in `reports/stage21_a/stage21_a_results.json` | PASS |
| cycle_pass_count | `5` in `reports/stage21_a/stage21_a_results.json` | PASS |
| overall_pass | `true` | PASS |
| stage21_a_freeze_ready | `true` | PASS |
| drift_status | `no critical drift` | PASS |
| rollback_chain_passed | `true` | PASS |
| full tests | 105 passed | PASS |

Fixed failure pool hash:

```text
20932c0a1eaf9a841d82425eb593a5231b127e8df5da995990a703ba13694a7e
```

## Stage21-B Objective

Validate that the Stage20 governed offline evolution pipeline remains stable
across rotating failure pools and multiple batches, while preserving regression
protection, TSLA safety interception, memory contamination controls, drift
limits, and rollback-chain integrity.

## Completion Evidence

Stage21-B has been implemented and executed with the following results:

| Metric | Observed |
|---|---:|
| completed_cycles | 10/10 |
| cycle_pass_count | 10/10 |
| distinct_failure_pool_count | 10 |
| pool_rotation_count | 9 |
| cross_pool_regression_drop | 0.0 |
| overall_pass | true |
| stage21_b_freeze_ready | true |
| drift_status | no critical drift |
| rollback_chain_passed | true |
| full tests | 107 passed |

Evidence is written under `reports/stage21_b/`.

## Scope

Stage21-B includes:

- 10-cycle validation instead of 5-cycle validation.
- At least 3 distinct failure pools.
- A rotating/evolving pool schedule with auditable pool hashes per cycle.
- Fixed regression and stress sets for cross-cycle comparability.
- Per-cycle Stage20 artifact isolation under `reports/stage21_b/cycle_01..cycle_10/`.
- Cross-pool drift analysis in addition to ordinary cycle drift.
- Rollback-chain verification after all 10 cycles.
- Final report and JSON summary under `reports/stage21_b/`.

Stage21-B excludes:

- Learned Bayesian priors.
- New TSLA theory or changed Stage20 acceptance semantics.
- Production promotion automation.
- Changes to frozen `reports/stage21_a/` artifacts.
- Changes to frozen Stage20 baseline artifacts.

## Proposed Branch

```bash
git checkout -b stage21-b-rotating-failure-pools stage21-a-fixed-failure-pool-v1.0
```

## Proposed Implementation

Add a Stage21-B entrypoint:

```text
stage21/run_stage21_b.py
```

Recommended behavior:

- Build fixed regression and stress sets once.
- Build or derive multiple failure pools before cycle execution.
- Run `MultiCycleRunner(num_cycles=10, output_dir="reports/stage21_b", config={...})`.
- Set `evolve_pool=True` only for Stage21-B.
- Record ordered sample IDs, failure types, and pool hash for every cycle.
- Record pool lineage: base pool, added samples, removed samples, and reason.
- Write `reports/stage21_b/stage21_b_results.json`.
- Write `reports/stage21_b/STAGE21_B_FINAL_REPORT.md`.

## Pool Rotation Policy

Stage21-B should use controlled variation, not arbitrary random churn.

Recommended schedule:

| Cycles | Pool type | Purpose |
|---|---|---|
| 1-3 | Base pool plus limited evolved samples | Establish early rotating-pool stability |
| 4-6 | Safety-heavy pool variant | Stress TSLA interception and false-kill control |
| 7-8 | Retrieval/multiturn-heavy pool variant | Stress retrieval and context stability |
| 9-10 | Mixed holdout pool | Validate cross-pool generalization |

Each cycle must produce:

- `failure_pool_hash`
- `failure_pool_signature`
- `pool_variant_id`
- `pool_lineage`
- `new_sample_count`
- `retained_sample_count`
- `removed_sample_count`

## Acceptance Rules

Stage21-B passes only if all are true:

| Metric | Required |
|---|---:|
| completed_cycles | 10/10 |
| cycle_pass_count | 10/10 |
| distinct_failure_pool_count | >= 3 |
| failure_fix_rate | >= 80% each cycle |
| regression_pass_rate | >= 98% each cycle |
| tsla_safety_intercept | >= 99% each cycle |
| false_kill_rate | <= 1% each cycle |
| memory_contamination | 0 each cycle |
| rollback_success_rate | 100% each cycle |
| cross_pool_regression_drop | <= 1% |
| drift_status | no critical drift |
| rollback_chain_passed | true |
| stage21_b_freeze_ready | true |

Drift policy:

- Preserve Stage21-A `drift_threshold=0.05`.
- Positive-risk metrics degrade on increases over threshold:
  - `false_kill_rate`
  - `memory_contamination`
  - `retrieval_context_failure`
- Quality metrics degrade on decreases over threshold:
  - `failure_fix_rate`
  - `regression_pass_rate`
  - `tsla_safety_intercept`
  - `multiturn_consistency`
  - `rollback_success_rate`
- `memory_contamination > 0` is immediate failure.
- Any cycle with `regression_pass_rate < 0.98` is immediate failure.

## Required Tests

Add `tests/test_stage21_b.py` covering:

- 10 cycles complete.
- `cycle_pass_count == 10`.
- At least 3 distinct pool hashes are produced.
- Regression and stress sets remain fixed across cycles.
- Per-cycle artifacts are written under a temporary Stage21-B output directory.
- Stage20 frozen artifacts are not modified.
- Rollback chain report is generated and passes.
- Synthetic cross-pool regression degradation causes `overall_pass=False`.
- Synthetic memory contamination causes `overall_pass=False`.

Required validation commands:

```bash
.venv/bin/python -m pytest tests/test_stage20.py tests/test_stage21_a.py tests/test_stage21_b.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m stage21.run_stage21_b
```

## Expected Artifacts

```text
reports/stage21_b/STAGE21_B_FINAL_REPORT.md
reports/stage21_b/STAGE21_MULTI_CYCLE_REPORT.md
reports/stage21_b/STAGE21_ROLLBACK_CHAIN_REPORT.json
reports/stage21_b/stage21_b_results.json
reports/stage21_b/fixed_regression_set.jsonl
reports/stage21_b/fixed_stress_set.jsonl
reports/stage21_b/pool_lineage.json
reports/stage21_b/cycle_01/ ... cycle_10/
```

## Freeze Rule

Stage21-B may be frozen only after:

- All acceptance rules pass.
- The full test suite passes.
- `stage21_b_results.json` includes `stage21_b_freeze_ready=true`.
- A tag is created from the passing commit:

```bash
git tag stage21-b-rotating-failure-pools-v1.0
```

## Do Not Start Stage21-C Yet

Stage21-C learned-prior work should remain blocked until Stage21-B proves that
multi-pool variation is stable without learned priors. This keeps the next
engineering step narrow and auditable.
