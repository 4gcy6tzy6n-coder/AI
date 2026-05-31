# Milestone Status

## Current Stage: Stage24 Preparation

The project has moved beyond the early Phase 1 roadmap. The current trusted
baseline chain is **Stage20 Baseline v1.0** revalidated by Stage20-R, followed
by the frozen **Stage21-A fixed failure pool validation baseline** and the
completed **Stage21-B rotating failure pool validation baseline** and
**Stage22 independent multi-pool holdout validation baseline**, followed by
**Stage23 bounded learned-prior validation baseline**.

## Current Gate

| Gate | Status | Evidence |
|---|---|---|
| Stage20 Baseline v1.0 frozen | Complete | `data/stage20/STAGE20_BASELINE_V1.md` |
| Stage20-R revalidation frozen | Complete | `stage20-baseline-v1.0-revalidated` |
| Test environment restored | Complete | `.venv` from `requirements.txt` |
| Stage20 unit tests | Complete | 33 passed |
| Stage21-A unit tests | Complete | 3 passed |
| Full test suite | Complete | 107 passed |
| Stage20 pipeline revalidation | Complete | `reports/stage20_r/stage20_pipeline_revalidation.json` |
| 20-8 rollout conflict cleaned | Complete | `data/stage20/STAGE20_ACCEPTANCE_REPORT.md` |
| Stage21-A fixed failure pool validation | Complete | `reports/stage21_a/stage21_a_results.json` |
| Stage21-A frozen and tagged | Complete | `stage21-a-fixed-failure-pool-v1.0` |
| Stage21-B rotating failure pool validation | Complete | `reports/stage21_b/stage21_b_results.json` |
| Stage21-B implementation tests | Complete | 2 passed |
| Stage20/Stage21 regression tests | Complete | 38 passed |
| Full test suite after Stage21-B | Complete | 107 passed |
| Stage21-B frozen and tagged | Complete | `stage21-b-rotating-failure-pools-v1.0` |
| Stage22 independent multi-pool holdout validation | Complete | `reports/stage22/stage22_results.json` |
| Stage22 implementation tests | Complete | 3 passed |
| Stage20/Stage21/Stage22 regression tests | Complete | 41 passed |
| Full test suite after Stage22 | Complete | 110 passed |
| Stage22-30 roadmap prepared | Complete | `docs/00_project_overview/stage22_to_stage30_roadmap.md` |
| Stage22 preparation plan | Complete | `docs/05_decisions/stage22_preparation.md` |
| Stage23 preparation plan | Complete | `docs/05_decisions/stage23_preparation.md` |
| Stage23 bounded learned-prior validation | Complete | `reports/stage23/stage23_results.json` |
| Stage23 implementation tests | Complete | 3 passed |
| Stage20/Stage21/Stage22/Stage23 regression tests | Complete | 44 passed |
| Full test suite after Stage23 | Complete | 113 passed |
| Stage24 preparation plan | Complete | `docs/05_decisions/stage24_preparation.md` |
| Stage30 completion definition prepared | Complete | `docs/05_decisions/stage30_completion_definition.md` |

## Completed Stage: Stage21-A

**Definition**: 5-cycle fixed failure pool multi-cycle validation.

Stage21-A must not introduce evolving failure pools, learned priors, or new
architecture changes. It validates whether the Stage20 governed offline
evolution loop remains stable across repeated cycles over the same failure pool.

Stage21-A result:

| Metric | Observed |
|---|---:|
| completed_cycles | 5/5 |
| cycle_pass_count | 5/5 |
| overall_pass | true |
| stage21_a_freeze_ready | true |
| drift_status | no critical drift |
| rollback_chain_passed | true |
| fixed_failure_pool_hash | `20932c0a1eaf9a841d82425eb593a5231b127e8df5da995990a703ba13694a7e` |

## Completed Stage: Stage21-B

**Definition**: rotating/evolving failure pool multi-batch drift validation.

Stage21-B moves from repeated validation over one fixed pool to controlled
pool variation. It must prove that the governed Stage20 offline evolution loop
remains stable when the failure pool changes across cycles, without enabling
Stage21-C learned priors or changing the Stage20 baseline behavior.

Stage21-B result:

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

## Stage21-B Entry Conditions

| Condition | Required | Status |
|---|---|---|
| Git repository restored/initialized | Yes | Met |
| pytest and pytest-cov runnable | Yes | Met |
| Stage20 pipeline revalidated | Yes | Met |
| Historical 20-8 rollout conflict documented | Yes | Met |
| Stage21-A completed and frozen | Yes | Met |
| Stage21-A branch and tag pushed | Yes | Met |

## Stage21-B Success Thresholds

| Metric | Threshold |
|---|---:|
| completed_cycles | 10/10 |
| pool_rotation_count | >= 3 distinct pools |
| per-cycle failure_fix_rate | >= 80% |
| per-cycle regression_pass_rate | >= 98% |
| per-cycle tsla_safety_intercept | >= 99% |
| per-cycle false_kill_rate | <= 1% |
| per-cycle memory_contamination | 0 |
| rollback_success_rate | 100% |
| cross-pool regression stability | no critical regression |
| drift_status | no critical drift |
| stage21_b_freeze_ready | true |

## Completed Stage: Stage22

**Definition**: independent multi-pool holdout validation.

Stage22 should use Stage21-B pool lineage as evidence, then define independent
holdout failure pools that are not merely sequential rotations of the Stage21-B
base pool. Stage22 entry is now open from `stage21-b-rotating-failure-pools-v1.0`.

Stage22 result:

| Metric | Observed |
|---|---:|
| completed_cycles | 12/12 |
| cycle_pass_count | 12/12 |
| independent_holdout_pool_count | 4 |
| cycles_per_holdout_pool | 3 each |
| cross_pool_regression_drop | 0.0 |
| overall_pass | true |
| stage22_freeze_ready | true |
| drift_status | no critical drift |
| rollback_chain_passed | true |

## Completed Stage: Stage23

**Definition**: learned-prior validation under bounded governance.

Stage23 may evaluate learned Bayesian prior candidates only after Stage22 is
committed, tagged, and pushed. Learned priors must remain reversible,
auditable, and safety-neutral or better than the frozen heuristic baseline.

Stage23 result:

| Metric | Observed |
|---|---:|
| baseline_chain_complete | true |
| learned_prior_candidate_count | 1 |
| prior_comparison_cycles | 12 |
| learned_prior_safety_regression | 0 |
| expected_cause_probability_delta | +1.34 pp |
| overall_pass | true |
| stage23_freeze_ready | true |
| drift_status | no critical drift |
| rollback_chain_passed | true |

## Next Stage: Stage24

**Definition**: adversarial and stress horizon validation.

Stage24 should stress the governed loop with safety-heavy, retrieval-heavy,
multiturn-heavy, memory-boundary, and mixed adversarial holdouts. Stage25
promotion governance remains blocked until Stage24 freezes.

## Long-Horizon Target: Stage30

Stage30 is now defined as the governed production baseline freeze target. It
remains blocked until Stage21-B through Stage29 are completed, frozen, and
tagged. The current Stage30 preparation artifacts are:

- `docs/00_project_overview/stage22_to_stage30_roadmap.md`
- `docs/05_decisions/stage30_completion_definition.md`
- `docs/05_decisions/stage30_preparation_plan.md`

## Recent Updates

- 2026-05-31: Stage21-A completed, frozen, tagged, and pushed as `stage21-a-fixed-failure-pool-v1.0`.
- 2026-05-31: Stage22-30 roadmap and Stage30 completion definition prepared.
- 2026-05-31: Stage21-B rotating failure pool validation completed with 10/10 cycles and `stage21_b_freeze_ready=true`.
- 2026-05-31: Stage21-B frozen/tagged and Stage22 independent multi-pool holdout preparation documented.
- 2026-05-31: Stage22 independent multi-pool holdout validation completed with 12/12 cycles and `stage22_freeze_ready=true`.
- 2026-05-31: Stage23 bounded learned-prior validation completed with `stage23_freeze_ready=true`.
- 2026-05-30: Stage20-R completed; Stage20 Baseline v1.0 revalidated for Stage21-A entry.
- 2026-05-11: Stage20 Baseline v1.0 frozen.
- 2026-04-17: Project initialized.
