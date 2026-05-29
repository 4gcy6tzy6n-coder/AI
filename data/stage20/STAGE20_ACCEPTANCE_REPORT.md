# Stage 20-A Acceptance Hardening Report

**Generated**: 2026-05-11T14:10:04.089160+00:00
**Version**: Stage20_Baseline_v1.0

---

## Acceptance Verdict: **PASS**

## Acceptance Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| failure_fix_rate | 100.00% | >= 80.00% | PASS |
| regression_pass_rate | 100.00% | >= 98.00% | PASS |
| tsla_safety_intercept | 100.00% | >= 99.00% | PASS |
| false_kill_rate | 0.00% | <= 1.00% | PASS |
| memory_contamination | 0.00% | == 0.00% | PASS |
| retrieval_context_failure | 0.00% | <= 2.00% | PASS |
| multiturn_consistency | 100.00% | >= 95.00% | PASS |
| rollback_success_rate | 100.00% | >= 100.00% | PASS |

## Failure Distribution by Type

- **knowledge_miss**: 15.0% #######
- **memory_write_error**: 10.0% #####
- **multiturn_anomaly**: 15.0% #######
- **retrieval_mismatch**: 15.0% #######
- **safety_boundary_error**: 10.0% #####
- **tsla_false_pass**: 10.0% #####
- **tsla_over_block**: 10.0% #####
- **unnatural_generation**: 15.0% #######

## Risk Level Distribution

- **critical**: 20.0%
- **high**: 20.0%
- **low**: 30.0%
- **medium**: 30.0%

## Pipeline Step Summary

- **20-0_trigger**: PASS
- **20-1_failure_pool**: PASS
- **20-2_5_btsta_diagnoses**: PASS
- **20-6_replay**: PASS
- **20-7_shadow**: PASS
- **20-8_rollout**: FAIL
- **20-9_freeze**: PASS

## Baseline Freeze

- **Version**: Stage20_Baseline_v1.0
- **Frozen at**: 2026-05-11T14:10:04.088616+00:00
- **Components frozen**: 14
- **Rollback runbook**: data/stage20/stage20_rollback_runbook.md