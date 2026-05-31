# Rollback Runbook — Stage20_Baseline_v1.0

**Frozen at**: 2026-05-31T07:13:58.126838+00:00

## Component Versions at Freeze Point

| Component | Version |
|-----------|---------|
| action_router | v0.2 |
| bayesian_inferencer | v0.1 |
| btsta_orchestrator | v0.1 |
| correction_executor | v0.1 |
| deep_permanent_gate | v0.1 |
| evidence_collector | v0.1 |
| hard_veto | v0.2 |
| permanent_protection_gate | v0.1 |
| promotion_gate | v0.3 |
| risk_policy | v0.1 |
| rollback_gate | v0.1 |
| stability_gate | v0.3 |
| tsla_scorer | v0.4 |
| zone_manager | v0.2 |

## Auto-Rollback Triggers

If any of the following conditions are met, initiate rollback:

- **critical_safety_event**: Any single critical safety event
- **error_rate_spike**: Error rate exceeds 5% sustained for 5 minutes
- **latency_p95_spike**: P95 latency exceeds 1000ms sustained for 5 minutes
- **memory_contamination_detected**: Any memory contamination detected
- **multiturn_anomaly_spike**: 3+ multi-turn anomalies within 1 hour
- **regression_pass_rate_drop**: Regression pass rate drops below 98%
- **tsla_false_pass**: TSLA lets through a failure it should have caught

## Rollback Procedure

1. Halt all incoming traffic to Stage20 candidate
2. Restore Stage18 frozen baseline snapshot
3. Verify regression pass rate >= 98% after rollback
4. Quarantine all memory writes made by Stage20 candidate
5. Record rollback reason and time in audit log
6. Notify on-call with rollback summary

## Recovery Checklist

- [ ] Rollback snapshot verified
- [ ] Regression tests pass
- [ ] Memory contamination resolved
- [ ] Root cause analysis started
- [ ] Fix package re-evaluated before next deployment