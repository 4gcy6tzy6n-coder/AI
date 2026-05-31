# Stage30 Operator Runbook

Stage30 preserves Stage27 safe defaults. Read-only inspection is allowed; production writes require governed promotion gates and rollback readiness.

## Required Checks

- Baseline chain complete: `True`
- Rollback chain passed: `True`
- Production shadow passed: `True`
- Critical drift count: `0`
- Memory contamination: `0`

## Blocked Operations

- Uncontrolled online learning
- Direct mutation of frozen baselines
- Production writes outside governed promotion gates
- Rollback bypass