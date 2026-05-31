# Stage27 Operator Runbook

All workflows default to read-only or dry-run behavior. Apply-mode operations are blocked in Stage27.

## Inspect Baseline Chain

- Command: `stage27 inspect --read-only`
- Mode: `read_only`
- Guarded: `False`
- Blocked: `False`
- Expected pass: `True`

## Validate Audit Cockpit

- Command: `stage27 validate --audit-cockpit`
- Mode: `read_only`
- Guarded: `False`
- Blocked: `False`
- Expected pass: `True`

## Promote Candidate Guarded

- Command: `stage27 promote --candidate safe --dry-run`
- Mode: `dry_run`
- Guarded: `True`
- Blocked: `False`
- Expected pass: `True`

## Rollback Candidate

- Command: `stage27 rollback --candidate unsafe --dry-run`
- Mode: `dry_run`
- Guarded: `True`
- Blocked: `False`
- Expected pass: `True`

## Block Unsafe Operation

- Command: `stage27 promote --candidate unsafe --apply`
- Mode: `apply`
- Guarded: `False`
- Blocked: `True`
- Expected pass: `True`

## Unsafe Operation Blocks

- `unsafe_apply_promotion` targeting `promotion`: blocked=True
- `unsafe_mutate_frozen_baseline` targeting `frozen_baseline`: blocked=True
- `unsafe_online_learning` targeting `online_learning`: blocked=True
- `unsafe_skip_rollback_gate` targeting `rollback_gate`: blocked=True