# Optimization Idea Library: Fast-WAM RobotWin

Last updated: 2026-04-23

## Baseline Reference
- clean_mean_success_rate: 0.9172 (from 20260402_034912, sigma_shift=null, replan_steps=24, num_inference_steps=10)
- Target: 0.9631 (delta: +0.0459)
- Weak tasks: open_microwave(0.49/0.35), hanging_mug(0.59/0.71), turn_switch(0.64/0.65), place_can_basket(0.70/0.61)

## Ideas

### IDEA-001 (SELECTED for iteration 1): Temporal Ensemble + sigma_shift=3.0
- **Type**: CODE + PARAM
- **Priority**: HIGH
- **Risk**: LOW (zero additional compute)
- **Description**:
  1. sigma_shift: null → 3.0 (already changed in working tree)
     - Lower shift = more uniform denoising schedule → more fine-grain steps
     - Better precision at end of denoising chain
  2. Temporal Ensemble in deploy_policy.py:
     - Store full action_chunk from previous inference call
     - Track episode step counter per episode
     - For ALL steps in execution window j (0-23), check if prev_chunk also covers that global step
     - With action_horizon=32 and replan_steps=24: prev_chunk covers 8 overlapping steps
     - Blend: action = 0.7*new_chunk[i] + 0.3*prev_chunk[idx] for overlapping steps
     - Reset prev_chunk on each episode reset
  3. In addition, add EMA smoothing within the action queue
     - action_queue[i] = alpha*raw_action[i] + (1-alpha)*action_queue[i-1], alpha=0.9
     - Reduces high-frequency noise between consecutive actions
- **Hypothesis**: sigma_shift=3.0 improves precision for all 50 tasks (+1-2%). Temporal ensemble smooths trajectory boundaries (+1-3% on weak tasks). EMA smoothing reduces jitter (+0.5-1%). Combined: +3-6% overall.
- **Status**: IN_PROGRESS

### IDEA-002: Reduce replan_steps
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: HIGH (eval timeout risk - 3x more inference calls)
- **Description**: replan_steps 24 → 8
- **Status**: PENDING (timing risk)

### IDEA-003 (original IDEA-003): Sigma Shift Tuning
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: sigma_shift: null → 3.0 (already part of IDEA-001 above)
- **Status**: MERGED INTO IDEA-001

### IDEA-004: Extended action_horizon for perfect ensemble coverage
- **Type**: CODE + PARAM
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Set action_horizon=48 at inference; with replan_steps=24, 100% of execution steps get ensemble coverage from 2 chunks
- **Status**: PENDING

### IDEA-005: Multi-inference ensemble
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: HIGH (2-3x timing cost)
- **Description**: Run infer_action N times and average actions
- **Status**: PENDING (timing risk)

## Red Line Audit

| Idea | R1 | R2 | R3 | R4 | R5 | R6 | Decision |
|------|----|----|----|----|----|----|---------|
| IDEA-001 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-002 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED (timing risk) |
| IDEA-004 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-005 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED (timing risk) |

## Iteration Log

| Iter | Idea | Type | Before | After | Delta | Status | Key Takeaway |
|------|------|------|--------|-------|-------|--------|--------------|
