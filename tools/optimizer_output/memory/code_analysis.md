# Code Analysis: Fast-WAM (RobotWin Optimization)

## Baseline: ROBOTWIN_OVERALL_SUCCESS = 0.9172 (clean_mean_success_rate)

## Evaluation Parameters (from sim_robotwin.yaml + train.yaml)
- action_horizon = 32
- replan_steps = 24 (execute 24 of 32 predicted steps before replanning)
- num_inference_steps = 10 (action denoising steps)
- sigma_shift = null → uses default action_infer_shift=5.0
- seed = 42 (fixed, deterministic)
- rand_device = cpu

## Weakness Analysis
Precision tasks with lowest success rates (clean):
- open_microwave: 0.49 (hardest)
- hanging_mug: 0.59
- turn_switch: 0.64
- place_can_basket: 0.70

These fail due to: imprecise contact, compounding errors in 24-step open-loop execution

## Flow Matching Scheduler (sigma_shift analysis)
With shift=5.0 and 10 steps:
- Last step: t=357→0 (35.7% of total trajectory in one jump)
- Last 2 steps: account for 55.6% of total
- Highly back-loaded schedule

With shift=3.0 and 10 steps:
- Last step: ~18-25% of total
- More balanced integration

More balanced step distribution → more precise final action values → better performance for precision tasks.

## Key File: configs/sim_robotwin.yaml (line 26: sigma_shift: null → can be changed to 3.0)
