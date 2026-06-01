# Code Analysis: Fast-WAM RoboTwin Optimization

## Pipeline Summary

Fast-WAM uses Mixture-of-Transformers (MoT) coupling video expert (Wan2.2-TI2V-5B 30-layer) + action expert (ActionDiT 30-layer, 1024 hidden). At inference:
1. Stitch 3 camera views into 384x320 composite image
2. Encode image via VAE → latent tokens
3. Encode text prompt via T5 → context tokens
4. Append proprioception (joint state) to context
5. Prefill video KV cache from first frame (no future imagination!)
6. Run flow-matching denoiser for `num_inference_steps=10` steps (action-only)
7. Output `action_horizon=32` actions
8. Execute `replan_steps=24` actions before next inference call

## Key Parameters (from previous best run 20260402_034912)

| Parameter | Value | Source |
|-----------|-------|--------|
| num_inference_steps | 10 | sim_robotwin.yaml:eval_num_inference_steps |
| replan_steps | 24 | sim_robotwin.yaml:EVALUATION.replan_steps |
| action_horizon | 32 (num_frames-1) | derived |
| action_infer_shift | 5.0 | fastwam.yaml |
| video_infer_shift | 5.0 | fastwam.yaml |

## Weak Tasks (primary optimization targets)

| Task | Clean | Random | Avg |
|------|-------|--------|-----|
| open_microwave | 0.49 | 0.35 | 0.42 |
| hanging_mug | 0.59 | 0.71 | 0.65 |
| turn_switch | 0.64 | 0.65 | 0.645 |
| place_can_basket | 0.70 | 0.61 | 0.655 |
| move_stapler_pad | 0.81 | 0.67 | 0.74 |
| pick_diverse_bottles | 0.82 | 0.87 | 0.845 |
| move_can_pot | 0.88 | 0.93 | 0.905 |
| place_object_basket | 0.82 | 0.87 | 0.845 |

## IDEA: Reduce replan_steps (receding-horizon control)

**Hypothesis**: Changing `replan_steps` from 24 to a smaller value (e.g., 8-16) means:
- After computing 32-action chunk, only execute N before replanning
- More frequent observations → less action drift
- Better error correction for precision tasks

**Expected impact**: Improves precision tasks like `open_microwave`, `hanging_mug`, `turn_switch`

**How to change**: Modify `configs/sim_robotwin.yaml`: `replan_steps: 24` → `replan_steps: 8` (or 12)

**Risk**: Slower total evaluation time (more inference calls per episode)
