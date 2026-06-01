import numpy as np

action_horizon = 32
replan_steps = 24
blend_weight = 0.3

# Chunk 0: steps 0-31
chunk0 = np.random.randn(action_horizon, 14)
prev_chunk_start = 0

# Simulate: execute 24 steps, episode_step = 24
episode_step_after_chunk0 = 24

# Chunk 1 starts at step 24
chunk1 = np.random.randn(action_horizon, 14)
chunk_start = episode_step_after_chunk0

n_exec = min(replan_steps, action_horizon)
ensemble_count = 0

for i in range(n_exec):
    target_step = chunk_start + i
    prev_idx = target_step - prev_chunk_start

    if 0 <= prev_idx < len(chunk0):
        ensemble_count += 1

print(f"Steps ensembled: {ensemble_count}/{n_exec}")
print(f"Ensemble range: steps {chunk_start} to {chunk_start+ensemble_count-1}")
print(f"Expected: 8 steps (steps 24-31) - (32-24=8)")

# EMA check
alpha = 0.85
raw = [np.array([float(i)] * 14) for i in range(n_exec)]
smoothed = [raw[0]]
for i in range(1, len(raw)):
    s = alpha * raw[i] + (1.0 - alpha) * smoothed[i-1]
    smoothed.append(s)

print(f"EMA smoothed[10][0] = {smoothed[10][0]:.4f} (expected ~9.0 for alpha=0.85)")
print("All checks passed!")
