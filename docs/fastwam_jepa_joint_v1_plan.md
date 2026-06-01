# FastWAM-JEPA-Joint v1 Plan

## 1. Goal

本实验目标是新增一个 **FastWAM-JEPA-Joint v1** 模型。

它不是普通的 V-JEPA policy，也不是把 V-JEPA visual tokens 简单塞进 `text/context` 里作为额外条件。

本实验必须明确基于 **FastWAM-Joint** 的思想：

> action tokens 和 future visual/state tokens 要在 joint transformer / MoT-style attention 中交互。

也就是说，目标是实现：

```text
FastWAM-Joint 的 V-JEPA feature-space 版本
```

而不是：

```text
ActionDiT + extra V-JEPA context tokens
```

---

## 2. Core Idea

原 FastWAM-Joint：

```text
RGB video
→ Wan VAE
→ video latent

noisy future video latent
+
noisy action
+
text/proprio context
→ Video-DiT + ActionDiT + MoT joint attention
→ video flow target
→ action flow target
```

FastWAM-JEPA-Joint v1：

```text
RGB current video
→ frozen V-JEPA2 encoder
→ current V-JEPA visual tokens

future video
→ frozen V-JEPA2 encoder
→ target future V-JEPA tokens

action
+ noise
+ timestep
→ noisy action
→ ActionDiT.pre_dit()
→ action tokens

main joint tokens:
[current V-JEPA visual tokens]
[future query/state tokens]
[action tokens]

condition context:
[text tokens + proprio token]

main joint tokens
+ condition context
→ VJepaACJointPredictor
→ updated action tokens
→ predicted future V-JEPA tokens

updated action tokens
→ ActionDiT.post_dit()
→ predicted action flow target
```

Training loss:

```text
L_total = L_action_flow + lambda_future * L_future_vjepa
```

其中：

```text
L_action_flow:
  保留 FastWAM 原 action diffusion / flow matching loss

L_future_vjepa:
  predicted future V-JEPA tokens
  对齐
  frozen V-JEPA2 encoder(future video)
```

---

## 3. Key Design Constraints

### 3.1 This must be a joint version

不接受只做：

```text
V-JEPA visual tokens
→ project to text_dim
→ concat 到 context
→ ActionDiT cross-attention
```

这种方案最多只能作为 baseline，不是本实验目标。

v1 目标必须是：

```text
action tokens
current V-JEPA visual tokens
future query/state tokens
```

进入同一个 joint transformer / MoT-style attention 结构中交互。

---

### 3.2 VJepaACJointPredictor must not process raw noisy_action

这是硬约束。

`VJepaACJointPredictor` 不应该直接接收：

```text
noisy_action: [B, T_a, A]
```

正确流程是：

```text
action
+ noise
+ timestep
→ noisy_action: [B, T_a, A]

noisy_action
+ timestep
+ context/context_mask
→ ActionDiT.pre_dit()
→ action_pre

action_pre["tokens"]
→ action_tokens: [B, T_a, D_h]

action_tokens
+ current V-JEPA visual tokens
+ future query/state tokens
→ VJepaACJointPredictor
→ updated_action_tokens

updated_action_tokens
+ action_pre
→ ActionDiT.post_dit()
→ pred_action_flow_target: [B, T_a, A]
```

原因：

- `noisy_action` 是机器人动作空间里的连续数值；
- `action_tokens` 才是 transformer hidden token；
- 原 FastWAM-Joint 也是先 `pre_dit()`，再进入 joint/MoT；
- 这样可以复用 ActionDiT 的 action encoder、time embedding、context projection 和 action head。

---

### 3.3 text/proprio remains condition context

`text/proprio` 不作为主 joint token 混入 visual/action/future token 序列。

v1 保持原 FastWAM 风格：

```text
text prompt
→ T5 encoder
→ text context tokens

proprio
→ proprio_encoder
→ proprio token

concat(text context tokens, proprio token)
→ condition context
```

更准确地说：

```text
condition context tokens: [B, L_c, D_t]
condition mask:           [B, L_c]
```

这里的 `text + proprio` 是 token 维度拼接，不是数值相加。

在 joint predictor 中：

```text
main tokens:
[current V-JEPA visual tokens]
[future query/state tokens]
[action tokens]

condition context:
[text tokens + proprio token]

joint transformer block:
main tokens 做 self-attention
main tokens 通过 cross-attention 使用 condition context
```

---

## 4. Role of V-JEPA2 and V-JEPA2-AC

### 4.1 V-JEPA2 encoder is the key component

本项目里最关键的是 **V-JEPA2 encoder**。

它用于替换原 FastWAM 里的 Wan VAE 视觉表征：

```text
RGB video
→ V-JEPA2 encoder
→ V-JEPA visual/state tokens
```

这样模型使用的是更紧凑、更语义化、更偏物理状态的视觉表示，而不是偏像素重建的 VAE latent。

第一版默认：

```text
V-JEPA2 encoder frozen
```

---

### 4.2 V-JEPA2-AC predictor is used as a design reference

V-JEPA2-AC 原版更接近：

```text
current V-JEPA state
+ candidate action sequence
+ robot state
→ action-conditioned world model
→ future V-JEPA states
```

它本身不生成 action，动作通常由外部 planner / MPC / CEM 提供。

本项目不是直接复刻 V-JEPA2-AC。  
本项目是：

```text
FastWAM action diffusion
+
V-JEPA2-AC-style future feature prediction
+
FastWAM-Joint-style joint attention
```

因此 v1 中的 predictor 应称为：

```text
V-JEPA2-AC-style joint predictor
```

而不是直接假设能无缝使用官方 V-JEPA2-AC predictor 权重。

---

### 4.3 Future query tokens are a v1 engineering simplification

官方 V-JEPA2-AC 更接近自回归 / block-causal future prediction：

```text
z_t + a_t → z_{t+1}
z_{t+1} + a_{t+1} → z_{t+2}
...
```

v1 为了先跑通 FastWAM-Joint + V-JEPA feature-space，可以采用并行预测方案：

```text
current visual tokens
+ action tokens
+ future query tokens
+ condition context
→ predicted future V-JEPA tokens
```

这里的 `future query tokens` 是 learnable output slots：

```text
future query tokens: [N_f, D_h]
expanded to:         [B, N_f, D_h]
```

它们不是 attention 里的 Q 矩阵本身，而是一组可学习的未来状态输出槽位。

训练完成后，future query tokens 作为模型参数固定；推理时复制到 batch 维度后参与预测。

注意：

```text
future query tokens 是 v1 的工程简化方案，
不是官方 V-JEPA2-AC 的核心机制。
```

后续版本可以改成更接近官方 V-JEPA2-AC 的 autoregressive/block-causal predictor。

---

## 5. Tensor Symbols

建议统一使用以下符号，避免混淆：

```text
B      batch size
T_c    current video frame count
T_f    future video frame count
H_img  image height
W_img  image width

N_v    current V-JEPA visual token count
N_f    future V-JEPA token count / future query token count

D_v    V-JEPA feature dimension
D_h    joint transformer hidden dimension
D_t    text/context token dimension
D_p    proprio dimension

T_a    action horizon
A      action dimension

L      text token count
L_c    condition context token count after appending proprio
```

Important:

```text
Use D_h for hidden dimension.
Do not use H for hidden dimension, because H may be confused with image height.
```

---

## 6. Main Data Flow

### 6.1 Current visual path

```text
current video: [B, 3, T_c, H_img, W_img]
→ frozen VJepaEncoderWrapper
→ current_visual_tokens: [B, N_v, D_v]
→ visual_adapter
→ current_visual_joint_tokens: [B, N_v, D_h]
```

---

### 6.2 Future target path, training only

```text
future video: [B, 3, T_f, H_img, W_img]
→ frozen VJepaEncoderWrapper
→ target_future_tokens: [B, N_f, D_v]
→ detach()
```

This branch exists only during training.

---

### 6.3 Action path

Training:

```text
ground_truth_action: [B, T_a, A]
+ noise
+ timestep
→ noisy_action: [B, T_a, A]
→ ActionDiT.pre_dit(noisy_action, timestep, context, context_mask)
→ action_pre
→ action_tokens = action_pre["tokens"]: [B, T_a, D_h]
```

Inference:

```text
initial action noise: [B, T_a, A]
→ ActionDiT.pre_dit(noisy_action, timestep, context, context_mask)
→ action_tokens: [B, T_a, D_h]
```

---

### 6.4 Condition context path

```text
text context: [B, L, D_t]

optional proprio: [B, T, D_p]
→ proprio_encoder
→ proprio token: [B, 1, D_t]

concat(text context, proprio token)
→ condition context: [B, L_c, D_t]
→ condition mask: [B, L_c]
```

This condition context is used by:

```text
ActionDiT.pre_dit()
VJepaACJointPredictor cross-attention
```

---

### 6.5 Future query path

```text
learnable_future_query_tokens: [N_f, D_h]
→ expand over batch
→ future_query_tokens: [B, N_f, D_h]
```

These tokens are the output slots for future V-JEPA prediction.

---

### 6.6 Joint predictor path

Main joint tokens:

```text
current_visual_joint_tokens: [B, N_v, D_h]
future_query_tokens:         [B, N_f, D_h]
action_tokens:               [B, T_a, D_h]
```

Concatenated main token sequence:

```text
main_tokens: [B, N_v + N_f + T_a, D_h]
```

Condition context:

```text
condition_context: [B, L_c, D_t]
condition_mask:    [B, L_c]
```

VJepaACJointPredictor outputs:

```text
updated_action_tokens: [B, T_a, D_h]
future_hidden_tokens:  [B, N_f, D_h]
```

Then:

```text
updated_action_tokens
→ ActionDiT.post_dit(updated_action_tokens, action_pre)
→ pred_action_flow: [B, T_a, A]

future_hidden_tokens
→ future_feature_projection
→ pred_future_tokens: [B, N_f, D_v]
```

---

## 7. Losses

### 7.1 Action loss

Use original FastWAM action diffusion / flow matching loss.

```text
pred_action_flow:   [B, T_a, A]
target_action_flow: [B, T_a, A]
```

Compute masked / weighted MSE as in original FastWAM.

---

### 7.2 Future V-JEPA loss

```text
pred_future_tokens:   [B, N_f, D_v]
target_future_tokens: [B, N_f, D_v]
```

Use:

```text
L1
or
SmoothL1
```

Target must be detached:

```python
target_future_tokens = target_future_tokens.detach()
```

---

### 7.3 Total loss

```python
loss_total = loss_action + lambda_future * loss_future_vjepa
```

Suggested start:

```text
lambda_future = 0.1 or 0.5
```

Do not let future loss dominate action loss at the beginning.

---

## 8. Training vs Inference

### 8.1 Training

Training has:

```text
ground-truth action
future video
target future V-JEPA tokens
loss_action
loss_future_vjepa
```

Training output:

```text
pred_action_flow
pred_future_tokens
loss_total
```

---

### 8.2 Inference

Inference does not have:

```text
ground-truth action
future video
target future V-JEPA tokens
loss
```

Inference starts from:

```text
current observation
text/task condition
current proprio
initial action noise
```

Inference loop:

```text
noisy_action
→ ActionDiT.pre_dit()
→ action_tokens
→ VJepaACJointPredictor
→ updated_action_tokens
→ ActionDiT.post_dit()
→ pred_action_flow
→ scheduler step
→ updated noisy_action
→ repeat
→ final action
```

During robot deployment:

```text
B is usually 1
```

During batched simulation evaluation:

```text
B can be larger than 1
```

---

## 9. New Modules

### 9.1 VJepaEncoderWrapper

Suggested file:

```text
src/fastwam/models/vjepa/vjepa_encoder_wrapper.py
```

Required v1 behavior:

```text
input:  video [B, 3, T, H_img, W_img]
output: tokens [B, N, D_v]
```

v1 must support dummy mode:

```text
dummy mode does not load real V-JEPA2
dummy mode does not import external/vjepa2
dummy mode does not require CUDA
dummy mode only checks interface and shape
```

Real V-JEPA2 loading is TODO for server-side integration.

---

### 9.2 VJepaACJointPredictor

Suggested file:

```text
src/fastwam/models/vjepa/vjepa_ac_joint_predictor.py
```

Inputs:

```text
current_visual_joint_tokens: [B, N_v, D_h]
future_query_tokens:         [B, N_f, D_h]
action_tokens:               [B, T_a, D_h]
condition_context:           [B, L_c, D_t]
condition_mask:              [B, L_c]
```

Outputs:

```text
updated_action_tokens: [B, T_a, D_h]
future_hidden_tokens:  [B, N_f, D_h]
pred_future_tokens:    [B, N_f, D_v]
```

Hard rule:

```text
VJepaACJointPredictor must not directly process raw noisy_action.
It receives action_tokens from ActionDiT.pre_dit().
```

---

### 9.3 FastWAMJEPAJoint

Suggested file:

```text
src/fastwam/models/wan22/fastwam_jepa_joint.py
```

Responsibilities:

```text
reuse ActionDiT.pre_dit()
reuse ActionDiT.post_dit()
reuse action scheduler
reuse text/proprio context logic
call VJepaEncoderWrapper
call VJepaACJointPredictor
compute loss_action
compute loss_future_vjepa
return loss_total and loss_dict
```

---

## 10. What to Reuse

Reuse:

```text
ActionDiT.pre_dit()
ActionDiT.post_dit()
FastWAM action diffusion / flow scheduler
FastWAM text/context processing
FastWAM proprio_encoder / _append_proprio_to_context()
FastWAM-Joint / MoT design idea
```

ActionDiT should not be trained from scratch if original FastWAM weights are available.

---

## 11. What Not to Modify

Do not modify default behavior of:

```text
src/fastwam/models/wan22/fastwam.py
src/fastwam/models/wan22/fastwam_joint.py
src/fastwam/models/wan22/action_dit.py
src/fastwam/models/wan22/mot.py
external/vjepa2/*
```

Only minimal import / registry changes are allowed later if needed.

---

## 12. Dummy Mode Policy

Dummy mode is not for performance.

Dummy mode is only for checking:

```text
import works
class construction works
forward path works
shape matches
loss returns scalar
no obvious interface error
```

Minimum dummy requirement:

```text
VJepaEncoderWrapper(dummy=True)
input video: [B, 3, T, H_img, W_img]
output tokens: [B, N, D_v]
```

Dummy output can be simple deterministic zeros or simple projection.  
It does not need to represent real V-JEPA semantics.

---

## 13. Codex Task Order

### Task 0: Read-only plan

Read this document and relevant code.  
Do not modify files.

Must confirm:

```text
this is FastWAM-Joint feature-space version
not context-only baseline
VJepaACJointPredictor receives action_tokens, not raw noisy_action
text/proprio remains condition context
```

---

### Task 1: Dummy VJepaEncoderWrapper

Create:

```text
src/fastwam/models/vjepa/__init__.py
src/fastwam/models/vjepa/vjepa_encoder_wrapper.py
```

Only dummy mode.

---

### Task 2: VJepaACJointPredictor skeleton

Create:

```text
src/fastwam/models/vjepa/vjepa_ac_joint_predictor.py
```

Implement shape-compatible joint predictor.

Must take action tokens, not raw noisy action.

---

### Task 3: FastWAMJEPAJoint skeleton

Create:

```text
src/fastwam/models/wan22/fastwam_jepa_joint.py
```

Connect:

```text
VJepaEncoderWrapper
VJepaACJointPredictor
ActionDiT.pre_dit()
ActionDiT.post_dit()
action scheduler
text/proprio context
```

---

### Task 4: Loss integration

Implement:

```text
loss_action
loss_future_vjepa
loss_total
loss_dict
```

---

### Task 5: Shape test

Create:

```text
tests/test_fastwam_jepa_joint_shapes.py
```

Test:

```text
dummy forward works
training_loss returns scalar loss_total
pred_action_flow shape is correct
pred_future_tokens shape is correct
original FastWAM files remain untouched
```

---

## 14. v1 Success Criteria

v1 is successful if:

```text
original FastWAM / FastWAMJoint behavior is unchanged
FastWAMJEPAJoint can be imported
dummy forward runs
dummy training_loss returns scalar loss
action branch shape matches original FastWAM
future token prediction shape matches target V-JEPA tokens
server can later replace dummy encoder with real frozen V-JEPA2 encoder
small server-side sanity training does not NaN
```

---

## 15. Out of Scope for v1

Do not implement in v1:

```text
real V-JEPA2 weight loading on local machine
official V-JEPA2-AC predictor weight loading
V-JEPA feature decoding to RGB video
future feature diffusion
full autoregressive/block-causal V-JEPA2-AC rollout
fine-tuning V-JEPA2 encoder
goal-image planning
MPC/CEM action optimization
full benchmark comparison
```