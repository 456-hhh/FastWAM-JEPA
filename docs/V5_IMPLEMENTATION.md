# FastWAM-JEPA-IDM V5

## Architecture

V5 replaces the Wan VAE and VideoDiT with a frozen V-JEPA2 ViT-G encoder and a
30-layer JEPA Visual DiT. The original release ActionDiT remains the action
expert. Visual and action experts interact through the existing Mixture of
Transformers (MoT) attention path. JEPA tokens are never appended to text
context, and V5 has no future predictor, JEPA adapter, KV teacher, or KV
distillation loss.

Production Visual DiT defaults are 30 layers, hidden size 768, FFN size 3072,
24 heads, head dimension 128, V-JEPA dimension 1408, and text/proprio context
dimension 4096. Its Q/K/V attention width is 3072, matching release ActionDiT.
The expensive `4096->3072` context K/V projections are shared at the Visual DiT
level, while every block has its own visual Q/O, self-attention, FFN,
modulation, norms, and cross-attention call. This keeps the exact cross-attention
dimensions without duplicating context K/V tensors 30 times in parameters or
checkpoints.
With the production dimensions this implementation has exactly 598,926,208
Visual DiT parameters (about 0.60B).

## Temporal Contract

The decision anchor is `t`. The dataset keeps `num_frames=33` and
`action_video_freq_ratio=4`, so V5 consumes video indices `[0,1,2,3,4]`, which
represent raw observations `[t,t+4,t+8,t+12,t+16]`. It predicts the 16-action
chunk `[a(t),...,a(t+15)]`; rollout executes four actions and replans. Sixteen
actions cover the four-action visual stride through `t+16` without retaining
the unused second half of the original 32-action target.

The current clip is the algorithmic repeat `[o(t),o(t)]`. The future clip is a
single four-frame clip `[o(t+4),o(t+8),o(t+12),o(t+16)]`; it is not split into
two clips because one V-JEPA forward must preserve the encoder's two temporal
tubelet groups.

## Cameras And Token Order

Dataset video must be exactly `[B,3,T,224,448]`. It is split into 224x224
agentview and wrist images before each camera is independently resized to
256x256 by the shared frozen V-JEPA wrapper.

Official V-JEPA2 `PatchEmbed3D` applies `Conv3d` and then
`flatten(2).transpose(1,2)` to `[B,D,T,H,W]`. Therefore the token order is
temporal-major with H/W row-major inside each tubelet group. V5 strictly
reshapes current 256-token outputs as `[1,16,16,1408]` and future 512-token
outputs as `[2,16,16,1408]`. Spatial-only adaptive pooling maps 16x16 to 6x6.
The V5 sequence order is temporal, camera (`agentview`, then `wrist`), then 6x6
row-major: 72 tokens each for `z0`, `z1`, and `z2`, or 216 total.

Learned camera, temporal-group, and 2D spatial embeddings are combined with
3D RoPE over `(time,y,x)`. Cameras share RoPE coordinates and are distinguished
by camera embeddings.

## Visual Causality

Visual attention follows a block-causal mask: `z0` reads only `z0`, `z1` reads
`z0/z1`, and `z2` reads `z0/z1/z2`. Both cameras and all 6x6 positions within a
temporal group communicate fully. In joint MoT attention, visual queries cannot
read action keys. Action queries read all 216 visual keys and all 16 action
keys.

## Context

Cached text is `[B,L,4096]` with its real boolean validity mask. Padding is
zeroed, but its mask is never silently converted to all-true. Current normalized
proprioception is projected by the release `Linear(8,4096)` and appended as one
valid context token. Both Visual DiT and ActionDiT cross-attend this same base
context.

## Stage 1

Frozen V-JEPA encodes clean `z0` and ground-truth `z1/z2`. Gaussian flow noise
is added only to the 144 future tokens. Token-wise timestep modulation uses zero
for `z0` and the sampled visual timestep for future tokens. Visual DiT predicts
the future flow target and MSE is computed only on future tokens. ActionDiT is
not instantiated on the GPU in Stage1.

## Stage 2

Visual loss is unchanged. The IDM branch independently chooses clean future
conditioning for 50% of samples and flow-noised ground-truth future conditioning
for 50%. `z0` remains clean. Release ActionDiT and proprio are frozen, but the
ActionDiT forward is not wrapped in `no_grad`: action loss backpropagates through
visual K/V into Visual DiT. `L_total = L_visual + L_action`.

## Stage 3

Stage3 strictly initializes from Stage2. Visual DiT, all 30 ActionDiT layers and
head, and proprio projection are trainable; V-JEPA remains frozen. The two
training branches and unit loss weights remain unchanged. Separate default
learning rates are `2e-5`, `1e-6`, and `2e-6` for visual, action, and proprio.

## Inference

Inference accepts only current agentview/wrist RGB, text/context mask, and
current proprio. First, Visual DiT denoises 144 Gaussian future tokens while the
external `z0` tensor remains bitwise unchanged. Next, the predicted
`[z0,z1,z2]` sequence is run once through `MoT.prefill_video_cache` with all
visual timesteps zero. Finally, ActionDiT denoises 16 Gaussian action tokens
through `MoT.forward_action_with_video_cache`. There is no ground-truth future,
current-only mode, or alternate future-source API.

## Fail-Fast And Checkpoints

V5 enables strict dataset exceptions and strict V-JEPA/release/module loading.
All temporal, camera, token, action, and context shapes are asserted. Nonfinite
predictions, losses, and gradients terminate the run. CUDA is mandatory for
production training/evaluation; tiny mode is only enabled by explicit
`--tiny`.

Checkpoints save each model parameter once: Stage1 stores Visual DiT and
proprio; Stage2/3 also store ActionDiT. Optimizer, LR scheduler, global step,
dataloader epoch/progress, and per-rank Python/NumPy/Torch/CUDA RNG states are
stored for exact resume. Metadata records the resolved CLI, git commit, temporal
and camera contracts, parameter counts, and SHA256 values for V-JEPA, release
FastWAM, dataset stats, and the parent stage checkpoint. A mismatch is fatal.

## Server Commands

Stage1:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train_fastwam_jepa_idm_v5_stage1_visual.py \
  --libero-data-root "$LIBERO_ROOT" --dataset-stats-path "$STATS" \
  --release-checkpoint "$RELEASE" --vjepa-repo "$VJEPA_REPO" \
  --vjepa-checkpoint "$VJEPA_CKPT" --batch-size 4 --steps 5000 \
  --output-dir runs/v5_stage1_visual
```

Stage2:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train_fastwam_jepa_idm_v5_stage2_interface.py \
  --libero-data-root "$LIBERO_ROOT" --dataset-stats-path "$STATS" \
  --release-checkpoint "$RELEASE" --vjepa-repo "$VJEPA_REPO" \
  --vjepa-checkpoint "$VJEPA_CKPT" --stage1-checkpoint "$STAGE1" \
  --batch-size 4 --steps 1000 --output-dir runs/v5_stage2_interface
```

Stage3:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train_fastwam_jepa_idm_v5_stage3_joint.py \
  --libero-data-root "$LIBERO_ROOT" --dataset-stats-path "$STATS" \
  --release-checkpoint "$RELEASE" --vjepa-repo "$VJEPA_REPO" \
  --vjepa-checkpoint "$VJEPA_CKPT" --stage2-checkpoint "$STAGE2" \
  --batch-size 4 --steps 5000 --output-dir runs/v5_stage3_joint
```

Rollout:

```bash
python tools/evaluate_fastwam_jepa_idm_v5_libero_rollout.py \
  --checkpoint "$STAGE3" --release-checkpoint "$RELEASE" \
  --dataset-stats-path "$STATS" --libero-data-root "$LIBERO_ROOT" \
  --vjepa-repo "$VJEPA_REPO" --vjepa-checkpoint "$VJEPA_CKPT" \
  --libero-suite libero_spatial --task-id 0 --num-episodes 5 \
  --output-json runs/v5_stage3_joint/libero_spatial_task0.json
```

Sanity:

```bash
python tools/sanity_fastwam_jepa_idm_v5.py --tiny
python tools/sanity_fastwam_jepa_idm_v5_real.py \
  --checkpoint "$STAGE2" --release-checkpoint "$RELEASE" \
  --dataset-stats-path "$STATS" --libero-data-root "$LIBERO_ROOT" \
  --vjepa-repo "$VJEPA_REPO" --vjepa-checkpoint "$VJEPA_CKPT"
```
