# ShapeNN — Claude Code Project Memory

## Goal
Build an Industrial-Grade 2D Shape Matching Network (Translation x, y + Rotation theta).
- **Target:** Sub-pixel translation accuracy (< 0.5 px), robust rotation estimation for random 3-10 sided polygons.
- **Success criteria:** >95% validation accuracy, avg translation error < 0.5 px.

## Environment
- **Python env:** `~/Mythra/envs/dnn_env` (NOT nn_env — GEMINI.md is outdated on this)
  - Activate: `source ~/Mythra/envs/dnn_env/bin/activate`
- **Hardware:** RTX 4090, 24 GB VRAM (NOT RTX 2060 — GEMINI.md is outdated)
- **GPU safe batch size:** 128 with AMP enabled (batch=256 OOMs, batch=192 is marginal)

## Current Architecture — SiameseUNet (network.py)
Two independent encoders (query + template), then:
1. **Correlation bottleneck:** `corr = normalize(q5) * normalize(t5)`, `diff = abs(q5-t5)`, concat -> DoubleConv(2048->1024)
2. **Dual skip connections:** decoder receives `cat([q_i, t_i])` at every scale (not just query)
3. **Output:** 128x128 heatmap logit -> sigmoid -> Gaussian peak at match location, zero for negatives

Architecture channels: 64->128->256->512->1024 (81.7M parameters)
```
Up layers: up1=Up(2048,512), up2=Up(1024,256), up3=Up(512,128), up4=Up(256,64)
```

## Current Training Setup (train_shapeNN.py)
- **Loss:** Weighted MSE + 0.5 * BCE classification + 2.0 * DSNT coordinate loss (spatial softargmax)
  - Weighted MSE: `weight = 1 + 50*target`
  - BCE MUST use `binary_cross_entropy_with_logits` (not `binary_cross_entropy`) — autocast-safe
  - **DSNT coord loss:** `spatial_softargmax(logits) -> expected (x,y) -> smooth_L1_loss(pred, gt)` — directly optimizes coordinate accuracy for positive samples
- **Sigma annealing:** linearly from 2.0 -> 1.0 over EPOCHS (sharper targets as training progresses)
- **LR:** 1e-4
- **Early stopping:** val_loss-based, patience=8 (NOT accuracy — accuracy oscillates due to hard threshold)
- **Hard negatives:** 50% of pairs use 180 deg flips to break symmetry ambiguity
- **Dataset:** 100K precomputed samples in train_data.pt (targets_vector with actual rendered centroid, NOT intended continuous position)
- **generate_heatmap_target:** Vectorized with broadcasting (no Python for-loop)

## Current State (as of 2026-02-21) — TARGETS MET

### Current checkpoint: `siamese_unet.pth`
- Trained with DSNT coord loss + sigma annealing (2.0->1.0) + centroid-corrected GT
- Best at epoch 15: val_loss=0.1193, accuracy=99.8%, CoordErr=0.073px
- Early stopped at epoch 23

### Evaluation (Dense=True, 200 samples)
```
Avg Rotation Error:    2.54 deg
Avg Translation Error: 0.07 px   ✓ (was 1.53 px)
Rotation Accuracy (<10 deg): 100.0%  ✓ TARGET MET
Rotation Accuracy (<2 deg):  40.5%   (was 63%, traded for translation precision)
Translation Accuracy (<1.0 px): 99.5%  ✓
Translation Accuracy (<0.5 px): 99.5%  ✓ TARGET MET (was 14%)
```

### Checkpoints available
- `siamese_unet.pth` — current best (DSNT-trained, targets met)
- `siamese_unet_sigma2_best.pth` — previous sigma=2 fine-tune (pre-DSNT)
- `siamese_unet_sigma4_best.pth` — original sigma=4 model

### Changes that achieved the targets (2026-02-21)
1. **DSNT spatial softargmax coordinate loss** (train_shapeNN.py)
   - `spatial_softargmax()` converts logits -> probability distribution -> expected (x,y) coordinates
   - `smooth_l1_loss` on predicted vs ground truth pixel coordinates for positive samples
   - Weight: COORD_LOSS_WEIGHT=2.0
   - This directly optimizes coordinate accuracy — heatmap MSE alone cannot do this
2. **Sigma annealing** (train_shapeNN.py)
   - Linearly decreases from SIGMA_START=2.0 to SIGMA_END=1.0 over EPOCHS
3. **Vectorized generate_heatmap_target** (generate_polygon_dataset.py)
   - Replaced Python for-loop with broadcasting
4. **Fixed ground truth labels** (generate_static_dataset.py)
   - Actual rendered polygon centroid via `cv2.moments()` instead of intended (tx, ty)
   - Eliminates ~0.3px systematic label noise from integer vertex quantization
5. **Spatial softargmax in evaluate_no_gui.py**
   - Replaced COM-based peak extraction; evaluation GT also uses rendered centroid

## What Was Tried and Why It Failed

### 1. Regression heads (sin/cos for rotation)
- Tried before this codebase. Converged to (0,0). Abandoned. DO NOT retry.

### 2. Simple bottleneck `cat([q5, t5])` (old network.py)
- Caused severe overfitting: train loss 0.04, val loss 1.2
- Fixed by correlation bottleneck (normalize+multiply, abs diff)

### 3. Query-only skip connections in decoder
- Caused translation to be completely broken: 26 px avg error
- Model defaulted to center because it had no spatial template reference
- Fixed by dual skips: `cat([q_i, t_i])` at every decoder level

### 4. Accuracy-based early stopping
- Accuracy oscillates wildly (e.g., ep9=85%, ep11=49%) due to hard threshold `max_val < 0.3`
- When model briefly outputs 0.31 for negatives instead of 0.29, ALL negatives fail -> 50% accuracy
- Fixed by switching to val_loss-based early stopping

### 5. `F.binary_cross_entropy` inside autocast
- Raises RuntimeError. ALWAYS use `F.binary_cross_entropy_with_logits` with raw logits.

### 6. BATCH_SIZE=256 or 192
- OOMs on this architecture (81.7M params). Max safe is 128 with AMP.

### 7. Hard negative ratio 30%
- Model learned rotation but couldn't resolve 180 deg ambiguity on symmetric shapes
- Fixed by increasing to 50% hard negatives (180 deg flips +/-20 deg)

### 8. argparse default not matching constant
- Had `BATCH_SIZE = 256` at top but `default=64` in argparse -> always ran at 64
- Keep both constants and argparse defaults in sync

### 9. Python stdout buffering with nohup redirect
- Without `-u` flag, no output appears for minutes
- Always run: `python -u train_shapeNN.py`

### 10. Heatmap MSE loss alone insufficient for sub-pixel accuracy
- Pure pixel-wise MSE optimizes "does my heatmap look like the target Gaussian"
- Does NOT directly optimize "is my peak coordinate correct"
- A Gaussian shifted by 0.5px overlaps heavily with the target -> low MSE but 0.5px error
- Fixed by adding DSNT spatial softargmax coordinate loss (smooth L1 on expected coordinates)

### 11. Integer vertex quantization in ground truth
- `apply_affine_transform` returns `int32` vertices -> rendered shape center != intended (tx, ty)
- Creates ~0.3px systematic label noise (irreducible error floor)
- Fixed by computing actual rendered centroid via `cv2.moments()` in generate_static_dataset.py

## Key Files
- `network.py` — SiameseUNet architecture
- `train_shapeNN.py` — Training loop with DSNT coord loss + sigma annealing
- `evaluate_no_gui.py` — Evaluation script (dense=True for accurate eval, softargmax extraction)
- `shape_dataset.py` — ShapeMatchingDatasetPrecomputed
- `generate_polygon_dataset.py` — generate_heatmap_target() (vectorized) + geometry helpers
- `generate_static_dataset.py` — Generates train_data.pt / val_data.pt (centroid-corrected GT)
- `training_log.csv` — Training history (now includes coord_err and sigma columns)
- `siamese_unet.pth` — Current best checkpoint
- `siamese_unet_sigma4_best.pth` — Backup of sigma=4 trained best model

## Inference (evaluate_no_gui.py)
- Dense search: rotates template 0-359 deg in 1 deg steps (chunk size 36), picks highest heatmap activation
- Sub-pixel: spatial softargmax on predicted heatmap (logits -> softmax -> weighted coordinate sum)
- Run: `python evaluate_no_gui.py --dense`
- OOM note: dense=False (72 angles in one batch) OOMs; dense=True (chunks of 36) is fine

## Git Remote
https://github.com/MythraV/shape_match_NN.git
Commit and push after significant milestones.
