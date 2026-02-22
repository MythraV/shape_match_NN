import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import csv
import time
import datetime
import subprocess
import argparse

from shape_dataset import ShapeMatchingDatasetSimple, ShapeMatchingDatasetPrecomputed
from network import SiameseUNet
from generate_polygon_dataset import generate_heatmap_target

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 1e-4  # Fine-tuning LR
EPOCHS = 30
IMG_SIZE = 128
DATASET_SIZE = 100000
LOSS_SCALE = 10.0
SIGMA_START = 2.0
SIGMA_END = 1.0
COORD_LOSS_WEIGHT = 2.0


def spatial_softargmax(logits, img_size):
    """Differentiable spatial softargmax (DSNT) for sub-pixel coordinate extraction.

    Args:
        logits: (B, 1, H, W) raw logits from the model
        img_size: spatial size (H=W)
    Returns:
        coords: (B, 2) predicted (x, y) in pixel space
    """
    B = logits.shape[0]
    logits_flat = logits.view(B, -1)  # (B, H*W)
    probs = F.softmax(logits_flat, dim=1)  # (B, H*W)

    x = torch.arange(0, img_size, device=logits.device).float()
    y = torch.arange(0, img_size, device=logits.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xx_flat = xx.reshape(-1)  # (H*W,)
    yy_flat = yy.reshape(-1)  # (H*W,)

    pred_x = (probs * xx_flat.unsqueeze(0)).sum(dim=1)  # (B,)
    pred_y = (probs * yy_flat.unsqueeze(0)).sum(dim=1)  # (B,)

    return torch.stack([pred_x, pred_y], dim=1)  # (B, 2)


def evaluate(model, loader, device, sigma, desc="Eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_coord_err = 0.0
    total_pos = 0
    steps = 0

    with torch.no_grad():
        for template, query, targets_vector in loader:
            template = template.to(device)
            query = query.to(device)
            targets_vector = targets_vector.to(device)

            with autocast(enabled=True):
                raw_heatmaps = generate_heatmap_target(
                    batch_size=template.size(0),
                    img_size=IMG_SIZE,
                    targets=targets_vector[:, 0:2],
                    device=device,
                    sigma=sigma
                )
                match_flag = targets_vector[:, 2].view(-1, 1, 1, 1)
                final_targets = raw_heatmaps * match_flag

                pred_logits = model(template, query)
                pred_heatmap = torch.sigmoid(pred_logits)

                diff = (pred_heatmap - final_targets) ** 2
                weights = 1.0 + (50.0 * final_targets)
                loc_loss = (diff * weights).mean()

                # Classification loss (autocast-safe): raw logits -> BCEWithLogitsLoss
                logits_flat = pred_logits.flatten(1)
                logits_global = logits_flat.topk(4, dim=1).values.mean(dim=1)
                match_flag_1d = targets_vector[:, 2]
                cls_loss = F.binary_cross_entropy_with_logits(logits_global, match_flag_1d)

                # Coordinate loss for positives via softargmax
                pos_mask = match_flag_1d > 0.5
                if pos_mask.any():
                    pred_coords = spatial_softargmax(pred_logits[pos_mask], IMG_SIZE)
                    center = IMG_SIZE / 2.0
                    gt_x = center + targets_vector[pos_mask, 0] * center
                    gt_y = center + targets_vector[pos_mask, 1] * center
                    gt_coords = torch.stack([gt_x, gt_y], dim=1)
                    coord_loss = F.smooth_l1_loss(pred_coords, gt_coords)
                    coord_err = torch.sqrt(((pred_coords - gt_coords)**2).sum(dim=1)).mean()
                    total_coord_err += coord_err.item() * pos_mask.sum().item()
                    total_pos += pos_mask.sum().item()
                else:
                    coord_loss = torch.tensor(0.0, device=device)

                loss = (loc_loss + 0.5 * cls_loss + COORD_LOSS_WEIGHT * coord_loss) * LOSS_SCALE

            total_loss += loss.item()
            steps += 1

            # Accuracy: for positives, peak within 5px; for negatives, max < 0.3
            B = pred_heatmap.shape[0]
            pred_flat_acc = pred_heatmap.view(B, -1)
            max_vals, max_idxs = pred_flat_acc.max(dim=1)
            peak_y = (max_idxs // IMG_SIZE).float()
            peak_x = (max_idxs % IMG_SIZE).float()

            center = IMG_SIZE / 2.0
            gt_x = center + targets_vector[:, 0] * center
            gt_y = center + targets_vector[:, 1] * center

            loc_err = torch.sqrt((peak_x - gt_x) ** 2 + (peak_y - gt_y) ** 2)
            match_flag_b = targets_vector[:, 2]
            pos_mask_acc = match_flag_b > 0.5

            pos_correct = pos_mask_acc & (loc_err < 5.0) & (max_vals > 0.1)
            neg_correct = (~pos_mask_acc) & (max_vals < 0.3)

            total_correct += (pos_correct.sum() + neg_correct.sum()).item()
            total_samples += B

            if steps >= 100:
                break

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_coord_err = total_coord_err / total_pos if total_pos > 0 else 0.0
    return total_loss / steps, accuracy, avg_coord_err


def train_unet(train_path, val_path, output_dir):
    print(f"--- Starting Training on {DEVICE} ---")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {output_dir}")

    torch.backends.cudnn.benchmark = True

    # 1. Dataset Preparation
    if not os.path.exists(train_path):
        print(f"Dataset {train_path} not found. Generating...")
        if train_path == "train_data.pt":
            cmd = ["python", "generate_static_dataset.py", "--size", str(DATASET_SIZE), "--out", train_path]
            subprocess.check_call(cmd)
        else:
            raise FileNotFoundError(f"Training data not found at {train_path}.")

    print(f"Loading Training Data from {train_path}...")
    train_dataset = ShapeMatchingDatasetPrecomputed(train_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    if not os.path.exists(val_path):
        if val_path == "val_data.pt":
            print("Generating validation set...")
            cmd = ["python", "generate_static_dataset.py", "--size", "5000", "--out", val_path]
            subprocess.check_call(cmd)
        else:
            raise FileNotFoundError(f"Validation data not found at {val_path}")

    print(f"Loading Validation Data from {val_path}...")
    val_dataset = ShapeMatchingDatasetPrecomputed(val_path)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = SiameseUNet(n_channels=1).to(DEVICE)
    if os.path.exists("siamese_unet.pth"):
        print("Loading existing checkpoint...")
        try:
            model.load_state_dict(torch.load("siamese_unet.pth", map_location=DEVICE))
            print("Checkpoint loaded.")
        except Exception as e:
            print(f"Checkpoint incompatible ({e}), starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler()

    log_file = "training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "epoch", "train_loss", "val_loss", "val_accuracy", "lr"])

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(EPOCHS):
        # Sigma annealing: linearly decrease from SIGMA_START to SIGMA_END
        sigma = SIGMA_START - (SIGMA_START - SIGMA_END) * (epoch / max(EPOCHS - 1, 1))

        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (template, query, targets_vector) in enumerate(train_loader):
            template = template.to(DEVICE, non_blocking=True)
            query = query.to(DEVICE, non_blocking=True)
            targets_vector = targets_vector.to(DEVICE, non_blocking=True)

            with autocast(enabled=True):
                raw_heatmaps = generate_heatmap_target(
                    batch_size=template.size(0),
                    img_size=IMG_SIZE,
                    targets=targets_vector[:, 0:2],
                    device=DEVICE,
                    sigma=sigma
                )
                match_flag = targets_vector[:, 2].view(-1, 1, 1, 1)
                final_targets = raw_heatmaps * match_flag

                pred_logits = model(template, query)
                pred_heatmap = torch.sigmoid(pred_logits)

                diff = (pred_heatmap - final_targets) ** 2
                weights = 1.0 + (50.0 * final_targets)
                loc_loss = (diff * weights).mean()

                # Classification loss (autocast-safe): raw logits -> BCEWithLogitsLoss
                logits_flat = pred_logits.flatten(1)
                logits_global = logits_flat.topk(4, dim=1).values.mean(dim=1)
                match_flag_1d = targets_vector[:, 2]
                cls_loss = F.binary_cross_entropy_with_logits(logits_global, match_flag_1d)

                # Coordinate regression loss for positive samples via DSNT softargmax
                pos_mask = match_flag_1d > 0.5
                if pos_mask.any():
                    pred_coords = spatial_softargmax(pred_logits[pos_mask], IMG_SIZE)
                    center = IMG_SIZE / 2.0
                    gt_x = center + targets_vector[pos_mask, 0] * center
                    gt_y = center + targets_vector[pos_mask, 1] * center
                    gt_coords = torch.stack([gt_x, gt_y], dim=1)
                    coord_loss = F.smooth_l1_loss(pred_coords, gt_coords)
                else:
                    coord_loss = torch.tensor(0.0, device=DEVICE)

                loss = (loc_loss + 0.5 * cls_loss + COORD_LOSS_WEIGHT * coord_loss) * LOSS_SCALE

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time

        val_loss, val_accuracy, val_coord_err = evaluate(model, val_loader, DEVICE, sigma)
        scheduler.step(val_loss)

        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lr = optimizer.param_groups[0]['lr']
        print(f"[{ts}] Ep {epoch+1} ({epoch_time:.1f}s) | Train: {epoch_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_accuracy*100:.1f}% | CoordErr: {val_coord_err:.3f}px | sigma: {sigma:.2f} | LR: {lr:.2e}")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([ts, epoch+1, epoch_loss, val_loss, f"{val_accuracy*100:.1f}", lr, f"{val_coord_err:.3f}", f"{sigma:.2f}"])

        # Use val_loss for model selection — more reliable than accuracy metric
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0

            torch.save(model.state_dict(), "siamese_unet.pth")

            if output_dir:
                drive_path = os.path.join(output_dir, "siamese_unet_best.pth")
                torch.save(model.state_dict(), drive_path)
                print(f"Saved Best Model to {drive_path} (Val: {best_val_loss:.4f}, Acc: {val_accuracy*100:.1f}%, CoordErr: {val_coord_err:.3f}px)")
            else:
                print(f"Saved Best Model (Val: {best_val_loss:.4f}, Acc: {val_accuracy*100:.1f}%, CoordErr: {val_coord_err:.3f}px)")
        else:
            patience += 1
            if patience >= 8:
                print("Early stopping.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='train_data.pt', help='Path to training .pt file')
    parser.add_argument('--val_data', type=str, default='val_data.pt', help='Path to validation .pt file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    train_unet(args.train_data, args.val_data, args.output_dir)
