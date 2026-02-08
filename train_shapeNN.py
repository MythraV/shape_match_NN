import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import csv
import time
import datetime
import subprocess

from shape_dataset import ShapeMatchingDatasetSimple, ShapeMatchingDatasetPrecomputed
from network import SiameseUNet
from generate_polygon_dataset import generate_heatmap_target

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # Increased due to AMP
LEARNING_RATE = 2e-4
EPOCHS = 50
IMG_SIZE = 128
DATASET_SIZE = 100000
LOSS_SCALE = 100.0

def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for template, query, targets_vector in loader:
            template = template.to(device)
            query = query.to(device)
            targets_vector = targets_vector.to(device)
            
            # AMP not strictly needed for eval but good for consistency
            with autocast(enabled=True):
                raw_heatmaps = generate_heatmap_target(
                    batch_size=template.size(0),
                    img_size=IMG_SIZE,
                    targets=targets_vector[:, 0:2], 
                    device=device,
                    sigma=2.0
                )
                match_flag = targets_vector[:, 2].view(-1, 1, 1, 1)
                final_targets = raw_heatmaps * match_flag
                
                pred_logits = model(template, query)
                pred_heatmap = torch.sigmoid(pred_logits)
                
                diff = (pred_heatmap - final_targets) ** 2
                weights = 1.0 + (50.0 * final_targets)
                loss = (diff * weights).mean() * LOSS_SCALE
                
            total_loss += loss.item()
            steps += 1
            if steps >= 100: break
    return total_loss / steps

def train_unet():
    print(f"--- Starting Optimized Training on {DEVICE} ---")
    
    # Optimizations
    torch.backends.cudnn.benchmark = True
    
    # 1. Dataset Preparation
    train_pt = "train_data.pt"
    if not os.path.exists(train_pt):
        print(f"Pre-computed dataset {train_pt} not found. Generating...")
        cmd = ["python", "generate_static_dataset.py", "--size", str(DATASET_SIZE), "--out", train_pt]
        subprocess.check_call(cmd)
    
    # Load entire dataset into RAM (CPU)
    # Pinned memory helps transfer speed
    train_dataset = ShapeMatchingDatasetPrecomputed(train_pt)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Generate Validation set separately
    val_pt = "val_data.pt"
    if not os.path.exists(val_pt):
        print("Generating validation set...")
        cmd = ["python", "generate_static_dataset.py", "--size", "5000", "--out", val_pt]
        subprocess.check_call(cmd)
        
    val_dataset = ShapeMatchingDatasetPrecomputed(val_pt)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = SiameseUNet(n_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler() # For AMP

    log_file = "training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "epoch", "train_loss", "val_loss", "lr"])

    best_loss = float('inf')
    patience = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Tqdm helps monitor speed per batch in real-time if running interactively,
        # but for logs we use print.
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
                    sigma=2.0
                )
                match_flag = targets_vector[:, 2].view(-1, 1, 1, 1)
                final_targets = raw_heatmaps * match_flag

                pred_logits = model(template, query)
                pred_heatmap = torch.sigmoid(pred_logits)

                diff = (pred_heatmap - final_targets) ** 2
                weights = 1.0 + (50.0 * final_targets)
                loss = (diff * weights).mean() * LOSS_SCALE

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            if batch_idx % 200 == 0 and batch_idx > 0:
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[{ts}] E[{epoch+1}] Step[{batch_idx}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Eval
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_loss)
        
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lr = optimizer.param_groups[0]['lr']
        print(f"[{ts}] Ep {epoch+1} ({epoch_time:.1f}s) | Train: {epoch_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([ts, epoch+1, epoch_loss, val_loss, lr])

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), "siamese_unet.pth")
            print(f"Saved Best Model (Val: {best_loss:.4f})")
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping.")
                break

if __name__ == "__main__":
    train_unet()
