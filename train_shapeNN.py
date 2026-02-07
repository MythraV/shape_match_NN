import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
import time

from shape_dataset import ShapeMatchingDatasetSimple
from network import SiameseCorrelationUNet
from generate_polygon_dataset import generate_heatmap_target

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16          
LEARNING_RATE = 2e-4
EPOCHS = 50
IMG_SIZE = 128
DATASET_LEN = 30000
LOSS_SCALE = 100.0

def seed_worker_train(worker_id):
    # Training: Time-based randomness for infinite variety
    seed = (int(time.time() * 1000) + worker_id) % 2**32
    np.random.seed(seed)

def seed_worker_fixed(worker_id):
    # Validation/Test: Fixed deterministic seed
    # We add worker_id to ensure workers don't generate identical batches
    seed = (42 + worker_id) % 2**32
    np.random.seed(seed)

def seed_worker_test(worker_id):
    # Test: Different fixed deterministic seed
    seed = (12345 + worker_id) % 2**32
    np.random.seed(seed)

def evaluate(model, loader, device, description="Eval"):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for template, query, targets_vector in loader:
            template = template.to(device)
            query = query.to(device)
            targets_vector = targets_vector.to(device)
            
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
            
            # Limit evaluation to 100 batches to save time if dataset is huge
            if steps >= 100:
                break
                
    return total_loss / steps

def train_unet():
    print(f"--- Starting Discriminative Training on {DEVICE} ---")
    
    # 1. Training Loader (Infinite/Random)
    train_dataset = ShapeMatchingDatasetSimple(image_size=IMG_SIZE, length=DATASET_LEN)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        worker_init_fn=seed_worker_train
    )

    # 2. Validation Loader (Fixed Seed 42) - "Unseen 1"
    val_dataset = ShapeMatchingDatasetSimple(image_size=IMG_SIZE, length=2000)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=seed_worker_fixed
    )

    # 3. Test Loader (Fixed Seed 12345) - "Unseen 2"
    # To verify if Val set is weird
    test_dataset = ShapeMatchingDatasetSimple(image_size=IMG_SIZE, length=2000)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker_test
    )

    model = SiameseCorrelationUNet(n_channels=1).to(DEVICE)
    if os.path.exists("siamese_unet.pth"):
        print("Loading existing checkpoint...")
        try:
            model.load_state_dict(torch.load("siamese_unet.pth", map_location=DEVICE))
        except:
            print("Checkpoint incompatible, starting fresh.")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    log_file = "training_log.csv"
    log_exists = os.path.exists(log_file)
    csv_file = open(log_file, "a", newline="")
    log_writer = csv.writer(csv_file)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "train_eval_loss", "val_loss", "test_loss", "lr"])

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (template, query, targets_vector) in enumerate(train_loader):
            template = template.to(DEVICE)
            query = query.to(DEVICE)
            targets_vector = targets_vector.to(DEVICE)

            raw_heatmaps = generate_heatmap_target(
                batch_size=template.size(0),
                img_size=IMG_SIZE,
                targets=targets_vector[:, 0:2], 
                device=DEVICE,
                sigma=2.0 
            )
            
            match_flag = targets_vector[:, 2].view(-1, 1, 1, 1)
            final_targets = raw_heatmaps * match_flag

            optimizer.zero_grad()
            pred_logits = model(template, query)
            pred_heatmap = torch.sigmoid(pred_logits)

            diff = (pred_heatmap - final_targets) ** 2
            weights = 1.0 + (50.0 * final_targets)
            loss = (diff * weights).mean() * LOSS_SCALE

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"E[{epoch+1}] Step[{batch_idx}] | Loss: {loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader)
        
        # --- Evaluation Phase ---
        print(f"Evaluating Epoch {epoch+1}...")
        
        # 1. Train Eval (Seen/Dynamic Distribution)
        # We re-use train_loader but in eval mode (no grad). 
        # Since it's random, this tests generalization to the "Training Distribution".
        train_eval_loss = evaluate(model, train_loader, DEVICE, "Train Eval")
        
        # 2. Validation (Fixed Seed 42)
        val_loss = evaluate(model, val_loader, DEVICE, "Validation")
        
        # 3. Test (Fixed Seed 12345)
        test_loss = evaluate(model, test_loader, DEVICE, "Test")
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        print(f"==> Ep {epoch+1} | Train: {epoch_loss:.4f} | TrEval: {train_eval_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f} | LR: {current_lr:.2e}")
        
        log_writer.writerow([epoch + 1, epoch_loss, train_eval_loss, val_loss, test_loss, current_lr])
        csv_file.flush()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "siamese_unet.pth")
            print(f"Saved Best Model (Val: {best_loss:.4f})")

    csv_file.close()
    print(f"--- Training Complete ---")

if __name__ == "__main__":
    train_unet()