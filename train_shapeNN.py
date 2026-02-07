import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv

from shape_dataset import ShapeMatchingDatasetSimple
from network import SiameseCorrelationUNet
from generate_polygon_dataset import generate_heatmap_target

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16          
LEARNING_RATE = 2e-4
EPOCHS = 50
IMG_SIZE = 128
DATASET_LEN = 30000
LOSS_SCALE = 100.0 # Scaling loss to be in a more standard range

def train_unet():
    print(f"--- Starting Discriminative Training on {DEVICE} ---")

    train_dataset = ShapeMatchingDatasetSimple(image_size=IMG_SIZE, length=DATASET_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = ShapeMatchingDatasetSimple(image_size=IMG_SIZE, length=2000)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SiameseCorrelationUNet(n_channels=1).to(DEVICE)
    if os.path.exists("siamese_unet.pth"):
        print("Loading existing checkpoint...")
        try:
            model.load_state_dict(torch.load("siamese_unet.pth", map_location=DEVICE))
        except:
            print("Checkpoint incompatible, starting fresh.")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Setup CSV Logging
    log_file = "training_log.csv"
    log_exists = os.path.exists(log_file)
    csv_file = open(log_file, "a", newline="")
    log_writer = csv.writer(csv_file)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (template, query, targets_vector) in enumerate(train_loader):
            template = template.to(DEVICE)
            query = query.to(DEVICE)
            targets_vector = targets_vector.to(DEVICE)

            # 1. Generate Raw Heatmaps
            raw_heatmaps = generate_heatmap_target(
                batch_size=template.size(0),
                img_size=IMG_SIZE,
                targets=targets_vector[:, 0:2], 
                device=DEVICE,
                sigma=2.0 
            )
            
            # 2. MASK using the "Match Flag"
            match_flag = targets_vector[:, 2].view(-1, 1, 1, 1)
            final_targets = raw_heatmaps * match_flag

            optimizer.zero_grad()
            pred_logits = model(template, query)
            pred_heatmap = torch.sigmoid(pred_logits)

            # 3. Weighted Loss + Scaling
            diff = (pred_heatmap - final_targets) ** 2
            weights = 1.0 + (50.0 * final_targets)
            loss = (diff * weights).mean() * LOSS_SCALE

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"E[{epoch+1}] Step[{batch_idx}] | Loss: {loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for template, query, targets_vector in val_loader:
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
                
                pred_logits = model(template, query)
                pred_heatmap = torch.sigmoid(pred_logits)
                
                diff = (pred_heatmap - final_targets) ** 2
                weights = 1.0 + (50.0 * final_targets)
                v_loss = (diff * weights).mean() * LOSS_SCALE
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        print(f"==> End Epoch {epoch+1} | Train: {epoch_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        # Log to CSV
        log_writer.writerow([epoch + 1, epoch_loss, avg_val_loss, current_lr])
        csv_file.flush()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "siamese_unet.pth")
            print(f"Saved Best Model")

    csv_file.close()
    print(f"--- Training Complete ---")

if __name__ == "__main__":
    train_unet()
