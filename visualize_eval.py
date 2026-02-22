import torch
import numpy as np
import cv2
import os
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- LOCAL IMPORTS ---
from network import SiameseUNet
from generate_polygon_dataset import create_random_polygon, apply_affine_transform

# --- CONFIG ---
MODEL_PATH = "siamese_unet_best.pth" 
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_dist(img_uint8):
    """Matches the training preprocessing exactly"""
    edge_inv = 255 - img_uint8
    dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
    dist = np.exp(-0.05 * dist) 
    return torch.from_numpy(dist).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

def predict_and_profile(model, points_base, query_tensor):
    """
    Runs a dense search (0-360) and returns the full score profile + best heatmap.
    """
    # 1. Generate templates for every 5 degrees (Coarse-ish search for visualization)
    angles = np.arange(0, 360, 5) 
    center = np.array([IMG_SIZE//2, IMG_SIZE//2])
    batch_templates = []
    
    for angle in angles:
        # Regenerate template to avoid rotation artifacts
        points_temp = apply_affine_transform(points_base, float(angle), 1.0, (0, 0), center)
        img_temp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        if len(points_temp) > 0:
            cv2.polylines(img_temp, [points_temp], True, 255, 1)
        batch_templates.append(preprocess_dist(img_temp))

    # Stack into a batch [72, 1, 128, 128]
    t_batch = torch.cat(batch_templates, dim=0)
    q_batch = query_tensor.repeat(len(angles), 1, 1, 1)

    # 2. Batch Inference
    with torch.no_grad():
        logits = model(t_batch, q_batch)
        heatmaps = torch.sigmoid(logits)
    
    # 3. Extract Scores
    # Max value in heatmap = Confidence for that angle
    scores = heatmaps.view(len(angles), -1).max(dim=1).values.cpu().numpy()
    
    # 4. Find Best
    best_idx = np.argmax(scores)
    best_angle = angles[best_idx]
    best_heatmap = heatmaps[best_idx].squeeze().cpu().numpy()
    
    # Simple peak finding (Argmax)
    py, px = np.unravel_index(best_heatmap.argmax(), best_heatmap.shape)
    
    return px, py, best_angle, best_heatmap, angles, scores

def generate_report(num_samples=20):
    print(f"Loading model from {MODEL_PATH}...")
    model = SiameseUNet(n_channels=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Model file not found! Please train first.")
        return
    model.eval()

    results = []
    center = np.array([IMG_SIZE//2, IMG_SIZE//2])

    print(f"Running inference on {num_samples} samples...")
    
    for i in range(num_samples):
        # Generate random geometry
        points_base = create_random_polygon(3, 8, (20, 40), center)
        gt_angle = np.random.uniform(0, 360)
        gt_tx, gt_ty = np.random.uniform(-20, 20), np.random.uniform(-20, 20)
        
        # Create Query Image
        points_query = apply_affine_transform(points_base, gt_angle, 1.0, (gt_tx, gt_ty), center)
        img_query = np.full((IMG_SIZE, IMG_SIZE), 50, dtype=np.uint8)
        cv2.fillPoly(img_query, [points_query], 200) # Simple fill
        img_query = cv2.GaussianBlur(img_query, (3,3), 0.5) # Blur
        
        # Prepare Tensor
        t_query = torch.from_numpy(img_query).float().div(255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Run Model
        px, py, pred_angle, heatmap, search_angles, search_scores = predict_and_profile(model, points_base, t_query)
        
        # Calculate Errors
        gt_cx, gt_cy = IMG_SIZE//2 + gt_tx, IMG_SIZE//2 + gt_ty
        trans_err = np.sqrt((gt_cx - px)**2 + (gt_cy - py)**2)
        
        rot_err = abs(gt_angle - pred_angle) % 360
        if rot_err > 180: rot_err = 360 - rot_err
        
        results.append({
            'img_query': img_query,
            'heatmap': heatmap,
            'points_base': points_base,
            'gt': (gt_cx, gt_cy, gt_angle),
            'pred': (px, py, pred_angle),
            'errs': (trans_err, rot_err),
            'profile': (search_angles, search_scores)
        })

    # Sort by Rotation Error (Best to Worst)
    results.sort(key=lambda x: x['errs'][1])

    # Select Best 3 and Worst 3
    to_plot = results[:3] + results[-3:]
    labels = ["BEST MATCH"] * 3 + ["WORST FAILURE"] * 3

    # Plotting
    fig, axes = plt.subplots(len(to_plot), 3, figsize=(15, 3 * len(to_plot)))
    
    for idx, (res, label_text) in enumerate(zip(to_plot, labels)):
        img = res['img_query']
        hmap = res['heatmap']
        gt_cx, gt_cy, gt_ang = res['gt']
        px, py, pred_ang = res['pred']
        t_err, r_err = res['errs']
        angles, scores = res['profile']
        
        # --- Column 1: Visual Overlay ---
        ax_img = axes[idx, 0]
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Draw GT (Green)
        pts_gt = apply_affine_transform(res['points_base'], gt_ang, 1.0, (gt_cx-IMG_SIZE//2, gt_cy-IMG_SIZE//2), center)
        cv2.polylines(vis_img, [pts_gt], True, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Draw Pred (Red if fail, Blue if pass)
        color = (255, 0, 0) if r_err > 10 else (0, 100, 255)
        pts_pred = apply_affine_transform(res['points_base'], pred_ang, 1.0, (px-IMG_SIZE//2, py-IMG_SIZE//2), center)
        cv2.polylines(vis_img, [pts_pred], True, color, 2, cv2.LINE_AA)
        
        ax_img.imshow(vis_img)
        ax_img.set_title(f"{label_text}\nRot Err: {r_err:.1f}° | Trans Err: {t_err:.1f}px", fontweight='bold')
        ax_img.axis('off')

        # --- Column 2: Heatmap ---
        ax_map = axes[idx, 1]
        im_map = ax_map.imshow(hmap, cmap='magma', vmin=0, vmax=1)
        ax_map.set_title("Network Confidence")
        ax_map.scatter([px], [py], c='cyan', marker='x', s=100)
        ax_map.axis('off')
        
        # --- Column 3: Search Profile ---
        ax_plot = axes[idx, 2]
        ax_plot.plot(angles, scores, color='blue', linewidth=2)
        ax_plot.axvline(x=gt_ang, color='green', linestyle='--', label='GT')
        ax_plot.axvline(x=pred_ang, color='red', linestyle='--', label='Pred')
        ax_plot.set_title("Rotation Search Profile")
        ax_plot.set_ylabel("Conf")
        ax_plot.grid(True, alpha=0.3)
        if idx == 0: ax_plot.legend()

    plt.tight_layout()
    plt.savefig("diagnostic_report.png")
    print("Saved diagnostic_report.png")

if __name__ == "__main__":
    generate_report(num_samples=20)