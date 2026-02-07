import torch
import numpy as np
import torchvision.transforms.functional as TF
from shape_dataset import ShapeMatchingDatasetSimple 
from network import SiameseUNet 
import cv2
import os

MODEL_PATH = "siamese_unet.pth" 
IMG_SIZE = 128

def find_peak_subpixel(heatmap):
    """
    Finds the peak of the heatmap with sub-pixel accuracy using a local centroid.
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
    if max_val < 0.1:
        return -1.0, -1.0, max_val
    
    x_p, y_p = max_loc
    r = 5
    y_start = max(0, y_p - r)
    y_end = min(heatmap.shape[0], y_p + r + 1)
    x_start = max(0, x_p - r)
    x_end = min(heatmap.shape[1], x_p + r + 1)
    
    window = heatmap[y_start:y_end, x_start:x_end].copy()
    window[window < 0.3 * max_val] = 0
    
    M = cv2.moments(window)
    if M["m00"] == 0:
        return float(x_p), float(y_p), max_val
    
    cx = M["m10"] / M["m00"] + x_start
    cy = M["m01"] / M["m00"] + y_start
    
    return cx, cy, max_val

def predict_pose_search(model, template_tensor, query_tensor, device):
    """
    Two-stage search for better accuracy.
    """
    # 1. Coarse Search (10 deg steps)
    angles_coarse = np.arange(0, 360, 10)
    
    def get_scores(target_angles):
        batch_t = []
        for a in target_angles:
            batch_t.append(TF.rotate(template_tensor, -float(a), interpolation=TF.InterpolationMode.BILINEAR))
        batch_t = torch.stack(batch_t).squeeze(1).to(device)
        batch_q = query_tensor.repeat(len(target_angles), 1, 1, 1).to(device)
        with torch.no_grad():
            logits = model(batch_t, batch_q)
            heatmaps = torch.sigmoid(logits)
        scores = heatmaps.view(len(target_angles), -1).max(dim=1).values
        return scores, heatmaps

    scores_c, _ = get_scores(angles_coarse)
    best_c_idx = torch.argmax(scores_c).item()
    best_c_angle = angles_coarse[best_c_idx]
    
    # Debug: print scores for first sample
    # print(f"Coarse Scores: {scores_c.cpu().numpy()}")

    # 2. Fine Search (+/- 10 deg in 1 deg steps)
    angles_fine = np.arange(best_c_angle - 10, best_c_angle + 11, 1)
    angles_fine = np.mod(angles_fine, 360)
    
    scores_f, heatmaps_f = get_scores(angles_fine)
    best_f_idx = torch.argmax(scores_f).item()
    best_f_angle = angles_fine[best_f_idx]
    best_conf = scores_f[best_f_idx].item()
    
    # print(f"Best Fine Angle: {best_f_angle}, Conf: {best_conf}")
    
    best_map = heatmaps_f[best_f_idx].squeeze().cpu().numpy()
    px, py, _ = find_peak_subpixel(best_map)
    
    return px, py, best_f_angle, best_conf, best_map

def evaluate_model(num_samples=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseCorrelationUNet(n_channels=1).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    from generate_polygon_dataset import apply_affine_transform, create_random_polygon
    center = np.array([IMG_SIZE//2, IMG_SIZE//2])
    
    rot_errors = []
    trans_errors = []
    
    correct_rot_10 = 0
    correct_rot_2 = 0
    correct_trans_1 = 0
    correct_trans_05 = 0

    print(f"Starting evaluation over {num_samples} samples...")

    for i in range(num_samples):
        # Generate complex polygons (3-10 sides)
        points_base = create_random_polygon(3, 10, (int(IMG_SIZE*0.15), int(IMG_SIZE*0.3)), center)
        
        gt_angle = np.random.uniform(0, 360)
        gt_tx = np.random.uniform(-30, 30)
        gt_ty = np.random.uniform(-30, 30)
        
        # Query
        points_query = apply_affine_transform(points_base, gt_angle, 1.0, (gt_tx, gt_ty), center)
        img_query = np.full((IMG_SIZE, IMG_SIZE), np.random.randint(0, 50), dtype=np.uint8)
        fg = np.random.randint(150, 255)
        cv2.fillPoly(img_query, [points_query], fg)
        cv2.polylines(img_query, [points_query], True, fg, 1, cv2.LINE_AA)
        noise = np.random.normal(0, 3, img_query.shape).astype(np.int16)
        img_query = np.clip(img_query + noise, 0, 255).astype(np.uint8)
        img_query = cv2.GaussianBlur(img_query, (3, 3), 0.5)
        t_query = torch.from_numpy(img_query.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)

        # Template
        points_temp = apply_affine_transform(points_base, 0.0, 1.0, (0, 0), center)
        img_temp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.polylines(img_temp, [points_temp], True, 255, 1)
        edge_inv = 255 - img_temp
        dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
        dist = np.exp(-0.05 * dist) 
        t_temp = torch.from_numpy(dist).float().unsqueeze(0).unsqueeze(0).to(device)

        px, py, pa, conf, best_map = predict_pose_search(model, t_temp, t_query, device)
        
        pred_tx = px - IMG_SIZE//2
        pred_ty = py - IMG_SIZE//2
        
        angle_err = abs(pa - gt_angle) % 360
        if angle_err > 180: angle_err = 360 - angle_err
        
        trans_err = np.sqrt((pred_tx - gt_tx)**2 + (pred_ty - gt_ty)**2)

        if i < 10:
            print(f"Sample {i}: GT Ang={gt_angle:.1f}, Pred Ang={pa:.1f}, Err={angle_err:.1f} | GT Pos=({gt_tx:.1f}, {gt_ty:.1f}), Pred Pos=({pred_tx:.1f}, {pred_ty:.1f}), Err={trans_err:.2f} | Conf={conf:.3f}")
        
        rot_errors.append(angle_err)
        trans_errors.append(trans_err)
        
        if angle_err < 10: correct_rot_10 += 1
        if angle_err < 2:  correct_rot_2 += 1
        if trans_err < 1.0: correct_trans_1 += 1
        if trans_err < 0.5: correct_trans_05 += 1
        
        if i == 0:
            # Save a sample heatmap
            heatmap_img = (best_map * 255).astype(np.uint8)
            cv2.imwrite("sample_heatmap.png", heatmap_img)

    print(f"\n--- Results ---")
    print(f"Avg Rotation Error:    {np.mean(rot_errors):.2f} deg")
    print(f"Avg Translation Error: {np.mean(trans_errors):.2f} px")
    print(f"Rotation Accuracy (<10 deg): {correct_rot_10/num_samples*100:.1f}%")
    print(f"Rotation Accuracy (<2 deg):  {correct_rot_2/num_samples*100:.1f}%")
    print(f"Translation Accuracy (<1.0 px): {correct_trans_1/num_samples*100:.1f}%")
    print(f"Translation Accuracy (<0.5 px): {correct_trans_05/num_samples*100:.1f}%")

if __name__ == "__main__":
    evaluate_model(200)