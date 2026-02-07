import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF

# --- IMPORTS ---
from shape_dataset import ShapeMatchingDatasetSimple 
from network import SiameseUNet 
from generate_polygon_dataset import apply_affine_transform, create_random_polygon

MODEL_PATH = "siamese_unet.pth" 
IMG_SIZE = 128

def find_peak(heatmap):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
    return max_loc[0], max_loc[1], max_val

def predict_pose_search(model, template_tensor, query_tensor, device):
    """
    Rotates template 36 times (0-350 deg), compares all against query.
    Returns: best_x, best_y, best_angle, confidence
    """
    # 1. Create Batch of Rotated Templates
    angles = np.arange(0, 360, 10) # 10 degree steps
    batch_templates = []
    
    for angle in angles:
        # Rotate template (Distance Field)
        rot_t = TF.rotate(template_tensor, float(angle), interpolation=TF.InterpolationMode.BILINEAR)
        batch_templates.append(rot_t)
        
    batch_templates = torch.stack(batch_templates).squeeze(1).to(device) # (36, 1, 128, 128)
    batch_queries = query_tensor.repeat(len(angles), 1, 1, 1).to(device) # (36, 1, 128, 128)
    
    # 2. Batch Inference
    with torch.no_grad():
        logits = model(batch_templates, batch_queries)
        heatmaps = torch.sigmoid(logits) 
        
    # 3. Find Winner
    # Max pool spatial dims to get score per angle
    scores = heatmaps.view(len(angles), -1).max(dim=1).values
    best_idx = torch.argmax(scores).item()
    best_conf = scores[best_idx].item()
    best_angle = angles[best_idx]
    
    # 4. Get Position from winner heatmap
    best_map = heatmaps[best_idx].squeeze().cpu().numpy()
    pred_x, pred_y, _ = find_peak(best_map)
    
    return pred_x, pred_y, best_angle, best_conf

def visualize_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = SiameseUNet(n_channels=1).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    except:
        print("Model not found. Run training first.")
        return
    model.eval()

    # --- Generate a Test Case ---
    center = np.array([IMG_SIZE//2, IMG_SIZE//2])
    points_base = create_random_polygon(3, 6, (30, 50), center)
    
    gt_angle = np.random.uniform(0, 360)
    gt_tx = np.random.randint(-30, 30)
    gt_ty = np.random.randint(-30, 30)
    
    # Create Query Image
    points_query = apply_affine_transform(points_base, gt_angle, 1.0, (gt_tx, gt_ty), center)
    img_query = np.full((IMG_SIZE, IMG_SIZE), 50, dtype=np.uint8)
    cv2.fillPoly(img_query, [points_query], 200)
    img_query = cv2.GaussianBlur(img_query, (3,3), 0.5)
    
    t_query = torch.from_numpy(img_query.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)

    # Create Template (At 0 degrees)
    points_temp = apply_affine_transform(points_base, 0.0, 1.0, (0, 0), center)
    img_temp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.polylines(img_temp, [points_temp], True, 255, 1)
    
    # Preprocess Dist Transform
    edge_inv = 255 - img_temp
    dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
    dist = np.exp(-0.05 * dist) 
    t_temp = torch.from_numpy(dist).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- RUN SEARCH ---
    print(f"Ground Truth: Angle={gt_angle:.1f}, Tx={gt_tx}, Ty={gt_ty}")
    px, py, pa, conf = predict_pose_search(model, t_temp, t_query, device)
    
    print(f"Prediction:   Angle={pa:.1f}, Px={px}, Py={py}, Conf={conf:.3f}")
    
    # --- VISUALIZATION ---
    img_color = cv2.cvtColor(img_query, cv2.COLOR_GRAY2BGR)
    
    # Draw GT
    gt_cx = IMG_SIZE//2 + gt_tx
    gt_cy = IMG_SIZE//2 + gt_ty
    cv2.circle(img_color, (int(gt_cx), int(gt_cy)), 3, (0, 255, 0), -1)
    
    # Draw Pred
    cv2.drawMarker(img_color, (int(px), int(py)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    # Overlay Template at Predicted Pose
    shift_x = px - IMG_SIZE//2
    shift_y = py - IMG_SIZE//2
    pred_cnt = apply_affine_transform(points_base, pa, 1.0, (shift_x, shift_y), center)
    cv2.polylines(img_color, [pred_cnt], True, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(img_color, f"Conf: {conf:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img_color, f"AngErr: {abs(gt_angle-pa):.1f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Result", cv2.resize(img_color, (0,0), fx=3, fy=3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_inference()