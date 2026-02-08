import torch
import numpy as np
import os
import cv2
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Import the generation logic
from generate_polygon_dataset import create_random_polygon, apply_affine_transform

def generate_single_sample(img_size):
    center = np.array([img_size//2, img_size//2])
    
    # 1. Base Geometry (3-10 sides)
    points_base = create_random_polygon(
        min_sides=3, max_sides=10, 
        radius_range=(int(img_size*0.15), int(img_size*0.3)), 
        canvas_center=center
    )
    
    # 2. Determine Match
    is_match = (np.random.rand() > 0.5)
    
    # 3. Angles
    query_angle = np.random.uniform(-180, 180)
    if is_match:
        template_angle = query_angle + np.random.uniform(-5, 5)
    else:
        # 30% Hard Negative
        if np.random.rand() < 0.3:
            offset = 180 + np.random.uniform(-20, 20)
        else:
            offset = np.random.uniform(10, 350)
        template_angle = query_angle + offset

    # 4. Transforms
    tx = np.random.uniform(-30, 30)
    ty = np.random.uniform(-30, 30)
    
    # Query: Rotated + Translated + Tiny Augmentation
    query_angle_aug = query_angle + np.random.uniform(-1, 1)
    points_query = apply_affine_transform(points_base, query_angle_aug, 1.0, (tx, ty), center)
    
    # Template: Rotated + Centered
    points_template = apply_affine_transform(points_base, template_angle, 1.0, (0, 0), center)

    # 5. Render Template (Distance Field) -> Return as float32
    img_temp = np.zeros((img_size, img_size), dtype=np.uint8)
    if len(points_template) > 0:
        cv2.polylines(img_temp, [points_template], True, 255, 1)
    edge_inv = 255 - img_temp
    dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
    dist = np.exp(-0.05 * dist)
    
    # 6. Render Query (Image) -> Return as uint8 to save space
    bg = np.random.randint(0, 100)
    img_query = np.full((img_size, img_size), bg, dtype=np.uint8)
    if len(points_query) > 0:
        fg = np.random.randint(120, 255)
        cv2.fillPoly(img_query, [points_query], fg)
        cv2.polylines(img_query, [points_query], True, fg, 1, cv2.LINE_AA)
    
    noise = np.random.normal(0, 5, img_query.shape).astype(np.int16)
    img_query = np.clip(img_query + noise, 0, 255).astype(np.uint8)
    img_query = cv2.GaussianBlur(img_query, (3, 3), 0.5)

    # 7. Targets
    if is_match:
        # Normalize targets to [-1, 1] relative to center? 
        # The training code expects: target_tx = tx / (img_size/2)
        # But wait, generate_heatmap_target uses raw range? 
        # Let's check ShapeMatchingDatasetSimple in shape_dataset.py
        # It does: target_tx = tx / (self.img_size / 2.0)
        target_tx = tx / (img_size / 2.0)
        target_ty = ty / (img_size / 2.0)
        targets = np.array([target_tx, target_ty, 1.0], dtype=np.float32)
    else:
        targets = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return dist.astype(np.float32), img_query, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100000, help='Number of samples')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--out', type=str, default='train_data.pt')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()

    print(f"Generating {args.size} samples with {args.jobs} jobs...")
    
    results = Parallel(n_jobs=args.jobs)(
        delayed(generate_single_sample)(args.img_size) for _ in tqdm(range(args.size))
    )
    
    # Unpack
    print("Stacking tensors...")
    templates = np.array([r[0] for r in results]) # (N, 128, 128)
    queries = np.array([r[1] for r in results])   # (N, 128, 128)
    targets = np.array([r[2] for r in results])   # (N, 3)
    
    print("Saving to .pt file...")
    data = {
        'templates': torch.from_numpy(templates).unsqueeze(1), # (N, 1, H, W)
        'queries': torch.from_numpy(queries).unsqueeze(1),     # (N, 1, H, W)
        'targets': torch.from_numpy(targets)                   # (N, 3)
    }
    
    torch.save(data, args.out)
    print(f"Saved {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)")

if __name__ == "__main__":
    main()
