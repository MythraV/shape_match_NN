import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

# Import helpers
from generate_polygon_dataset import create_random_polygon, apply_affine_transform

class ShapeMatchingDatasetPrecomputed(Dataset):
    def __init__(self, pt_file_path):
        print(f"Loading dataset from {pt_file_path}...")
        data = torch.load(pt_file_path)
        self.templates = data['templates'] # Float32
        self.queries = data['queries']     # Uint8
        self.targets = data['targets']     # Float32
        print(f"Loaded {len(self.templates)} samples.")
        
    def __len__(self):
        return len(self.templates)

    def __getitem__(self, idx):
        # Template is already float32 (distance field)
        template = self.templates[idx]
        
        # Query is uint8, need to normalize to [0, 1] float32
        query = self.queries[idx].float() / 255.0
        
        target = self.targets[idx]
        
        return template, query, target

class ShapeMatchingDatasetSimple(Dataset):
    def __init__(self, image_size=128, length=2000):
        self.img_size = image_size
        self.length = length
        self.center = np.array([image_size//2, image_size//2])
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Base Geometry
        # We use full complexity (3-10 sides) to make shapes unique
        points_base = create_random_polygon(
            min_sides=3, max_sides=10, 
            radius_range=(int(self.img_size*0.15), int(self.img_size*0.3)), 
            canvas_center=self.center
        )
        
        # 2. Determine Case: Match (50%) vs No-Match (50%)
        is_match = (np.random.rand() > 0.5)
        
        # 3. Setup Angles
        query_angle = np.random.uniform(-180, 180)
        
        if is_match:
            # Aligned: Template is within +/- 5 deg of Query
            template_angle = query_angle + np.random.uniform(-5, 5)
        else:
            # Misaligned: Template is > 10 deg away
            offset = np.random.uniform(10, 180) * (1 if np.random.rand() > 0.5 else -1)
            template_angle = query_angle + offset

        # 4. Apply Transforms
        tx = np.random.uniform(-30, 30)
        ty = np.random.uniform(-30, 30)
        
        # Query: Rotated + Translated + Tiny Augmentation
        # Add a tiny bit of random rotation to query itself to not be perfectly aligned to grid
        query_angle_aug = query_angle + np.random.uniform(-1, 1)
        points_query = apply_affine_transform(points_base, query_angle_aug, 1.0, (tx, ty), self.center)
        
        # Template: Rotated + Centered (No Translation)
        points_template = apply_affine_transform(points_base, template_angle, 1.0, (0, 0), self.center)

        # 5. Render Template (Distance Field)
        img_temp = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        if len(points_template) > 0:
            cv2.polylines(img_temp, [points_template], True, 255, 1)
        tensor_temp = self.preprocess_distance_transform(img_temp)

        # 6. Render Query (Filled Polygon + Blur + Noise)
        bg = np.random.randint(0, 100)
        img_query = np.full((self.img_size, self.img_size), bg, dtype=np.uint8)
        
        if len(points_query) > 0:
            fg = np.random.randint(120, 255)
            # Fill the shape
            cv2.fillPoly(img_query, [points_query], fg)
            # Add AA edge
            cv2.polylines(img_query, [points_query], True, fg, 1, cv2.LINE_AA)

        # Add Noise & Blur to mimic camera
        noise = np.random.normal(0, 5, img_query.shape).astype(np.int16)
        img_query = np.clip(img_query + noise, 0, 255).astype(np.uint8)
        img_query = cv2.GaussianBlur(img_query, (3, 3), 0.5)

        tensor_query = img_query.astype(np.float32) / 255.0
        tensor_query = torch.from_numpy(tensor_query).unsqueeze(0)

        # 7. Targets
        if is_match:
            # Target: [tx, ty, Flag=1.0]
            target_tx = tx / (self.img_size / 2.0)
            target_ty = ty / (self.img_size / 2.0)
            targets = torch.tensor([target_tx, target_ty, 1.0], dtype=torch.float32)
        else:
            # Target: [0, 0, Flag=0.0]
            targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        return tensor_temp, tensor_query, targets

    def preprocess_distance_transform(self, edge_img):
        edge_inv = 255 - edge_img
        dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
        dist = np.exp(-0.05 * dist) 
        return torch.from_numpy(dist).float().unsqueeze(0)