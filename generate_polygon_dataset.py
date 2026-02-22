import cv2
import numpy as np
import torch

def create_random_polygon(min_sides, max_sides, radius_range, canvas_center):
    '''
    Generates a list of (x,y) points representing a random shape.
    '''
    if min_sides < 3: return []

    num_verts = np.random.randint(min_sides, max_sides+1)
    
    angles = np.random.uniform(0, 2*np.pi, size=num_verts)
    angles.sort()

    radius = np.random.randint(radius_range[0], radius_range[1]+1, size=num_verts)
    pts = np.array([(r*np.cos(theta), r*np.sin(theta)) for r, theta in zip(radius, angles)])

    # Move to canvas center 
    pts = pts + np.array(canvas_center)
    return pts.astype(np.int32)


def apply_affine_transform(vertices, rotation_deg, scale, translate_xy, canvas_center):
    '''
    Apply affine transform (Rotation + Scale + Translation) to points.
    '''
    if vertices.shape[1] == 1: vertices = vertices.reshape(-1,2)
    
    vertices = vertices - canvas_center
    
    # Rotate 
    rads = np.deg2rad(rotation_deg)
    c, s = np.cos(rads), np.sin(rads)
    R = np.array([[c, -s], [s, c]])

    verts_rot = scale * R @ vertices.T  
    verts_new = verts_rot.T + canvas_center + translate_xy

    return verts_new.astype(np.int32)

def generate_heatmap_target(batch_size, img_size, targets, device, sigma=2.0):
    """
    Generates Gaussian heatmaps centered at targets (vectorized).
    targets: (B, 2) normalized [-1, 1]
    Returns: (B, 1, H, W)
    """
    x = torch.arange(0, img_size, device=device).float()
    y = torch.arange(0, img_size, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)

    center = img_size / 2.0
    px = (center + targets[:, 0] * center).view(-1, 1, 1)  # (B, 1, 1)
    py = (center + targets[:, 1] * center).view(-1, 1, 1)  # (B, 1, 1)

    dist_sq = (xx.unsqueeze(0) - px)**2 + (yy.unsqueeze(0) - py)**2
    heatmaps = torch.exp(-dist_sq / (2.0 * sigma**2))

    return heatmaps.unsqueeze(1)  # (B, 1, H, W)