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
    Generates Gaussian heatmaps centered at targets.
    targets: [tx, ty] normalized [-1, 1]
    """
    x = torch.arange(0, img_size, device=device).float()
    y = torch.arange(0, img_size, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    heatmap_batch = []
    center = img_size / 2.0
    
    for i in range(batch_size):
        px = center + targets[i, 0] * center
        py = center + targets[i, 1] * center
        
        dist_sq = (xx - px)**2 + (yy - py)**2
        heatmap = torch.exp(-dist_sq / (2.0 * sigma**2))
        heatmap_batch.append(heatmap)
        
    return torch.stack(heatmap_batch).unsqueeze(1) # (B, 1, H, W)