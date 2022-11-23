import numpy as np
import torch

def get_ball_scene_dataloader(n_scenes, n_balls, max_radius=.5):
    while True:
        if isinstance(n_balls, int):
            L = n_balls
        else:
            L = np.random.randint(n_balls[0], n_balls[1]+1)
        yield generate_3d_scenes(B=n_scenes, L=L, radius_range=(.2,max_radius))

def generate_3d_scenes(B, L, radius_range):
    """Generate a batch of 3D scenes with L balls."""
    max_radius = radius_range[1]
    radii = torch.rand(B, L, device='cuda') * (max_radius - radius_range[0]) + radius_range[0]
    centers = torch.rand(B, L, 3, device='cuda') * 2 - 1
    colors = torch.rand(L, 3, device='cuda')

    def rgba_and_sdf(rgba_coords, sdf_coords=None):
        """coords: (N,3)
        rgba: (B,N,4), sdf: (B,N)"""
        N = rgba_coords.shape[0]
        rgba = torch.zeros((B,N,4), device=rgba_coords.device)
        signed_dists = (rgba_coords-centers.unsqueeze(2)).norm(dim=-1) - radii.view(B,L,1) # (B,L,N)
        in_ball = (signed_dists < 0).max(dim=1) # (B,N), (B,N)
        L_indices = in_ball.indices[in_ball.values] # (*)
        rgb = colors[L_indices].reshape(-1,3) # (*,3)
        rgba[in_ball.values] = torch.cat((rgb, torch.ones_like(rgb[...,:1])), dim=-1) # (*,4)
        if sdf_coords is None:
            sdf = (signed_dists.amin(dim=1) + max_radius) / (2.8+max_radius) * 2 - 1 # (B,N), scale from -1 to 1
        else:
            signed_dists = (sdf_coords-centers.unsqueeze(2)).norm(dim=-1) - radii.view(B,L,1) # (B,L,N2)
            sdf = (signed_dists.amin(dim=1) + max_radius) / (2.8+max_radius) * 2 - 1 # (B,N2), scale from -1 to 1

        return rgba, sdf.unsqueeze_(-1)

    return rgba_and_sdf