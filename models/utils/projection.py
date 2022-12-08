import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras
)

def project_points(points, azimuth, elevation, distance, image_size, device):

    depths = points[:, :, 2:]
    z_nears = torch.max(depths, 1).values.flatten()
    z_fars =  torch.min(depths, 1).values.flatten()

    R, T = look_at_view_transform(distance, elevation, azimuth)

    cameras = FoVOrthographicCameras(znear=z_nears, zfar=z_fars, device=device, R=R, T=T)

    projections = cameras.transform_points_screen(points, image_size=image_size)[:, :, :2]

    return projections.round().long()

def map_points_to_plane(feat_map, pixel_coords):

    aligned_feats = []
    for i in range(feat_map.shape[0]):

        xs = pixel_coords[i, :, 0]
        ys = pixel_coords[i, :, 1]

        f = feat_map[i, :, xs, ys]
        aligned_feats.append(f)

    return torch.stack(aligned_feats, dim=0)
