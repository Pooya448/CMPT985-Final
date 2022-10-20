import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import *
from .MLP import *
from .utils.projection import project_points, map_points_to_plane

class NormNet(nn.Module):
    """docstring for NormNet."""

    def __init__(self, opt, device):
        super(NormNet, self).__init__()

        self.encoder = UNet(opt['unet'])
        self.regressor = MLP(opt['mlp'], opt['mlp']['channels'])
        self.device = device

    def forward(self, input_image, spatial_feat, occ_feat, points, azimuth):

        unet_featmap = self.encoder(input_image)

        pixel_coords = project_points(points=points, azimuth=azimuth, elevation=0, distance=1, image_size=(512, 512), device=self.device)

        ### pixel alignment implement here
        pixel_aligned_feats_norm = map_points_to_plane(unet_featmap, pixel_coords)

        regressor_in = torch.cat([pixel_aligned_feats_norm, occ_feat, spatial_feat], 1)
        norm_pred = self.regressor(regressor_in)

        return norm_pred, pixel_aligned_feats_norm
