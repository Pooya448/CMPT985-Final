import torch
import torch.nn as nn
import torch.nn.functional as F
from models.UNet import *
from models.MLP import *
from .utils.projection import project_points, map_points_to_plane

class ColorNet(nn.Module):
    """docstring for ColorNet."""

    def __init__(self, opt, device):
        super(ColorNet, self).__init__()

        self.encoder = UNet(opt['unet'])
        self.regressor = MLP(opt['mlp'], opt['mlp']['channels'])
        self.device = device

    def forward(self, input_image, spatial_feat, occ_feat, norm_feat, points, azimuth):

        unet_featmap = self.encoder(input_image)

        pixel_coords = project_points(points=points, azimuth=azimuth, elevation=0, distance=1, image_size=(512, 512), device=self.device)

        ### pixel alignment implement here
        pixel_aligned_feats_color = map_points_to_plane(unet_featmap, pixel_coords)

        regressor_in = torch.cat([pixel_aligned_feats_color, occ_feat, norm_feat, spatial_feat], 1)
        color_pred = self.regressor(regressor_in)

        return color_pred, pixel_aligned_feats_color
