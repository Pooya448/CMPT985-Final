import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SHGNet import SHGNet
from models.MLP import *
from .utils.projection import project_points, map_points_to_plane

class OccNet(nn.Module):
    """docstring for OccNet."""

    def __init__(self, opt, device):
        super(OccNet, self).__init__()

        self.encoder = SHGNet(opt['shg'])
        self.regressor = MLP(opt['mlp'], opt['mlp']['channels'])
        self.device = device

    def forward(self, input_image, spatial_feat, points, azimuth):

        shg_featmap = self.encoder(input_image)

        pixel_coords = project_points(points=points, azimuth=azimuth, elevation=0, distance=1, image_size=(512, 512), device=self.device)

        pixel_aligned_feats_occ = map_points_to_plane(shg_featmap, pixel_coords)  #based in bilinear interpolation

        regressor_in = torch.cat([pixel_aligned_feats_occ, spatial_feat], 1)
        occ_pred = self.regressor(regressor_in)

        return occ_pred, pixel_aligned_feats_occ
