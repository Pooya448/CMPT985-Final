import os
import torch
import numpy as np
from tqdm.notebook import tqdm
# import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Pointclouds

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

class DR():
    def __init__(self, opt, device):
        self.renderer_type = opt['renderer_type']
        self.cam_type = opt['cam_type']
        self.im_size = opt['im_size']
        self.composition_type = opt['comp_type']
        self.radius = opt['radius']
        self.points_per_pixel = opt['points_per_pixel']
        self.device = device
        self.renderer = self.create_renderer()

    def render(self, points, occ, col, norm):

        colored_norms = 0.5 * norm + 0.5

        print(f"\n\n\n\n render points shape: {len(points.tolist())}")
        # print(f"\n\n\n\n render points shape: {norm.shape}")
        # print(f"\n\n\n\n render points shape: {col.shape}")
        # points = points.tolist()
        # norm = norm.tolist()
        # col = col.tolist()
        # occ = occ.tolist()

        col_ptc = Pointclouds(points=points, normals=norm, features=col)
        norm_ptc = Pointclouds(points=points, normals=norm, features=colored_norms)
        occ_ptc = Pointclouds(points=points, normals=norm, features=occ) # Further Check -> Checked!

        # Dimension [B x im_size x im_size x 4] -> 4th channel is alpha
        col_render = self.renderer(col_ptc)

        # Dimension [B x im_size x im_size x 3]
        norm_render = self.renderer(norm_ptc)

        # Dimension [B x im_size x im_size x 1]
        occ_render = self.renderer(occ_ptc)



        return col_render, norm_render, occ_render

    def load_points(self, path, device):
        pointcloud = np.load(path)
        verts = torch.Tensor(pointcloud['verts']).to(device)
        rgb = torch.Tensor(pointcloud['rgb']).to(device)
        norms = torch.Tensor(pointcloud['normals']).to(device)
        pt_cloud = Pointclouds(points=[verts], normals=norms, features=[rgb])
        return pt_cloud

    def create_pointcloud(self, pointcloud, device):
        verts = torch.Tensor(pointcloud['verts']).to(device)
        rgb = torch.Tensor(pointcloud['rgb']).to(device)
        norms = torch.Tensor(pointcloud['normals']).to(device)
        pt_cloud = Pointclouds(points=[verts], normals=norms, features=[rgb])
        return pt_cloud

    def create_camera(self, cam_type, device):
        if cam_type == 'orthographic':
            R, T = look_at_view_transform(20, 10, 0)
            cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
            return cameras
        elif cam_type == 'perspective':
            cameras = FoVPerspectiveCameras(device=device)
            return cameras

    def get_raster_settings(self, im_size, radius, points_per_pixel):

        raster_settings = PointsRasterizationSettings(
            image_size=im_size,
            radius=radius,
            points_per_pixel=points_per_pixel
        )

        return raster_settings

    def create_renderer(self):

        raster_settings = self.get_raster_settings(self.im_size, self.radius, self.points_per_pixel)

        cameras = self.create_camera(self.cam_type, self.device)

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

        if self.composition_type == 'alpha':
            compositor = AlphaCompositor()
        elif self.composition_type == 'norm_weight':
            compositor = NormWeightedCompositor()

        if self.renderer_type == 'simple':
            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=compositor
            ).to(self.device)
        elif self.renderer_type == 'pulsar':
            renderer = PulsarPointsRenderer(
                rasterizer=rasterizer,
                n_channels=4
            ).to(self.device)

        return renderer
