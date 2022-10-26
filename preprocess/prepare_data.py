# To Do -> Here render the dataset to get RGB (textured) and norm images for 2d supervision

import argparse

import os
import cv2
import sys
import torch
import matplotlib.pyplot as plt

from torchvision.io import read_image, write_png

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    PointsRenderer,
    PulsarPointsRenderer,
    MeshRasterizer,
    PointsRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    Textures,
    AlphaCompositor,
    NormWeightedCompositor
)

from glob import glob
import numpy as np

from utils.geometry import load_tex_mesh, sample_from_mesh


class MGNPrep(object):

    def __init__(self, data_dir, num_views, cam_type, n_sample=50000):
        self.device = torch.device('mps')
        self.data_dir = data_dir
        self.subjects = self.list_subjects()
        self.num_views = num_views
        self.step = 360 // num_views
        self.render_angles = list(range(0, 360, self.step))
        self.cam_type = cam_type
        self.n_sample = n_sample

    def prepare_data(self):

        for sub in self.subjects:

            print(f"Rendering subject id: {sub.split('/')[-1]}")

            obj_path = os.path.join(sub, "scan.obj")
            tex_path = os.path.join(sub, "scan_tex.jpg")

            mesh = load_tex_mesh(obj_path, tex_path, self.device)

            all_points, all_norms, all_colors = sample_from_mesh(mesh, n=self.n_sample)

            torch.save(all_points, os.path.join(sub, "points.pt"))
            torch.save(all_norms, os.path.join(sub, "normals.pt"))
            torch.save(all_colors, os.path.join(sub, "colors.pt"))

            points, normals, _ = sample_from_mesh(mesh, n=1000000)

            points = points.squeeze()
            normals = normals.squeeze()

            norm_colors = 0.5 * normals + 0.5
            norm_ptc = Pointclouds(points=[points], normals=[normals], features=[norm_colors])

            save_dir_mesh = os.path.join(sub, f"mesh_render_{self.num_views}")
            os.makedirs(save_dir_mesh, exist_ok=True)

            save_dir_norm = os.path.join(sub, f"norm_render_{self.num_views}")
            os.makedirs(save_dir_norm, exist_ok=True)

            for angle in self.render_angles:

                point_renderer = self.get_point_renderer(camera=self.cam_type, azimuth=angle)
                mesh_renderer = self.get_mesh_renderer(camera=self.cam_type, azimuth=angle)

                mesh_image = mesh_renderer(mesh)

                file_path = os.path.join(save_dir_mesh, f'{angle}.png')
                write_png(input=mesh_image[0, ..., :3].permute((2, 0, 1)).cpu().to(dtype=torch.uint8), filename=file_path)


                norms_image = point_renderer(norm_ptc)

                im = norms_image[0, ..., :3].cpu()
                im *= (255.0/im.max())

                file_path = os.path.join(save_dir_norm, f'{angle}.png')
                write_png(input=im.permute((2, 0, 1)).to(dtype=torch.uint8), filename=file_path)


    def list_subjects(self):
        search_dir = os.path.join(self.data_dir, "*")
        subjects = sorted(glob(search_dir))
        return subjects


    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device


    def get_point_renderer(self, camera, azimuth, elevation=0):

        R, T = look_at_view_transform(1, elevation, azimuth)

        if camera == 'orth':
            cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)
        if camera == 'pers':
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = 0.003,
            points_per_pixel = 10
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        return renderer

    def get_mesh_renderer(self, camera, azimuth, elevation=0):

        R, T = look_at_view_transform(1.9, elevation, azimuth)

        if camera == 'orth':
            cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)
        if camera == 'pers':
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        ### Light front
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        return renderer

parser = argparse.ArgumentParser(
    description='Render MGN Data.'
)
parser.add_argument('--data', '-d', type=str, help='Path to MGN data.')
parser.add_argument('--cam', '-c', type=str, help='Camera Type.')
parser.add_argument('--views', '-v', type=int, help='Number of views.')
args = parser.parse_args()

prep = MGNPrep(data_dir=args.data, num_views=args.views, cam_type=args.cam)
prep.prepare_data()
