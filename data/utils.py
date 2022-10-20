import os
import cv2
import sys
import torch

from pytorch3d.io import load_objs_as_meshes, load_obj
from torchvision.io import read_image, write_png
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes

from pytorch3d.renderer import (
    TexturesUV,
    TexturesVertex,
    Textures,
)

def loc(points, device):
    vert = points.detach().cpu().numpy().reshape((-1, 3))
    bb_min, bb_max = np.min(vert, axis=0), np.max(vert, axis=0)
    loc = np.array([(bb_min[0] + bb_max[0]) / 2, (bb_min[1] + bb_max[1]) / 2, (bb_min[2] + bb_max[2]) / 2])

    return torch.from_numpy(loc).to(device=device, dtype=torch.float32)


def sample_from_mesh(mesh, n=100000):
    return sample_points_from_meshes(meshes=mesh, num_samples=n, return_normals=True, return_textures=True)

def normalize_mesh_target(mesh, target_points, device):

    bbox = mesh.get_bounding_boxes().squeeze()

    mesh_loc = torch.Tensor([(bbox[0, 0] + bbox[0, 1]) / 2, (bbox[1, 0] + bbox[1, 1]) / 2, (bbox[2, 0] + bbox[2, 1]) / 2]).to(device=device, dtype=torch.float32)
    target_loc = loc(target_points, device)
    diff = target_loc - mesh_loc

    diff = diff.unsqueeze(dim=0).repeat([mesh.verts_packed().shape[0], 1])
    n_mesh = mesh.offset_verts(diff)

    return n_mesh

def normalize_mesh(mesh, device):

    bbox = mesh.get_bounding_boxes().squeeze()

    loc = -1 * torch.Tensor([(bbox[0, 0] + bbox[0, 1]) / 2, (bbox[1, 0] + bbox[1, 1]) / 2, (bbox[2, 0] + bbox[2, 1]) / 2]).to(device=device, dtype=torch.float32)
    loc = loc.unsqueeze(dim=0).repeat([mesh.verts_packed().shape[0], 1])

    n_mesh = mesh.offset_verts(loc)

    return n_mesh

def load_tex_mesh(obj, tex, device):

    tex_im = read_image(tex).permute((1, 2, 0)).to(device, dtype=torch.float32)

    verts, faces, aux = load_obj(obj, device=device)

    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_im = tex_im[None, ...]  # (1, H, W, 3)

    tex_uv = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=tex_im)

    tex_mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex_uv)

    tex_mesh = normalize_mesh(tex_mesh, device)

    return tex_mesh
