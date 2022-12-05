from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
from torchvision.io import read_image
from .utils import *

class MGNData(Dataset):


    def __init__(self, mode, data_path='', split_file ='', batch_size = 64, num_workers = 12, split = False, num_view=360, n_subject=None, **kwargs):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.num_view = num_view
        self.angle_step = 360 / self.num_view
        self.n_sample = 2000
        self.n = 50000

        if split == True:
            split_file = self.split_data()

        self.split = np.load(split_file)[self.mode]

        if n_subject == None:
            self.n_subject = len(self.split)
        else:
            self.n_subject = n_subject

        self.subjects = [os.path.join(self.data_dir, self.split[i]) for i in range(self.n_subject) if
                     os.path.exists(os.path.join(self.data_dir, self.split[i]))]

        if self.n_subject == 1:
            print(f"Subject id - {self.mode} : {self.split[0]}")

    def __len__(self):
        return self.n_subject * self.num_view

    def __getitem__(self, idx):

        subject_id = idx // self.num_view
        view_id = idx % self.num_view
        view_angle = int(view_id * self.angle_step)
        subject_path = self.subjects[subject_id]

        reg_path = os.path.join(subject_path, "registration.pkl")

        with open(reg_path, 'rb') as f:
            reg_data = pickle.load(f, encoding='latin1')

        beta = torch.from_numpy(reg_data['betas'])
        pose = torch.from_numpy(reg_data['pose'])
        trans = torch.from_numpy(reg_data['trans'])
        # gender = reg_data['gender']

        mesh_im_path = os.path.join(subject_path, f"mesh_render_360/{view_angle}.png")
        norm_im_path = os.path.join(subject_path, f"norm_render_360/{view_angle}.png")
        # col_im_path = os.path.join(subject_path, f"point_color_render_360/{view_angle}.png")

        mesh_im = read_image(mesh_im_path).float() # C x W x H
        norm_im = read_image(norm_im_path).float()

        # obj_path = os.path.join(subject_path, "scan.obj")
        # tex_path = os.path.join(subject_path, "scan_tex.jpg")
        #
        # mesh = load_tex_mesh(obj_path, tex_path, self.device)

        points = torch.load(os.path.join(subject_path, "points.pt"), map_location=self.device)
        normals = torch.load(os.path.join(subject_path, "normals.pt"), map_location=self.device)
        colors = torch.load(os.path.join(subject_path, "colors.pt"), map_location=self.device)
        # sdf = torch.load(os.path.join(subject_path, "sdf.pt"))

        perm = torch.randperm(self.n)
        idxs = perm[:self.n_sample]

        # print("points", points.shape)
        # points = points[idxs]
        # normals = normals[idxs]
        # colors = colors[idxs]
        points = points[:, idxs, :]
        # print("after points", points.shape)
        # print("normals shape", normals.shape, idxs)
        normals = normals[:, idxs, :]
        colors = colors[:, idxs, :]
        # sdf = sdf[idxs]

        # col_im = read_image(col_im_path).permute((1, 2, 0))

        # points_path = os.path.join(subject_path, "points.pt")
        # normals_path = os.path.join(subject_path, "normals.pt")
        # colors_path = os.path.join(subject_path, "colors.pt")
        #
        # points = torch.load(points_path, map_location='cpu')
        # normals = torch.load(normals_path, map_location='cpu')
        # colors = torch.load(colors_path, map_location='cpu')

        occ = torch.ones((self.n_sample, 1))
        joints = torch.zeros((171, 3))

        data_dict = {

        ### SMPL Params
            'beta' : beta,
            'theta' : pose,
            'trans' : trans,
            # 'gender' : gender,

        ### 2D Data
            'im_mesh' : mesh_im,
            'im_norm' : norm_im,

        ### 3D Data
            'points' : points,
            'gt_norm' : normals,
            'gt_col' : colors,
            'gt_occ' : occ,

        ### Render & Projection
            'azimuth' : view_angle,
            'elevation' : 0,
            'distance' : 1,

            'joints' : joints,
        }
        return data_dict

    def split_data(self, ratio=0.2):

        subjects = [x for x in os.listdir(self.data_dir) if str.isnumeric(x)]
        train_subjects, val_subjects = train_test_split(subjects, test_size=ratio, shuffle=True)

        split_path = os.path.join(self.data_dir, 'split_file.npz')
        np.savez(split_path, train=np.array(train_subjects), val=np.array(val_subjects))

        return split_path

    def get_loader(self, shuffle=True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
