from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch

class ARCHData(Dataset):


    def __init__(self, mode,  data_path = '', split_file = '',
                 batch_size = 64, num_workers = 12,**kwargs):

        self.path = data_path
        self.split = np.load(split_file)[mode]

        self.data = ['/{}'.format(self.split[i]) for i in range(len(self.split)) if
                     os.path.exists(os.path.join(data_path, '{}.npz'.format(self.split[i])))]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path  + self.data[idx] +'.npz'
        return {'path': self.data[idx]} #todo: add all data

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
