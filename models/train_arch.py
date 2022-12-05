"""This is for training tailornet meshes"""
from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
import ipdb
from .DR import DR
from .LEAP import query_leap

class Trainer(object):

    def __init__(self, occ_net, norm_net, col_net, rbf, device, train_dataset, val_dataset,  batch_size, opt):
        self.device = device
        self.occ_net = occ_net.to(device)
        self.norm_net = norm_net.to(device)
        self.col_net = col_net.to(device)
        self.rbf = rbf   #todo: not implemented yet -> Done
        self.batch_size = batch_size
        self.opt = opt

        #optimizer
        if opt['training']['optimizer'] == 'Adam':
            self.occ_opt = optim.Adam(self.occ_net.parameters(), lr=1e-4)
            self.norm_opt = optim.Adam(self.norm_net.parameters(), lr=1e-4)
            self.col_opt = optim.Adam(self.col_net.parameters(), lr=1e-4)

        if opt['training']['optimizer']  == 'Adadelta':
            self.occ_opt = optim.Adadelta(self.occ_net.parameters())
            self.norm_opt = optim.Adadelta(self.norm_net.parameters())
            self.col_opt = optim.Adadelta(self.col_net.parameters())


        if opt['training']['optimizer']  == 'RMSprop':
            self.occ_opt = optim.RMSprop(self.occ_net.parameters(), momentum=0.9)
            self.norm_opt = optim.RMSprop(self.norm_net.parameters(), momentum=0.9)
            self.col_opt = optim.RMSprop(self.col_net.parameters(), momentum=0.9)


        self.loss_3d_occ = nn.HuberLoss(reduction='mean', delta=1.0)  #todo: mean or sum?
        self.loss_3d_col = nn.L1Loss()
        self.loss_3d_norm = nn.L1Loss()  #todo: this or angle based loss?
        self.loss_2d_norm =  nn.L1Loss()  #todo: GT data? -> Done
        self.loss_2d_col =  nn.L1Loss()  #todo: this or perceptual loss?

        self.renderer = DR(opt['renderer'], self.device)

        root_dir = opt['experiment']['root_dir']
        exp_name = opt['experiment']['exp_name']
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        #self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.exp_path = '{}/{}/'.format(root_dir, exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.train_min = None
        self.bacth_size = opt['training']['batch_size']


    def train_step(self,batch, ep=None):

        self.occ_net.train()
        self.occ_opt.zero_grad()
        self.norm_net.train()
        self.norm_opt.zero_grad()
        self.col_net.train()
        self.col_opt.zero_grad()

        loss, loss_dict = self.compute_loss(batch, ep)
        loss.backward()

        self.occ_opt.step()
        self.norm_opt.step()
        self.col_opt.step()

        return loss.item(), loss_dict

    @staticmethod
    def loss_total( loss_dict, weight_dict=None):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = loss_dict[k]  #todo: add weights here, maybe we will need weighted loss, other redundant

        tot_loss = list(w_loss.values())
        return torch.stack(tot_loss).sum()

    def compute_loss(self, batch, ep=None):
        """one forward pass and loss calculation for a batch"""
        device = self.device
        #read image and pose landmarks(for now both angles and joint location) and GT data
        im = batch.get("im_mesh").to(device)  #Is this one channel or RBG image? -> Rendered mesh or rendered pointcloud
        points = batch.get("points").squeeze(1).to(device)  #todo: canonical or posed?
        theta = batch.get("theta").to(device)
        joints = batch.get("joints").to(device)
        beta = batch.get("beta").to(device)  #todo: do we need this? maybe yes for new model

        smpl_body = {}
        smpl_body['betas'] = beta
        smpl_body['pose_body'] = theta[:, 3:66]
        smpl_body['pose_hand'] = torch.zeros((self.bacth_size, 90))
        smpl_body['gender'] = 'neutral'\

        #supervisioon data
        gt_occ = batch.get("gt_occ").squeeze(1).to(device)
        gt_norm = batch.get("gt_norm").squeeze(1).to(device)
        gt_col = batch.get("gt_col").squeeze(1).to(device)
        im_norm = batch.get("im_norm").to(device)
        azimuth = batch.get("azimuth").to(device)

        print(f"shape of norms: {gt_norm.shape}")
        print(f"shape of cols: {gt_col.shape}")
        print(f"shape of occ: {gt_occ.shape}")
        print(f"shape of ims {im_norm.shape}")

        #use leap for weights and canonicalization
        # point_weights, can_points = query_leap(points, self.opt['leap_path'], smpl_body, self.opt['body_model_path'], self.batch_size, self.device, canonical_points=False, vis=False)

        #create spatial feature using landmarks(joints in our case)
        f_ps = self.rbf(points, joints)

        pred_occ, f_po = self.occ_net(im, f_ps, points, azimuth)  #predicted occupancy and pixel aligned image features
        # self.cuda_status('after occ')

        pred_norm, f_pn = self.norm_net(im, f_ps, f_po, points, azimuth)  #predicted normal and pixel aligned image features
        # self.cuda_status('after norm')

        pred_col, f_pc = self.col_net(im, f_ps, f_po, f_pn, points, azimuth)  #predicted normal and pixel aligned image features
        # self.cuda_status('after color')

        ###calculate all the loss here
        loss_dict = {}
        loss_dict['3d_occ'] = self.loss_3d_occ(pred_occ, gt_occ)
        print(f"pred occ shape: {pred_occ.shape}")
        print(f"gt occ shape: {gt_occ.shape}")
        print(f"pred norm shape: {pred_occ.shape}")
        print(f"gt norm shape: {gt_occ.shape}")
        loss_dict['3d_norm']  = self.loss_3d_norm(pred_norm, gt_norm)
        loss_dict['3d_col'] = self.loss_3d_col(pred_col, gt_col)

        #render predicted images
        pred_im_col, pred_im_norm, pred_im_occ = self.renderer.render(points, pred_col, pred_occ, pred_norm)

        loss_dict['2d_col'] = self.loss_2d_col(pred_im_col, im)
        loss_dict['2d_norm'] = self.loss_2d_norm(pred_im_norm, im_norm)

        total_loss = self.loss_total(loss_dict)
        return total_loss, loss_dict

    def train_model(self, epochs, eval=True):
        loss = 0
        start = self.load_checkpoint()
        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()
            if epoch % 100 == 0:   #save every 100 epochs
                self.save_checkpoint(epoch)

            for batch in train_data_loader:
                loss, loss_dict = self.train_step(batch, epoch)
                print("Current loss: {},   ".format(loss))
                print("Individual loss: ", loss_dict)
                sum_loss += loss
            batch_loss = sum_loss / len(train_data_loader)

            if self.train_min is None:
                self.train_min = batch_loss
            if batch_loss < self.train_min:
                self.save_checkpoint(epoch)
                for path in glob(self.exp_path + 'train_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'train_min={}'.format(epoch), [epoch, batch_loss])


            if eval:
                val_loss = self.compute_val_loss(epoch)
                print('validation loss:   ', val_loss)
                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    self.save_checkpoint(epoch)
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, batch_loss])
                self.writer.add_scalar('val loss batch avg', val_loss, epoch)

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', batch_loss, epoch)
            for k in loss_dict.keys():
                self.writer.add_scalar('training loss for last batch {} avg'.format(k), loss_dict[k] , epoch)
                #self.writer.add_scalar('training loss for batch {} avg'.format(k), loss_terms[k] , epoch)  #todo: add this if needed

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_occ_dict': self.occ_net.state_dict(),
                        'optimizer_occ_state_dict': self.occ_opt.state_dict(),
                        'model_state_col_dict': self.col_net.state_dict(),
                        'optimizer_col_state_dict': self.col_opt.state_dict(),
                        'model_state_norm_dict': self.norm_net.state_dict(),
                        'optimizer_norm_state_dict': self.norm_opt.state_dict()}, path,
                       _use_new_zipfile_serialization=False)


    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if True or len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.occ_opt.load_state_dict(checkpoint['model_state_occ_dict'])
        self.occ_opt.load_state_dict(checkpoint['optimizer_occ_state_dict'])
        self.col_net.load_state_dict(checkpoint['model_state_col_dict'])
        self.col_opt.load_state_dict(checkpoint['optimizer_col_state_dict'])
        self.norm_net.load_state_dict(checkpoint['model_state_norm_dict'])
        self.norm_opt.load_state_dict(checkpoint['optimizer_norm_state_dict'])

        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self, ep):

        self.col_net.eval()
        self.occ_net.eval()
        self.norm_net.eval()
        # return 5.
        sum_val_loss = 0
        num_batches = 15

        val_data_loader = self.val_dataset.get_loader()
        for batch in val_data_loader:
            loss, _= self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
        return sum_val_loss /len(val_data_loader)

    def cuda_status(self, phase):
        t = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        r = torch.cuda.memory_reserved(0) / (1024**2)
        a = torch.cuda.memory_allocated(0) / (1024**2)
        f = r-a  # free inside reserved

        print(phase)
        print(f"Total Memory -> {t}")
        print(f"Reserved Memory -> {r}")
        print(f"Allocated Memory -> {a}")
        print(f"Free Memory -> {f}")
