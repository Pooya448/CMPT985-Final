from .utils.shgnet_util import *
import torch.nn as nn

class SHGNet(nn.Module):
    """docstring for SHGNet."""

    def __init__(self, opt):
        super(SHGNet, self).__init__()
        self.n_stacks = opt['n_stacks']
        self.out_channels = opt['out_channels']
        self.in_channels = opt['in_channels']
        self.n_joints = opt['n_joints']

        self.HGs = Hourglass(self.in_channels, self.out_channels, self.n_joints)

        # self.Hourglass.append(Hourglass(self.in_channels, self.out_channels, self.n_joints))
        # self.Hourglass.append([Hourglass(self.out_channels, self.out_channels, self.n_joints) for _ in range(self.n_stacks - 1)])


    def forward(self, x):
        im = x
        x = HG(x)
        return x


#
# class SHGNet(nn.Module):
#     """docstring for SHGNet."""
#
#     def __init__(self, opt):
#         super(SHGNet, self).__init__()
#
#         self.n_stacks = opt['n_stacks']
#         self.n_modules = opt['n_modules']
#         self.depth = opt['depth']
#         self.out_channels = opt['out_channels']
#         self.in_channels = opt['in_channels']
#         self.n_joints = opt['n_joints']
#
#         self.pre_hourglass = nn.Sequential(
#             nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             ResidualBlock(in_channels=64, out_channels=128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             ResidualBlock(128, 128),
#             ResidualBlock(128, self.out_channels)
#         )
#
#         self.post_hourglass = nn.Sequential(
#             nn.Conv2d(self.n_joints, self.out_channels, kernel_size=1, stride=1, padding=0),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.BatchNorm2d(num_features=self.out_channels),
#             nn.ReLU(),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.BatchNorm2d(num_features=self.out_channels),
#             nn.ReLU(),
#         )
#
#         self.hgArray = nn.ModuleList([])
#         self.llArray = nn.ModuleList([])
#         self.linArray = nn.ModuleList([])
#         self.htmapArray = nn.ModuleList([])
#         self.llBarArray = nn.ModuleList([])
#         self.htmapBarArray = nn.ModuleList([])
#
#         for i in range(self.n_stacks):
#             self.hgArray.append(Hourglass(self.depth, self.out_channels, self.n_modules))
#
#             self.llArray.append(
#                 nn.ModuleList([ResidualBlock(self.out_channels, self.out_channels) for _ in range(self.n_modules)]))
#
#             self.linArray.append(self.lin(self.out_channels, self.out_channels))
#             self.htmapArray.append(nn.Conv2d(self.out_channels, self.n_joints, kernel_size=1, stride=1, padding=0))
#
#         for i in range(self.n_stacks - 1):
#             self.llBarArray.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0))
#             self.htmapBarArray.append(nn.Conv2d(self.n_joints, self.out_channels, kernel_size=1, stride=1, padding=0))
#
#     def forward(self, x):
#         inter = self.pre_hourglass(x)
#         outHeatmap = []
#
#         for i in range(self.n_stacks):
#             ll = self.hgArray[i](inter)
#             for j in range(self.n_modules):
#                 ll = self.llArray[i][j](ll)
#             ll = self.linArray[i](ll)
#             htmap = self.htmapArray[i](ll)
#             outHeatmap.append(htmap)
#
#             if i < self.n_stacks - 1:
#                 ll_ = self.llBarArray[i](ll)
#                 htmap_ = self.htmapBarArray[i](htmap)
#                 inter = inter + ll_ + htmap_
#
#         out = self.post_hourglass(outHeatmap[-1])
#         return out
#
#     def lin(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU()
#         )
