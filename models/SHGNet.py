from .utils.shgnet_util import *
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, 
                          int(out_dim/2), 
                          1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), 
                          int(out_dim/2), 
                          3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = nn.Conv2d(int(out_dim/2),
                          out_dim, 
                          1, relu=False)
        self.skip_layer = nn.Conv2d(inp_dim,
                               out_dim, 
                               1,relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

# class Hourglass(nn.Module):
#     def __init__(self, n, f, bn=None, increase=0):
#         super(Hourglass, self).__init__()
#         nf = f + increase
#         self.up1 = Residual(f, f)
#         # Lower branch
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.low1 = Residual(f, nf)
#         self.n = n
#         # Recursive hourglass
#         if self.n > 1:
#             self.low2 = Hourglass(n-1, nf, bn=bn)
#         else:
#             self.low2 = Residual(nf, nf)
#         self.low3 = Residual(nf, f)
#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, x):
#         up1  = self.up1(x)
#         pool1 = self.pool1(x)
#         low1 = self.low1(pool1)
#         low2 = self.low2(low1)
#         low3 = self.low3(low2)
#         up2  = self.up2(low3)
#         return up1 + up2

class SHGNet(nn.Module):
    """docstring for SHGNet."""

    def __init__(self, opt, device):
        super(SHGNet, self).__init__()
        self.n_stacks = opt['n_stacks']
        self.out_channels = opt['out_channels']
        self.in_channels = opt['in_channels']
        self.n_joints = opt['n_joints']

        # self.HGs = Hourglass(self.in_channels, self.out_channels, self.n_joints)
        print(self.out_channels)
        self.HGs = Hourglass(1, self.out_channels).to(device)


        # self.Hourglass.append(Hourglass(self.in_channels, self.out_channels, self.n_joints))
        # self.Hourglass.append([Hourglass(self.out_channels, self.out_channels, self.n_joints) for _ in range(self.n_stacks - 1)])


    def forward(self, x):
        im = x
        x = self.HGs(x)
        return x

from torch import nn

Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x

        # print(f"shapeeeeeeeee {x.shape}")
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(3, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(3, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2
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
