# import torch.nn as nn
#
# class Hourglass(nn.Module):
#     """docstring for Hourglass."""
#
#     def __init__(self, in_channels, out_channels, n_joints):
#         super(Hourglass, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_joints = n_joints
#
#         self.n_cube = 3
#         self.n_block = 3
#
#         self.pre = PreHourglass(self.in_channels, self.out_channels)
#         self.post = PostHourglass(self.out_channels, self.out_channels, self.n_joints)
#
#         self.down_array = nn.ModuleList([DownCube(self.out_channels, self.out_channels) for _ in range(self.n_cube)])
#         self.up_array = nn.ModuleList([UpCube(scale=2) for _ in range(self.n_cube)])
#
#         self.feat_ex = FeatCube(self.n_block, self.out_channels)
#
#         self.skip_upsample = nn.UpsamplingNearest2d(scale_factor=2)
#
#     def forward(self, x):
#
#         input = x
#
#         res = []
#
#         feat_pre, res_pre = self.pre(x)
#         res.append(res_pre)
#
#         feat = feat_pre
#
#         for i, cube in enumerate(self.down_array):
#             f, r = cube(feat)
#             res.append(r)
#             feat = f
#
#         feat = self.feat_ex(feat, res[-1])
#
#         for i, cube in enumerate(self.up_array):
#             feat = cube(feat, res[-i + 3])
#
#         feat = self.skip_upsample(feat)
#
#         output = self.post(feat, input)
#
#         return output
#
#
#
# class PostHourglass(nn.Module):
#     """docstring for PostHourglass."""
#
#     def __init__(self, in_channels, out_channels, n_joints):
#         super(PostHourglass, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_joints = n_joints
#
#         self.bottle_neck = BottleneckBlock(in_channels=in_channels, out_channels=out_channels)
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU()
#         )
#         self.heatmap_conv = nn.Conv2d(self.out_channels, self.n_joints, kernel_size=1, stride=1, padding=0)
#         self.output_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
#         self.heatmap_conv_reverse = nn.Conv2d(self.n_joints, self.out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x, input):
#         x = self.bottle_neck(x)
#         x = self.block(x)
#
#         heatmap = self.heatmap_conv(x)
#         out = self.output_conv(x)
#
#         heatmap_rev = self.heatmap_conv_reverse(heatmap)
#
#         output = heatmap_rev + out + input
#         return output
#
#
# class PreHourglass(nn.Module):
#     """docstring for PreHourglass"""
#
#     def __init__(self, in_channels, out_channels):
#         super(PreHourglass, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.pre_res = nn.Sequential(
#
#             nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             BottleneckBlock(in_channels=64, out_channels=128),
#         )
#
#         self.feat_extract = nn.Sequential(
#
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             BottleneckBlock(in_channels=128, out_channels=128),
#             BottleneckBlock(in_channels=128, out_channels=out_channels),
#         )
#
#         self.res = nn.Sequential(
#
#             BottleneckBlock(in_channels=128, out_channels=128),
#             BottleneckBlock(in_channels=128, out_channels=out_channels),
#         )
#
#     def forward(self, x):
#
#         x = self.pre_res(x)
#
#         feat = self.feat_extract(x)
#         residual = self.res(x)
#
#         return feat, residual
#
# class DownCube(nn.Module):
#     """docstring for DownCube"""
#
#     def __init__(self, in_channels, out_channels):
#         super(DownCube, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.bottle_neck = BottleneckBlock(in_channels=in_channels, out_channels=out_channels)
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.res = BottleneckBlock(in_channels=out_channels, out_channels=out_channels)
#
#     def forward(self, x):
#
#         x = self.bottle_neck(x)
#
#         feat = self.max_pool(x)
#         residual = self.res(x)
#
#         return feat, residual
#
# class UpCube(nn.Module):
#     """docstring for UpCube"""
#
#     def __init__(self, scale):
#         super(UpCube, self).__init__()
#
#         self.upsample = nn.UpsamplingNearest2d(scale_factor=scale)
#
#     def forward(self, x, res):
#
#         x = self.upsample(x)
#         return x + res
#
# class FeatCube(nn.Module):
#     """docstring for FeatCube"""
#
#     def __init__(self, n_block, channels):
#         super(FeatCube, self).__init__()
#
#         self.channels = channels
#         self.n_block = n_block
#
#         self.blocks = nn.ModuleList([BottleneckBlock(in_channels=channels, out_channels=channels) for _ in range(self.n_block)])
#
#     def forward(self, x, res):
#
#         for block in self.blocks:
#             x = block(x)
#
#         return x + res
#
# class BottleneckBlock(nn.Module):
#     """docstring for BottleneckBlock"""
#
#     def __init__(self, in_channels, out_channels):
#         super(BottleneckBlock, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#         self.block = nn.Sequential(
#
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
#
#             nn.BatchNorm2d(out_channels // 2),
#             nn.ReLU(),
#             nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
#
#             nn.BatchNorm2d(out_channels // 2),
#             nn.ReLU(),
#             nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
#         )
#
#     def forward(self, x):
#
#         if self.in_channels != self.out_channels:
#             identity = self.skip_conv(x)
#         else:
#             identity = x
#
#         x = self.block(x)
#
#         return x + identity
#
# #
# # class ResidualBlock(nn.Module):
# #     """docstring for ResBlock."""
# #
# #     def __init__(self, in_channels, out_channels):
# #         super(ResidualBlock, self).__init__()
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
# #
# #         self.res_block = nn.Sequential(
# #
# #             nn.BatchNorm2d(in_channels),
# #             nn.ReLU(),
# #             nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
# #
# #             nn.BatchNorm2d(out_channels // 2),
# #             nn.ReLU(),
# #             nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
# #
# #             nn.BatchNorm2d(out_channels // 2),
# #             nn.ReLU(),
# #             nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
# #         )
# #
# #     def forward(self, x):
# #         if self.in_channels != self.out_channels:
# #             skip = self.conv_skip(x)
# #         else:
# #             skip = x
# #
# #         return skip + self.res_block(x)
# #
# # class Hourglass(nn.Module):
# #     """docstring for Hourglass."""
# #
# #     def __init__(self, depth, n_channels, n_module):
# #
# #         super(Hourglass, self).__init__()
# #
# #         self.depth = depth
# #         self.n_channels = n_channels
# #         self.n_module = n_module
# #
# #         reslist_1 = [ResidualBlock(n_channels, n_channels) for _ in range(self.n_module)]
# #         reslist_2 = [ResidualBlock(n_channels, n_channels) for _ in range(self.n_module)]
# #         reslist_3 = [ResidualBlock(n_channels, n_channels) for _ in range(self.n_module)]
# #
# #         self.resnet1 = nn.Sequential(*reslist_1)
# #         self.resnet2 = nn.Sequential(*reslist_2)
# #         self.resnet3 = nn.Sequential(*reslist_3)
# #
# #         self.subHourglass = None
# #         self.res_waist = None
# #
# #         if self.depth > 1:
# #             self.subHourglass = Hourglass(self.depth - 1, self.n_channels, self.n_module)
# #         else:
# #             res_waist_list = [ResidualBlock(n_channels, n_channels) for _ in range(self.n_module)]
# #             self.res_waist = nn.Sequential(*res_waist_list)
# #
# #     def forward(self, x):
# #         up = self.resnet1(x)
# #
# #         down1 = nn.MaxPool2d(kernel_size=2, stride=2)(x)
# #         down1 = self.resnet2(down1)
# #
# #         if self.depth > 1:
# #             down2 = self.subHourglass(down1)
# #         else:
# #             down2 = self.res_waist(down1)
# #
# #         down3 = self.resnet3(down2)
# #
# #         down = nn.UpsamplingNearest2d(scale_factor=2)(down3)
# #
# #         return up + down
