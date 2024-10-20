# ------------------------------------------------------------------------
# DHD
# Copyright (c) 2024 Zhechao Wang. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self,channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1

import numpy as np
class conv_mask_uniform(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, interplate='none'):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        # self.mask = masker # shape [1, 256, 32, 32] 有值的地方call 1, 没值的地方call 0

        self.interpolate = interplate
        self.r = 7
        self.padding_interpolate = 3

        self.Lambda = nn.Parameter(torch.tensor(3.0))
        square_dis = np.zeros((self.r, self.r))
        center_point = (square_dis.shape[0] // 2, square_dis.shape[1] // 2)

        for i in range(square_dis.shape[0]):
            for j in range(square_dis.shape[1]):
                square_dis[i][j] = (i - center_point[0]) ** 2 + (j - center_point[1]) ** 2

        square_dis[center_point[0]][center_point[1]] = 100000.0

        self.square_dis = nn.Parameter(torch.Tensor(square_dis), requires_grad=False)

    def forward(self, x, mask):

        y = super().forward(x)

        # y = x

        self.out_h, self.out_w = y.size(-2), y.size(-1)

        kernel = (-(self.Lambda ** 2) * self.square_dis.detach()).exp()
        kernel = kernel / (kernel.sum() + 10 ** (-5))

        kernel = kernel.expand((self.out_channels, 1, kernel.size(0), kernel.size(1)))
        interpolated = F.conv2d(y * mask, kernel, stride=1, padding=self.padding_interpolate, groups=self.out_channels)

        out = y * mask + interpolated * (1 - mask)

        return out


class CompressNet(nn.Module):
    def __init__(self, channel):
        super(CompressNet, self).__init__()
        self.conv1_1 = nn.Conv2d(channel, 64, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.conv1_2(x_1))

        return x_1
class stack_channel(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, interplate='none'):
        super(stack_channel, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=bias)

        # 这里把定常的核函数确定好
        square_dis = np.zeros((out_channels, kernel_size, kernel_size))
        # 将对应的索引打出来
        for i in range(out_channels):
            square_dis[i, i // 3, i % 3] = 1

        self.square_dis = nn.Parameter(torch.Tensor(square_dis), requires_grad=False)

    def forward(self, x):
        kernel = self.square_dis.detach().unsqueeze(1)
        stack = F.conv2d(x, kernel, stride=1, padding=1, groups=1)

        return stack

class myCompressNet(nn.Module):
    def __init__(self, channel):
            super(myCompressNet, self).__init__()
            self.conv1_1 = nn.Conv2d(channel, channel // 8, kernel_size=1, stride=1, padding=0)
            self.bn1_1 = nn.BatchNorm2d(channel // 8)

            self.conv1_2 = nn.Conv2d(channel // 8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
            x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
            x_1 = F.relu(self.conv1_2(x_1))

            return x_1
class LayerNorm(nn.Module):
    """
    From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm_a = LayerNorm(dim*2, eps=1e-6, data_format="channels_first")
        self.norm_v = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.a = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, a, v):
        a = self.norm_a(a)
        a = self.a(a.unsqueeze(0))#.squeeze(0)#self.a(a)
        v = self.norm_v(v)
        v = self.v(v.unsqueeze(0))#.squeeze(0)
        att = self.proj(a * v)

        return att
class attention_collaboration(nn.Module):
    def __init__(self, dim=64):
        super(attention_collaboration, self).__init__()
        self.convatt = ConvMod(dim)
        self.masker = conv_mask_uniform(64, 64, kernel_size=3, padding=1)
    def interplot_f(self, feature, masker):
        masker_t = torch.zeros_like(feature)
        masker_t[:, masker[0], masker[1]] = 1
        masker_f = masker_t[None, :, :, :].float()
        inter = self.masker(feature.unsqueeze(0), masker_f)
        return torch.squeeze(inter)
    def forward(self, ego_conf, nb_conf, delta=0.25):
        w = ego_conf.shape[-2]
        h = ego_conf.shape[-1]
        ego_request = 1 - ego_conf  # 取出不自信的内容
        att_map = ego_request * nb_conf  # 自己不自信的*其他人自信的
        K = int(h * w * delta)
        _, indices = torch.topk(att_map.reshape(-1), k=K, largest=True, sorted=False)
        cols = indices % h
        rows = indices // h
        mask = (rows, cols)
        return mask

