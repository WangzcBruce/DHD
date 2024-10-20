# ------------------------------------------------------------------------
# DHD
# Copyright (c) 2024 Zhechao Wang. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dhd.models.sisw_core import PixelWeightedFusionSoftmax, conv_mask_uniform, \
    stack_channel, myCompressNet, attention_collaboration

class SISW(nn.Module):
    def __init__(self):
        super(SISW, self).__init__()
        self.PixelWeightedFusion = PixelWeightedFusionSoftmax(64 * 2)
        self.compress = myCompressNet(64)
        self.stack = stack_channel(1, 9, kernel_size=3, padding=1)
        self.attcoll = attention_collaboration()
        self.masker = conv_mask_uniform(64, 64, kernel_size=3, padding=1)
    def generate_entropy(self, input_agent):
        w = input_agent.shape[-2]
        h = input_agent.shape[-1]
        batch_nb = input_agent.reshape(-1, 1, 1, 1)
        stack = self.stack(input_agent).permute(2, 3, 1, 0).contiguous().reshape(-1, 9, 1, 1)
        p = F.sigmoid((stack - batch_nb)).mean(dim=1).reshape(w, h)
        return p

    def interplot_f(self, feature, masker):
        masker_t = torch.zeros_like(feature)
        masker_t[:, masker[0], masker[1]] = 1
        masker_f = masker_t[None, :, :, :].float()
        inter = self.masker(feature.unsqueeze(0), masker_f)
        return torch.squeeze(inter)
    def generate_entropy_noise(self, input_agent):
        w = input_agent.shape[-2]
        h = input_agent.shape[-1]
        batch_nb = input_agent.reshape(-1, 1, 1, 1)
        stack = self.stack(input_agent).permute(2, 3, 1, 0).contiguous().reshape(-1, 9, 1, 1)
        p = F.sigmoid((stack - batch_nb)).mean(dim=1).reshape(w, h)
        noise_level = 1e-8
        noise = torch.randn(p.size(), device='cuda') * noise_level #torch.randn(noise_shape, device=device)
        p_noise = p + noise
        return p_noise
    def forward(self, bevs, istrain=True):
        B_T, N, C, H, W = bevs.shape
        local_com_mat = bevs
        T = 3
        batch_size = B_T // T
        local_com_mat = local_com_mat.view(batch_size, T, N, C, H, W)
        local_com_mat_update = torch.ones_like(local_com_mat)
        bandwidth = []
        for b in range(batch_size):
            num_agent = N
            for t in range(T):
                for i in range(1):
                    tg_agent = local_com_mat[b, t, i]  # 256x32x32
                    neighbor_feat_list = list()
                    neighbor_feat_list.append(tg_agent)
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, t, j], 0)
                            if nb_agent.min() + nb_agent.max() == 0:
                                neighbor_feat_list.append(nb_agent[0])
                            else:
                                tg_agent_com = self.compress(torch.unsqueeze(tg_agent, 0))
                                warp_feat_trans_com = self.compress(nb_agent)
                                if istrain:
                                    tg_entropy = self.generate_entropy_noise(tg_agent_com)
                                    nb_entropy = self.generate_entropy_noise(warp_feat_trans_com)
                                else:
                                    tg_entropy = self.generate_entropy(tg_agent_com)
                                    nb_entropy = self.generate_entropy(warp_feat_trans_com)
                                selection = self.attcoll(tg_entropy,nb_entropy)
                                bandwidth.append(len(selection[0])/40000)
                                warp_feat_interplot = self.interplot_f(nb_agent.squeeze(0), selection)
                                neighbor_feat_list.append(warp_feat_interplot.squeeze(0))
                    tmp_agent_weight_list = list()
                    sum_weight = 0
                    for k in range(num_agent):
                        try:
                            cat_feat = torch.cat([tg_agent, neighbor_feat_list[k]], dim=0)
                        except:
                            print(tg_agent.shape)
                            print(neighbor_feat_list[k].shape)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                    agent_weight_list = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight.expand([64, -1, -1])
                        agent_weight_list.append(AgentWeight)
                    agent_wise_weight_feat = 0
                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k] * neighbor_feat_list[k]
                    # # feature update and hid renew
                    local_com_mat_update[b, t, i] = agent_wise_weight_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = local_com_mat_update[:, :, 0, :, :, :].view(-1, C, H, W)
        return (feat_fuse_mat, np.mean(bandwidth).item())

