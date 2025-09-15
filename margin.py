import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list.float()))
        m_list = m_list * (max_m / torch.max(m_list))
        self.m_list = m_list.to(device)
        self.s = s  # 缩放因子
        self.device = device

    def forward(self, x, target):

        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), True)

        batch_m = self.m_list[target]
        batch_m = batch_m.view(-1, 1).repeat(1, x.size(1))


        x_m = x - batch_m * index.float()
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s * output, target)