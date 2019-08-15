import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn

class TripletLoss(torch.nn.Module):

    def __init__(self, metric = 2, alpha = 0.5):

        super(TripletLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha

    def forward(self, h1, h2, h3):

        dis_positive = torch.norm(h1 - h2, dim = 1, p = self.metric)
        dis_negative = torch.norm(h1 - h3, dim = 1, p = self.metric)
        #print(dis_positive, dis_negative, dis_positive.shape, dis_negative.shape)

        loss_vals = torch.max(torch.zeros_like(dis_positive), dis_positive - dis_negative + self.alpha)
        good = (loss_vals < 1e-5).sum()
        bad = h1.shape[0] - good

        return torch.sum(loss_vals), good, bad

if __name__ == '__main__':

    pass