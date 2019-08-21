import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn
import random

class TripletLoss(torch.nn.Module):

    def __init__(self, p = 2, alpha = 100, cosine = False):

        super(TripletLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.cosine = cosine

    def forward(self, h1, h2, h3, sent1, sent2):

        #word_differences = 1. - torch.cosine_similarity(sent1, sent2, dim = 2)
        word_differences = torch.norm(sent1 - sent2, dim = 2, p = self.p)

        if not self.cosine:
            dis_positive = torch.norm(h1 - h2, dim = 1, p = self.p)
            dis_negative = torch.norm(h1 - h3, dim = 1, p = self.p)

        else:
            dis_positive = 1. - torch.nn.functional.cosine_similarity(h1, h2)
            dis_negative = 1. - torch.nn.functional.cosine_similarity(h1, h3)
            dis_positive = torch.clamp(dis_positive, 1e-5, 0.999)
            dis_negative = torch.clamp(dis_negative, 1e-5, 0.999)
        #print(dis_positive, dis_negative, dis_positive.shape, dis_negative.shape)

        loss_vals = torch.max(torch.zeros_like(dis_positive), dis_positive - dis_negative + self.alpha)
        good = (loss_vals < 1e-5).sum()
        bad = h1.shape[0] - good

        return torch.sum(loss_vals) + 0.5 * torch.mean(word_differences), good, bad

if __name__ == '__main__':

    pass