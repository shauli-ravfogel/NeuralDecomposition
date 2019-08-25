import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn
import random

class TripletLoss(torch.nn.Module):

    def __init__(self, p = 2, alpha = 0.2, cosine = False):

        super(TripletLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.cosine = cosine

    def forward(self, h1, h2, h3):

        h1 = h1 / torch.norm(h1, dim = 1, p = self.p, keepdim = True)
        h2 = h2 / torch.norm(h2, dim = 1, p = self.p, keepdim = True)
        h3 = h3 / torch.norm(h3, dim = 1, p = self.p, keepdim = True)

        if not self.cosine:
            dis_positive = torch.norm(h1 - h2, dim = 1, p = self.p)
            dis_negative = torch.norm(h1 - h3, dim = 1, p = self.p)

        else:
            dis_positive = 1. - torch.nn.functional.cosine_similarity(h1, h2)
            dis_negative = 1. - torch.nn.functional.cosine_similarity(h1, h3)
            dis_positive = torch.clamp(dis_positive, 1e-5, 0.999)
            dis_negative = torch.clamp(dis_negative, 1e-5, 0.999)
        #print(dis_positive, dis_negative, dis_positive.shape, dis_negative.shape)

        differences = dis_positive - dis_negative
        loss_vals = torch.max(torch.zeros_like(dis_positive), differences + self.alpha)
        #loss_vals = F.softplus(dis_positive - dis_negative)
        good = (loss_vals < 1e-5).sum()
        bad = h1.shape[0] - good

        return torch.sum(loss_vals), torch.mean(differences), good, bad



class BatchHardTripletLoss(torch.nn.Module):

    def __init__(self, p = 2, alpha = 0.1, normalize = False, cosine = False, softplus = False):

        super(BatchHardTripletLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.normalize = normalize
        self.cosine = cosine
        self.softplus = softplus

    def get_mask(self, labels, positive = True):

        diffs =  labels[None, :] - torch.t(labels[None, :])

        if positive:

            mask = diffs == 0

        else:

            mask = diffs != 0

        if positive:
            mask[range(len(mask)), range(len(mask))] = 0
        return mask



    def forward(self, h1, h2, h3):

        if self.normalize or self.cosine:

            h1 = h1 / torch.norm(h1, dim = 1, p = self.p, keepdim = True)
            h2 = h2 / torch.norm(h2, dim = 1, p = self.p, keepdim = True)

        labels = torch.arange(0, h1.shape[0]).cuda()
        labels = torch.cat((labels, labels), dim = 0)
        batch = torch.cat((h1, h2), dim = 0)

        if not self.cosine:
            dists = torch.norm((batch[:, None, :] - batch), dim = 2, p = self.p)
        else:
            dists = 1. - batch @ torch.t(batch)

        dists = torch.clamp(dists, min = 1e-7)

        mask_anchor_positive = self.get_mask(labels, positive = True).float()
        mask_anchor_negative = self.get_mask(labels, positive=False).float()

        anchor_positive_dist = mask_anchor_positive * dists
        hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)
        max_anchor_negative_dist, _ = torch.max(dists, dim=1, keepdim=True)

        anchor_negative_dist = dists + max_anchor_negative_dist * (1 - mask_anchor_negative)
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

        differences = hardest_positive_dist - hardest_negative_dist

        if not self.softplus:
            triplet_loss = torch.max(differences + self.alpha, torch.zeros_like(differences))
        else:
            triplet_loss = F.softplus(differences)

        relevant = triplet_loss[triplet_loss > 1e-5]
        good = (triplet_loss < 1e-5).sum()
        bad = batch.shape[0] - good
        mean_norm_squared = torch.mean(torch.norm(batch, dim = 1)**2)

        return torch.mean(relevant) + 0 * mean_norm_squared, torch.mean(differences), good, bad, torch.sqrt(mean_norm_squared)

if __name__ == '__main__':

    pass