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
        bad = h1.loss_vals[0] - good

        return torch.sum(loss_vals), torch.mean(differences), good, bad



class BatchHardTripletLoss(torch.nn.Module):

    def __init__(self, p = 2, alpha = 0.1, normalize = False, mode = "euc", softplus = False, k = 15):

        super(BatchHardTripletLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.normalize = normalize
        self.mode = mode
        self.softplus = softplus
        self.k = k

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

        if self.normalize or self.mode == "cosine":

            h1 = h1 / torch.norm(h1, dim = 1, p = self.p, keepdim = True)
            h2 = h2 / torch.norm(h2, dim = 1, p = self.p, keepdim = True)

        labels = torch.arange(0, h1.shape[0])
        labels = torch.cat((labels, labels), dim = 0)
        batch = torch.cat((h1, h2), dim = 0)

        if self.mode == "euc":
            dists = torch.norm((batch[:, None, :] - batch), dim = 2, p = self.p)
        elif self.mode == "cosine":
            dists = 1. - batch @ torch.t(batch)

        dists = torch.clamp(dists, min = 1e-7)

        mask_anchor_positive = self.get_mask(labels, positive = True).float()
        mask_anchor_negative = self.get_mask(labels, positive=False).float()

        anchor_positive_dist = mask_anchor_positive * dists
        hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)
        max_anchor_negative_dist, _ = torch.max(dists, dim=1, keepdim=True)

        anchor_negative_dist = dists + max_anchor_negative_dist * (1 - mask_anchor_negative)
        k = int(np.random.choice(range(1, self.k + 1)))
        #print(k)
        #hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)
        try:
            hardest_negative_dist, _ = torch.kthvalue(anchor_negative_dist, dim=1, k = k, keepdim=True)
        except Exception as e:
                print(e)
                print(k)
                print(h1.shape, h2.shape)
                print(labels.shape)
                print(batch.shape)
                print(dists.shape)
                print(mask_anchor_positive.shape)
                print(mask_anchor_negative.shape)
                print(anchor_positive_dist.shape)
                print(anchor_negative_dist.shape)
                exit()
        differences = hardest_positive_dist - hardest_negative_dist

        if not self.softplus:
            triplet_loss = torch.max(differences + self.alpha, torch.zeros_like(differences))
        else:
            triplet_loss = F.softplus(differences)

        relevant = triplet_loss[triplet_loss > 1e-5]
        good = (triplet_loss < 1e-5).sum()
        bad = batch.shape[0] - good
        mean_norm_squared = torch.mean(torch.norm(batch, dim = 1)**2)

        return torch.mean(relevant) + 1e-4 * mean_norm_squared, torch.mean(differences), good, bad, torch.sqrt(mean_norm_squared)



class BatchAllTripletLoss(torch.nn.Module):

    def __init__(self, p = 2, alpha = 1, normalize = False, mode = "euc", softplus = True, k = 15):

        super(BatchAllTripletLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.normalize = normalize
        self.mode = mode
        self.softplus = softplus
        self.k = k

    def get_mask(self, labels):

        diffs = labels[None, :] - torch.t(labels[None, :])
        mask = (diffs == 0).float()
        mask = mask - torch.eye(mask.shape[0])
        return mask

    def forward(self, h1, h2, h3):

        if self.normalize or self.mode == "cosine":

            h1 = h1 / torch.norm(h1, dim = 1, p = self.p, keepdim = True)
            h2 = h2 / torch.norm(h2, dim = 1, p = self.p, keepdim = True)

        labels = torch.arange(0, h1.shape[0])
        labels = torch.cat((labels, labels), dim = 0)
        batch = torch.cat((h1, h2), dim = 0)

        if self.mode == "euc":
            dists = torch.norm((batch[:, None, :] - batch), dim = 2, p = self.p)
        elif self.mode == "cosine":
            dists = 1. - batch @ torch.t(batch)

        dists = torch.clamp(dists, min = 1e-7)

        anchor_positive_dist = dists.unsqueeze(2)
        anchor_negative_dist = dists.unsqueeze(1)
        differences = anchor_positive_dist - anchor_negative_dist
        triplet_loss = differences + self.alpha

        mask = self.get_mask(labels).float()
        triplet_loss = mask * triplet_loss
        triplet_loss = torch.max(triplet_loss, torch.zeros_like(triplet_loss))

        # Remove negative losses (i.e. the easy triplets)
        relevant = triplet_loss[triplet_loss > 1e-5]
        good = (triplet_loss < 1e-5).sum()
        bad = np.prod(triplet_loss.shape) - good
        mean_norm_squared = torch.mean(torch.norm(batch, dim = 1)**2)

        return torch.mean(relevant) + 1e-4 * mean_norm_squared, torch.mean(differences), good, bad, torch.sqrt(mean_norm_squared)




if __name__ == '__main__':

    pass