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


class HardNegativeSampler(object):

    def __init__(self, k = 5):

        self.k = k

    def _get_mask(self, labels, positive = True):

        diffs =  labels[None, :] - (labels[None, :]).T

        if positive:

            mask = diffs == 0

        else:

            mask = diffs != 0

        if positive:
            mask[range(len(mask)), range(len(mask))] = 0
        return mask

    def get_distances(self, labels, dists):

        mask_anchor_positive = self._get_mask(labels, positive = True)
        mask_anchor_negative = self._get_mask(labels, positive = False)
        anchor_positive_dist = mask_anchor_positive * dists
        hardest_positive_idx = np.argmax(anchor_positive_dist, axis=1)
        max_anchor_negative_dist = np.max(dists, axis=1, keepdims=True)

        anchor_negative_dist = dists + max_anchor_negative_dist * (1 - mask_anchor_negative)
        k = int(np.random.choice(range(1, self.k + 1)))
        hardest_negatives_idx = np.argpartition(anchor_negative_dist, k, axis = 1)[:,k]

        return hardest_positive_idx, hardest_negatives_idx


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    dist[dist != dist] = 0  # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)



class BatchHardTripletLoss2(torch.nn.Module):

    def __init__(self, p = 2, alpha = 0.1, normalize = False, mode = "euc", final = "softplus", k = 5):

        super(BatchHardTripletLoss2, self).__init__()
        self.p = p
        self.alpha = alpha
        self.normalize = normalize
        self.mode = mode
        self.final = final
        self.sampler = HardNegativeSampler(k = k)
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

    def forward(self, h1, h2, sent1, sent2, index, evaluation = False):

        if self.normalize or self.mode == "cosine":

            h1 = h1 / torch.norm(h1, dim = 1, p = self.p, keepdim = True)
            h2 = h2 / torch.norm(h2, dim = 1, p = self.p, keepdim = True)

        sent1, sent2 = np.array(sent1, dtype = object), np.array(sent2, dtype = object)
        labels = torch.arange(0, h1.shape[0])#.cuda()
        labels = torch.cat((labels, labels), dim = 0)
        batch = torch.cat((h1, h2), dim = 0)

        sents = np.concatenate((sent1, sent2), axis = 0)

        if self.mode == "euc":
            #dists = torch.norm((batch[:, None, :] - batch), dim = 2, p = self.p)
            dists = pairwise_distances(batch)
        elif self.mode == "cosine":
            dists = 1. - batch @ torch.t(batch)

        dists = torch.clamp(dists, min = 1e-7)

        try:

            hardest_positive_idx, hardest_negatives_idx = self.sampler.get_distances(labels.detach().cpu().numpy(), dists.detach().cpu().numpy())
            hardest_positive_idx, hardest_negatives_idx = torch.tensor(hardest_positive_idx).cuda(), torch.tensor(hardest_negatives_idx).cuda()

            hardest_negative_dist = dists.gather(1, hardest_negatives_idx.view(-1,1))
            hardest_positive_dist = dists.gather(1, hardest_positive_idx.view(-1,1))


            if evaluation and index == 0:

                hardest_negative_indices = hardest_negatives_idx.detach().cpu().numpy().squeeze()
                neg_sents = sents[hardest_negative_indices]
                with open("negatives.txt", "w") as f:
                    for (anchor_sent, hard_sent) in zip(sents, neg_sents):
                        f.write(anchor_sent + "\n")
                        f.write("-----------------------------------------\n")
                        f.write(hard_sent + "\n")
                        f.write("==========================================================\n")
        except Exception as e:
                print(e)
                print(self.k)
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

        if self.final == "plus":
            triplet_loss = torch.max(differences + self.alpha, torch.zeros_like(differences))
        elif self.final == "softplus":
            triplet_loss = F.softplus(differences, beta = 3)
        elif self.final == "softmax":

            z = torch.max(hardest_positive_dist, hardest_negative_dist)
            pos = torch.exp(hardest_positive_dist - z)
            neg = torch.exp(hardest_negative_dist - z)
            triplet_loss = (pos / (pos + neg))
        else:
            #alpha = 0.01
            #triplet_loss = torch.max(hardest_positive_dist/hardest_negative_dist, torch.ones_like(differences)*(1-alpha)) - (1-alpha)
            triplet_loss = hardest_positive_dist - hardest_negative_dist

        relevant = triplet_loss[triplet_loss > 1e-5]
        good = (hardest_positive_dist < hardest_negative_dist).sum() #(triplet_loss < 1e-5).sum()
        bad = batch.shape[0] - good
        mean_norm_squared = torch.mean(torch.norm(batch, dim = 1)**2)

        #print(relevant.shape, relevant)
        #print("***************************************")

        return torch.mean(relevant), torch.mean(differences), good, bad, torch.sqrt(mean_norm_squared)



















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

        h1,h2 = h1.cpu(), h2.cpu()

        labels = torch.arange(0, h1.shape[0])
        labels = torch.cat((labels, labels), dim = 0)
        batch = torch.cat((h1, h2), dim = 0)

        if self.mode == "euc":
            dists = pairwise_distances(batch)
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








class SoftCCALoss(torch.nn.Module):

    def __init__(self, p = 2, running_average = True):

        super(SoftCCALoss, self).__init__()
        self.p = p
        self.running_average = running_average

    def forward(self, X, Y, r = 1e-6, alpha = 500, eps = 1e-7):

        m1 = torch.mean(X, dim=0, keepdim=True)
        m2 = torch.mean(Y, dim=0, keepdim=True)

        X = X - m1
        Y = Y - m2
        N, d = X.shape
        S11 = ((torch.t(X) @ X) / (N - 1)) + r * torch.eye(d).float().cuda()
        S22 = ((torch.t(Y)  @ Y) / (N - 1)) + r * torch.eye(d).float().cuda()
        #S12 = (torch.t(Y) @ X) / (N - 1)

        corr_term = 0.5 * torch.norm(X - Y, p = 2, dim = 1).mean()#**2

        # add variance penalty

        var_S1 = torch.diag(S11)
        var_S2 = torch.diag(S22)

        S11 = S11 -torch.diag(torch.diag(S11)) + torch.diag(1./(var_S1 + eps) - var_S1)
        S22 = S22 -torch.diag(torch.diag(S22)) + torch.diag(1./(var_S2 + eps) - var_S2)
        #S11 = S11 - torch.eye(S11.shape[0]).cuda()
        #S22 = S22 - torch.eye(S22.shape[0]).cuda()

        decorrelation_term = 0.5 * (torch.norm(S11, p = 1) + torch.norm(S22, p = 1)) * (1/d**2) * alpha

        #print(corr_term, alpha * decorrelation_term)

        loss = 0.05 * (corr_term + decorrelation_term) # * (1./d**2)

        if np.random.random() < 0:
            print(torch.norm(X, dim = 1).mean())
            print(torch.diag(S11)[:10])
            print("----------------")
            print(S11[:,0][:10])
            print(alpha * decorrelation_term, corr_term)
            print("=============================")
        return loss


if __name__ == '__main__':

    pass