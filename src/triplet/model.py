import torch
from torch import nn
from pytorch_revgrad import RevGrad



class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


class Siamese(nn.Module):

    def __init__(self, cca_model, final_dim, pair_repr, adversary = False):

        super(Siamese, self).__init__()

        self.cca_model = cca_model
        self.pair_repr = pair_repr
        self.adversary = adversary

        if self.cca_model is None:
            self.layer_sizes = [2048, 1024, final_dim]
        else:
            self.layer_sizes = [self.cca_model.final, final_dim]

        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(self.layer_sizes,self.layer_sizes[1:])):

            layers.append(nn.BatchNorm1d(layer_dim))
            #if i == 0:
            #    layers.append(GaussianNoise(stddev=0.001))
            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))
            if i != len(self.layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        if self.adversary:

            adversary = []
            adversary.append(RevGrad())
            adversary.append(nn.Linear(final_dim + self.layer_sizes[0], 512))
            adversary.append(nn.Linear(512, 256))
            adversary.append(nn.Linear(256, 128))
            adversary.append(nn.Linear(128, 2))

            self.adversary = nn.Sequential(*adversary)

    def process(self, X):

        if self.cca_model is not None:

            X = self.cca_model(X)

        H = self.layers(X)
        return X,H

    def pair2vec(self, h1, h2):

        if self.pair_repr == "diff":
            return h2 - h1
        elif self.pair_repr == "abs-diff":
            return torch.abs(h2 - h1)
        elif self.pair_repr == "product":
            return h1 * h2
        elif self.pair_repr == "abs-product":
            return torch.abs(h1 * h2)
        elif self.pair_repr == "plus":
            return h1 + h2
        elif self.pair_repr == "abs-plus":
            return torch.abs(h1 + h2)

    def input_for_adversary(self, w1, h1, h2):

        elmo_vecs = w1
        concat = torch.cat((h1,h2), dim = 0)

        idx = torch.randperm(concat.shape[0])
        concat = concat[idx] # row shuffling
        true_labels = idx < w1.shape[0]
        transformed_vecs = concat[:w1.shape[0]] # randomly choose either h1[i] or h2[i] for i = 1...n
        final = torch.cat((elmo_vecs,transformed_vecs), dim = 1)

        return (final, true_labels)

    def forward(self, w1, w2, w3, w4): #(w1, w2 at the same ind), (w3, w4 at the same ind)

        w1, h1 = self.process(w1)
        w2, h2 = self.process(w2)
        w3, h3 = self.process(w3)
        w4, h4 = self.process(w4)

        p1 = self.pair2vec(h1, h3)
        p2 = self.pair2vec(h2, h4)

        if self.adversary:
            adv_inp, adv_labels = self.input_for_adversary(w1, h1, h2)
            adv_preds = self.adversary(adv_inp)
        else:
            adv_labels, adv_preds = None, None

        output_dict = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "h1": h1, "h2": h2, "h3": h3, "h4": h4,
                       "p1": p1, "p2": p2, "adv_preds": adv_preds, "adv_labels": adv_labels}

        return (w1, w2, w3, w4), (h1,h2,h3,h4), (p1, p2), (adv_preds, )



class SoftCCANetwork(nn.Module):

    def __init__(self, dim = 2048, final = 256):

        super(SoftCCANetwork, self).__init__()
        self.final = final
        layer_sizes = [dim, 1500, final]
        layer_sizes = [dim, final]
        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))

            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        #layers = [nn.Linear(dim, final, bias = False)]
        self.layers = nn.Sequential(*layers)


    def normalize(self, h):

        stds = torch.std(h, dim = 0, keepdim = True)
        means = torch.mean(h, dim = 0, keepdim = True)

        return (h - means) / stds

    def forward(self, x):

        h = self.layers(x)
        return h



if __name__ == '__main__':

    train_size = 5000
    dim = 2048
    final_dim = 256
    batch_size = 3
    net = Siamese(None, final_dim, "diff", adversary = True)
    x1 = torch.rand((batch_size, dim)) - 0.5
    h = torch.rand((batch_size, final_dim)) - 0.5
    h2 = torch.rand((batch_size, final_dim)) - 0.5

    final, labels = net.input_for_adversary(x1, h, h2)
    print(final.shape)
    print(labels.shape)