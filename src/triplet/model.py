import torch
from torch import nn
from pytorch_revgrad import RevGrad


class Siamese(nn.Module):

    def __init__(self, cca_model, final_dim, pair_repr):

        super(Siamese, self).__init__()

        self.cca_model = cca_model
        self.pair_repr = pair_repr
        if self.cca_model is None:
            layer_sizes = [2048, 2048, 1024, 512, final_dim]
        else:
            layer_sizes = [self.cca_model.final, final_dim]#, final_dim]

        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))
            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        pos_network = []
        pos_network.append(RevGrad())
        pos_network.append(nn.Linear(final_dim, 512))
        pos_network.append(nn.Linear(512, 256))
        pos_network.append(nn.Linear(256, 128))
        pos_network.append(nn.Linear(128, 50))

        self.pos_net = nn.Sequential(*pos_network)

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

    def forward(self, w1, w2, w3, w4): #(w1, w2 at the same ind), (w3, w4 at the same ind)

        w1, h1 = self.process(w1)
        w2, h2 = self.process(w2)
        w3, h3 = self.process(w3)
        w4, h4 = self.process(w4)

        p1 = self.pair2vec(h1, h3)
        p2 = self.pair2vec(h2, h4)

        return (w1, w2, w3, w4), (h1,h2,h3,h4), (p1, p2)





class SoftCCANetwork(nn.Module):

    def __init__(self, dim = 2048, final = 256):

        super(SoftCCANetwork, self).__init__()
        self.final = final
        layer_sizes = [dim, 1800, 1500, final]
        #layer_sizes = [dim, final]
        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = False))

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
    net = Siamese()
    x1 = torch.rand(dim) - 0.5
    x2 = torch.rand(dim) - 0.5

    h = net(x1,x2)

    print(h)
