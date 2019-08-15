import torch
from torch import nn
from pytorch_revgrad import RevGrad

class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 500):

        super(Siamese, self).__init__()

        layers = []
        layers.append(nn.Linear(dim, 1500))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(1500, 1500))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(1500, 1000))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(1000, 1000))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(1000, final))

        self.layers = nn.Sequential(*layers)

        pos_network = []
        pos_network.append(RevGrad())
        pos_network.append(nn.Linear(final, 512))
        #pos_network.append(nn.ReLU())
        pos_network.append(nn.Linear(512, 256))
        #pos_network.append(nn.ReLU())
        pos_network.append(nn.Linear(256, 128))
        #pos_network.append(nn.ReLU())
        pos_network.append(nn.Linear(128, 50))

        self.pos_net = nn.Sequential(*pos_network)

    def forward(self, x1, x2):

        h1, h2 = self.layers(x1), self.layers(x2)
        return h1 - h2

if __name__ == '__main__':

    train_size = 5000
    dim = 2048
    net = Siamese()
    x1 = torch.rand(dim) - 0.5
    x2 = torch.rand(dim) - 0.5

    h = net(x1,x2)

    print(h)