import torch
from torch import nn
from pytorch_revgrad import RevGrad

class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 256):

        super(Siamese, self).__init__()

        layers = []
        layers.append(nn.Linear(dim, 1500))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1500, 1200))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1200, 900))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(900, 700))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(700, final))

        layers = [nn.Linear(dim, final, bias = False)]
        self.layers = nn.Sequential(*layers)

        final_net = []
        final_net.append(nn.Linear(final, final, bias = False))
        #final_net.append(nn.ReLU())
        #final_net.append(nn.Linear(final, final, bias=True))
        self.final_net = nn.Sequential(*final_net)

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
        return self.final_net(h1 * h2)

        outer = h1[:, :, None] @ h2[:, None, :]
        #h1, h2 = self.layers(x1), self.layers(x2)
        outer = outer.view(x1.shape[0], h1.shape[1]**2)
        return outer
        return torch.abs(h1 - h2)
        return self.final_net(torch.cat((h1, h2), dim = 1))
        diff = torch.abs(h1 - h2)
        return self.final_net(diff)
        return diff

if __name__ == '__main__':

    train_size = 5000
    dim = 2048
    net = Siamese()
    x1 = torch.rand(dim) - 0.5
    x2 = torch.rand(dim) - 0.5

    h = net(x1,x2)

    print(h)
