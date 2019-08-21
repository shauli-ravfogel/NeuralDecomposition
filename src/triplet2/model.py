import torch
from torch import nn
from pytorch_revgrad import RevGrad

class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 250):

        super(Siamese, self).__init__()

        layers = []
        layers.append(nn.Linear(dim, 1500))
        layers.append(nn.ReLU())
        #layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(1500, 1000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1000, 500))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(500, 250))
        layers.append(nn.Linear(250, final))

        layers = [nn.Linear(dim, 1500, bias = False)]
        layers.append(nn.Linear(1500, 1000, bias = False))
        layers.append(nn.Linear(1000, 500, bias=False))
        layers.append(nn.Linear(500, final, bias=False))
        self.layers = nn.Sequential(*layers)


    def forward(self, sent_vecs):

        transformed =  self.layers(sent_vecs) # (sent_length, 2048)
        #normalized =  transformed / transformed.norm(dim = 2)[..., None]
        #distances = (transformed[0, ...] @ torch.t(transformed[0, ...]))[None, ...]
        distances = torch.norm(transformed[..., None, :] - transformed, dim=3, p=2) # (sent_length, sent_length)
        return torch.flatten(distances, start_dim = 1), transformed # (sent_length^2)

if __name__ == '__main__':

    dataset = dataset.Dataset("sample.hdf5")
    model = Siamese()
    vecs1, vecs2 = dataset[0]
    print(model(vecs1[0]))
