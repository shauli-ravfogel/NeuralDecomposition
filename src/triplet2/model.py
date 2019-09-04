import torch
from torch import nn
from pytorch_revgrad import RevGrad

class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 512):

        super(Siamese, self).__init__()

        layer_sizes = [dim, 1500, 1024, final]
        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            layers.append(nn.BatchNorm1d(layer_dim))
            #if i == 0:
            #    layers.append(GaussianNoise(stddev=0.001))
            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))
            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def process_word(self, word_vec):

        return self.layers(word_vec)

    def forward(self, sent_vecs):

        print(sent_vecs)
        exit()

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
