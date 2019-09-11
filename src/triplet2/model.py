import torch
from torch import nn
from pytorch_revgrad import RevGrad
import transformer
import numpy as np


CUDA = False

class SkipConectionLinear(nn.Module):

    def __init__(self, in_dim, out_dim, substract = True):
        super(SkipConectionLinear, self).__init__()

        if in_dim != out_dim:
            raise Exception("in_dim must equal out_dim in skip connection.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.substract = substract

    def forward(self, x):

            return x + self.linear(x)

class Siamese(nn.Module):

    def __init__(self, cca_network, dim = 2048, final_dim = 128, word_dropout = 0, self_attention = False):

        super(Siamese, self).__init__()
        self.self_attention = self_attention
        self.cca_network = cca_network
        self.word_dropout = word_dropout
        self.pair_repr = "diff"
        layer_sizes = [dim, final_dim]
        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            if layer_dim != next_layer_dim:
                layers.append(nn.Linear(layer_dim, next_layer_dim, bias = False))
            else:
                layers.append(SkipConectionLinear(layer_dim, next_layer_dim))
            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        #self.self_attention_layer = nn.TransformerEncoderLayer(d_model = final_dim, nhead=1, dim_feedforward=512, dropout=0.05)
        self.self_attention_layer = transformer.MultiHeadedAttention(h = 1, d_model = final_dim)
        print(self.layers)

    def process(self, word_vec):

        return word_vec, self.layers(word_vec)

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

    def process_sentence(self, sent_vecs):

        x = self.layers(sent_vecs)
        return self.self_attention_layer(x,x,x)

    def process_batch(self, batch_transformed1, batch_transformed2, sent_lengths):

        max_len = batch_transformed1.shape[1]
        row_idx = torch.arange(batch_transformed1.shape[0])

        valid_idx = (torch.arange(max_len)[None, :] < sent_lengths[:, None]).float() #(mask > 1e-6).float()
        choice = torch.multinomial(valid_idx, 2)
        l,m = choice[:, 0], choice[:, 1]

        w1 = batch_transformed1[row_idx, l, :].squeeze()
        w2 = batch_transformed1[row_idx, m, :].squeeze()
        w3 = batch_transformed2[row_idx, l, :].squeeze()
        w4 = batch_transformed2[row_idx, m, :].squeeze()

        #p1 = self.pair2vec(w2, w1)
        #p2 = self.pair2vec(w4, w3)
        #return (p1, p2)

        return (w1, w3) if np.random.random() < 0.5 else (w2, w4)


    def forward(self, sent_vecs1, sent_vecs2, lengths):

        transformed1 = self.layers(sent_vecs1)
        transformed2 = self.layers(sent_vecs2)

        transformed1 = self.self_attention_layer(transformed1, transformed1, transformed1)
        transformed2 = self.self_attention_layer(transformed2, transformed2, transformed2)
        p1, p2 = self.process_batch(transformed1, transformed2, lengths)

        return p1, p2

if __name__ == '__main__':

    dataset = dataset.Dataset("sample.hdf5")
    model = Siamese()
    vecs1, vecs2 = dataset[0]
    print(model(vecs1[0]))
