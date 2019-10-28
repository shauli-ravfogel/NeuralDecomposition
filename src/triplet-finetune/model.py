import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
from typing import List

class ELMOEncoder(nn.Module):

    def __init__(self, elmo_options_path = "../../data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                 elmo_weights_path = "../../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                 finetune = True,
                 attention = False,
                 layer_sizes = [1024, 512],
                 reduce = "mean"): #TODO: self attetion is broken, gives None when using a mask

        super().__init__()

        grad = True if finetune else False
        self.elmo = Elmo(elmo_options_path, elmo_weights_path, num_output_representations = 2, requires_grad = grad)
        self.self_attention = torch.nn.MultiheadAttention(layer_sizes[-1], num_heads=1, dropout=0.0)
        self.attention = attention
        self.reduce = reduce

        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))

            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, batch: List[List[str]]):

        batch_embeds = self.elmo(batch_to_ids(batch))
        lengths = torch.Tensor([len(sent) for sent in batch]).float()

        batch_embeds, batch_mask = batch_embeds["elmo_representations"], batch_embeds["mask"]
        batch_mask = batch_mask.byte().bool()

        last_layer = batch_embeds[-1] #(batch size, max_seq len, 1024)
        last_layer = last_layer.transpose(0,1) #(seq len, batch_size, 1024)

        last_layer = self.layers(last_layer)  # apply final transformation

        if self.attention:
            attn_output, attn_output_weights = self.self_attention(last_layer, last_layer, last_layer, key_padding_mask = None)
            h = attn_output
        else:
            h = last_layer

        # to get a single vector, reduce over the sequence length dimension.

        h = h.transpose(0,1) #(batch_size, seq_len, 1024)

        if self.reduce == "mean":
            sum = torch.sum(h * batch_mask[..., None].float(), dim=1)
            final = sum / lengths[:, None]
        else:
            raise Exception("Unknown reduce method")

        return final


if __name__ == '__main__':

    encoder = ELMOEncoder()
    sent1 = ["hello", ",", "how", "are", "you", "today", "?"]
    sent2 = ["good", ",", "thanks", "!"]
    sent3 = ["yes", "!"]
    batch = [sent1, sent2, sent3]
    batch_ids = batch_to_ids(batch)


    encoded = encoder(batch)
    print(encoded.shape)
    print(encoded)