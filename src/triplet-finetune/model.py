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
                 attention = True,
                 layer_sizes = [2048, 75],
                 reduce = "mean"):

        super().__init__()

        grad = True if finetune else False
        self.elmo = Elmo(elmo_options_path, elmo_weights_path, num_output_representations = 2, requires_grad = grad)
        self.self_attention = torch.nn.MultiheadAttention(1024, num_heads=1, dropout=0.0)
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

        if self.attention:
            attn_output, attn_output_weights = self.self_attention(last_layer, last_layer, last_layer, key_padding_mask = batch_mask)
            h = attn_output
        else:
            h = last_layer

        # to get a single vector, reduce over the sequence length dimension.

        if self.reduce == "mean":
            sum = torch.sum(h, dim=0)
            final = sum / lengths[:, None]
        else:
            raise Exception("Unknown reduce method")

        final = final.unsqueeze(0)  # add the dummy seq len dimension

        return final, final



encoder = ELMOEncoder()
sent1 = ["hello", ",", "how", "are", "you", "today", "?"]
sent2 = ["good", ",", "thanks", "!"]
sent3 = ["yes", "!"]
batch = [sent1, sent2, sent3]
batch_ids = batch_to_ids(batch)
print(type(batch_ids))
print(batch_ids.shape)

encoded = encoder(batch)
padded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=False, padding_value=0)
print(padded)