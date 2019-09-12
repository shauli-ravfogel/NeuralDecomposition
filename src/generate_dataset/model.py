from allennlp.commands.elmo import ElmoEmbedder
from model_interface import ModelInterface
import numpy as np
import torch.nn as nn
import torch


class Elmo(ModelInterface):

    def __init__(self, elmo_options, elmo_weights, cuda_device, layers):
        options_file = elmo_options
        weight_file = elmo_weights
        self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=cuda_device)
        self.layers = layers

    def run(self, sents):
        embeddings = self.elmo.embed_batch(sents)

        vecs = []

        for i in range(len(sents)):
            sent_embs = np.concatenate([embeddings[i][layer] for layer in self.layers], axis = 1)
            vecs.append(sent_embs)

        return vecs


class ElmoRandom(ModelInterface):

    def __init__(self, elmo_options, elmo_weights, cuda_device, layers):
        options_file = elmo_options
        weight_file = elmo_weights
        self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=cuda_device)
        self.elmo.elmo_bilm.apply(model_init_fn)
        self.layers = layers


def model_init_fn(init_fn=torch.nn.init.xavier_normal_):
    def initialize_weights(model):
        if type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            init_fn(model.weight_hh_l0)
            init_fn(model.weight_ih_l0)
        elif hasattr(model, 'weight'):  # type(model) in [nn.Linear, nn.Conv1d]:
            init_fn(model.weight.data)
    return initialize_weights