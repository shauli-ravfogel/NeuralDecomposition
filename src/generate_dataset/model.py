from allennlp.commands.elmo import ElmoEmbedder
from model_interface import ModelInterface
import numpy as np

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
