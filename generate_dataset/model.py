from typing import List, TypeVar
import numpy as np
from model_interface import ModelInterface
from allennlp.commands.elmo import ElmoEmbedder, batch_to_ids
import utils


class Elmo(ModelInterface):

    def __init__(self, elmo_options, elmo_weights):
        options_file = elmo_options
        weight_file = elmo_weights
        self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)

    def run(self, sents, layer=-1):
        embeddings = self.elmo.embed_batch(sents)

        vecs = []

        for i in range(len(sents)):
            sent_embs = embeddings[i][layer]
            vecs.append(sent_embs)

        return vecs
