from allennlp.commands.elmo import ElmoEmbedder
from model_interface import ModelInterface


class Elmo(ModelInterface):

    def __init__(self, elmo_options, elmo_weights, cuda_device):
        options_file = elmo_options
        weight_file = elmo_weights
        self.elmo = ElmoEmbedder(options_file, weight_file)#, cuda_device=cuda_device)

    def run(self, sents, layer=-1):
        embeddings = self.elmo.embed_batch(sents)

        vecs = []

        for i in range(len(sents)):
            sent_embs = embeddings[i][layer]
            vecs.append(sent_embs)

        return vecs
