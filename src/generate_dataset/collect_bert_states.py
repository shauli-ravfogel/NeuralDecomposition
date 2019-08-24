import numpy as np
import sys
import pickle
from typing import List
import utils
import h5py

import torch

from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder

from tqdm import tqdm


sys.path.append("../../../src/generate_dataset")
FUNCTION_WORDS = utils.DEFAULT_PARAMS["function_words"]


config = BertConfig(vocab_size_or_config_json_file=30522)
bert_model = BertModel(config)
token_embedder = BertEmbedder(bert_model)

bert_name = 'bert-base-uncased'
token_indexer = PretrainedBertIndexer(pretrained_model=bert_name, use_starting_offsets=True)
vocab = Vocabulary()
tlo_embedder = BertEmbedder(bert_model, top_layer_only=True)


def get_equivalent_sentences(equivalent_sentences_path="data/interim/bert_online_sents_same_pos4.pickle")\
        -> List[List[List[str]]]:
    # equivalent_sentences_path is the path to a file containing 150k groups of equivalent sentences.
    # each group contains k=15 sentences, represented as a list of lists of string. e.g., if the length of the sentences in the first group is L=20, then sentences[0] is a KxL=15x20 list, where position i,j contains the jth word in the ith sentence.

    with open(equivalent_sentences_path, "rb") as f:
        sentences = pickle.load(f)  # a list of groups. each group is a list of lists of strings
        sentences = list(sentences.values())

    return sentences


def get_bert_states(sentence_group: List[List[str]]):
    instances = []
    for sen in sentence_group:
        toks = [Token(w) for w in sen]

        instance = Instance({"tokens": TextField(toks, {"bert": token_indexer})})
        instances.append(instance)

    batch = Batch(instances)
    batch.index_instances(vocab)

    padding_lengths = batch.get_padding_lengths()
    tensor_dict = batch.as_tensor_dict(padding_lengths)
    tokens = tensor_dict["tokens"]

    bert_vectors = tlo_embedder(tokens["bert"], offsets=tokens["bert-offsets"])

    return bert_vectors.data.numpy()


def save_bert_states(equivalent_sentences: List[List[List[str]]]):
    data = []

    with h5py.File('data/interim/encoded_sents_bert2.150k.hdf5', 'w') as h5:
        for i, group_of_equivalent_sentences in tqdm(enumerate(equivalent_sentences)):
            bert_states = get_bert_states(group_of_equivalent_sentences)
            # if the length (num of words) of the group i is L, and there are K=15 sentences in the group, then bert_states is a numpy array of dims KxLxD where D is the size of the bert vectors.

            L = len(group_of_equivalent_sentences[0])  # group's sentence length
            content_indices = np.array([i for i in range(L) if group_of_equivalent_sentences[0][i] not in FUNCTION_WORDS])

            sents = np.array(group_of_equivalent_sentences, dtype=object)

            data.append(bert_states)
            g = h5.create_group(str(i))
            g.attrs['group_size'], g.attrs['sent_length'] = sents.shape
            g.create_dataset('vecs', data=bert_states, compression=True, chunks=True)
            dt = h5py.special_dtype(vlen=str)
            g.create_dataset('sents', data=sents, dtype=dt, compression=True, chunks=True)
            g.create_dataset('content_indices', data=content_indices, compression=True, chunks=True)


if __name__ == "__main__":
    all_groups = get_equivalent_sentences()
    save_bert_states(all_groups)
