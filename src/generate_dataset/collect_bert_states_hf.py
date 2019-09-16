import argparse
import numpy as np
# import sys
import pickle
from typing import List, Tuple, Dict
import utils
import h5py
from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder

from tqdm import tqdm

import torch
import torch.nn.functional as F
from allennlp.nn import util

# sys.path.append("../../../src/generate_dataset")
FUNCTION_WORDS = utils.DEFAULT_PARAMS["function_words"]


def get_equivalent_sentences(equivalent_sentences_path: str, num_sentences: int) -> List[List[List[str]]]:
    # equivalent_sentences_path is the path to a file containing 150k groups of equivalent sentences.
    # each group contains k=15 sentences, represented as a list of lists of string.
    # e.g., if the length of the sentences in the first group is L=20,
    # then sentences[0] is a KxL=15x20 list, where position i,j contains the jth word in the ith sentence.

    with open(equivalent_sentences_path, "rb") as f:
        sentences = pickle.load(f)  # a list of groups. each group is a list of lists of strings
        sentences = list(sentences.values())

    return sentences[:num_sentences]


def tokenize(original_sentence: List[str], tokenizer) -> Dict[int, int]:
    """
    Parameters
    ----------
    Returns
    -------
    bert_tokens: The sentence, tokenized by BERT tokenizer.
    orig_to_tok_map: An output dictionary consisting of a mapping (alignment) between indices in the original tokenized sentence, and indices in the sentence tokenized by the BERT tokenizer. See https://github.com/google-research/bert
    """

    bert_tokens = ["[CLS]"]
    orig_to_tok_map = {}
    is_subword = []

    for i, w in enumerate(original_sentence):
        tokenized_w = tokenizer.tokenize(w)
        has_subwords = len(tokenized_w) > 1
        is_subword.append(has_subwords)
        bert_tokens.extend(tokenized_w)

        orig_to_tok_map[i] = len(bert_tokens) - 1

    bert_tokens.append("[SEP]")

    return orig_to_tok_map


def result2layers(batched_layers, layer):
    data = []
    for batch in batched_layers[layer]:
        data.append(batch.unsqueeze(0))
    return torch.cat(data, dim=0)


def get_bert_states(sentence_group: List[List[str]], model, tokenizer, layer: int):
    instances = []
    for sen in sentence_group:
        text = ' '.join(sen)

        instance = tokenizer.encode(text)
        instances.append([instance, sen])

    batch = torch.tensor([x[0] for x in instances])

    last_layer, _, rest_layers = model(batch)
    all_layers = (rest_layers + (last_layer,))
    # all_layers = torch.cat(all_layers)
    result2layers(all_layers, layer)

    return bert_vectors.data.numpy()


def save_bert_states(embedder, equivalent_sentences: List[List[List[str]]], output_file: str,
                     layer: int):
    with h5py.File(output_file, 'w') as h5:
        for i, group_of_equivalent_sentences in tqdm(enumerate(equivalent_sentences)):
            bert_states = get_bert_states(group_of_equivalent_sentences, embedder, layer)
            # if the length (num of words) of the group i is L, and there are K=15 sentences in the group,
            # then bert_states is a numpy array of dims KxLxD where D is the size of the bert vectors.

            L = len(group_of_equivalent_sentences[0])  # group's sentence length
            content_indices = np.array(
                [i for i in range(L) if group_of_equivalent_sentences[0][i] not in FUNCTION_WORDS])

            sents = np.array(group_of_equivalent_sentences, dtype=object)

            # data.append(bert_states)
            g = h5.create_group(str(i))
            g.attrs['group_size'], g.attrs['sent_length'] = sents.shape
            g.create_dataset('vecs', data=bert_states, compression=True, chunks=True)
            dt = h5py.special_dtype(vlen=str)
            g.create_dataset('sents', data=sents, dtype=dt, compression=True, chunks=True)
            g.create_dataset('content_indices', data=content_indices, compression=True, chunks=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-sentences', dest='input_sentences', type=str,
                        default='data/interim/bert_online_sents_same_pos4.pickle',
                        help='equivalent sentences to parse with bert')
    parser.add_argument('--bert-model', dest='bert_model', type=str,
                        default='bert-base-uncased',
                        help='bert model type to use. bert-base-uncased / bert-large-uncased / ...')
    parser.add_argument('--output-file', dest='output_file', type=str,
                        default='data/interim/encoder_bert/sents_bert.hdf5',
                        help='output file where the encoded vectors are stored')
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=30522,
                        help='The size of bert\'s vocabulary')
    parser.add_argument('--num-sentences', dest='num_sentences', type=int, default=999999999,
                        help='The amount of group sentences to use')
    parser.add_argument('--layer', dest='layer', type=int, default=-1,
                        help='The layer of bert to persist')

    args = parser.parse_args()
    all_groups = get_equivalent_sentences(args.input_sentences, args.num_sentences)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    # Models can return full list of hidden-states & attentions weights at each layer
    model = BertModel.from_pretrained(args.bert_model,
                                      output_hidden_states=True,
                                      output_attentions=False)
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_attentions = model(input_ids)[-2:]

    # config = BertConfig(vocab_size_or_config_json_file=args.vocab_size)
    # bert_model = BertModel.from_pretrained(args.bert_model)
    #
    # token_indexer = PretrainedBertIndexer(pretrained_model=args.bert_model, use_starting_offsets=True)
    # vocab = Vocabulary()
    # tlo_embedder = BertLayerEmbedder(bert_model).eval()
    #
    # save_bert_states(tlo_embedder, all_groups, args.output_file, args.layer)
