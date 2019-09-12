#!/usr/bin/python
# -*- coding: utf-8 -*-

from syntactic_extractor import SyntacticExtractor
from allennlp.commands.elmo import ElmoEmbedder
from typing import List, Tuple, Dict
import copy
import numpy as np
import random
from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertModel, PretrainedBertEmbedder

from random_elmo import RandomElmoEmbedder

import sys
sys.path.append('src/generate_dataset')
from collect_bert_states import BertLayerEmbedder

random.seed(0)
from collections import Counter, defaultdict
from tqdm.auto import tqdm


class Embedder(object):
    def __init__(self):
        pass

    def _load_sents(self, wiki_path, num_sents, max_length=35) -> List[List[str]]:
        print("Loading sentences...")

        with open(wiki_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            lines = [line.strip().split(" ") for line in lines]

        if max_length is not None:
            lines = list(filter(lambda sentence: len(sentence) < max_length, lines))

        lines = lines[:num_sents]

        return lines

    def _embedder(self, sentence: List[str]) -> np.ndarray:
        raise NotImplementedError()

    def get_data(self, wiki_path: str, num_sents: int) -> List[Tuple[List[np.ndarray], str]]:
        sentences = self._load_sents(wiki_path, num_sents)
        embeddings_and_sents = self.run_embedder(sentences)
        return embeddings_and_sents

    def run_embedder(self, sentences: List[List[str]]) -> List[Tuple[List[np.ndarray], str]]:
        raise NotImplementedError()


class EmbedElmo(Embedder):

    def __init__(self, params: Dict, device: int = 0):

        Embedder.__init__(self)
        elmo_options_path = params['elmo_options_path']
        elmo_weights_path = params['elmo_weights_path']
        self.embedder = self._load_elmo(elmo_weights_path, elmo_options_path, device=device)

    def _load_elmo(self, elmo_weights_path, elmo_options_path, device=0):

        print("Loading ELMO...")
        return ElmoEmbedder(elmo_options_path, elmo_weights_path, cuda_device=device)

    def run_embedder(self, sentences: List[List[str]]) -> List[Tuple[np.ndarray, str]]:

        print("Running ELMO...")

        elmo_embeddings = []

        temp_list = []
        for sent in tqdm(sentences, ascii=True):
            temp_list.append(sent)

            if len(temp_list) > 500:
                batch_embeds = self.embedder.embed_batch(temp_list)
                for sent, emb in zip(temp_list, batch_embeds):
                    elmo_embeddings.append((emb, sent))
                temp_list = []

        all_embeddings = []

        for (sent_emn, sent_str) in elmo_embeddings:
            last_layer = sent_emn[-1, :, :]
            second_layer = sent_emn[-2, :, :]
            concatenated = np.concatenate([second_layer, last_layer], axis=1)
            all_embeddings.append((concatenated, sent_str))

        return all_embeddings

    def _embedder(self, sentence):
        return self._embedder(sentence)


class EmbedRandomElmo(EmbedElmo):
    def __init__(self, params: Dict, random_emb, random_lstm, device: int = 0):

        Embedder.__init__(self)
        elmo_options_path = params['elmo_options_path']
        elmo_weights_path = params['elmo_weights_path']
        self.embedder = self._load_random_elmo(elmo_weights_path, elmo_options_path, random_emb, random_lstm, device=device)

    def _load_random_elmo(self, elmo_weights_path, elmo_options_path, random_emb, random_lstm, device=0):

        print("Loading Random ELMO...")
        return RandomElmoEmbedder(elmo_options_path, elmo_weights_path, cuda_device=device,
                                  random_emb=random_emb, random_lstm=random_lstm)

class EmbedBert(Embedder):
    def __init__(self, params: Dict, device: int = 0):
        Embedder.__init__(self)

        bert_name = 'bert-large-uncased'
        bert_model = BertModel.from_pretrained(bert_name)
        self.token_indexer = PretrainedBertIndexer(pretrained_model=bert_name, use_starting_offsets=True)
        self.vocab = Vocabulary()
        # self.embedder = BertLayerEmbedder(bert_model).eval()
        self.embedder = PretrainedBertEmbedder(bert_name)

    def run_embedder(self, sentences: List[List[str]]) -> List[Tuple[List[np.ndarray], str]]:
        print("Running Bert...")

        bert_embeddings = []

        for sent in tqdm(sentences, ascii=True):
            bert_embeddings.append((self._embedder(sent), sent))

        return bert_embeddings

    def _embedder(self, sentence: List[str]) -> np.ndarray:
        toks = [Token(w) for w in sentence]

        instance = Instance({"tokens": TextField(toks, {"bert": self.token_indexer})})

        batch = Batch([instance])
        batch.index_instances(self.vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        bert_vectors = self.embedder(tokens["bert"], offsets=tokens["bert-offsets"])
        return bert_vectors.data.numpy()[0]
