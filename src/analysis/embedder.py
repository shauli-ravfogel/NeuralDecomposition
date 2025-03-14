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

        if len(temp_list) > 0:
            batch_embeds = self.embedder.embed_batch(temp_list)
            for sent, emb in zip(temp_list, batch_embeds):
                elmo_embeddings.append((emb, sent))

        all_embeddings = []

        for (sent_emn, sent_str) in elmo_embeddings:
            last_layer = sent_emn[-1, :, :]
            second_layer = sent_emn[-2, :, :]
            concatenated = np.concatenate([second_layer, last_layer], axis=1)
            all_embeddings.append((concatenated, sent_str))

        return all_embeddings

    def _embedder(self, sentence):
        return self._embedder(sentence)


class BertEmbedder(Embedder):

        def __init__(self, device, layers=[1, 16, "mean"):
        
                self.cuda_device = device
                self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
                self.model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking', output_hidden_states = True, output_attentions = True)
                #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True, output_attentions = True)
                self.model.eval()
                self.model.to('cuda:{}'.format(self.cuda_device))
                self.num_vecs = len(layers)            
                self.use_mean, self.layers = "mean" in layers, [l for l in layers if l != "mean"]         

        def _tokenize(self, original_sentence: List[str]) -> Tuple[List[str], Dict[int, int]]:
    
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
                has_subwords = False
                is_subword = []
        
                for i, w in enumerate(original_sentence):
            
                        tokenized_w = self.tokenizer.tokenize(w)
                        has_subwords = len(tokenized_w) > 1
                        is_subword.append(has_subwords)
                        bert_tokens.extend(tokenized_w)
            
                        orig_to_tok_map[i] = len(bert_tokens) - 1
                  
                bert_tokens.append("[SEP]")
        
                return (bert_tokens, orig_to_tok_map)

        def run_embedder(self, sentences: List[List[str]]) -> List[Tuple[np.ndarray, str]]:

                print("Running BERT...")

                bert_embeddings = []
                
                for sent in tqdm(sentences, ascii=True):
                
                         bert_tokens, orig_to_tok_map = self._tokenize(sent)
                         sent_len = len(sent)
                         embeddings = np.zeros((sent_len, self.num_vecs * 1024))

                         
                         indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
                         tokens_tensor = torch.tensor([indexed_tokens]).to('cuda:{}'.format(self.cuda_device)) 
                           
                         with torch.no_grad():

                                all_hidden_states, all_attentions = self.model(tokens_tensor)[-2:]
                                layers = [all_hidden_states[layer].squeeze() for layer in self.layers]
                                vecs = torch.cat(layers, dim = 1)  #.detach().cpu().numpy()
       	       	       	       	
                                if self.use_mean:

                                    mean_vecs = torch.mean(torch.cat(all_hidden_states, dim = 0), dim = 0)#.detach().cpu().numpy()
                                    vecs = torch.cat((vecs, mean_vecs), dim = 1)
                                
                                vecs = vecs.detach().cpu().numpy()

                                for j in range(sent_len):  
                                        
                                        embeddings[j] = vecs[orig_to_tok_map[j]]
                         
                         bert_embeddings.append((embeddings, sent))     

                return bert_embeddings   
    
    
    
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
