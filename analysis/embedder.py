#!/usr/bin/python
# -*- coding: utf-8 -*-

from syntactic_extractor import SyntacticExtractor
from allennlp.commands.elmo import ElmoEmbedder
from typing import List, Tuple
import copy
import numpy as np
import random
random.seed(0)
from collections import Counter, defaultdict
from tqdm.auto import tqdm


class Embedder(object):

        def __init__(self, elmo_options_path: str, elmo_weights_path: str, wiki_path: str, num_sents: int, device = 0):
        
                self.elmo = self._load_elmo(elmo_weights_path, elmo_options_path, device = device)
                self.sentences = self._load_sents(wiki_path, num_sents)
        
        def get_data(self) -> List[Tuple[List[np.ndarray], str]]:
        
                embeddings_and_sents = self._run_elmo(self.sentences)
                return embeddings_and_sents
        
        def _load_elmo(self, elmo_weights_path, elmo_options_path, device = 0):
        
                print("Loading ELMO...")

                return ElmoEmbedder(elmo_options_path, elmo_weights_path, cuda_device=device)
                return ElmoEmbedder(elmo_options_path, elmo_weights_path)        

        def _run_elmo(self, sentences: List[List[str]]) -> List[Tuple[List[np.ndarray], str]]:
        
                print("Running ELMO...")
                
                elmo_embeddings = []
                
                for sent in tqdm(sentences, ascii=True):
                
                        elmo_embeddings.append((self.elmo.embed_sentence(sent), sent))
                        
                all_embeddings = []
 
                for (sent_emn, sent_str) in elmo_embeddings:
                
                        last_layer = sent_emn[-1, :, :]
                        second_layer = sent_emn[-2, :, :]
                        concatenated = np.concatenate([second_layer, last_layer], axis = 1)
                        all_embeddings.append((concatenated, sent_str))
                
                return all_embeddings
                
        def _load_sents(self, wiki_path, num_sents, max_length = 35) -> List[List[str]]:
        
                print("Loading sentences...")
                
                with open(wiki_path, "r", encoding = "utf8") as f:
                        lines = f.readlines()
                        lines =  [line.strip().split(" ") for line in lines]
                        
                if max_length is not None:
                        lines = list(filter(lambda sentence: len(sentence) < max_length, lines))
                
                lines = lines[:num_sents]
                
                return lines
