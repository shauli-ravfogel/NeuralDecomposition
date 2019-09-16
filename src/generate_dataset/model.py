from allennlp.commands.elmo import ElmoEmbedder
from model_interface import ModelInterface
import numpy as np
import torch.nn as nn
import torch

import sys
sys.path.append('src/analysis')
from random_elmo import RandomElmoEmbedder


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


class Bert(ModelInterface):

        def __init__(self, cuda_device, layers=[1, 16, -1]):
        
                self.cuda_device = cuda_device
                self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
                self.model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking', output_hidden_states = True, output_attentions = True)
                #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True, output_attentions = True)
                self.model.eval()
                self.model.to('cuda:{}'.format(self.cuda_device))
                self.layers = layers
                self.use_mean = not isinstance(layers, list)                

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

        def run(self, sents: np.ndarray):
        
                num_sents, sent_len = sents.shape
                
                if not self.use_mean:
                    embeddings = np.zeros((num_sents, sent_len, len(self.layers) * 1024))
                else:
                    embeddings = np.zeros((num_sents, sent_len, 1024))
 
                for i, sent in enumerate(sents):
                
                        bert_tokens, orig_to_tok_map = self._tokenize(sent)
                        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda:{}'.format(self.cuda_device))
                        # Predict hidden states features for each layer
                        with torch.no_grad():

                                all_hidden_states, all_attentions = self.model(tokens_tensor)[-2:]
                                #last_hidden = all_hidden_states[-1].squeeze()#.detach().cpu().numpy()
                                #first_hidden = all_hidden_states[1].squeeze()#.detach().cpu().numpy()
                                #middle_hidden = all_hidden_states[int(len(all_hidden_states)/2)].squeeze()#.detach().cpu().numpy()
                                
                                if not self.use_mean:
                                    layers = [all_hidden_states[layer].squeeze() for layer in self.layers]
                                    vecs = torch.cat(layers, dim = 1).detach().cpu().numpy()
 
                                else:
                                    vecs = torch.mean(torch.cat(all_hidden_states, dim = 0), dim = 0).detach().cpu().numpy()

                                for j in range(sent_len):  
                                        
                                        embeddings[i,j] = vecs[orig_to_tok_map[j]]
  
                return embeddings




class ElmoRandom(ModelInterface):

    def __init__(self, elmo_options, elmo_weights, cuda_device, rand_emb, rand_lstm, layers):
        options_file = elmo_options
        weight_file = elmo_weights
        self.elmo = RandomElmoEmbedder(options_file, weight_file, cuda_device=cuda_device,
                                       random_emb=rand_emb, random_lstm=rand_lstm)
        self.layers = layers

    def run(self, sents):
        embeddings = self.elmo.embed_batch(sents)

        vecs = []

        for i in range(len(sents)):
            sent_embs = np.concatenate([embeddings[i][layer] for layer in self.layers], axis = 1)
            vecs.append(sent_embs)

        return vecs

# def model_init_fn(init_fn=torch.nn.init.xavier_normal_):
#     def initialize_weights(model):
#         if type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
#             init_fn(model.weight_hh_l0)
#             init_fn(model.weight_ih_l0)
#         elif hasattr(model, 'weight'):  # type(model) in [nn.Linear, nn.Conv1d]:
#             init_fn(model.weight.data)
#     return initialize_weights
