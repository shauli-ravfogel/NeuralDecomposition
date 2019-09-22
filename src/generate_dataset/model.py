from allennlp.commands.elmo import ElmoEmbedder
from model_interface import ModelInterface
import numpy as np
from typing import List, Tuple, Dict
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM


class Elmo(ModelInterface):

    def __init__(self, elmo_options, elmo_weights, cuda_device, layers, only_fwd = True):
        options_file = elmo_options
        weight_file = elmo_weights
        self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=cuda_device)
        self.layers = layers
        self.only_fwd = only_fwd

    def run(self, sents):
        embeddings = self.elmo.embed_batch(sents)

        vecs = []

        for i in range(len(sents)):
            if not self.only_fwd:
                sent_embs = np.concatenate([embeddings[i][layer] for layer in self.layers], axis = 1)
            
            else:
                sent_embs = np.concatenate([embeddings[i][layer] if layer !=0 else embeddings[i][layer][:512] for layer in self.layers], axis = 1)

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

        def run(self, sents: np.ndarray):
        
                num_sents, sent_len = sents.shape
                embeddings = np.zeros((num_sents, sent_len, self.num_vecs * 1024))
 
                for i, sent in enumerate(sents):
                
                        bert_tokens, orig_to_tok_map = self._tokenize(sent)
                        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda:{}'.format(self.cuda_device))
                        # Predict hidden states features for each layer
                        with torch.no_grad():

                                all_hidden_states, all_attentions = self.model(tokens_tensor)[-2:]
                                    
                                layers = [all_hidden_states[layer].squeeze() for layer in self.layers]
                                vecs = torch.cat(layers, dim = 1)  #.detach().cpu().numpy()
       	       	       	       	
                                if self.use_mean:

                                    mean_vecs = torch.mean(torch.cat(all_hidden_states, dim = 0), dim = 0)#.detach().cpu().numpy()
                                    vecs = torch.cat((vecs, mean_vecs), dim = 1)

                                vecs = vecs.detach().cpu().numpy()

                                for j in range(sent_len):  
            
                                        embeddings[i,j] = vecs[orig_to_tok_map[j]]
  
                return embeddings
                                



                
                
                
                
                
                
