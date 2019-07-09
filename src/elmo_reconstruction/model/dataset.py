from torch.utils import data
import numpy as np
import torch
import pickle
from allennlp.commands.elmo import ElmoEmbedder

class Dataset(data.Dataset):

        def __init__(self, data_location, elmo_options, elmo_weights):    
        
                #with open(data_location, "r") as f:
                        #self.lines = f.readlines()
                self.sentences = self._load_data(data_location)
                #self.sentences = self.sentences[:]
                options_file = elmo_options
                weight_file = elmo_weights
                self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=-1)
                
        def _load_data(self, data_location):
                
                with open(data_location, "rb") as f:
                
                      sentences = pickle.load(f)
                
                d = sentences
                #d = {k:v for k,v in sentences.items() if k < 200}
                return d
                
        def __len__(self):

                       return len(self.sentences)

        def __getitem__(self, index):
  
                sents = self.sentences[index]
                vecs = np.array(self.elmo.embed_batch(sents))
                
                sent_length = len(sents[0])                
                embds = vecs[:, 0, :, :]
                #contextualized_embds = np.concatenate((vecs[1, :, :], vecs[2, :, :]), axis = 1)
                layer1, layer2 = vecs[:, 1, :, :], vecs[:, 2, :, :]

                return (torch.from_numpy(embds).cuda(), torch.from_numpy(layer1).cuda(), torch.from_numpy(layer2).cuda())
