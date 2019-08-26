from torch.utils import data
import numpy as np
import torch

class Dataset(data.Dataset):

        def __init__(self, data_location):    
        
                with open(data_location, "r") as f:
                        self.lines = f.readlines()
                        self.lines = self._load_data(self.lines)
                        self.lines = self.lines[:]

        def _from_string(self, vec_str):
        
                return np.array([float(x) for x in vec_str.split(" ")])
                
        def _load_data(self, lines):
                
                all_data = []
                
                for i, line in enumerate(lines):

                        if i > 10000: break
                        
                        vecs, sent = line.strip().split("\t")
                        vecs = vecs.split("*")
                        sent = sent.split(" ")
                        vecs = [self._from_string(v) for v in vecs]

                        all_data.append({"sent": sent, "vecs": vecs})
                        
                return all_data
                
        def __len__(self):

                       return len(self.lines)

        def __getitem__(self, index):
  
                data_dictionary = self.lines[index]
                sent, vecs = data_dictionary["sent"], data_dictionary["vecs"]
                
                return [torch.from_numpy(v).float().cuda() for v in vecs]
