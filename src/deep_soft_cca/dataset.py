from torch.utils import data
import numpy as np
import torch
import pickle

class Dataset(data.Dataset):
    def __init__(self, data_path):

        with open(data_path, "rb") as f:

            self.data = pickle.load(f)
        print("Training set size is {}".format(len(self.data)))




    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        with torch.no_grad():

            word_repres = self.data[index]["vecs"]
            return [torch.from_numpy(w).float() for w in word_repres]