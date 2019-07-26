from torch.utils import data
import numpy as np
import torch
import pickle

class Dataset(data.Dataset):
    def __init__(self, view1_location, view2_location):

        self.view1, self.view2 = self._load_data(view1_location, view2_location)
        print("Training set size is {}".format(len(self.view1)))

    def _from_string(self, vec_str):

        return np.array([float(x) for x in vec_str.split(" ")])

    def _load_data(self, view1_location, view2_location):

        with open(view1_location, "rb") as f:

            view1 = pickle.load(f)

        with open(view2_location, "rb") as f:

            view2 = pickle.load(f)

        return view1, view2

    def __len__(self):

        return len(self.view1)

    def __getitem__(self, index):
        with torch.no_grad():
            return (torch.from_numpy(self.view1[index]).float().cuda(), torch.from_numpy(self.view2[index]).float().cuda())