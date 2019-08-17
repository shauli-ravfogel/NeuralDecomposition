from torch.utils import data
import numpy as np
import torch
import pickle

class Dataset(data.Dataset):
    def __init__(self, views_path):

        self.view1, self.view2, self.positions = self._load_data(views_path)
        print("Training set size is {}".format(len(self.view1)))

    def _from_string(self, vec_str):

        return np.array([float(x) for x in vec_str.split(" ")])

    def _load_data(self, views_path):

        with open(views_path, "rb") as f:

            views = pickle.load(f)

        view1, view2, positions = map(np.squeeze, map(np.asarray, zip(*views)))        
        return view1, view2, positions

    def __len__(self):

        return len(self.view1)

    def __getitem__(self, index):

        with torch.no_grad():

            x1 = self.view1[index]
            x2 = self.view2[index]
            ind = self.positions[index]
            #x1 = np.random.rand(*x1.shape) - 0.5
            #x2 = np.random.rand(*x2.shape) - 0.5

            return ((torch.from_numpy(x1).float().cuda(), ind), (torch.from_numpy(x2).float().cuda(), ind))
