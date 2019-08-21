from torch.utils import data
import numpy as np
import torch
import pickle
import h5py

class Dataset(data.Dataset):
    def __init__(self, data_path):

        with open(data_path, "rb") as f:

            #self.data = h5py.File(data_path, 'r')
            #self.keys = list(self.data.keys())
            self.data = pickle.load(f)

        print("Training set size is {}".format(len(self.data)))


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        with torch.no_grad():

            vecs1, vecs2 = np.random.choice(self.data, size = 2, replace = False)

            """
            i = self.keys[index]
            j = self.keys[(index + np.random.choice(range(15))) % len(self)]
            group1, group2 = self.data[i], self.data[j]
            vecs1 = group1["vecs"][:,:,:]
            vecs2 = group2["vecs"][:,:,:]
             """
            return torch.from_numpy(vecs1).float().cuda(), torch.from_numpy(vecs2).float().cuda()

if __name__ == '__main__':

    dataset = Dataset("sample.hdf5")
    for i in range(1000):
        dataset[i]
        print(i)