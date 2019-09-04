from torch.utils import data
import numpy as np
import torch
import pickle
import h5py



def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).cuda()], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = tuple(map(lambda x:
                    pad_tensor(x, pad=max_len, dim=self.dim), batch))
        # stack all

        xs = torch.stack(batch, dim=0)
        return xs

    def __call__(self, batch):
        return self.pad_collate(batch)




class Dataset(data.Dataset):
    def __init__(self, data_path):

        with open(data_path, "rb") as f:

            self.data = pickle.load(f)

        print("Dataset set size is {}".format(len(self.data)))


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        with torch.no_grad():

            instance = self.data[index]
            sent1_vecs, sent2_vecs = instance["vecs"]
            sent2_vecs = sent1_vecs
            instance["vecs"] = (torch.from_numpy(sent1_vecs).float().cuda(), torch.from_numpy(sent2_vecs).float().cuda())
            return instance["vecs"][0]

if __name__ == '__main__':

    dataset = Dataset("sample.hdf5")
    for i in range(1000):
        dataset[i]
        print(i)