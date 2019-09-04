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
    #print(vec.shape, dim, pad)
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

    def pad_collate(self, batch_data):
        """
        args:
            batch_data - list of (x: tensor, y: tensor, x_str: str, y_str: str, length: int)

        """

        X, Y, X_str, Y_str, lengths = list(zip(*batch_data))
        X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first = True)
        Y_padded = torch.nn.utils.rnn.pad_sequence(Y, batch_first = True)

        return (X_padded, Y_padded, X_str, Y_str, lengths)

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
            print(instance.keys())
            exit()
            sent1_vecs, sent2_vecs = instance["vecs"]
            x,y = torch.from_numpy(sent1_vecs).float().cuda(), torch.from_numpy(sent2_vecs).float().cuda()
            x_sent, y_sent = instance["sent1"], instance["sent2"]
            length = torch.LongTensor(instance["sent_length"]).cuda()

            return (x,y,x_sent,y_sent, length)

if __name__ == '__main__':

    dataset = Dataset("sample.hdf5")
    for i in range(1000):
        dataset[i]
        print(i)