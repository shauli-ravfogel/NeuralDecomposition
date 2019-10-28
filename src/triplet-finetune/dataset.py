from torch.utils import data
import numpy as np
import torch
import pickle
import h5py


CUDA = False

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):

        pass

    def pad_collate(self, batch_data):
        """
        args:
            batch_data - list of (x: tensor, y: tensor, x_str: str, y_str: str, length: int)

        """

        sents = []
        ids = []

        for i, group in enumerate(batch_data):
            for sent in group:

                sents.append(sent)
                ids.append(i)

        sents, ids = np.array(sents), np.array(ids)

        return sents, torch.LongTensor(ids)

    def __call__(self, batch):
        return self.pad_collate(batch)




class Dataset(data.Dataset):
    def __init__(self, data_path, filter_func = False):

        with open(data_path, "rb") as f:

            self.sents = list(pickle.load(f).values())

        print("Dataset set size is {}".format(len(self.sents)))


    def __len__(self):

        return len(self.sents)

    def __getitem__(self, index):

        return self.sents[index]

if __name__ == '__main__':

    dataset = Dataset("bert_online_sents_same_pos5.pickle")
    generator = data.DataLoader(dataset, batch_size=64, drop_last = False, shuffle=True, collate_fn=PadCollate())
    for x in generator:

        print(x)
        exit()