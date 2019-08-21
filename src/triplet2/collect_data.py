
from typing import List
import typing
import numpy as np
import h5py
import tqdm
import pickle
import random

Equivalent_sentences_group = typing.NamedTuple("equivalent_sentences",
                                    [('vecs', np.ndarray), ('sents', List[List[str]]),
                                     ("content_indices", List[int])])

def collect_data(path):

    data_file = h5py.File(path, 'r')
    pbar = tqdm.tqdm(total=35000, ascii = True)
    data = []
    i = 0
    keys = list(data_file.keys())
    output_filename = "data.35k.pickle"

    print("Collecting data...")

    for i in range(35000):

        key = keys[i]
        group = data_file[key]
        data.append(group["vecs"][:,:])
        pbar.update(1)

        if i % 1000 == -1:

            with open(output_filename, "wb") as f:
                pickle.dump(data, f)

    print("Collected {} instances from {} sentences".format(len(data), i))


    with open(output_filename, "wb") as f:
        pickle.dump(data, f)

    f.close()



if __name__ == '__main__':

    collect_data("../../data/interim/encoded_sents.150k.hdf5")