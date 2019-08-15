
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

def collect_data(path, num_examples, num_examples_per_group):

    f = h5py.File(path, 'r')
    pbar = tqdm.tqdm(total=num_examples)
    data = []
    i = 0
    keys = f.keys()

    print("Collecting data...")

    while len(data) < num_examples:

        j,k = np.random.choice(keys, size = 2, replace = False)
        group1, group2 = f[j], f[k]  # group has the same interface as Equivalent_sentences_group
        group_data = generate_training_instances(group1, group2, num_examples_per_group)
        data.extend(group_data)
        pbar.update(num_examples_per_group)
        i += 1

    print("Collected {} instances from {} sentences".format(len(data), i))

    output_filename = "data.pickle"

    with open(output_filename, "wb") as f:
        pickle.dump(data, f)

    f.close()


def generate_training_instances(group1: Equivalent_sentences_group, group2: Equivalent_sentences_group, num_examples_per_group: int):

    vecs1, sents1, content_idx1 = group1["vecs"], group1["sents"], group1["content_indices"]
    vecs2, sents2, content_idx2 = group2["vecs"], group2["sents"], group2["content_indices"]
    group_size, sent1_len, sent2_len = group1.attrs["group_size"], group1.attrs["sent_length"], group2.attrs["sent_length"]

    data = []

    for ind in range(num_examples_per_group):
        # sample indices for sentences in both groups.

        i, j = np.random.choice(range(group_size), size = 2, replace = False)
        k = np.random.choice(range(group_size))

        # sample word indices

        l, m = np.random.choice(range(min(sent1_len, sent2_len)), size = 2, replace = True)
        w1, w2 = vecs1[i,l], vecs1[j, m]
        w3, w4 = vecs1[j, l], vecs1[i, m]
        w5, w6 = vecs2[k,l], vecs2[k, m]

        instance =  (w1, w2, w3, w4, w5, w6)
        data.append(instance)

    return data