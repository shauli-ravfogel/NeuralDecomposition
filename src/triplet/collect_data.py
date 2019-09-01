
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

def collect_data(path, num_examples, num_examples_per_group,  min_length = 12, max_length = 30):

    f = h5py.File(path, 'r')
    pbar = tqdm.tqdm(total=num_examples)
    data = []
    i = 0
    num_sents = 150000
    keys = list(f.keys())
    output_filename = "data_dist_decay.pickle"
    sents_collected = 0

    print("Collecting data...")
    while len(data) < num_examples:

        j = keys[i % num_sents]
        group = f[j]# group has the same interface as Equivalent_sentences_group
        sent_length = group["sents"].shape[1]

        if sent_length > min_length and sent_length < max_length:
            group_data = generate_training_instances(group, num_examples_per_group, i%num_sents)
            data.extend(group_data)
            pbar.update(len(group_data))
            sents_collected += 1

        i += 1

        if i % 1500 == 0 and i > 0:
            with open(output_filename, "wb") as f2:
                pickle.dump(data, f2)

    print("Collected {} instances from {} sentences".format(len(data), sents_collected))

    with open(output_filename, "wb") as f:
        pickle.dump(data, f)

    f.close()


def generate_training_instances(group: Equivalent_sentences_group, num_examples_per_group: int, sent_id, filter_func_words=True, decay_by_distance = True, sigma = 7):

    vecs, sents, content_idx = group["vecs"], group["sents"], group["content_indices"]
    group_size, sent_len = group.attrs["group_size"], group.attrs["sent_length"]

    data = []

    for ind in range(num_examples_per_group):

        i, j = np.random.choice(range(group_size), size = 2, replace = False)

        # sample word indices

        if not decay_by_distance:
            l, m = np.random.choice(range(sent_len), size = 2, replace = True)
        else:
            l = np.random.choice(range(sent_len))
            diff = int(np.random.randn()*sigma)
            m = max(0, min(sent_len-1, l + diff))

        if l > m:

            l,m = m,l

        #if (sents1[i,l] == sents1[j, l]) or (sents1[i,m] == sents1[j,m]): continue
        sent_i_str = " ".join(sents[i][:l]) + " *" + sents[i, l] + "* " + " ".join(sents[i][l + 1:m]) + " *" + sents[i, m] + "* " + " ".join(sents[i, m + 1:])
        sent_j_str = " ".join(sents[j][:l]) + " *" + sents[j, l] + "* " + " ".join(sents[j][l + 1:m]) + " *" + sents[j, m] + "* " + " ".join(sents[j, m + 1:])

        w1, w2 = vecs[i,l], vecs[j, m]
        w3, w4 = vecs[j, l], vecs[i, m]

        word1_is_function_word = l not in content_idx
        word2_is_function_word = m not in content_idx

        instance =  {"vecs": (w1, w2, w3, w4), "sent1":  sent_i_str, "sent2": sent_j_str, "indices": (l,m), "sent_id": sent_id, "word1_is_function": word1_is_function_word, "word2_is_function": word2_is_function_word}
        data.append(instance)

    return data


if __name__ == '__main__':

    #collect_data("sample.hdf5", 22000, 100)
    collect_data("../../data/interim/encoded_sents.150k.hdf5", 1700000, 25)