
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

MODE = "words"

def collect_data(path, num_examples, num_examples_per_group,  min_length = 12, max_length = 35):

    f = h5py.File(path, 'r')
    pbar = tqdm.tqdm(total=num_examples, ascii = True)
    data = []
    i = 0
    write_freq = 600000
    num_sents = 132000
    keys = list(f.keys())
    output_filename = "elmo.words.30.bag_of_words."
    sents_collected = 0
    write_count =  0
    total_collected = 0

    print("Collecting data...")
    while total_collected < num_examples:

        j = keys[i % num_sents]
        group = f[j]# group has the same interface as Equivalent_sentences_group
        sent_length = group["sents"].shape[1]
        if len(data) % write_freq == 0 and len(data) > 0:

            with open(output_filename + str(write_count) + ".pickle", "wb") as f2:
                pickle.dump(data, f2)
                write_count += 1
                data = []

        if sent_length > min_length and sent_length < max_length:
            group_data = generate_training_instances(group, num_examples_per_group, i%num_sents)
            #print(len(group_data))
            data.extend(group_data)
            if MODE == "word":
                pbar.update(len(group_data))
            else:
                #pbar.update(group_data[0]["vecs"][0].shape[0] * num_examples_per_group)
                pbar.update(len(group_data))
                total_collected += len(group_data)

            sents_collected += 1

        i += 1

        if i % 5000 == 0 and i > 0 and MODE == "word":
            with open(output_filename+".pickle", "wb") as f2:
                pickle.dump(data, f2)

    print("Collected {} instances from {} sentences".format(len(data), sents_collected))

    with open(output_filename+str(write_count)+".pickle", "wb") as f:
        pickle.dump(data, f)

    f.close()


def generate_training_instances(group: Equivalent_sentences_group, num_examples_per_group: int, sent_id, filter_func_words=True, decay_by_distance = True, sigma = 12):

    vecs, sents, content_idx = group["vecs"], group["sents"], group["content_indices"]
    group_size, sent_len = group.attrs["group_size"], group.attrs["sent_length"]

    data = []

    if MODE == "words":
        for ind in range(num_examples_per_group):

            if np.random.random() < 0.3:

                i = 0
                j = np.random.choice(range(1, group_size))

            else:

                i,j = np.random.choice(range(group_size), size = 2, replace = False)
             
            # sample word indices

            if not decay_by_distance:
                l, m = np.random.choice(range(sent_len), size = 2, replace = True)
            else:
                l = np.random.choice(range(sent_len))
                diff = int(np.random.randn()*sigma)
                if diff == 0:
                  diff = 1 if np.random.rand() < 0.5 else -1

                #m = max(0, min(sent_len-1, l + diff))
                if (l + diff > sent_len -1) or (l + diff < 0):
                    diff *= -1

                if (sent_len - 1 < l + diff) or (0 > l + diff):

                    l,m = np.random.choice(range(sent_len), replace = False, size = 2)
                else:
                    m = l + diff 

            if l > m:

                l,m = m,l

            #if (sents1[i,l] == sents1[j, l]) or (sents1[i,m] == sents1[j,m]): continue
            sent_i_str = " ".join(sents[i][:l]) + " *" + sents[i, l] + "* " + " ".join(sents[i][l + 1:m]) + " *" + sents[i, m] + "* " + " ".join(sents[i, m + 1:])
            sent_j_str = " ".join(sents[j][:l]) + " *" + sents[j, l] + "* " + " ".join(sents[j][l + 1:m]) + " *" + sents[j, m] + "* " + " ".join(sents[j, m + 1:])

            w1, w2 = vecs[i,l][1024:], vecs[j, m][1024:]
            w3, w4 = vecs[j, l][1024:], vecs[i, m][1024:]
            sent1_bag_of_words = np.mean(vecs[i][:, :1024], axis = 0)
            sent2_bag_of_words = np.mean(vecs[j][:, :1024], axis = 0)
            sent1_bag_content = np.mean(vecs[i, content_idx[...], :1024], axis = 0)
            sent2_bag_content = np.mean(vecs[j, content_idx[...], :1024], axis = 0)
            #print(sent1_bag_of_words.shape, sent1_bag_content.shape)
            #exit()

            word1_is_function_word = l not in content_idx
            word2_is_function_word = m not in content_idx
            words = (sents[i][l], sents[j][m], sents[j,l], sents[i,m])

            instance =  {"vecs": (w1, w2, w3, w4), "sent1":  sent_i_str, "sent2": sent_j_str, "sent1_mean": sent1_bag_of_words, "sent2_mean": sent2_bag_of_words, "sent1_mean_content": sent1_bag_content, "sent2_mean_content": sent2_bag_content, "words": words, "indices": (l,m), "sent_len": sent_len, "sent_id": sent_id, "word1_is_function": word1_is_function_word, "word2_is_function": word2_is_function_word}
            data.append(instance)
    else:

        for ind in range(num_examples_per_group):
            i, j = np.random.choice(range(group_size), size=2, replace=False)
            vecs_i, vecs_j = vecs[i][...], vecs[j][...]
            sent_i_str, sent_j_str = " ".join(sents[i][...]), " ".join(sents[j][...])
            instance = {"vecs": (vecs_i, vecs_j), "sent1": sent_i_str, "sent2": sent_j_str, "sent_id": sent_id, "content_idx": content_idx[...], "sent_length": sent_len}
            #print(sent_i_str)
            #print(sent_j_str)
            #print("======================")
            data.append(instance)

    return data


if __name__ == '__main__':

    #collect_data("sample.hdf5", 7, 3)
    #collect_data("../../data/interim/elmo_states.wiki.hdf5", 5000000, 30)
    collect_data("../../data/interim/elmo_states.all_layers.wiki.hdf5", 2400000, 30)
