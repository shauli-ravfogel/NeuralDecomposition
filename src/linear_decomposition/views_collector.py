import h5py
import typing
from typing import Dict, Tuple, List
import numpy as np
import tqdm
import pickle

Equivalent_sentences_group = typing.NamedTuple("equivalent_sentences",
                                               [('vecs', np.ndarray), ('sents', List[List[str]]),
                                                ("content_indices", List[int])])


class CollectorBase(object):

    def __init__(self, path, output_dir, view_size, method, exclude_function_words=True):
        """
                Parameters
                -------------------------
                Path: str, required.
                      The path to the dataset that contain equivalent sentences.
                      The file is HDF5 format. Each group (listed by a string index) is one set of equivalent sentences, and contains datasets "vecs" (ELMO representations of the words), "sents" and "content indices" (indices of content words)
                     (The foramt is described above in 'Equivalent_sentences_group'")
                """

        self.path = path
        self.output_dir = output_dir
        self.f = h5py.File(path, 'r')
        self.view_size = view_size
        self.exclude_function_words = exclude_function_words
        self.method = method

    def read_one_group(self, equivalent_sentences: Equivalent_sentences_group) \
            -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
                Parameters
                -----------------------
                equivalent_sentences: Equivalent_sentences_group, requiered.
                ----------------------
                Return
                        A list of tuples extracted from this group.
                        each tuples contains (view1_instance, view2_instance, index_in_the_sentence)
                """

        raise NotImplementedError

    def collect_views(self):
        pbar = tqdm.tqdm(total=self.view_size)
        views = []
        i = 0

        print("Collecting views...")

        while len(views) < self.view_size:
            group = self.f[str(i)]  # group has the same interface as Equivalent_sentences_group
            vecs, sents, content_idx = group["vecs"], group["sents"], group["content_indices"]
            group_size, sent_len = group.attrs["group_size"], group.attrs["sent_length"]
            group_data = self.read_one_group(vecs, sents, content_idx, sent_len, group_size)
            views.extend(group_data)
            pbar.update(len(group_data))
            i += 1

        print("Collected {} pairs from {} sentences".format(len(views), i))

        output_filename = self.output_dir \
                          + "/views.sentences:{}.pairs:{}.mode:{}.no-func-words:{}".format(i, len(views),
                                                                                           self.method,
                                                                                           self.exclude_function_words)

        with open(output_filename, "wb") as f:
            pickle.dump(views, f)

        self.close_file()

    def close_file(self):
        self.f.close()


class SimpleCollector(CollectorBase):

    def __init__(self, *args):

        super(SimpleCollector, self).__init__(*args)

    def read_one_group(self, vecs: np.ndarray, sents: np.ndarray, content_idx: np.ndarray, sent_len: int,
                       group_size: int) -> List[Tuple[np.ndarray, np.ndarray, int]]:

        view1 = vecs[:, :, :]  # (num sents, num_indices, 2048)
        view2 = vecs[1:, :, :]  # (num_sents-1, num_indices, 2048)
        view1_words = sents[:, :]  # (num_sents, sent_length)
        view2_words = sents[1:, :]  # (num_sents-1, sent_length

        data = []

        for word_index in range(sent_len):

            is_function_word = word_index not in content_idx
            if self.exclude_function_words and is_function_word: continue

            view1_vecs = view1[:, word_index, :]  # (group_size, 2048)
            view2_vecs = view2[:, word_index, :]  # (group_size - 1, 2048)
            view1_words_at_index = view1_words[:, word_index]  # (group_size,)
            view2_words_at_index = view2_words[:, word_index]  # (group_size - 1,)
            idx = [word_index] * min(view1_vecs.shape[0], view2_vecs.shape[0])
            examples = list(zip(view1_vecs, view2_vecs, view1_words_at_index, view2_words_at_index, idx))

            if self.exclude_function_words or (not is_function_word):
                data.extend(examples)
            else:  # function word & we include function words
                data.append(examples[0])  # don't append multiple occurrences of the same function word.

        return data


class AveragedCollector(CollectorBase):

    def __init__(self, *args):

        super(AveragedCollector, self).__init__(*args)

    def read_one_group(self, vecs: np.ndarray, sents: np.ndarray, content_idx: np.ndarray, sent_len: int,
                       group_size: int) -> List[Tuple[np.ndarray, np.ndarray, int]]:

        view1 = vecs[::2, :, :]  # (num sents/2 +-1, num_indices, 2048)
        view2 = vecs[1::2, :, :]  # (num_sents/2 +-1, num_indices, 2048)
        view1_words = sents[::2, :]  # (num_sents/2 +-1, sent_length)
        view2_words = sents[1::2, :]  # (num_sents/2 +- 1, sent_length

        data = []

        for word_index in range(sent_len):

            if self.exclude_function_words and (word_index not in content_idx): continue

            view1_vecs = view1[:, word_index, :]
            view2_vecs = view2[:, word_index, :]

            m1 = np.mean(view1_vecs, axis=0)[None, :]
            m2 = np.mean(view2_vecs, axis=0)[None, :]

            data.append((m1, m2, word_index))

        return data


class SentenceCollector(CollectorBase):

    def __init__(self, *args):

        super(SentenceCollector, self).__init__(*args)

    def read_one_group(self, vecs: np.ndarray, sents: np.ndarray, content_idx: np.ndarray, sent_len: int,
                       group_size: int) -> List[Tuple[np.ndarray, np.ndarray, int]]:

        view1 = vecs[:, :, :]  # (num sents, num_indices, 2048)
        view2 = vecs[1:, :, :]  # (num_sents-1, num_indices, 2048)
        view1_words = sents[:, :]  # (num_sents, sent_length)
        view2_words = sents[1:, :]  # (num_sents-1, sent_length

        data = []

        if self.exclude_function_words:
            view1, view2 = view1[:, content_idx, :], view2[:, content_idx, :]
            view1_words, view2_words = view1_words[:, content_idx], view2_words[:, content_idx]

        n = min(len(view1), len(view2))

        for i in range(n):
            # average over all positions in the ith pairs of sentences.
            m1 = np.mean(view1[i, :, :], axis=0)  # (2048,)
            m2 = np.mean(view2[i, :, :], axis=0)  # (2048,)
            data.append((m1, m2, -1))  # -1 since this is in the sentence level

        return data
