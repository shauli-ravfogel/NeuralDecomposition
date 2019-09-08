import model
from typing import Dict, List
import numpy as np
import utils
import tqdm
import random
import h5py

FUNCTION_WORDS = utils.DEFAULT_PARAMS['function_words']

class ModelRunner(object):

    def __init__(self, model: model.ModelInterface,
                 equivalent_sentences_dict: Dict[int, List[List[str]]],
                 output_file: str,
                 persist=True):

        self.model = model
        self.equivalent_sentences_dict = equivalent_sentences_dict
        self.output_file = output_file
        self.persist = persist

    def run(self):

        print("Running neural model on equivalent sentences...")

        print(type(self.equivalent_sentences_dict.items()))
        
        with h5py.File(self.output_file, 'w') as h5:
                for i, group_of_equivalent_sentences in tqdm.tqdm(enumerate(self.equivalent_sentences_dict.values()), ascii = True):
                        vecs = self.model.run(group_of_equivalent_sentences)

                        L = len(group_of_equivalent_sentences[0])  # group's sentence length
                        content_indices = np.array([i for i in range(L) if group_of_equivalent_sentences[0][i] not in FUNCTION_WORDS])
                        sents = np.array(group_of_equivalent_sentences, dtype=object)

                        # data.append(bert_states)
                        g = h5.create_group(str(i))
                        g.attrs['group_size'], g.attrs['sent_length'] = sents.shape
                        g.create_dataset('vecs', data=vecs, compression=True, chunks=True)
                        dt = h5py.special_dtype(vlen=str)
                        g.create_dataset('sents', data=sents, dtype=dt, compression=True, chunks=True)
                        g.create_dataset('content_indices', data=content_indices, compression=True, chunks=True)
                







               
class TuplesModelRunner(object):

    def __init__(self, model: model.ModelInterface,
                 equivalent_sentences_dict: Dict[int, List[List[str]]],
                 output_file: str,
                 persist=True):

        self.model = model
        self.equivalent_sentences_dict = equivalent_sentences_dict
        self.output_file = output_file
        self.persist = persist

    def run(self, num_examples_per_sentence=4, num_equivalents=5, num_indices=1):

        print("Running neural model on equivalent sentences...")

        N = len(self.equivalent_sentences_dict)

        with open(self.output_file, "w") as f:

            for i in tqdm.tqdm(range(N)):

                equivalent_sentences = self.equivalent_sentences_dict[i][:num_equivalents]
                vecs = self.model.run(equivalent_sentences)

                sent_length = len(equivalent_sentences[0])

                # Create positive examples

                for j in range(num_examples_per_sentence):
                    indices = np.random.choice(range(sent_length), size=num_indices)
                    sent1_ind, sent2_ind = np.random.choice(range(num_equivalents), size=2, replace = False)
                    sent1_vecs, sent2_vecs = vecs[sent1_ind][indices], vecs[sent2_ind][indices]

                    sent1_str, sent2_str = " ".join(equivalent_sentences[sent1_ind]), " ".join(
                        equivalent_sentences[sent2_ind])
                    sent1_vecs_str = "*".join([utils.to_string(v) for v in sent1_vecs])
                    sent2_vecs_str = "*".join([utils.to_string(v) for v in sent2_vecs])
                    to_write = [utils.to_string(indices), sent1_str, sent2_str, sent1_vecs_str, sent2_vecs_str, "1"]
                    if self.persist:
                        f.write("\t".join(to_write) + "\n")

                # Create negative examples

                equivalent_sentences2 = random.choice(list(self.equivalent_sentences_dict.values()))
                vecs2 = self.model.run(equivalent_sentences2)
                sent2_length = len(equivalent_sentences2[0])

                for j in range(num_examples_per_sentence):
                    max_length = min(sent_length, sent2_length)

                    indices = np.random.choice(range(max_length), size=num_indices)
                    sent1_ind, sent2_ind = np.random.choice(range(num_equivalents)), np.random.choice(
                        range(num_equivalents))
                    sent1_vecs, sent2_vecs = vecs[sent1_ind][indices], vecs2[sent2_ind][indices]

                    sent1_str, sent2_str = " ".join(equivalent_sentences[sent1_ind]), " ".join(
                        equivalent_sentences2[sent2_ind])
                    sent1_vecs_str = "*".join([utils.to_string(v) for v in sent1_vecs])
                    sent2_vecs_str = "*".join([utils.to_string(v) for v in sent2_vecs])
                    to_write = [utils.to_string(indices), sent1_str, sent2_str, sent1_vecs_str, sent2_vecs_str, "0"]
                    if self.persist:
                        f.write("\t".join(to_write) + "\n")
