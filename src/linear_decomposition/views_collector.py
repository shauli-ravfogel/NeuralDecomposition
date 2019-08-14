import h5py
import typing
from typing import Dict, Tuple
import numpy as np
import tqdm

Equivalent_sentences_group = typing.NamedTuple("equivalent_sentences",
                                    [('vecs', np.ndarray), ('sents', List[List[str]]),
                                     ("content_indices", List[int])])
                                     
class CollectorBase(object):

        def __init__(self, path, view_size, output_filename, exclude_function_words = True):
        
                """
                Parameters
                -------------------------
                Path: str, required.
                      The path to the dataset that contain equivalent sentences.
                      The file is HDF5 format. Each group (listed by a string index) is one set of equivalent sentences, and contains datasets "vecs" (ELMO representations of the words), "sents" and "content indices" (indices of content words)
                     (The foramt is described above in 'Equivalent_sentences_group'")
                """
        
                self.path = path
                self.f = h5py.File(path, 'r')
                self.view_size = view_size
                self.output_filename = output_filename
                self.exclude_function_words = exclude_function_words
        
        def read_one_group(equivalent_sentences: Equivalent_sentences_group) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        
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
        
                pbar = tqdm.tqdm(total = self.view_size)
                views = []
                i = 0
                
                print("Collecting views...")
                
                while len(views) < self.view_size:
               
                        group = self.f[str(i)] # group has the same interface as Equivalent_sentences_group
                        group_data = self.read_one_group(group)
                        views.extend(group_data)
                        pbar.update(1)
                        i += 1
        
        
                print("Collected {} instances from {} sentences".format(len(views), i))
                
                with open("output_filename", "wb") as f:
                
                        pickle.dump(views, f)
                
        def close_file(self):
        
                self.f.close()










class SimpleCollector(CollectorBase):

        def __init__(self, path, view_size, output_filename, exclude_function_words):
        
        
                super(SimpleCollector, self).__init__(path, view_size, output_filename, exclude_function_words)
                
        def read_one_group(equivalent_sentences: Equivalent_sentences_group) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        
                vecs, sents = equivalent_sentences["vecs"], equivalent_sentences["vecs"
                group_size, sent_length = equivalent_sentences.attrs["group_size"], equivalent_sentences.attrs["sent_length"]
                
                if self.exclude_function_words:
                
                        indices = equivalent_sentences["content_indices"]
                        vecs = vecs[:, indices, :]
                        sents = sents[:, indices]
                else:
                
                        indices = np.arange(start = 0, stop = sent_length, dtype = int)
                
                view1 = vecs[::2, indices, :]  #(num sents/2+-1, num_indices, 2048)
                view2 = vecs[1::2, indices, :] # (num_sents/2+-1, num_indices, 2048)
                view1_words = sents[::2, indices] #(num_sents/2+-1, sent_length)
                view2_words = sents[1::2, indices] # (num_sents/2+-1, sent_length
                
                assert view1.shape == view1_words.shape
                assert view2.shape == view2_words.shape
                
                data = []
                
                for word_index in range(sent_length):
                      
                                
                      
        
        
        
        
        
        
        
        
        
        
        
        
        
                
               
