import model
from typing import Dict, List
import numpy as np
import utils
import tqdm
import random


class ModelRunner(object):

    def __init__(self, model: model.ModelInterface,
                 data_filename: str,
                 output_file: str,
                 persist=True):
                 
        self.data_filename = data_filename
        self.model = model
        self.output_file = output_file
        self.sentences = utils.read_sentences(self.data_filename)
        self.persist = persist

    def run(self):

        print("Running neural model...")
        
        N = len(self.sentences)

        with open(self.output_file, "w") as f:

            for i in tqdm.tqdm(range(N)):
            
                sents = [self.sentences[i]]
                vecs = self.model.run(sents)[0]
                sent_length = len(sents[0])
                sent_str = " ".join(sents[0])
                vec_str = "*".join([utils.to_string(v) for v in vecs])
                #print("writing")
                #print(vec_str)
                #exit()
                f.write(vec_str + "\t" + sent_str + "\n")
