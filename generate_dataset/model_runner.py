import model
from typing import Dict, List
import numpy as np
import utils
import tqdm
import random

class ModelRunner(object):

	def __init__(self, model: model.ModelInterface, equivalent_sentences_dict: Dict[int, List[List[str]]]):
	
		self.model = model
		self.equivalent_sentences_dict = equivalent_sentences_dict
	
	def run(self, num_examples_per_sentence = 4, num_equivalents = 5, num_indices = 1):
	
		print("Running neural model on equivalent sentences...")
		
		N = len(self.equivalent_sentences_dict)
		
		with open(utils.DEFAULT_PARAMS["dataset_filename"], "w") as f:
		
			for i in tqdm.tqdm(range(N)):
			
				equivalent_sentences = self.equivalent_sentences_dict[i][:num_equivalents]
				vecs = self.model.run(equivalent_sentences)
			
				sent_length = len(equivalent_sentences[0])
			
				# Create positive examples
			
				for j in range(num_examples_per_sentence):
			
					indices = np.random.choice(range(sent_length), size = num_indices)
					sent1_ind, sent2_ind = np.random.choice(range(num_equivalents), size = 2)
					sent1_vecs, sent2_vecs = vecs[sent1_ind][indices], vecs[sent2_ind][indices]
				
					sent1_str, sent2_str = " ".join(equivalent_sentences[sent1_ind]), " ".join(equivalent_sentences[sent2_ind])
					sent1_vecs_str = "*".join([utils.to_string(v) for v in sent1_vecs])
					sent2_vecs_str = "*".join([utils.to_string(v) for v in sent2_vecs])
					to_write = [utils.to_string(indices), sent1_str, sent2_str, sent1_vecs_str, sent2_vecs_str, "1"]
					f.write("\t".join(to_write) + "\n")
					
				# Create negative examples
				
				equivalent_sentences2 = random.choice(list(self.equivalent_sentences_dict.values()))
				vecs2 = self.model.run(equivalent_sentences2)
				sent2_length  = len(equivalent_sentences2[0])
				
				for j in range(num_examples_per_sentence):
					
					max_length = min(sent_length, sent2_length)
					
					indices = np.random.choice(range(max_length), size = num_indices)
					sent1_ind, sent2_ind = np.random.choice(range(num_equivalents)), np.random.choice(range(num_equivalents))
					sent1_vecs, sent2_vecs = vecs[sent1_ind][indices], vecs2[sent2_ind][indices]
				
					sent1_str, sent2_str = " ".join(equivalent_sentences[sent1_ind]), " ".join(equivalent_sentences2[sent2_ind])
					sent1_vecs_str = "*".join([utils.to_string(v) for v in sent1_vecs])
					sent2_vecs_str = "*".join([utils.to_string(v) for v in sent2_vecs])
					to_write = [utils.to_string(indices), sent1_str, sent2_str, sent1_vecs_str, sent2_vecs_str, "0"]
					f.write("\t".join(to_write) + "\n")				
				
