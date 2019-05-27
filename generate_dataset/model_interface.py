from typing import List
import numpy as np

class ModelInterface(object):

	"""
	An interface for a wrapper that runs a neural model and outputs
	contextualized representation of each token.
	"""
	
	def __init__(self):
	
		pass
	
	def run(self, sents: List[List[str]]) -> List[np.ndarray]:
	
		"""
		sents: a list of N equivalent sentences, each is a list of len_sentence words
		return: a list of size N. each element is a numpy array of size len_sentence X representation size
		"""
	
		raise NotImplementedError
