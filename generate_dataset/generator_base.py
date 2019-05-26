from typing import List, Tuple, Dict, DefaultDict
import utils
import pickle
import tqdm

class EquivalentSentencesGenerator:

	"""
	An abstract class for generating syntactically-equivalent sentenes, that is, sentences sharing 
	the same underlying hierarchical structure (but possibly different semantics). 
	"""

	def __init__(self, data_filename: str, num_sentences: int):
	
		self.sentences = utils.read_sentences(data_filename)
		self.num_sentences = num_sentences
	
	def generate(self) -> Dict[int, List[List[str]]]:
	
		sents_dict = dict()
		print("Generating equivalent sentences...")
		
		for i, sentence in tqdm.tqdm(enumerate(self.sentences), total = len(self.sentences)):
		
			equivalent_sentences = self.get_equivalent_sentences(sentence)
			sents_dict[i] = equivalent_sentences

		pickleFile = open(utils.DEFAULT_PARAMS["sentences_dict_filename"], 'wb')
		pickle.dump(sents_dict, pickleFile)
			
		return sents_dict

	def get_equivalent_sentences(self, original_sentence: List[str]) -> List[List[str]]:

		raise NotImplementedError()


